import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from timm.models.vision_transformer import Block as ViTBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from utils.calibrator import QuantCalibrator
from quantizers._ste import *
from quant_layers import *
from types import MethodType
import logging
import math


def mlp_forward(self, x):
    """Forward for MLP sub-module with perturbation support for Fisher estimation."""
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-6
    return x


def vit_block_forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
    """ViT Block forward that routes through separate MLP forward for Fisher estimation."""
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def swin_block_forward_mlp(self, x):
    """Swin Block forward with separate MLP routing."""
    B, H, W, C = x.shape
    shortcut = x
    x = self.norm1(x)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = shortcut + self.drop_path(x)
    x = x.reshape(B, -1, C)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C)
    return x


def positive_percentile(tensor, pct):
    """Compute the pct-th percentile of positive values in tensor (for GELU clamping)."""
    mini_batch_size = 1
    tensor_too_large = True
    while tensor_too_large:
        try:
            t = tensor.view(mini_batch_size, -1)[0:1, :]
            t = t.view(-1)
            positive_mask = t > 0
            positive_tensor = torch.where(positive_mask, t, torch.tensor(float('nan')).to(t.device))
            sorted_tensor, _ = positive_tensor.sort(dim=0)
            tensor_too_large = False
        except:
            mini_batch_size *= 2
    counts = (~torch.isnan(sorted_tensor)).sum(dim=0, keepdim=True).float()
    ranks = ((counts * pct).ceil().long() - 1).clamp(min=0)
    result = torch.gather(sorted_tensor, 0, ranks).squeeze()
    return result.item()


class MLPReconstructor(QuantCalibrator):
    """
    Fisher-guided MLP Reconstruction for Vision Transformer PTQ.

    Replaces the perturbation Hessian from APHQ-ViT with Fisher Information Matrix
    (FIM) guidance. The same Fisher framework is shared with BlockReconstructor's
    DPLR-FIM loss for a unified FIM-guided PTQ pipeline.

    Args:
        model: model to reconstruct (GELU replaced with ReLU in MLP blocks)
        full_model: FP32 reference model (keeps original GELU)
        calib_loader: calibration data loader
        metric: Fisher metric ('fisher_diag', 'fisher_dplr', 'fisher_lr', 'mse', 'mae')
        temp: temperature for softmax in KL divergence
        k: rank of Fisher for low-rank approximation
        p1: weight for low-rank Fisher term
        p2: weight for diagonal Fisher term
        use_mean_hessian: whether to average Fisher gradient across batch
    """
    def __init__(self, model, full_model, calib_loader,
                 metric="fisher_diag", temp=20, k=1, p1=1.0, p2=1.0,
                 use_mean_hessian=True):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.metric = metric
        self.temperature = temp
        self.k = k
        self.p1 = p1
        self.p2 = p2
        self.use_mean_hessian = use_mean_hessian
        self.raw_pred_softmaxs = None
        self.blocks = {}
        self.full_blocks = {}

        # Collect ViT/Swin blocks (same types as BlockReconstructor)
        types_of_block = [
            timm.layers.patch_embed.PatchEmbed,
            timm.models.vision_transformer.Block,
            timm.models.swin_transformer.SwinTransformerBlock,
            timm.models.swin_transformer.PatchMerging,
        ]
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in types_of_block):
                self.blocks[name] = module
                self._prepare_module_data_init(module)
        for name, module in self.full_model.named_modules():
            if any(isinstance(module, t) for t in types_of_block):
                self.full_blocks[name] = module
                self._prepare_module_data_init(module)

    def _prepare_module_data_init(self, module):
        """Initialize data buffers on blocks for Fisher estimation."""
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.perturb_u = module.perturb_d = False
        # Replace MLP forward with perturbation-capable version
        if hasattr(module, 'mlp'):
            module.mlp.raw_input = module.mlp.tmp_input = None
            module.mlp.raw_out = module.mlp.tmp_out = None
            module.mlp.raw_grad = module.mlp.tmp_grad = None
            module.mlp.perturb_u = module.mlp.perturb_d = False
            module.mlp.forward = MethodType(mlp_forward, module.mlp)
            # Initialize fc1/fc2 input caches
            if hasattr(module.mlp, 'fc1'):
                module.mlp.fc1.raw_input = module.mlp.fc1.tmp_input = None
            if hasattr(module.mlp, 'fc2'):
                module.mlp.fc2.raw_input = module.mlp.fc2.tmp_input = None

    def _compute_raw_pred_softmaxs(self, device):
        """Compute softmax outputs of the FP model on calibration data.
        Cached for sharing with BlockReconstructor."""
        if self.raw_pred_softmaxs is not None:
            return
        logging.info('Computing raw_pred_softmaxs (shared Fisher base) ...')
        self.raw_pred_softmaxs = []
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.full_model(inp) / self.temperature
                raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                self.raw_pred_softmaxs.append(raw_pred_softmax)
            torch.cuda.empty_cache()

    def init_block_raw_data(self, full_block, device):
        """Collect raw inputs/outputs from the FP model's MLP."""
        logging.info('MLPRecon: initializing raw data ...')
        hooks = []
        hooks.append(full_block.mlp.register_forward_hook(self.outp_forward_hook))
        hooks.append(full_block.mlp.fc1.register_forward_hook(self.single_input_forward_hook))
        hooks.append(full_block.mlp.fc2.register_forward_hook(self.single_input_forward_hook))

        self._compute_raw_pred_softmaxs(device)

        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                self.full_model(inp) / self.temperature
            torch.cuda.empty_cache()

        full_block.mlp.raw_out = torch.cat(full_block.mlp.tmp_out, dim=0)
        full_block.mlp.fc1.raw_input = torch.cat(full_block.mlp.fc1.tmp_input, dim=0)
        full_block.mlp.fc2.raw_input = torch.cat(full_block.mlp.fc2.tmp_input, dim=0)
        full_block.mlp.fc1.tmp_input = full_block.mlp.fc2.tmp_input = full_block.mlp.tmp_out = None
        for hook in hooks:
            hook.remove()

    def init_block_fisher(self, block, full_block, device):
        """Compute Fisher Information at MLP output using KL divergence backward.

        This replaces APHQ's perturbation-based Hessian with principled Fisher estimation.
        For fisher_diag: one backward pass, grad.abs().mean() as importance.
        For fisher_dplr/lr: accumulate k gradient samples for low-rank approximation.
        """
        if self.metric in ['fisher_diag', 'fisher_dplr', 'fisher_lr']:
            # Compute full model prediction softmaxs (shared across blocks)
            self._compute_raw_pred_softmaxs(device)

            if self.metric == 'fisher_diag':
                self._compute_fisher_diag(full_block, device)
            elif self.metric in ['fisher_dplr', 'fisher_lr']:
                self._compute_fisher_lowrank(full_block, device)

            # Copy Fisher data from full_block to quantized block
            block.mlp.raw_grad = full_block.mlp.raw_grad.to(device)
            block.mlp.delta_out = full_block.mlp.delta_out.to(device) if full_block.mlp.delta_out is not None else None
            block.mlp.inverse_B = full_block.mlp.inverse_B if full_block.mlp.inverse_B is not None else None

            # Normalize gradient (same as APHQ's normalization)
            if self.use_mean_hessian:
                block.mlp.raw_grad = block.mlp.raw_grad * torch.sqrt(
                    block.mlp.raw_grad.numel() / block.mlp.raw_grad.pow(2).sum()
                )
        elif self.metric in ['mse', 'mae']:
            block.mlp.raw_grad = None
        else:
            raise NotImplementedError(f'Unknown metric: {self.metric}')

    def _compute_fisher_diag(self, full_block, device):
        """Compute diagonal Fisher: single backward pass, |grad| as importance."""
        logging.info('MLPRecon: computing diagonal Fisher ...')
        hook = full_block.mlp.register_full_backward_hook(self.grad_hook)
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            self.full_model.zero_grad()
            inp = inp.to(device)
            pred = self.full_model(inp) / self.temperature
            loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                self.raw_pred_softmaxs[i],
                reduction="batchmean"
            )
            loss.backward()
            torch.cuda.empty_cache()
        raw_grad = torch.cat(full_block.mlp.tmp_grad, dim=0)
        full_block.mlp.tmp_grad = None
        hook.remove()
        # Diagonal Fisher: flatten to (1, N*C) for loss consistency with low-rank mode
        raw_grad = raw_grad.abs().reshape(raw_grad.shape[0], -1)
        full_block.mlp.raw_grad = raw_grad.mean(dim=0, keepdim=True) if self.use_mean_hessian else raw_grad
        full_block.mlp.delta_out = None
        full_block.mlp.inverse_B = None
        del raw_grad
        torch.cuda.empty_cache()

    def _compute_fisher_lowrank(self, full_block, device):
        """Compute low-rank Fisher: accumulate k gradient/delta_out samples.

        For MLP reconstruction, delta_out is computed by comparing MLP outputs
        with and without small perturbation, since quantization doesn't exist yet.
        """
        logging.info(f'MLPRecon: computing low-rank Fisher (k={self.k}) ...')
        full_block.mlp.raw_grad = None
        full_block.mlp.delta_out = None

        for step in range(self.k):
            # Alternate perturbation directions for delta_out estimation
            full_block.mlp.perturb_u = (step % 2 == 0)
            full_block.mlp.perturb_d = (step % 2 == 1)

            # Forward to get perturbed output
            hook_out = full_block.mlp.register_forward_hook(self.outp_forward_hook)
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                self.full_model(inp) / self.temperature
            perturbed_out = torch.cat(full_block.mlp.tmp_out, dim=0)
            full_block.mlp.tmp_out = None
            hook_out.remove()

            # Backward to get gradient
            hook_grad = full_block.mlp.register_full_backward_hook(self.grad_hook)
            for i, (inp, target) in enumerate(self.calib_loader):
                self.model.zero_grad()
                self.full_model.zero_grad()
                inp = inp.to(device)
                pred = self.full_model(inp) / self.temperature
                loss = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    self.raw_pred_softmaxs[i],
                    reduction="batchmean"
                )
                loss.backward()
            raw_grad = torch.cat(full_block.mlp.tmp_grad, dim=0)
            full_block.mlp.tmp_grad = None
            hook_grad.remove()

            full_block.mlp.perturb_u = full_block.mlp.perturb_d = False

            # Accumulate Fisher data
            raw_grad_flat = raw_grad.reshape(raw_grad.shape[0], -1).abs()
            raw_grad_flat = raw_grad_flat.mean(dim=0).unsqueeze(0)  # (1, N)
            delta_out = (perturbed_out - full_block.mlp.raw_out.to(device)).abs().mean(dim=0).reshape(1, -1)  # (1, N)

            if full_block.mlp.raw_grad is None:
                full_block.mlp.raw_grad = raw_grad_flat
                full_block.mlp.delta_out = delta_out
            else:
                full_block.mlp.raw_grad = torch.cat([full_block.mlp.raw_grad, raw_grad_flat], dim=0)  # (k, N)
                full_block.mlp.delta_out = torch.cat([full_block.mlp.delta_out, delta_out], dim=0)  # (k, N)

            del raw_grad, perturbed_out, raw_grad_flat, delta_out
            torch.cuda.empty_cache()

        # Compute inverse_B for DPLR loss
        full_block.mlp.inverse_B = torch.linalg.inv(
            full_block.mlp.delta_out.to(device) @ full_block.mlp.delta_out.transpose(1, 0).to(device)
        )  # (k, k)
        torch.cuda.empty_cache()

    def reconstruct_single_block(self, name, block, device, ub,
                                 batch_size: int = 32, iters: int = 20000,
                                 lr: float = 4e-5, p: float = 2.0):
        """Reconstruct MLP weights using Fisher-guided optimization."""
        w_params = []
        for _name, module in block.named_modules():
            if 'fc1' in _name or 'fc2' in _name or 'norm2' in _name:
                w_params += [module.weight, module.bias]
        w_optimizer = torch.optim.Adam(w_params, lr=lr)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=iters, eta_min=0.)
        loss_func = MLPLossFunction(
            block, weight=2.0, rec_loss=self.metric, max_count=iters,
            p=p, p1=self.p1, p2=self.p2
        )
        for i in range(iters):
            idx = torch.randperm(block.mlp.fc1.raw_input.size(0))[:batch_size]
            cur_inp = block.mlp.fc1.raw_input[idx].to(device)
            cur_out = block.mlp.raw_out[idx].to(device)

            if self.metric == 'fisher_diag':
                cur_grad = block.mlp.raw_grad
            elif self.metric in ['fisher_dplr', 'fisher_lr']:
                cur_grad = block.mlp.raw_grad
            else:
                cur_grad = None

            w_optimizer.zero_grad()
            recon_out = block.mlp(cur_inp)
            fc2_inp = block.mlp.act(block.mlp.fc1(cur_inp))
            fc2_quant_inp = torch.clamp(fc2_inp, 0, ub)
            quant_out = block.mlp.fc2(fc2_quant_inp)
            err = loss_func(
                recon_out, cur_out, cur_grad, quant_out,
                delta_out=block.mlp.delta_out,
                inverse_B=block.mlp.inverse_B
            )
            err.backward()
            w_optimizer.step()
            w_scheduler.step()

        del block.mlp.fc1.raw_input, block.mlp.raw_out, block.mlp.raw_grad
        torch.cuda.empty_cache()

    def reconstruct_model(self, pct=0.9999):
        """Reconstruct all MLP blocks using Fisher guidance."""
        device = next(self.model.parameters()).device
        for name, block in self.blocks.items():
            if not hasattr(block, 'mlp'):
                continue
            logging.info('MLPRecon: reconstructing {} ...'.format(name))
            full_block = self.full_blocks[name]

            # Step 1: Collect raw data from FP model
            self.init_block_raw_data(full_block, device)

            # Step 2: Transfer raw data to quantized model
            block.mlp.fc1.raw_input = full_block.mlp.fc1.raw_input.to(device)
            block.mlp.raw_out = full_block.mlp.raw_out.to(device)

            # Step 3: Compute Fisher gradient at MLP output
            self.init_block_fisher(block, full_block, device)
            del full_block.mlp.raw_grad, full_block.mlp.delta_out, full_block.mlp.inverse_B

            # Step 4: Compute upper bound for GELU activation clamping
            ub = positive_percentile(full_block.mlp.fc2.raw_input, pct=pct)
            del full_block.mlp.fc1.raw_input, full_block.mlp.fc2.raw_input, full_block.mlp.raw_out
            logging.info('MLPRecon: {} ub = {:.4f}'.format(name, ub))

            # Step 5: Fisher-guided optimization
            self.reconstruct_single_block(name, block, device, ub=ub)
            logging.info('MLPRecon: finished {}.'.format(name))

        # Return raw_pred_softmaxs for sharing with BlockReconstructor
        return self.raw_pred_softmaxs


class MLPLossFunction:
    """Loss function for Fisher-guided MLP Reconstruction.

    Supports: mse, mae, fisher_diag, fisher_lr, fisher_dplr.
    fisher_diag: ((pred-tgt)^2 * |grad|).mean()  (diagonal Fisher weighting)
    fisher_dplr: p1 * low_rank_term + p2 * diag_term  (same as FIMA-Q block recon)
    fisher_lr: low-rank Fisher weighted loss only
    """

    def __init__(self, block, weight: float = 2.0, rec_loss: str = 'mse',
                 max_count: int = 2000, p: float = 2., p1: float = 1., p2: float = 1.):
        self.block = block
        self.rec_loss = rec_loss
        self.weight = weight
        self.p = p
        self.p1 = p1
        self.p2 = p2
        self.count = 0
        self.init_loss_1 = None
        self.init_loss_2 = None

    def __call__(self, pred, tgt, grad=None, quant_out=None,
                 delta_out=None, inverse_B=None):
        self.count += 1

        if self.rec_loss == 'mse':
            rec_loss = self._lp_loss(pred, tgt, p=self.p) / 10
            quant_loss = self._lp_loss(quant_out, tgt, p=self.p) / 10
        elif self.rec_loss == 'mae':
            rec_loss = self._lp_loss(pred, tgt, p=1.0) / 10
            quant_loss = self._lp_loss(quant_out, tgt, p=1.0) / 10
        elif self.rec_loss == 'fisher_diag':
            # Diagonal Fisher: match APHQ pattern ((pred-tgt)^2 * |grad|).sum(1).mean() / 10
            # grad from _compute_fisher_diag is (1, N*C) — reshape to (1, N, C) for per-token weighting
            N, C = pred.shape[1], pred.shape[2]
            grad_reshaped = grad.abs().reshape(1, N, C)
            rec_loss = ((pred - tgt).pow(2) * grad_reshaped).sum(1).mean() / 10
            quant_loss = ((quant_out - tgt).pow(2) * grad_reshaped).sum(1).mean() / 10
        elif self.rec_loss == 'fisher_lr':
            # Low-rank Fisher: (N*C)-flattened grad, inverse_B-weighted
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            A = cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            loss_lr = (A @ inverse_B @ A.transpose(1, 2)).mean()
            if self.init_loss_1 is None:
                self.init_loss_1 = loss_lr.detach()
            rec_loss = loss_lr / self.init_loss_1 / 10

            quant_cha = (quant_out - tgt).abs().reshape(quant_out.shape[0], -1)
            A_q = quant_cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            quant_loss_lr = (A_q @ inverse_B @ A_q.transpose(1, 2)).mean()
            quant_loss = quant_loss_lr / self.init_loss_1 / 10
        elif self.rec_loss == 'fisher_dplr':
            # DPLR-FIM: p1 * low_rank + p2 * diagonal
            N, C = pred.shape[1], pred.shape[2]
            grad_reshaped = grad.abs().reshape(1, N, C)
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            A = cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            loss_lr = (A @ inverse_B @ A.transpose(1, 2)).mean()
            loss_diag = ((pred - tgt).pow(2) * grad_reshaped).sum(1).mean()
            if self.init_loss_1 is None:
                self.init_loss_1 = loss_lr.detach()
                self.init_loss_2 = loss_diag.detach()
            rec_loss = (self.p1 * loss_lr / self.init_loss_1 + self.p2 * loss_diag / self.init_loss_2) / 10

            quant_cha = (quant_out - tgt).abs().reshape(quant_out.shape[0], -1)
            A_q = quant_cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            quant_loss_lr = (A_q @ inverse_B @ A_q.transpose(1, 2)).mean()
            quant_loss_diag = ((quant_out - tgt).pow(2) * grad_reshaped).sum(1).mean()
            quant_loss = (self.p1 * quant_loss_lr / self.init_loss_1 + self.p2 * quant_loss_diag / self.init_loss_2) / 10
        else:
            raise ValueError(f'Not supported rec_loss: {self.rec_loss}')

        total_loss = rec_loss + quant_loss * self.weight
        if self.count == 1 or self.count % 500 == 0:
            print('MLP Total loss:\t{:.3f} (rec:{:.3f}, quant:{:.3f})\tcount={}'.format(
                float(total_loss), float(rec_loss), float(quant_loss), self.count))
        return total_loss

    @staticmethod
    def _lp_loss(pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).sum(1).mean()
