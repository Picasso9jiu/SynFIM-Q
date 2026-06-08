import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear
import logging


class QuantCalibrator:
    def __init__(self, model, calib_loader, calib_metric="mse", temperature=20):
        self.model = model
        self.calib_loader = calib_loader
        self.calib_metric = calib_metric
        self.temperature = temperature
        self.raw_pred_softmaxs = None  # cached for sharing with BlockReconstructor

    def single_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())

    def double_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())

    def outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.cpu().detach())
    def outp_forward_hook2(self, module, inp, outp):
        module.tmp_out=outp

    def grad_hook(self, module, grad_input, grad_output):
        if module.tmp_grad is None:
            module.tmp_grad = []
        module.tmp_grad.append(grad_output[0].clone().cpu().detach())

    def _compute_all_fisher_grads(self, device):
        """Compute Fisher gradients for all quantized modules in one backward pass.

        Register backward hooks on all modules, run KL div backward,
        and collect Fisher gradients for Fisher-weighted calibration.
        This is efficient — one backward pass gives gradients for all modules.
        """
        logging.info('Calibrator: computing Fisher gradients for all modules ...')

        # Ensure all modules have tmp_grad initialized
        for name, module in self.model.named_modules():
            if hasattr(module, 'metric') and not module.calibrated:
                module.tmp_grad = []
                module.fisher_grad = None

        # Register backward hooks on all quantized modules
        grad_hooks = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'metric') and not module.calibrated:
                grad_hooks.append(module.register_full_backward_hook(self.grad_hook))

        # Compute KL divergence backward
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                self.raw_pred_softmaxs[i],
                reduction="batchmean"
            )
            loss.backward()
            torch.cuda.empty_cache()

        # Store Fisher gradients on each module
        for name, module in self.model.named_modules():
            if hasattr(module, 'metric') and not module.calibrated:
                if module.tmp_grad:
                    raw_grad = torch.cat(module.tmp_grad, dim=0)
                    # Diagonal Fisher: mean absolute gradient
                    module.fisher_grad = raw_grad.abs().mean(dim=0, keepdim=True)
                    module.tmp_grad = None
                else:
                    module.fisher_grad = None

        # Remove all hooks
        for hook in grad_hooks:
            hook.remove()
        torch.cuda.empty_cache()

    def _iter_reconstruction_blocks(self):
        block_types = (
            timm.layers.patch_embed.PatchEmbed,
            timm.models.vision_transformer.Block,
            timm.models.swin_transformer.SwinTransformerBlock,
            timm.models.swin_transformer.PatchMerging,
        )
        for name, module in self.model.named_modules():
            if any(isinstance(module, block_type) for block_type in block_types) or name.split('.')[-1] == 'head':
                yield name, module

    def _set_quant_module_mode(self, mode):
        previous_modes = {}
        for module in self.model.modules():
            if hasattr(module, 'mode'):
                previous_modes[module] = module.mode
                module.mode = mode
        return previous_modes

    @staticmethod
    def _restore_quant_module_mode(previous_modes):
        for module, mode in previous_modes.items():
            module.mode = mode

    def _set_quantizer_training_mode(self, enabled):
        previous_states = {}
        for module in self.model.modules():
            if hasattr(module, 'training_mode'):
                previous_states[module] = bool(module.training_mode)
                if enabled:
                    module.init_training()
                else:
                    module.end_training()
        return previous_states

    @staticmethod
    def _restore_quantizer_training_mode(previous_states):
        for module, enabled in previous_states.items():
            if enabled:
                module.init_training()
            else:
                module.end_training()

    def compute_block_residual_stats(self):
        """Measure the residual left by calibration for each reconstruction block.

        The statistic is only diagnostic/guidance data for BlockRecon. It does
        not update quantization parameters.
        """
        device = next(self.model.parameters()).device
        blocks = dict(self._iter_reconstruction_blocks())
        if not blocks:
            return {}

        logging.info('Calibrator: computing block calibration residual stats ...')

        raw_outputs = {name: [] for name in blocks}
        quant_outputs = {name: [] for name in blocks}
        quant_grads = {name: [] for name in blocks}
        quant_requires_grad = {name: [] for name in blocks}
        raw_softmaxs = []

        def capture_quant_output(block_name):
            def hook(module, inp, outp):
                quant_outputs[block_name].append(outp.detach().cpu())
                quant_requires_grad[block_name].append(bool(torch.is_tensor(outp) and outp.requires_grad))
                if torch.is_tensor(outp) and outp.requires_grad:
                    outp.register_hook(
                        lambda grad, name=block_name:
                            quant_grads[name].append(grad.detach().cpu())
                    )
            return hook

        raw_hooks = []
        previous_modes = self._set_quant_module_mode('raw')
        try:
            for name, block in blocks.items():
                raw_hooks.append(block.register_forward_hook(
                    lambda module, inp, outp, block_name=name:
                        raw_outputs[block_name].append(outp.detach().cpu())
                ))
            with torch.no_grad():
                for inp, _ in self.calib_loader:
                    inp = inp.to(device)
                    pred = self.model(inp) / self.temperature
                    raw_softmaxs.append(F.softmax(pred, dim=-1).detach())
        finally:
            for hook in raw_hooks:
                hook.remove()
            self._restore_quant_module_mode(previous_modes)

        quant_hooks = []
        previous_modes = self._set_quant_module_mode('quant_forward')
        previous_training_states = self._set_quantizer_training_mode(True)
        try:
            for name, block in blocks.items():
                quant_hooks.append(block.register_forward_hook(capture_quant_output(name)))
            for i, (inp, _) in enumerate(self.calib_loader):
                self.model.zero_grad(set_to_none=True)
                # Keep the image tensor in the graph so intermediate block
                # activations always expose gradients for residual sensitivity.
                inp = inp.to(device).detach().requires_grad_(True)
                pred = self.model(inp) / self.temperature
                loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_softmaxs[i], reduction="batchmean")
                loss.backward()
                torch.cuda.empty_cache()
        finally:
            for hook in quant_hooks:
                hook.remove()
            self._restore_quantizer_training_mode(previous_training_states)
            self._restore_quant_module_mode(previous_modes)

        stats = {}
        sensitivity_scores = []
        for name in blocks:
            if not raw_outputs[name] or not quant_outputs[name]:
                continue
            raw_out = torch.cat(raw_outputs[name], dim=0).float()
            quant_out = torch.cat(quant_outputs[name], dim=0).float()
            residual = quant_out - raw_out
            mse = residual.pow(2).mean()
            rel = mse.sqrt() / raw_out.std().clamp_min(1e-8)
            if quant_grads[name]:
                grad = torch.cat(quant_grads[name], dim=0).float().abs()
                grad_mean = grad.mean().clamp_min(1e-12)
            else:
                logging.warning(
                    'Calibrator residual {}: no block gradient captured; falling back to residual-only score.'.format(
                        name
                    )
                )
                grad_mean = torch.tensor(1e-12)
            score = rel * grad_mean.sqrt()
            stats[name] = {
                'mse': float(mse.item()),
                'rel': float(rel.item()),
                'fisher': float(grad_mean.item()),
                'score': float(score.item()),
                'grad_batches': len(quant_grads[name]),
                'requires_grad_batches': sum(quant_requires_grad[name]),
            }
            if 'patch_embed' not in name and name.split('.')[-1] != 'head':
                sensitivity_scores.append(score)

        if sensitivity_scores:
            ref_score = torch.stack(sensitivity_scores).median().clamp_min(1e-12)
        else:
            ref_score = torch.tensor(1.0)

        for name, stat in stats.items():
            norm = min(max(stat['score'] / float(ref_score.item()), 0.0), 4.0)
            stat['residual_norm'] = norm
            logging.info(
                'Calibrator residual {}: mse={:.6e}, rel={:.6e}, fisher={:.6e}, score={:.6e}, norm={:.3f}'.format(
                    name, stat['mse'], stat['rel'], stat['fisher'], stat['score'], stat['residual_norm']
                )
            )
            logging.info(
                'Calibrator residual {} grad capture: grad_batches={} requires_grad_batches={}'.format(
                    name, stat['grad_batches'], stat['requires_grad_batches']
                )
            )

        del raw_outputs, quant_outputs, quant_grads, quant_requires_grad, raw_softmaxs
        torch.cuda.empty_cache()
        return stats

    def batching_quant_calib(self, raw_pred_softmaxs=None):
        """Batch calibration with optional Fisher weighting.

        Args:
            raw_pred_softmaxs: pre-computed softmaxs from MLPReconstructor (shared Fisher).
        """
        device = next(self.model.parameters()).device

        # Compute or reuse raw_pred_softmaxs (needed for both standard and Fisher calibration)
        if raw_pred_softmaxs is not None:
            self.raw_pred_softmaxs = raw_pred_softmaxs
            logging.info('Calibrator: reusing shared raw_pred_softmaxs.')
        else:
            self.raw_pred_softmaxs = []
            with torch.no_grad():
                for inp, target in self.calib_loader:
                    inp = inp.to(device)
                    pred = self.model(inp)
                    raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
                torch.cuda.empty_cache()

        # Compute Fisher gradients if using Fisher-weighted calibration
        if self.calib_metric not in ['mse', 'mae']:
            self._compute_all_fisher_grads(device)

        total = sum(1 for name, module in self.model.named_modules() if hasattr(module, 'metric') and not module.calibrated)
        with tqdm(total=total) as progress_bar:
            for name, module in self.model.named_modules():
                if not hasattr(module, 'metric') or module.calibrated:
                    continue
                progress_bar.set_description(f"calibrating {name}")
                hooks = []
                hooks.append(module.register_forward_hook(self.outp_forward_hook))
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul):
                    hooks.append(module.register_forward_hook(self.double_input_forward_hook))
                for i, (inp, target) in enumerate(self.calib_loader):
                    self.model.zero_grad()
                    inp = inp.to(device)
                    pred = self.model(inp)
                torch.cuda.empty_cache()
                # replace cached raw_inputs, raw_outs
                module.raw_out = torch.cat(module.tmp_out, dim=0)
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                for hook in hooks:
                    hook.remove()
                module.tmp_input = module.tmp_out = None
                # run hyperparameter_searching with Fisher gradient if available
                with torch.no_grad():
                    fisher_grad = getattr(module, 'fisher_grad', None)
                    module.hyperparameter_searching(fisher_grad=fisher_grad)
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                    torch.cuda.empty_cache()
                progress_bar.update()
        # end calibration
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"

        # Clean up fisher gradients and cached softmaxs
        for name, module in self.model.named_modules():
            if hasattr(module, 'fisher_grad'):
                del module.fisher_grad
