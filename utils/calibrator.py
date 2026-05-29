import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
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