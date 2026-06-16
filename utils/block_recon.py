import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from timm.models.swin_transformer import window_partition, window_reverse
from utils.calibrator import QuantCalibrator
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import *
from types import MethodType
import logging
import random
import copy
import re
import statistics


def patch_embed_forward(self, x):
    B, C, H, W = x.shape
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    else:
        x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


def vit_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
        
    return x



def swin_block_forward(self, x):
    B, H, W, C = x.shape
    shortcut = x
    x = self.norm1(x)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = shortcut + self.drop_path(x)
    x = x.reshape(B, -1, C)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


def swin_patchmerging_forward(self, x):
    B, H, W, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    x = self.norm(x)
    x = self.reduction(x)
    if self.perturb:
        rand_perturb = torch.empty_like(x, dtype=torch.float).uniform_(1, 2) * self.r
        x = x + rand_perturb
    return x


class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, optim_batch_size,calib_loader, metric="mse", temp=20, k=1,
                 dis_mode='q', p1=1., p2=1., adaptive_k=True, adaptive_p=True,
                 block_residual_stats=None, logit_guard=True, logit_guard_batches=None,
                 logit_guard_loader=None, adaptive_candidate_select=True,
                 adaptive_candidate_margin=0.003, w_bit=None, a_bit=None,
                 calib_metric=None,
                 adaptive_3bit_select_profile='safe'):
        super().__init__(model, calib_loader)
        self.batch_size = optim_batch_size
        self.metric = metric
        self.k = k
        self.dis_mode = dis_mode
        self.p1 = p1
        self.p2 = p2
        self.adaptive_k = adaptive_k
        self.adaptive_p = adaptive_p
        self.block_residual_stats = block_residual_stats or {}
        self.logit_guard = logit_guard
        self.logit_guard_batches = logit_guard_batches
        self.optim_logit_guard_loader = calib_loader
        self.heldout_logit_guard_loader = logit_guard_loader
        self.adaptive_candidate_select = adaptive_candidate_select
        self.adaptive_candidate_margin = adaptive_candidate_margin
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.calib_metric = calib_metric
        self.adaptive_3bit_select_profile = adaptive_3bit_select_profile
        self.auto_base_k, self.auto_base_p1, self.auto_base_p2 = self.compute_auto_base_params()
        if self.use_3bit_std_prior():
            logging.info(
                'Using 3bit std_prior profile: residual-aware adaptive k/p is kept, with a mild activation-std prior for MSE-Calib ablation.'
            )
        self.block_recon_summary = []
        self.blocks = {}
        self.quanted_blocks = []
        self.raw_pred_softmaxs = None
        self.temperature = temp
        types_of_block = [
            timm.layers.patch_embed.PatchEmbed,
            timm.models.vision_transformer.Block,
            timm.models.swin_transformer.SwinTransformerBlock,
            timm.models.swin_transformer.PatchMerging,
        ]
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'head':
                self.blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)

    @staticmethod
    def _clone_state_dict(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.quanted_input = module.quanted_out = None
        module.delta_out = module.inverse_B = None
        module.r=1e-6
        if isinstance(module, timm.layers.patch_embed.PatchEmbed):
            module.forward = MethodType(patch_embed_forward, module)
        elif isinstance(module, timm.models.vision_transformer.Block):
            module.forward = MethodType(vit_block_forward, module)
        elif isinstance(module, timm.models.swin_transformer.SwinTransformerBlock):
            module.forward = MethodType(swin_block_forward, module)
        elif isinstance(module, timm.models.swin_transformer.PatchMerging):
            module.forward = MethodType(swin_patchmerging_forward, module)
        module.perturb = False
                
    def set_block_mode(self, block, mode='raw'):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    @staticmethod
    def _block_index(name):
        m = re.search(r'blocks\.(\d+)', name)
        return int(m.group(1)) if m else None

    def compute_auto_base_params(self):
        if not self.block_residual_stats or not (self.adaptive_k or self.adaptive_p):
            return self.k, self.p1, self.p2

        block_stats = [
            stat for name, stat in self.block_residual_stats.items()
            if self._block_index(name) is not None
        ]
        if not block_stats:
            return self.k, self.p1, self.p2

        norms = [float(stat.get('residual_norm', 1.0)) for stat in block_stats]
        rels = [float(stat.get('rel', 0.0)) for stat in block_stats]
        median_norm = statistics.median(norms)
        high_norm_ratio = sum(norm >= 1.15 for norm in norms) / len(norms)
        severe_norm_ratio = sum(norm >= 1.35 for norm in norms) / len(norms)
        median_rel = statistics.median(rels)

        base_k = self.k
        if median_norm < 0.95 and high_norm_ratio < 0.20:
            base_k = max(self.k - 1, 4)
        elif median_norm >= 1.25 or severe_norm_ratio >= 0.25:
            base_k = min(self.k + 1, 6)
        base_k = max(4, min(6, base_k))

        base_p1 = self.p1
        base_p2 = self.p2
        if median_norm >= 1.20 or severe_norm_ratio >= 0.25:
            base_p1 *= 1.02
        if severe_norm_ratio >= 0.40:
            base_p1 *= 1.02
        base_p1 = max(0.98, min(1.06, base_p1))
        base_p2 = max(0.99, min(1.01, base_p2))

        logging.info(
            'Auto adaptive base: k={} p1={:.3f} p2={:.3f} '
            '(median_norm={:.3f}, high_norm_ratio={:.3f}, severe_norm_ratio={:.3f}, median_rel={:.3f})'.format(
                base_k, base_p1, base_p2, median_norm, high_norm_ratio, severe_norm_ratio, median_rel
            )
        )
        return base_k, base_p1, base_p2

    def get_block_k(self, name):
        return self.select_block_k(name)[0]

    def use_3bit_std_prior(self):
        return (
            (self.w_bit == 3 or self.a_bit == 3)
            and self.adaptive_3bit_select_profile == 'std_prior'
            and self.calib_metric in (None, 'mse', 'mae')
        )

    def select_block_k(self, name):
        """Select Fisher low-rank update count from calibration residuals."""
        if not self.adaptive_k:
            return self.k, 'adaptive_k_disabled'
        base_k = self.auto_base_k
        stat = self.block_residual_stats.get(name)
        if stat is not None:
            residual_norm = float(stat.get('residual_norm', 1.0))
            if 'patch_embed' in name:
                # Patch embedding is input-sensitive; SynFIM-Q keeps one extra
                # Fisher refresh here while still relying on the guard.
                return min(base_k + 1, 7), 'patch_embed_input_sensitive'
            if 'head' in name:
                return base_k, 'head_base'
            idx = self._block_index(name)
            block_k = base_k
            reason = 'base'
            early_threshold = 1.00 if self.use_3bit_std_prior() else 1.05
            mid_threshold = 1.10 if self.use_3bit_std_prior() else 1.15
            if idx is not None and idx <= 6 and residual_norm >= early_threshold:
                if self.use_3bit_std_prior() and idx in (2, 3, 4) and residual_norm >= 1.02:
                    block_k = min(base_k + 2, 7)
                    reason = 'std_prior_core_residual_plus2'
                    return block_k, reason
                block_k = min(base_k + 1, 7)
                reason = 'std_prior_high_residual_plus1' if self.use_3bit_std_prior() else 'high_residual_plus1'
            elif idx is not None and idx <= 8 and residual_norm >= mid_threshold:
                block_k = min(base_k + 1, 7)
                reason = 'std_prior_mid_high_residual_plus1' if self.use_3bit_std_prior() else 'mid_high_residual_plus1'
            return block_k, reason
        if 'patch_embed' in name:
            return min(self.k + 3, 12), 'fallback_patch_embed'
        m = re.search(r'blocks\.(\d+)', name)
        if m:
            idx = int(m.group(1))
            if idx <= 2:
                return min(self.k + 3, 12), 'fallback_early_plus3'
            elif idx <= 8:
                return self.k, 'fallback_mid_base'
            else:
                return self.k, 'fallback_tail_base'
        if 'head' in name:
            return max(self.k - 2, 1), 'fallback_head_minus2'
        return self.k, 'fallback_base'

    def apply_3bit_std_prior_p1p2(self, name, block, p1, p2):
        if not self.use_3bit_std_prior() or block.raw_out is None:
            return p1, p2
        block_std = block.raw_out.std().item()
        idx = self._block_index(name)
        if 'patch_embed' in name:
            p1 *= 0.99
            p2 *= 1.02
        elif idx is not None and idx <= 6:
            if block_std < 0.65:
                p1 *= 0.92
                p2 *= 1.12
            elif block_std < 0.90:
                p1 *= 0.95
                p2 *= 1.08
            elif block_std > 1.20:
                p1 *= 1.025
                p2 *= 0.995
        elif idx is not None and idx <= 8:
            if block_std < 0.95:
                p1 *= 0.97
                p2 *= 1.05
        elif idx is not None:
            p2 = min(p2, 1.01)
        return max(0.90, min(1.10, p1)), max(0.99, min(1.12, p2))

    def compute_adaptive_p1p2(self, name, block):
        """
        Select adaptive p1/p2 from residual statistics.

        The fixed candidate always uses self.p1/self.p2. The adaptive
        candidate starts from the auto base and only applies mild residual
        changes, because previous runs showed aggressive p2 changes hurt W4A4.
        """
        if not self.adaptive_p:
            return self.p1, self.p2
        base_p1, base_p2 = self.auto_base_p1, self.auto_base_p2
        stat = self.block_residual_stats.get(name)
        if stat is not None:
            residual_norm = float(stat.get('residual_norm', 1.0))
            idx = self._block_index(name)
            if 'patch_embed' in name:
                return self.apply_3bit_std_prior_p1p2(name, block, base_p1 * 0.98, base_p2)
            if 'head' in name or idx is None:
                return self.apply_3bit_std_prior_p1p2(name, block, base_p1, base_p2)
            if residual_norm >= 1.0:
                boost = min((residual_norm - 1.0) * 0.18, 0.07)
                if idx <= 6:
                    return self.apply_3bit_std_prior_p1p2(name, block, base_p1 * (1.0 + boost), base_p2)
                return self.apply_3bit_std_prior_p1p2(name, block, base_p1 * (1.0 + 0.5 * boost), base_p2)
            if idx >= 8:
                damp = min((1.0 - residual_norm) * 0.04, 0.025)
                return self.apply_3bit_std_prior_p1p2(name, block, base_p1 * (1.0 - damp), base_p2)
            return self.apply_3bit_std_prior_p1p2(name, block, base_p1, base_p2)
        if block.raw_out is None:
            return base_p1, base_p2
        block_std = block.raw_out.std().item()
        if block_std > 2.5:
            return base_p1 * 1.3, base_p2 * 0.7
        elif block_std > 1.0:
            return base_p1, base_p2
        else:
            return base_p1 * 0.7, base_p2 * 1.3

    def strong_3bit_candidate_params(self, name, adaptive_params):
        if not (self.w_bit == 3 or self.a_bit == 3):
            return None
        if self.adaptive_3bit_select_profile != 'safe_plus':
            return None
        idx = self._block_index(name)
        if idx is None:
            return None
        block_k, _reason, block_p1, block_p2 = adaptive_params
        if idx in (5, 6):
            return (
                min(block_k + 1, 7),
                'strong_mid_residual_plus',
                min(block_p1 * 1.035, 1.080),
                block_p2,
            )
        if idx == 8:
            return (
                block_k,
                'strong_late_p1_probe',
                min(max(block_p1 * 1.020, 1.000), 1.030),
                block_p2,
            )
        if idx == 10:
            return (
                block_k,
                'strong_tail_logit_probe',
                min(max(block_p1 * 1.025, 1.000), 1.030),
                block_p2,
            )
        return None

    def replace_block(self, target_block, new_block):
        self._replace_block_recursive(self.model, target_block, new_block)

    def _replace_block_recursive(self, model, target_block, new_block):
        for name, child in model.named_children():
            if child is target_block:
                setattr(model, name, new_block)
            else:
                self._replace_block_recursive(child, target_block, new_block)
                
    def wrap_quantizers_in_net(self, block, name):
        logging.info('wraping quantizers in {} ...'.format(name))
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.n_V, module.crb_rows, module.in_features), 
                                                           round_mode='learned_hard_sigmoid')
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.weight.shape[0], -1), 
                                                           round_mode='learned_hard_sigmoid')
                module.w_quantizer.soft_targets = True

    def set_block_soft_targets(self, block, soft_targets):
        for _, module in block.named_modules():
            if hasattr(module, 'w_quantizer') and hasattr(module.w_quantizer, 'soft_targets'):
                module.w_quantizer.soft_targets = soft_targets

    def estimate_block_mse(self, block, device, mode='qdrop', batch_size=128):
        eval_inp = block.raw_input
        if mode != 'rinp' and block.quanted_input is not None:
            eval_inp = block.quanted_input
        if eval_inp is None or block.raw_out is None:
            return None

        previous_modes = {}
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                previous_modes[module] = module.mode
                module.mode = 'quant_forward'

        loss_sum = 0.0
        elem_count = 0
        with torch.no_grad():
            for b_st in range(0, eval_inp.shape[0], batch_size):
                b_ed = min(eval_inp.shape[0], b_st + batch_size)
                cur_inp = eval_inp[b_st:b_ed].to(device)
                cur_out = block.raw_out[b_st:b_ed].to(device)
                pred = block(cur_inp)
                diff = (pred - cur_out).float()
                loss_sum += diff.pow(2).sum().item()
                elem_count += diff.numel()

        for module, mode_name in previous_modes.items():
            module.mode = mode_name
        torch.cuda.empty_cache()
        return loss_sum / max(elem_count, 1)

    def block_guard_threshold(self, name):
        idx = self._block_index(name)
        if idx is None:
            return 1.0
        if idx >= 10:
            return 0.995
        if idx >= 8:
            return 0.999
        return 1.0

    def block_loss_revert_threshold(self, name):
        idx = self._block_index(name)
        if idx is None:
            return None
        if idx >= 10:
            return 2.15
        return None

    def block_logit_guard_profile(self, name):
        idx = self._block_index(name)
        if idx is None:
            if 'patch_embed' in name or name.split('.')[-1] == 'head':
                return {
                    'ce_tol': 0.015,
                    'true_prob_tol': 0.006,
                    'margin_tol': 0.015,
                    'top1_drop_tol': 0.006,
                    'flip_tol': 8,
                }
            return {
                'ce_tol': 0.020,
                'true_prob_tol': 0.008,
                'margin_tol': 0.020,
                'top1_drop_tol': 0.010,
                'flip_tol': 10,
            }
        if idx == 6:
            return {
                'ce_tol': 0.060,
                'true_prob_tol': 0.018,
                'margin_tol': 0.025,
                'top1_drop_tol': 0.012,
                'flip_tol': 10,
            }
        if idx >= 9:
            return {
                'ce_tol': 0.015,
                'true_prob_tol': 0.006,
                'margin_tol': 0.015,
                'top1_drop_tol': 0.006,
                'flip_tol': 8,
            }
        if idx >= 6:
            return {
                'ce_tol': 0.025,
                'true_prob_tol': 0.010,
                'margin_tol': 0.020,
                'top1_drop_tol': 0.010,
                'flip_tol': 8,
            }
        return {
            'ce_tol': 0.025,
            'true_prob_tol': 0.010,
            'margin_tol': 0.020,
            'top1_drop_tol': 0.010,
            'flip_tol': 10,
        }

    def select_logit_guard_loader(self, name):
        if self.heldout_logit_guard_loader is None:
            return self.optim_logit_guard_loader, 'optim'
        idx = self._block_index(name)
        if idx is None:
            if 'patch_embed' in name:
                return self.optim_logit_guard_loader, 'optim'
            return self.heldout_logit_guard_loader, 'heldout'
        if idx >= 8:
            return self.heldout_logit_guard_loader, 'heldout'
        return self.optim_logit_guard_loader, 'optim'

    def evaluate_logit_guard(self, device, name=None):
        guard_loader, guard_source = self.select_logit_guard_loader(name or '')
        previous_modes = {}
        for _, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                previous_modes[module] = module.mode
                module.mode = 'quant_forward'

        total = 0
        top1_correct = 0
        ce_sum = 0.0
        true_prob_sum = 0.0
        margin_sum = 0.0
        pred_chunks = []
        correct_chunks = []
        with torch.no_grad():
            for batch_idx, (inp, target) in enumerate(guard_loader):
                if self.logit_guard_batches is not None and batch_idx >= self.logit_guard_batches:
                    break
                inp = inp.to(device)
                target = target.to(device)
                logits = self.model(inp)
                probs = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)
                correct = pred.eq(target)
                batch_size = target.numel()
                total += batch_size
                top1_correct += correct.sum().item()
                ce_sum += F.cross_entropy(logits, target, reduction='sum').item()
                true_prob = probs.gather(1, target.view(-1, 1)).squeeze(1)
                true_prob_sum += true_prob.sum().item()
                masked_logits = logits.clone()
                masked_logits.scatter_(1, target.view(-1, 1), float('-inf'))
                competitor = masked_logits.max(dim=-1).values
                margin_sum += (logits.gather(1, target.view(-1, 1)).squeeze(1) - competitor).sum().item()
                pred_chunks.append(pred.detach().cpu())
                correct_chunks.append(correct.detach().cpu())
                torch.cuda.empty_cache()

        for module, mode_name in previous_modes.items():
            module.mode = mode_name
        torch.cuda.empty_cache()

        total = max(total, 1)
        return {
            'total': total,
            'top1': top1_correct / total,
            'ce': ce_sum / total,
            'true_prob': true_prob_sum / total,
            'margin': margin_sum / total,
            'pred': torch.cat(pred_chunks, dim=0) if pred_chunks else torch.empty(0, dtype=torch.long),
            'correct': torch.cat(correct_chunks, dim=0) if correct_chunks else torch.empty(0, dtype=torch.bool),
            'source': guard_source,
        }

    @staticmethod
    def format_logit_guard_stats(stats):
        if stats is None:
            return 'n/a'
        return 'top1={:.4f}, ce={:.4f}, true_prob={:.4f}, margin={:.4f}, source={}'.format(
            stats['top1'], stats['ce'], stats['true_prob'], stats['margin'], stats.get('source', 'n/a')
        )

    def should_revert_by_logit_guard(self, name, before_stats, after_stats):
        if before_stats is None or after_stats is None:
            return False, 'disabled'
        profile = self.block_logit_guard_profile(name)
        top1_drop = before_stats['top1'] - after_stats['top1']
        ce_increase = after_stats['ce'] - before_stats['ce']
        true_prob_drop = before_stats['true_prob'] - after_stats['true_prob']
        margin_drop = before_stats['margin'] - after_stats['margin']
        wrong_to_right = 0
        right_to_wrong = 0
        if (
            before_stats['pred'].numel() == after_stats['pred'].numel()
            and before_stats['correct'].numel() == after_stats['correct'].numel()
        ):
            flips = before_stats['pred'].ne(after_stats['pred']).sum().item()
            right_to_wrong = (before_stats['correct'] & ~after_stats['correct']).sum().item()
            wrong_to_right = (~before_stats['correct'] & after_stats['correct']).sum().item()
        else:
            flips = 0

        ce_harm = ce_increase > profile['ce_tol']
        confidence_harm = true_prob_drop > profile['true_prob_tol'] and margin_drop > profile['margin_tol']
        top1_harm = (
            self.should_use_label_flip_guard(name)
            and top1_drop > profile['top1_drop_tol']
            and ce_increase > -0.005
            and true_prob_drop > -0.002
        )
        harmful_flips = right_to_wrong - wrong_to_right
        flip_harm = (
            self.should_use_label_flip_guard(name)
            and harmful_flips > profile['flip_tol']
            and ce_increase > -0.005
            and true_prob_drop > -0.002
        )

        reasons = []
        if top1_harm:
            reasons.append('top1_drop={:.4f}>{:.4f}'.format(top1_drop, profile['top1_drop_tol']))
        if ce_harm:
            reasons.append('ce_increase={:.4f}>{:.4f}'.format(ce_increase, profile['ce_tol']))
        if confidence_harm:
            reasons.append(
                'confidence_margin_drop={:.4f}/{:.4f}>{:.4f}/{:.4f}'.format(
                    true_prob_drop, margin_drop, profile['true_prob_tol'], profile['margin_tol']
                )
            )
        if flip_harm:
            reasons.append(
                'harmful_flips={} (right_to_wrong={}, wrong_to_right={}) > {}'.format(
                    harmful_flips, right_to_wrong, wrong_to_right, profile['flip_tol']
                )
            )
        if reasons:
            return True, '; '.join(reasons)
        return False, (
            'top1_drop={:.4f}, ce_increase={:.4f}, true_prob_drop={:.4f}, '
            'margin_drop={:.4f}, pred_flips={}, right_to_wrong={}, wrong_to_right={}'
        ).format(top1_drop, ce_increase, true_prob_drop, margin_drop, flips, right_to_wrong, wrong_to_right)

    def should_use_label_flip_guard(self, name):
        idx = self._block_index(name)
        return idx is None or idx >= 9

    def has_strong_logit_improvement(self, name, before_stats, after_stats):
        if before_stats is None or after_stats is None:
            return False
        idx = self._block_index(name)
        if idx is not None and idx >= 9:
            return False
        top1_gain = after_stats['top1'] - before_stats['top1']
        ce_gain = before_stats['ce'] - after_stats['ce']
        true_prob_gain = after_stats['true_prob'] - before_stats['true_prob']
        margin_gain = after_stats['margin'] - before_stats['margin']
        if (
            before_stats['correct'].numel() == after_stats['correct'].numel()
            and before_stats['correct'].numel() > 0
        ):
            right_to_wrong = (before_stats['correct'] & ~after_stats['correct']).sum().item()
            wrong_to_right = (~before_stats['correct'] & after_stats['correct']).sum().item()
        else:
            right_to_wrong = wrong_to_right = 0
        return (
            top1_gain >= 0.003
            and ce_gain >= 0.010
            and margin_gain >= 0.010
            and true_prob_gain >= -0.002
            and wrong_to_right >= right_to_wrong
        )

    def has_tail_logit_override(self, name, before_stats, after_stats):
        """Allow a narrow block.10 keep when classifier logits clearly improve."""
        if before_stats is None or after_stats is None:
            return False
        idx = self._block_index(name)
        if idx != 10:
            return False
        top1_drop = before_stats['top1'] - after_stats['top1']
        ce_gain = before_stats['ce'] - after_stats['ce']
        true_prob_gain = after_stats['true_prob'] - before_stats['true_prob']
        margin_gain = after_stats['margin'] - before_stats['margin']
        if (
            before_stats['correct'].numel() == after_stats['correct'].numel()
            and before_stats['correct'].numel() > 0
        ):
            right_to_wrong = (before_stats['correct'] & ~after_stats['correct']).sum().item()
            wrong_to_right = (~before_stats['correct'] & after_stats['correct']).sum().item()
        else:
            right_to_wrong = wrong_to_right = 0
        return (
            ce_gain >= 0.025
            and true_prob_gain >= 0.006
            and margin_gain >= 0.035
            and top1_drop <= 0.004
            and (right_to_wrong - wrong_to_right) <= 4
        )

    def should_reconstruct_block(self, name, block_start=None, block_end=None,
                                 skip_patch_embed=False, skip_head=False):
        if skip_patch_embed and 'patch_embed' in name:
            return False
        if skip_head and name.split('.')[-1] == 'head':
            return False
        idx = self._block_index(name)
        if idx is None:
            return True
        if block_start is not None and idx < block_start:
            return False
        if block_end is not None and idx > block_end:
            return False
        return True

    def set_qdrop(self, block, prob):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    if hasattr(module.a_quantizer, 'drop_prob'):
                        module.a_quantizer.drop_prob = prob
                elif isinstance(module, MinMaxQuantMatMul):
                    if hasattr(module.A_quantizer, 'drop_prob'):
                        module.A_quantizer.drop_prob = prob
                    if hasattr(module.B_quantizer, 'drop_prob'):
                        module.B_quantizer.drop_prob = prob

    def init_block_raw_data(self, block, name, device, qinp=False, keep_gpu=True):
        self.init_block_raw_inp_outp(block, device)
        if qinp and 'patch_embed' not in name:
            self.init_block_quanted_input(block, device)
        
        if self.metric == "fisher_brecq":
            self.init_block_brecq_hessian(block, device)

        if 'patch_embed' in name:
            block.quanted_input = block.raw_input

        if keep_gpu:
            block.raw_input, block.raw_out = block.raw_input.to(device), block.raw_out.to(device)
            if block.quanted_input is not None:
                block.quanted_input = block.quanted_input.to(device)
            if block.quanted_out is not None:
                block.quanted_out = block.quanted_out.to(device)
            if block.raw_grad is not None:
                block.raw_grad = block.raw_grad.to(device)

    def init_block_raw_inp_outp(self, block, device):
        logging.info('initializing raw input and raw output ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        hooks = []
        hooks.append(block.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.register_forward_hook(self.single_input_forward_hook))
        need_calculate_raw_softmax = False
        if self.raw_pred_softmaxs is None and self.metric in ["fisher_brecq", "fisher_lr","fisher_diag","fisher_dplr"]:
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
                torch.cuda.empty_cache()
        block.raw_out = torch.cat(block.tmp_out, dim=0)
        block.raw_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input, block.tmp_out = None, None
        for hook in hooks:
            hook.remove()
        torch.cuda.empty_cache()

    def set_shared_raw_pred_softmaxs(self, raw_pred_softmaxs):
        """Share pre-computed raw_pred_softmaxs from MLPReconstructor.

        This avoids recomputation when Fisher is shared across stages.
        """
        self.raw_pred_softmaxs = raw_pred_softmaxs
        logging.info('BlockRecon: received shared raw_pred_softmaxs.')

    def init_block_quanted_input(self, block, device):
        logging.info('initializing quanted input ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        hook = block.register_forward_hook(self.single_input_forward_hook)
        with torch.no_grad():
            for i, (inp, target) in enumerate(self.calib_loader):
                inp = inp.to(device)
                pred = self.model(inp)
        torch.cuda.empty_cache()
        block.quanted_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input = None
        hook.remove()
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')

    def init_block_brecq_hessian(self, block, device):
        logging.info('initializing brecq-fim ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        hook = block.register_full_backward_hook(self.grad_hook)
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
            loss.backward()
            torch.cuda.empty_cache()
        raw_grads = torch.cat(block.tmp_grad, dim=0)
        block.raw_grad = raw_grads.abs().reshape(raw_grads.shape[0], -1)
        hook.remove()
        del raw_grads
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        torch.cuda.empty_cache()

    def new_fisher_ro(self, block, device):
        logging.info('updating fisher information matrix ...')
        hooks = []
        hooks.append(block.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.register_full_backward_hook(self.grad_hook))
        for i, (inp, target) in enumerate(self.calib_loader):
            self.model.zero_grad()
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
            loss.backward()
            torch.cuda.empty_cache()
        raw_grad = torch.cat(block.tmp_grad, dim=0)
        raw_grad = raw_grad.reshape(raw_grad.shape[0], -1).abs()
        raw_grad = raw_grad.mean(dim=0).unsqueeze(0) # (1, N)
        q_out = torch.cat(block.tmp_out, dim=0).to(block.raw_out.device)
        delta_out = (q_out - block.raw_out).abs().mean(dim=0).reshape(1, -1) # (1, N)
        block.tmp_grad = block.tmp_out = None
        for hook in hooks:
            hook.remove()
        
        if block.raw_grad is None:
            block.raw_grad = raw_grad
            block.delta_out = delta_out
        else:
            block.raw_grad = torch.cat([block.raw_grad, raw_grad], dim=0) # (k, N)
            block.delta_out = torch.cat([block.delta_out, delta_out], dim=0) # (k, N)
        block.inverse_B = torch.linalg.inv(block.delta_out.to(device) @ block.delta_out.transpose(1, 0).to(device)) # (k, k)
        # block.inverse_B = torch.eye(block.raw_grad.shape[0]).to(device)
        del raw_grad, delta_out
        torch.cuda.empty_cache()

    def fixed_candidate_params(self, name):
        return self.k, 'fixed_base', self.p1, self.p2

    @staticmethod
    def params_are_same(left, right):
        return (
            left[0] == right[0]
            and abs(left[2] - right[2]) < 5e-3
            and abs(left[3] - right[3]) < 5e-3
        )

    @staticmethod
    def _relative_mse(post_mse, pre_mse):
        if pre_mse is None or post_mse is None or pre_mse <= 0:
            return 0.0
        return post_mse / pre_mse

    @staticmethod
    def _metric_delta(lhs, rhs, key, default=0.0):
        lhs_stats = lhs.get('post_logit_guard') if lhs is not None else None
        rhs_stats = rhs.get('post_logit_guard') if rhs is not None else None
        if lhs_stats is None or rhs_stats is None:
            return default
        return float(lhs_stats.get(key, 0.0)) - float(rhs_stats.get(key, 0.0))

    def candidate_rel_mse(self, pre_mse, result):
        return self._relative_mse(result.get('post_recon_mse'), pre_mse)

    def adaptive_has_reliable_gain(self, pre_mse, fixed_result, adaptive_result, block_name=None):
        if adaptive_result is None or adaptive_result.get('should_revert'):
            return False
        if fixed_result is None or fixed_result.get('should_revert'):
            return True
        fixed_rel = self.candidate_rel_mse(pre_mse, fixed_result)
        adaptive_rel = self.candidate_rel_mse(pre_mse, adaptive_result)
        mse_gain = fixed_rel - adaptive_rel
        ce_gain = -self._metric_delta(adaptive_result, fixed_result, 'ce')
        top1_gain = self._metric_delta(adaptive_result, fixed_result, 'top1')
        true_prob_gain = self._metric_delta(adaptive_result, fixed_result, 'true_prob')
        margin_gain = self._metric_delta(adaptive_result, fixed_result, 'margin')
        candidate_name = adaptive_result.get('candidate', 'adaptive')

        if self.w_bit == 3 or self.a_bit == 3:
            mse_clear = mse_gain >= 0.002
            logits_not_worse = (
                ce_gain >= -0.006
                and top1_gain >= -0.004
                and true_prob_gain >= -0.003
                and margin_gain >= -0.010
            )
            logits_clear = (
                ce_gain >= 0.015
                or top1_gain >= 0.012
                or (true_prob_gain >= 0.004 and margin_gain >= 0.015)
            )
            safe_accept = (mse_clear and logits_not_worse) or (mse_gain >= -0.0005 and logits_clear)
            if safe_accept:
                return True

            if self.adaptive_3bit_select_profile == 'safe_plus':
                idx = self._block_index(block_name or '')
                if idx is None:
                    return False
                # Keep early 3bit blocks fixed unless the conservative safe
                # rule passes. The balanced run showed block 2 can overfit the
                # reconstruction logits and degrade the final test accuracy.
                if idx < 5 or idx == 11:
                    return False
                if candidate_name == 'strong_adaptive' and idx not in (5, 6, 8, 10):
                    return False
                mid_block_mse_tie_break = (
                    5 <= idx <= 6
                    and mse_gain >= 0.00010
                    and ce_gain >= -0.008
                    and top1_gain >= -0.004
                    and true_prob_gain >= -0.004
                    and margin_gain >= -0.012
                )
                late_neutral_gain = (
                    idx >= 8
                    and mse_gain >= 0.00005
                    and ce_gain >= -0.003
                    and top1_gain >= -0.003
                    and true_prob_gain >= -0.0015
                    and margin_gain >= -0.006
                )
                tail_logit_gain = (
                    idx == 10
                    and mse_gain >= -0.0010
                    and ce_gain >= 0.015
                    and top1_gain >= 0.000
                    and true_prob_gain >= 0.001
                    and margin_gain >= 0.020
                )
                strong_mid_gain = (
                    candidate_name == 'strong_adaptive'
                    and 5 <= idx <= 6
                    and mse_gain >= -0.0007
                    and ce_gain >= 0.008
                    and top1_gain >= -0.002
                    and true_prob_gain >= -0.001
                    and margin_gain >= 0.006
                )
                strong_late_gain = (
                    candidate_name == 'strong_adaptive'
                    and idx == 8
                    and mse_gain >= -0.0005
                    and ce_gain >= 0.006
                    and top1_gain >= -0.003
                    and true_prob_gain >= 0.000
                    and margin_gain >= -0.004
                )
                strong_tail_gain = (
                    candidate_name == 'strong_adaptive'
                    and idx == 10
                    and mse_gain >= -0.0015
                    and ce_gain >= 0.012
                    and top1_gain >= -0.001
                    and true_prob_gain >= 0.000
                    and margin_gain >= 0.015
                )
                return (
                    mid_block_mse_tie_break
                    or late_neutral_gain
                    or tail_logit_gain
                    or strong_mid_gain
                    or strong_late_gain
                    or strong_tail_gain
                )

            if self.adaptive_3bit_select_profile != 'balanced':
                return safe_accept

            # 3bit logits on the reconstruction set are noisy, but very small
            # local-MSE differences can hide useful adaptive candidates. The
            # balanced profile only admits candidates with multiple consistent
            # positive signals, keeping fixed as the default anchor.
            relaxed_logit_gain = (
                mse_gain >= -0.0008
                and ce_gain >= 0.010
                and top1_gain >= 0.003
                and true_prob_gain >= -0.001
                and margin_gain >= 0.010
            )
            small_mse_gain = (
                mse_gain >= 0.0002
                and ce_gain >= 0.000
                and top1_gain >= -0.004
                and true_prob_gain >= 0.001
                and margin_gain >= 0.005
            )
            return relaxed_logit_gain or small_mse_gain

        return adaptive_result['score'] > fixed_result['score'] + self.adaptive_candidate_margin

    def candidate_score(self, pre_mse, result):
        if result['should_revert']:
            return -1e9
        stats = result['post_logit_guard']
        if stats is None:
            return -self._relative_mse(result['post_recon_mse'], pre_mse)
        rel_mse = self._relative_mse(result['post_recon_mse'], pre_mse)
        return (
            2.0 * stats['top1']
            - stats['ce']
            + 0.10 * stats['true_prob']
            + 0.02 * stats['margin']
            - 0.05 * rel_mse
        )

    def run_reconstruction_candidate(self, name, block, device, candidate_name,
                                     block_k, block_k_reason, block_p1, block_p2,
                                     pre_recon_state, pre_runtime_state, pre_recon_mse, pre_logit_guard,
                                     batch_size, iters, weight, b_range, warmup, lr,
                                     quant_act, mode, drop_prob):
        block.load_state_dict(pre_recon_state, strict=True)
        if self.w_bit == 3 or self.a_bit == 3:
            for attr in ['raw_grad', 'delta_out', 'inverse_B', 'tmp_grad', 'tmp_out']:
                setattr(block, attr, pre_runtime_state.get(attr))
        elif self.metric in ["fisher_lr", "fisher_diag", "fisher_dplr"]:
            block.raw_grad = None
            block.delta_out = None
            block.inverse_B = None
            block.tmp_grad = None
            block.tmp_out = None
        self.set_block_soft_targets(block, True)
        self.set_block_mode(block, 'quant_forward')
        for _, module in block.named_modules():
            if hasattr(module, 'training_mode'):
                module.init_training()
        if mode == 'qdrop':
            self.set_qdrop(block, drop_prob)

        w_params, a_params = [], []
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        module.a_quantizer.scale.requires_grad = True
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.mode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        module.A_quantizer.scale.requires_grad = True
                        module.B_quantizer.scale.requires_grad = True
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.mode = 'raw'

        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=lr) if len(a_params) != 0 else None
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=iters, eta_min=0.) if len(a_params) != 0 else None

        logging.info(
            'Block {} candidate {}: k={} ({}) p1={:.3f} p2={:.3f}'.format(
                name, candidate_name, block_k, block_k_reason, block_p1, block_p2
            )
        )
        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters,
                                 rec_loss=self.metric if 'head' not in name else 'kl_div',
                                 b_range=b_range, decay_start=0, warmup=warmup, p1=block_p1, p2=block_p2)
        i_change = max(1, math.floor(iters / max(block_k, 1)))
        for it in range(iters):
            idx = torch.randperm(block.raw_input.size(0))[:batch_size]
            if mode == 'qdrop':
                cur_quant_inp = block.quanted_input[idx].to(device) if block.quanted_input is not None else block.raw_input[idx].to(device)
                cur_fp_inp = block.raw_input[idx].to(device)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < drop_prob, cur_quant_inp, cur_fp_inp)
            elif mode == 'rinp':
                cur_inp = block.raw_input[idx].to(device)
            elif mode == 'qinp':
                cur_inp = block.quanted_input[idx].to(device)
            cur_out = block.raw_out[idx].to(device)

            loss_func.update_fisher = False
            if loss_func.rec_loss in ["fisher_lr", "fisher_diag", "fisher_dplr"]:
                if self.dis_mode in ['q']:
                    if it % i_change == 0:
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                elif self.dis_mode in ['qf']:
                    if it in range(block_k):
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                cur_grad = block.raw_grad.to(device)
            elif self.metric == "fisher_brecq":
                cur_grad = block.raw_grad[idx].to(device)
            else:
                cur_grad = None
            w_optimizer.zero_grad()
            if quant_act:
                a_optimizer.zero_grad()
            out_quant = block(cur_inp)
            if 'head' not in name:
                err = loss_func(out_quant, cur_out, cur_grad)
            else:
                err = loss_func(out_quant, cur_out)
            err.backward()
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()

        torch.cuda.empty_cache()
        for _, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
            if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                module.a_quantizer.scale.requires_grad = False
            elif isinstance(module, MinMaxQuantMatMul):
                module.A_quantizer.scale.requires_grad = False
                module.B_quantizer.scale.requires_grad = False
        self.set_qdrop(block, 1.0)

        post_recon_mse = self.estimate_block_mse(block, device, mode=mode)
        post_logit_guard = self.evaluate_logit_guard(device, name=name) if self.logit_guard else None
        logit_should_revert, logit_guard_reason = self.should_revert_by_logit_guard(
            name, pre_logit_guard, post_logit_guard
        )
        strong_logit_improvement = self.has_strong_logit_improvement(name, pre_logit_guard, post_logit_guard)
        tail_logit_override = self.has_tail_logit_override(name, pre_logit_guard, post_logit_guard)

        last_total_loss = None
        last_rec_loss = None
        last_round_loss = None
        if loss_func.last_total_loss is not None:
            last_total_loss = LossFunction.to_float(loss_func.last_total_loss)
            last_rec_loss = LossFunction.to_float(loss_func.last_rec_loss)
            last_round_loss = LossFunction.to_float(loss_func.last_round_loss)

        guard_threshold = self.block_guard_threshold(name)
        loss_revert_threshold = self.block_loss_revert_threshold(name)
        mse_should_revert = (
            pre_recon_mse is not None
            and post_recon_mse is not None
            and post_recon_mse > pre_recon_mse * guard_threshold
        )
        loss_should_revert = (
            loss_revert_threshold is not None
            and last_total_loss is not None
            and last_total_loss > loss_revert_threshold
        )
        idx = self._block_index(name)
        weak_tail_gain_should_revert = (
            idx is not None
            and idx >= 10
            and pre_recon_mse is not None
            and post_recon_mse is not None
            and post_recon_mse > pre_recon_mse * 0.99
        )
        logit_override = strong_logit_improvement or tail_logit_override
        should_revert = (mse_should_revert or loss_should_revert) and not logit_override
        if weak_tail_gain_should_revert:
            should_revert = not tail_logit_override
        if logit_should_revert:
            should_revert = True

        result = {
            'candidate': candidate_name,
            'state': self._clone_state_dict(block),
            'total_loss': last_total_loss,
            'rec_loss': last_rec_loss,
            'round_loss': last_round_loss,
            'b': loss_func.last_b,
            'count': loss_func.count,
            'k': block_k,
            'k_reason': block_k_reason,
            'p1': block_p1,
            'p2': block_p2,
            'post_recon_mse': post_recon_mse,
            'post_logit_guard': post_logit_guard,
            'logit_guard': logit_guard_reason,
            'strong_logit_improvement': strong_logit_improvement,
            'tail_logit_override': tail_logit_override,
            'should_revert': should_revert,
        }
        result['score'] = self.candidate_score(pre_recon_mse, result)
        logging.info(
            'Block {} candidate {} result: score={:.6f}, revert={}, mse {:.6e}->{:.6e}, logit after=({}), logit_guard={}'.format(
                name,
                candidate_name,
                result['score'],
                should_revert,
                pre_recon_mse if pre_recon_mse is not None else float('nan'),
                post_recon_mse if post_recon_mse is not None else float('nan'),
                self.format_logit_guard_stats(post_logit_guard),
                logit_guard_reason,
            )
        )
        return result

    def reconstruct_single_block_with_candidate_select(self, name, block, device,
                                                       batch_size=32, iters=20000, weight=0.01,
                                                       b_range=(20, 2), warmup=0.2, lr=4e-5,
                                                       quant_act=False, mode='qdrop', drop_prob=1.0):
        block_name = name
        self.wrap_quantizers_in_net(block, name)
        self.set_block_soft_targets(block, False)
        pre_recon_state = self._clone_state_dict(block)
        if self.w_bit == 3 or self.a_bit == 3:
            pre_runtime_state = {
                'raw_grad': block.raw_grad.detach().clone() if torch.is_tensor(block.raw_grad) else block.raw_grad,
                'delta_out': block.delta_out.detach().clone() if torch.is_tensor(block.delta_out) else block.delta_out,
                'inverse_B': block.inverse_B.detach().clone() if torch.is_tensor(block.inverse_B) else block.inverse_B,
                'tmp_grad': block.tmp_grad,
                'tmp_out': block.tmp_out,
            }
        else:
            pre_runtime_state = {}
        pre_recon_mse = self.estimate_block_mse(block, device, mode=mode)
        pre_logit_guard = self.evaluate_logit_guard(device, name=block_name) if self.logit_guard else None
        logging.info('Block {} logit guard before: {}'.format(
            block_name, self.format_logit_guard_stats(pre_logit_guard)
        ))

        fixed_params = self.fixed_candidate_params(name)
        adaptive_k, adaptive_k_reason = self.select_block_k(name)
        adaptive_p1, adaptive_p2 = self.compute_adaptive_p1p2(name, block)
        adaptive_params = (adaptive_k, adaptive_k_reason, adaptive_p1, adaptive_p2)
        strong_params = self.strong_3bit_candidate_params(name, adaptive_params)
        if (
            (self.w_bit == 3 or self.a_bit == 3)
            and self.params_are_same(fixed_params, adaptive_params)
            and strong_params is None
        ):
            candidates = [('fixed', fixed_params)]
            logging.info('Block {} candidate selection uses fixed path: adaptive params match fixed params.'.format(name))
        else:
            candidates = [
                ('fixed', fixed_params),
                ('adaptive', adaptive_params),
            ]
            if strong_params is not None and not self.params_are_same(fixed_params, strong_params) and not self.params_are_same(adaptive_params, strong_params):
                candidates.append(('strong_adaptive', strong_params))
            logging.info(
                'Block {} candidate selection enabled: fixed k={} p1={:.3f} p2={:.3f}; adaptive k={} ({}) p1={:.3f} p2={:.3f}'.format(
                    name,
                    fixed_params[0],
                    fixed_params[2],
                    fixed_params[3],
                    adaptive_params[0],
                    adaptive_params[1],
                    adaptive_params[2],
                    adaptive_params[3],
                )
            )
            if strong_params is not None:
                logging.info(
                    'Block {} strong 3bit candidate: k={} ({}) p1={:.3f} p2={:.3f}'.format(
                        name,
                        strong_params[0],
                        strong_params[1],
                        strong_params[2],
                        strong_params[3],
                    )
                )

        results = []
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for candidate_name, params in candidates:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state.clone())
            if cuda_states is not None:
                torch.cuda.set_rng_state_all([state.clone() for state in cuda_states])
            block_k, block_k_reason, block_p1, block_p2 = params
            results.append(
                self.run_reconstruction_candidate(
                    name, block, device, candidate_name,
                    block_k, block_k_reason, block_p1, block_p2,
                    pre_recon_state, pre_runtime_state, pre_recon_mse, pre_logit_guard,
                    batch_size, iters, weight, b_range, warmup, lr,
                    quant_act, mode, drop_prob,
                )
            )

        best = max(results, key=lambda item: item['score'])
        if len(results) >= 2:
            fixed_result = next((item for item in results if item['candidate'] == 'fixed'), None)
            adaptive_results = [
                item for item in results
                if item['candidate'] in ('adaptive', 'strong_adaptive')
            ]
            if fixed_result is not None and adaptive_results:
                reliable_adaptive = [
                    item for item in adaptive_results
                    if self.adaptive_has_reliable_gain(pre_recon_mse, fixed_result, item, block_name)
                ]
                if reliable_adaptive:
                    best = max(reliable_adaptive, key=lambda item: item['score'])
                elif fixed_result['score'] > -1e8:
                    best = fixed_result
                else:
                    best = max(results, key=lambda item: item['score'])
        if best['should_revert']:
            guard_action = 'revert'
            block.load_state_dict(pre_recon_state, strict=True)
            self.set_block_soft_targets(block, False)
            logging.info(
                'Block {} candidate selection: all candidates rejected; revert to pre-reconstruction state.'.format(
                    block_name
                )
            )
        else:
            guard_action = 'keep_{}'.format(best['candidate'])
            block.load_state_dict(best['state'], strict=True)
            self.set_block_soft_targets(block, False)
            logging.info(
                'Block {} candidate selection: chose {} (score {:.6f}).'.format(
                    block_name, best['candidate'], best['score']
                )
            )

        if best['total_loss'] is not None:
            logging.info(
                'Block {} final loss: total={:.6f} rec={:.6f} round={:.6f} b={:.2f} count={} k={} k_reason={} p1={:.3f} p2={:.3f} guard={} mse_before={} mse_after={} logit_before=({}) logit_after=({})'.format(
                    block_name,
                    best['total_loss'],
                    best['rec_loss'],
                    best['round_loss'],
                    best['b'],
                    best['count'],
                    best['k'],
                    best['k_reason'],
                    best['p1'],
                    best['p2'],
                    guard_action,
                    '{:.6e}'.format(pre_recon_mse) if pre_recon_mse is not None else 'n/a',
                    '{:.6e}'.format(best['post_recon_mse']) if best['post_recon_mse'] is not None else 'n/a',
                    self.format_logit_guard_stats(pre_logit_guard),
                    self.format_logit_guard_stats(best['post_logit_guard']),
                )
            )
            self.block_recon_summary.append({
                'name': block_name,
                'total_loss': best['total_loss'],
                'rec_loss': best['rec_loss'],
                'round_loss': best['round_loss'],
                'b': best['b'],
                'count': best['count'],
                'k': best['k'],
                'k_reason': best['k_reason'],
                'p1': best['p1'],
                'p2': best['p2'],
                'guard': guard_action,
                'mse_before': pre_recon_mse,
                'mse_after': best['post_recon_mse'],
                'logit_before': pre_logit_guard,
                'logit_after': best['post_logit_guard'],
                'logit_guard': best['logit_guard'],
            })

        for result in results:
            del result['state']
        del pre_recon_state
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input
        torch.cuda.empty_cache()
            
    def reconstruct_single_block(self, name, block, device,
                                 batch_size: int = 32, iters: int = 20000, weight: float = 0.01,
                                 b_range: tuple = (20, 2), warmup: float = 0.2, lr: float = 4e-5, p: float = 2.0,
                                 quant_act = False, mode = 'qdrop', drop_prob: float = 1.0):
        if self.adaptive_candidate_select and (self.adaptive_k or self.adaptive_p):
            return self.reconstruct_single_block_with_candidate_select(
                name, block, device,
                batch_size=batch_size, iters=iters, weight=weight,
                b_range=b_range, warmup=warmup, lr=lr,
                quant_act=quant_act, mode=mode, drop_prob=drop_prob,
            )
        block_name = name
        self.wrap_quantizers_in_net(block, name)
        self.set_block_soft_targets(block, False)
        pre_recon_state = self._clone_state_dict(block)
        pre_recon_mse = self.estimate_block_mse(block, device, mode=mode)
        pre_logit_guard = self.evaluate_logit_guard(device, name=block_name) if self.logit_guard else None
        post_recon_mse = None
        post_logit_guard = None
        logit_guard_reason = 'disabled'
        guard_action = 'not_checked'
        logging.info('Block {} logit guard before: {}'.format(
            block_name, self.format_logit_guard_stats(pre_logit_guard)
        ))
        self.set_block_soft_targets(block, True)
        self.set_block_mode(block, 'quant_forward')
        for _name, module in block.named_modules():
            if hasattr(module, 'training_mode'):
                module.init_training()
        if mode == 'qdrop':
            self.set_qdrop(block, drop_prob)
        w_params, a_params = [], []
        for _name, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        module.a_quantizer.scale.requires_grad = True
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.mode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        module.A_quantizer.scale.requires_grad = True
                        module.B_quantizer.scale.requires_grad = True
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.mode = 'raw'
        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=lr) if len(a_params) != 0 else None
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=iters, eta_min=0.) if len(a_params) != 0 else None

        # --- Dynamic k and adaptive p1/p2 ---
        block_k, block_k_reason = self.select_block_k(name)
        block_p1, block_p2 = self.compute_adaptive_p1p2(name, block)
        if self.adaptive_k:
            stat = self.block_residual_stats.get(name)
            if stat is not None:
                logging.info('Block {} residual-aware dynamic k={} (base k={}, residual_norm={:.3f}, reason={})'
                             .format(name, block_k, self.auto_base_k, float(stat.get('residual_norm', 1.0)),
                                     block_k_reason))
            else:
                logging.info('Block {} dynamic k={} (base k={}, reason={})'
                             .format(name, block_k, self.auto_base_k, block_k_reason))
        if self.adaptive_p:
            stat = self.block_residual_stats.get(name)
            if stat is not None:
                logging.info('Block {} residual-aware adaptive p1={:.2f} p2={:.2f} (residual_norm={:.3f}, rel={:.3e})'
                             .format(name, block_p1, block_p2,
                                     float(stat.get('residual_norm', 1.0)),
                                     float(stat.get('rel', 0.0))))
            else:
                block_std = block.raw_out.std().item() if block.raw_out is not None else 0.0
                logging.info('Block {} adaptive p1={:.2f} p2={:.2f} (std={:.4f})'
                             .format(name, block_p1, block_p2, block_std))

        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters,
                                 rec_loss=self.metric if 'head' not in name else 'kl_div',
                                 b_range=b_range, decay_start=0, warmup=warmup, p1=block_p1, p2=block_p2)
        i_change = math.floor(iters / block_k)
        for it in range(iters):
            idx = torch.randperm(block.raw_input.size(0))[:batch_size]
            if mode == 'qdrop':
                cur_quant_inp = block.quanted_input[idx].to(device) if block.quanted_input is not None else block.raw_input[idx].to(device)
                cur_fp_inp = block.raw_input[idx].to(device)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < drop_prob, cur_quant_inp, cur_fp_inp)
            elif mode == 'rinp':
                cur_inp = block.raw_input[idx].to(device)
            elif mode == 'qinp':
                cur_inp = block.quanted_input[idx].to(device)
            cur_out = block.raw_out[idx].to(device)
            
            loss_func.update_fisher = False
            if loss_func.rec_loss in ["fisher_lr", "fisher_diag", "fisher_dplr"] :
                if self.dis_mode in ['q']:
                    if it % i_change == 0:
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                elif self.dis_mode in ['qf']:
                    if it in range(block_k):
                        self.new_fisher_ro(block, device)
                        loss_func.update_fisher = True
                cur_grad = block.raw_grad.to(device)
            elif self.metric == "fisher_brecq" :
                cur_grad = block.raw_grad[idx].to(device)
            else:
                cur_grad = None
            w_optimizer.zero_grad()
            if quant_act:
                a_optimizer.zero_grad()
            out_quant = block(cur_inp)
            if 'head' not in name:
                err = loss_func(out_quant, cur_out, cur_grad)
            else:
                err = loss_func(out_quant, cur_out)
            err.backward()
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
        torch.cuda.empty_cache()
        # Finish optimization, use hard rounding.
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
            if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                module.a_quantizer.scale.requires_grad = False
            elif isinstance(module, MinMaxQuantMatMul):
                module.A_quantizer.scale.requires_grad = False
                module.B_quantizer.scale.requires_grad = False
        self.set_qdrop(block, 1.0)
        post_recon_mse = self.estimate_block_mse(block, device, mode=mode)
        post_logit_guard = self.evaluate_logit_guard(device, name=block_name) if self.logit_guard else None
        logit_should_revert, logit_guard_reason = self.should_revert_by_logit_guard(
            block_name, pre_logit_guard, post_logit_guard
        )
        strong_logit_improvement = self.has_strong_logit_improvement(block_name, pre_logit_guard, post_logit_guard)
        tail_logit_override = self.has_tail_logit_override(block_name, pre_logit_guard, post_logit_guard)
        logging.info('Block {} logit guard after: {} ({})'.format(
            block_name, self.format_logit_guard_stats(post_logit_guard), logit_guard_reason
        ))
        last_total_loss = None
        last_rec_loss = None
        last_round_loss = None
        if loss_func.last_total_loss is not None:
            last_total_loss = LossFunction.to_float(loss_func.last_total_loss)
            last_rec_loss = LossFunction.to_float(loss_func.last_rec_loss)
            last_round_loss = LossFunction.to_float(loss_func.last_round_loss)
        if pre_recon_mse is not None and post_recon_mse is not None:
            guard_threshold = self.block_guard_threshold(block_name)
            loss_revert_threshold = self.block_loss_revert_threshold(block_name)
            mse_should_revert = post_recon_mse > pre_recon_mse * guard_threshold
            loss_should_revert = False
            weak_tail_gain_should_revert = False
            if (
                loss_revert_threshold is not None
                and last_total_loss is not None
                and last_total_loss > loss_revert_threshold
            ):
                loss_should_revert = True
            idx = self._block_index(block_name)
            if idx is not None and idx >= 10:
                weak_tail_gain_should_revert = post_recon_mse > pre_recon_mse * 0.99
            logit_override = strong_logit_improvement or tail_logit_override
            should_revert = (mse_should_revert or loss_should_revert) and not logit_override
            if weak_tail_gain_should_revert:
                should_revert = not tail_logit_override
            if logit_should_revert:
                should_revert = True
            if should_revert:
                guard_action = 'revert'
                logging.info(
                    'Block {} guard: revert AdaRound update (mse {:.6e} -> {:.6e}, threshold {:.6f}, final_loss {}, logit_guard {}).'.format(
                        block_name,
                        pre_recon_mse,
                        post_recon_mse,
                        guard_threshold,
                        '{:.6f}'.format(last_total_loss) if last_total_loss is not None else 'n/a',
                        logit_guard_reason,
                    )
                )
                block.load_state_dict(pre_recon_state, strict=True)
                self.set_block_soft_targets(block, False)
            else:
                guard_action = 'keep'
                logging.info(
                    'Block {} guard: keep AdaRound update (mse {:.6e} -> {:.6e}, threshold {:.6f}, logit_guard {}, strong_logit_improvement {}, tail_logit_override {}).'.format(
                        block_name, pre_recon_mse, post_recon_mse, guard_threshold,
                        logit_guard_reason, strong_logit_improvement, tail_logit_override
                    )
                )
        if last_total_loss is not None:
            logging.info(
                'Block {} final loss: total={:.6f} rec={:.6f} round={:.6f} b={:.2f} count={} k={} k_reason={} p1={:.3f} p2={:.3f} guard={} mse_before={} mse_after={} logit_before=({}) logit_after=({})'.format(
                    block_name,
                    last_total_loss,
                    last_rec_loss,
                    last_round_loss,
                    loss_func.last_b,
                    loss_func.count,
                    block_k,
                    block_k_reason,
                    block_p1,
                    block_p2,
                    guard_action,
                    '{:.6e}'.format(pre_recon_mse) if pre_recon_mse is not None else 'n/a',
                    '{:.6e}'.format(post_recon_mse) if post_recon_mse is not None else 'n/a',
                    self.format_logit_guard_stats(pre_logit_guard),
                    self.format_logit_guard_stats(post_logit_guard),
                )
            )
            self.block_recon_summary.append({
                'name': block_name,
                'total_loss': last_total_loss,
                'rec_loss': last_rec_loss,
                'round_loss': last_round_loss,
                'b': loss_func.last_b,
                'count': loss_func.count,
                'k': block_k,
                'k_reason': block_k_reason,
                'p1': block_p1,
                'p2': block_p2,
                'guard': guard_action,
                'mse_before': pre_recon_mse,
                'mse_after': post_recon_mse,
                'logit_before': pre_logit_guard,
                'logit_after': post_logit_guard,
                'logit_guard': logit_guard_reason,
            })
        del pre_recon_state
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input
        torch.cuda.empty_cache()
    

    def reconstruct_model(self, quant_act: bool = False, mode: str = 'qdrop', drop_prob: float = 1.0,
                          keep_gpu: bool = True, block_start=None, block_end=None,
                          skip_patch_embed=False, skip_head=False):
        device = next(self.model.parameters()).device
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'raw'
        logging.info(
            'Block reconstruction selection: block_start={}, block_end={}, skip_patch_embed={}, skip_head={}'.format(
                block_start, block_end, skip_patch_embed, skip_head
            )
        )
        for idx, name in enumerate(self.blocks.keys()):
            if not self.should_reconstruct_block(name, block_start, block_end, skip_patch_embed, skip_head):
                logging.info('skipping {} by reconstruction selection; using calibrated quantized forward for downstream blocks.'.format(name))
                self.quanted_blocks.append(name)
                continue
            block = self.blocks[name]
            logging.info('reconstructing {} ...'.format(name))
            self.init_block_raw_data(block, name, device, qinp=(mode != 'rinp'), keep_gpu=keep_gpu)
            logging.info('adaround training for {} ...'.format(name))
            self.reconstruct_single_block(name, block, device, quant_act=quant_act, mode=mode, drop_prob=drop_prob)
            self.quanted_blocks.append(name)
            logging.info('finished reconstructing {}.'.format(name))
        if self.block_recon_summary:
            logging.info('Block reconstruction final loss summary:')
            for item in self.block_recon_summary:
                logging.info(
                    '  {name}: total={total_loss:.6f}, rec={rec_loss:.6f}, round={round_loss:.6f}, k={k} ({k_reason}), p1={p1:.3f}, p2={p2:.3f}, guard={guard}, mse={mse_before}->{mse_after}, logit={logit_before}->{logit_after}, logit_guard={logit_guard}'.format(
                        name=item['name'],
                        total_loss=item['total_loss'],
                        rec_loss=item['rec_loss'],
                        round_loss=item['round_loss'],
                        k=item['k'],
                        k_reason=item.get('k_reason', 'n/a'),
                        p1=item['p1'],
                        p2=item['p2'],
                        guard=item['guard'],
                        mse_before='{:.6e}'.format(item['mse_before']) if item['mse_before'] is not None else 'n/a',
                        mse_after='{:.6e}'.format(item['mse_after']) if item['mse_after'] is not None else 'n/a',
                        logit_before=self.format_logit_guard_stats(item.get('logit_before')),
                        logit_after=self.format_logit_guard_stats(item.get('logit_after')),
                        logit_guard=item.get('logit_guard', 'n/a'),
                    )
                )
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'quant_forward'
            if hasattr(module, 'w_quantizer') and hasattr(module.w_quantizer, 'get_hard_value'):
                module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
                if hasattr(module.w_quantizer, 'alpha'):
                    del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"

        
class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p1: float = 2.,
                 p2: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p1 = p1
        self.p2 = p2
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.update_fisher = False
        self.last_total_loss = None
        self.last_rec_loss = None
        self.last_round_loss = None
        self.last_b = None
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    @staticmethod
    def to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=2.0)
            if self.count == 1:
                self.init_loss_1 = rec_loss.detach()
            rec_loss = rec_loss / self.init_loss_1
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0)
            if self.count == 1:
                self.init_loss_1 = rec_loss.detach()
            rec_loss = rec_loss / self.init_loss_1
        elif self.rec_loss == 'fisher_lr':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_1 = (cha * grad.abs()).mean(dim=-1).pow(2).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
            rec_loss = 2 * loss_1 / self.init_loss_1
        elif self.rec_loss == 'fisher_diag':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_2 = (cha.pow(2) * grad.abs().mean(dim=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_2 = loss_2.detach()
            rec_loss = 2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_dplr':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            A = cha.unsqueeze(1) @ grad.abs().transpose(0, 1)
            loss_1 = (A @ self.block.inverse_B @ A.transpose(1, 2)).mean()
            loss_2 = (cha.pow(2) * grad.abs().mean(dim=0)).mean()
            if self.count == 1 or self.update_fisher:
                self.init_loss_1 = loss_1.detach()
                self.init_loss_2 = loss_2.detach()
            rec_loss = self.p1 * loss_1 / self.init_loss_1 + self.p2 * loss_2 / self.init_loss_2
        elif self.rec_loss == 'fisher_brecq':
            cha = (pred - tgt).abs().reshape(pred.shape[0], -1)
            loss_1 = (cha.pow(2) * grad.pow(2)).mean()
            if self.count == 1:
                self.init_loss_1 = loss_1.detach()
            rec_loss = loss_1 / self.init_loss_1
        elif self.rec_loss == 'kl_div':
            rec_loss = F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(tgt, dim=-1).detach(), reduction="batchmean")
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = round_loss_pow2 = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if hasattr(module, 'w_quantizer'):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        self.last_total_loss = total_loss.detach()
        self.last_rec_loss = rec_loss.detach()
        self.last_round_loss = round_loss.detach() if torch.is_tensor(round_loss) else float(round_loss)
        self.last_b = float(b)
        if self.count == 1 or self.count % 500 == 0:
            total_value = self.to_float(self.last_total_loss)
            rec_value = self.to_float(self.last_rec_loss)
            round_value = self.to_float(self.last_round_loss)
            logging.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  total_value, rec_value, round_value, self.last_b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
