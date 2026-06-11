import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
import time
import torch.nn.functional as F

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.mlp_recon import MLPReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from datetime import datetime
import logging

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoints/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_tiny",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base',
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/4bit/fim_unified.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="D:/AI/IaS-ViT-main/dataset/imagenet",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--optim-size", default=1024,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--optim-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=64,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=0, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    # MLP Reconstruction args
    parser.add_argument('--reconstruct-mlp', action='store_true',
                        help='Run Fisher-guided MLP reconstruction before calibration.')
    parser.add_argument('--load-reconstruct-checkpoint', type=str, default=None,
                        help='Path to the reconstructed checkpoint.')
    parser.add_argument('--test-reconstruct-checkpoint', action='store_true',
                        help='validate the reconstructed checkpoint.')
    parser.add_argument("--recon-metric", type=str, default=argparse.SUPPRESS,
                        choices=['fisher_diag', 'fisher_dplr', 'fisher_lr', 'mse', 'mae'],
                        help='MLP reconstruction metric (Fisher-guided or baseline)')

    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument("--calib-metric", type=str, default=argparse.SUPPRESS,
                        choices=['mse', 'mae', 'fisher_diag'],
                        help='calibration metric (fisher_diag for Fisher-weighted scale search)')
    parser.add_argument("--optim-metric", type=str, default=argparse.SUPPRESS,
                        choices=['fisher_brecq', 'fisher_lr', 'fisher_diag', 'fisher_dplr', 'mse', 'mae'],
                        help='optimization metric')
    parser.add_argument('--optim-mode', type=str, default=argparse.SUPPRESS, choices=['qinp', 'rinp', 'qdrop'],
                        help='`qinp`:use quanted input; `rinp`: use raw input; `qdrop` use qdrop input;')
    parser.add_argument('--drop-prob', type=float, default=argparse.SUPPRESS,
                        help='dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.')
    parser.add_argument('--k', type=int, default=argparse.SUPPRESS, help='The rank of Fisher')
    parser.add_argument('--p1', type=float, default=argparse.SUPPRESS, help='The proportion of low rank')
    parser.add_argument('--p2', type=float, default=argparse.SUPPRESS, help='The proportion of diag')
    parser.add_argument('--dis-mode', type=str, default=argparse.SUPPRESS, choices=['q','qf'],
                        help='the mode of getting gradient. `q`: use quantization; `qf` Take the first k times (default:Uniformly obtain k times);')
    parser.add_argument('--adaptive-k', action='store_true', default=argparse.SUPPRESS,
                        help='Enable layered dynamic Fisher rank (k varies by block depth)')
    parser.add_argument('--adaptive-p', action='store_true', default=argparse.SUPPRESS,
                        help='Enable adaptive p1/p2 weights based on activation std')
    parser.add_argument('--no-adaptive-k', action='store_true', default=argparse.SUPPRESS,
                        help='Disable layered dynamic Fisher rank (use global k for all blocks)')
    parser.add_argument('--no-adaptive-p', action='store_true', default=argparse.SUPPRESS,
                        help='Disable adaptive p1/p2 weights (use global p1/p2 for all blocks)')
    adaptive_candidate_group = parser.add_mutually_exclusive_group()
    adaptive_candidate_group.add_argument('--adaptive-candidate-select', action='store_true', default=argparse.SUPPRESS,
                                          help='Enable fixed/adaptive per-block candidate selection.')
    adaptive_candidate_group.add_argument('--no-adaptive-candidate-select', action='store_true', default=argparse.SUPPRESS,
                                          help='Disable fixed/adaptive per-block candidate selection.')
    parser.add_argument('--adaptive-candidate-margin', type=float, default=None,
                        help='Score margin required for adaptive candidate to beat fixed candidate.')
    parser.add_argument('--pct', type=float, default=argparse.SUPPRESS,
                        help='clamp percentile of mlp.fc2 input for GELU clamping.')
    parser.add_argument('--diagnose-residual-only', action='store_true',
                        help='Run calibration and block residual diagnostics, then exit before block reconstruction.')
    parser.add_argument('--recon-block-start', type=int, default=None,
                        help='Only reconstruct transformer blocks with index >= this value.')
    parser.add_argument('--recon-block-end', type=int, default=None,
                        help='Only reconstruct transformer blocks with index <= this value.')
    parser.add_argument('--skip-patch-embed', action='store_true',
                        help='Skip patch_embed during block reconstruction.')
    parser.add_argument('--skip-head', action='store_true',
                        help='Skip classifier head during block reconstruction.')
    logit_guard_group = parser.add_mutually_exclusive_group()
    logit_guard_group.add_argument('--logit-guard', action='store_true', default=argparse.SUPPRESS,
                                   help='Enable full-model logits/confidence guard for block reconstruction.')
    logit_guard_group.add_argument('--no-logit-guard', action='store_true', default=argparse.SUPPRESS,
                                   help='Disable full-model logits/confidence guard for block reconstruction.')
    parser.add_argument('--logit-guard-batches', type=int, default=None,
                        help='Number of calibration batches used by the logits/confidence guard. Default: all optim batches.')
    parser.add_argument('--logit-guard-size', type=int, default=0,
                        help='Number of held-out calibration samples used by the logits/confidence guard. Default 0 reuses optim samples.')
    parser.add_argument('--logit-guard-seed-offset', type=int, default=1009,
                        help='Seed offset for held-out logit guard samples.')
    parser.add_argument('--no-logit-bias-correction', action='store_true',
                        help='Disable guarded teacher-logit bias correction after block reconstruction.')
    parser.add_argument('--bias-correction-size', type=int, default=512,
                        help='Number of samples used to estimate the post-reconstruction head bias correction.')
    parser.add_argument('--bias-correction-guard-size', type=int, default=512,
                        help='Number of held-out samples used to keep/revert head bias correction.')
    parser.add_argument('--bias-correction-seed-offset', type=int, default=2029,
                        help='Seed offset for head bias correction samples.')
    parser.add_argument('--bias-correction-max-abs', type=float, default=0.08,
                        help='Maximum absolute logit bias correction per class.')
    return parser


def seed_all(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        recon_suffix = '_recon' if args.reconstruct_mlp else ''
        auto_name = '{}_w{}_a{}_optimsize_{}_{}{}{}{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, cfg.optim_metric,
            '' if cfg.optim_metric in ['mse', 'mae'] else '_dis_mode_' + cfg.dis_mode,
            '' if cfg.optim_metric not in['fisher_lr', 'fisher_dplr'] else '_rank_' + str(cfg.k),
            recon_suffix,
            cfg.optim_mode)
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)
    return save_path


def load_model(model, args, device, mode='calibrate', ckpt_path=None):
    assert mode in ['calibrate', 'optimize']
    if ckpt_path is None:
        ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + '.' + attr + '.scale'
                getattr(module, attr).scale.data = ckpt[ckpt_name].clone()

    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model


def set_quant_mode(model, mode):
    previous_modes = {}
    for module in model.modules():
        if hasattr(module, 'mode'):
            previous_modes[module] = module.mode
            module.mode = mode
    return previous_modes


def restore_quant_mode(previous_modes):
    for module, mode in previous_modes.items():
        module.mode = mode


def get_classifier_head(model):
    head = getattr(model, 'head', None)
    if head is not None and hasattr(head, 'bias'):
        return head
    for name, module in model.named_modules():
        if name.split('.')[-1] == 'head' and hasattr(module, 'bias'):
            return module
    return None


def collect_teacher_logit_bias_delta(model, teacher_model, data_loader, device, max_abs):
    previous_modes = set_quant_mode(model, 'quant_forward')
    teacher_model.eval()
    model.eval()
    delta_sum = None
    count = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            teacher_logits = teacher_model(data).float()
            quant_logits = model(data).float()
            batch_delta = (teacher_logits - quant_logits).sum(dim=0)
            delta_sum = batch_delta if delta_sum is None else delta_sum + batch_delta
            count += data.size(0)
            torch.cuda.empty_cache()
    restore_quant_mode(previous_modes)
    if delta_sum is None or count == 0:
        return None
    delta = delta_sum / count
    delta = delta - delta.mean()
    return delta.clamp(min=-max_abs, max=max_abs)


def evaluate_logit_metrics(model, data_loader, criterion, device, teacher_model=None):
    previous_modes = set_quant_mode(model, 'quant_forward')
    if teacher_model is not None:
        teacher_model.eval()
    model.eval()
    total = 0
    loss_sum = 0.0
    kl_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * data.size(0)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1_sum += prec1.item() * data.size(0)
            top5_sum += prec5.item() * data.size(0)
            if teacher_model is not None:
                teacher_logits = teacher_model(data)
                kl = F.kl_div(
                    F.log_softmax(output, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction='batchmean',
                )
                kl_sum += kl.item() * data.size(0)
            total += data.size(0)
            torch.cuda.empty_cache()
    restore_quant_mode(previous_modes)
    total = max(total, 1)
    return {
        'loss': loss_sum / total,
        'kl': kl_sum / total if teacher_model is not None else None,
        'top1': top1_sum / total,
        'top5': top5_sum / total,
    }


def format_logit_metrics(metrics):
    if metrics is None:
        return 'n/a'
    kl_text = 'n/a' if metrics['kl'] is None else '{:.6f}'.format(metrics['kl'])
    return 'top1={:.3f}, top5={:.3f}, ce={:.6f}, teacher_kl={}'.format(
        metrics['top1'], metrics['top5'], metrics['loss'], kl_text
    )


def guarded_head_logit_bias_correction(model, teacher_model, correction_loader, guard_loader,
                                       criterion, device, max_abs):
    if teacher_model is None:
        logging.info('Logit bias correction skipped: teacher model is unavailable.')
        return False
    head = get_classifier_head(model)
    if head is None or head.bias is None:
        logging.info('Logit bias correction skipped: classifier head bias is unavailable.')
        return False

    old_bias = head.bias.detach().clone()
    delta = collect_teacher_logit_bias_delta(model, teacher_model, correction_loader, device, max_abs)
    if delta is None:
        logging.info('Logit bias correction skipped: no correction samples.')
        return False
    delta = delta.to(head.bias.device, dtype=head.bias.dtype)

    before = evaluate_logit_metrics(model, guard_loader, criterion, device, teacher_model=teacher_model)
    logging.info('Logit bias correction guard before: {}'.format(format_logit_metrics(before)))

    candidates = [0.25, 0.50, 0.75, 1.00]
    best_alpha = 0.0
    best_metrics = before
    best_score = before['loss'] + (0.25 * before['kl'] if before['kl'] is not None else 0.0)
    for alpha in candidates:
        head.bias.data.copy_(old_bias + alpha * delta)
        metrics = evaluate_logit_metrics(model, guard_loader, criterion, device, teacher_model=teacher_model)
        score = metrics['loss'] + (0.25 * metrics['kl'] if metrics['kl'] is not None else 0.0)
        logging.info(
            'Logit bias correction candidate alpha={:.2f}: {}'.format(
                alpha, format_logit_metrics(metrics)
            )
        )
        top1_ok = metrics['top1'] >= before['top1'] - 0.05
        ce_ok = metrics['loss'] <= before['loss'] + 0.001
        kl_ok = metrics['kl'] is None or metrics['kl'] <= before['kl'] + 0.0005
        ce_gain = before['loss'] - metrics['loss']
        kl_gain = 0.0 if metrics['kl'] is None else before['kl'] - metrics['kl']
        clear_gain = ce_gain >= 0.002 or kl_gain >= 0.001
        if top1_ok and ce_ok and kl_ok and clear_gain and score < best_score:
            best_alpha = alpha
            best_metrics = metrics
            best_score = score

    if best_alpha > 0:
        head.bias.data.copy_(old_bias + best_alpha * delta)
        logging.info(
            'Logit bias correction keep: alpha={:.2f}, before=({}), after=({}), max_abs_delta={:.6f}'.format(
                best_alpha,
                format_logit_metrics(before),
                format_logit_metrics(best_metrics),
                float(delta.abs().max().detach().cpu()),
            )
        )
        return True

    head.bias.data.copy_(old_bias)
    logging.info(
        'Logit bias correction revert: no candidate improved guarded CE/KL without hurting Top-1. before=({})'.format(
            format_logit_metrics(before)
        )
    )
    return False

def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))

    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")

    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.optim_size = args.optim_size if hasattr(args, 'optim_size') else cfg.optim_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.optim_batch_size = args.optim_batch_size if hasattr(args, 'optim_batch_size') else cfg.optim_batch_size
    cfg.calib_metric = args.calib_metric if hasattr(args, 'calib_metric') else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, 'optim_metric') else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, 'optim_mode') else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else cfg.drop_prob
    cfg.recon_metric = args.recon_metric if hasattr(args, 'recon_metric') else getattr(cfg, 'recon_metric', 'fisher_diag')
    cfg.pct = args.pct if hasattr(args, 'pct') else getattr(cfg, 'pct', 0.9999)
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    cfg.k = args.k if hasattr(args, 'k') else cfg.k
    cfg.p1 = args.p1 if hasattr(args, 'p1') else cfg.p1
    cfg.p2 = args.p2 if hasattr(args, 'p2') else cfg.p2
    cfg.dis_mode = args.dis_mode if hasattr(args, 'dis_mode') else cfg.dis_mode
    # Adaptive k/p controls follow the config by default.
    cfg.adaptive_k = getattr(cfg, 'adaptive_k', False)
    if hasattr(args, 'no_adaptive_k'):
        cfg.adaptive_k = False
    elif hasattr(args, 'adaptive_k'):
        cfg.adaptive_k = True
    cfg.adaptive_p = getattr(cfg, 'adaptive_p', False)
    if hasattr(args, 'no_adaptive_p'):
        cfg.adaptive_p = False
    elif hasattr(args, 'adaptive_p'):
        cfg.adaptive_p = True
    cfg.adaptive_candidate_select = getattr(cfg, 'adaptive_candidate_select', False)
    if hasattr(args, 'adaptive_candidate_select'):
        cfg.adaptive_candidate_select = True
    elif hasattr(args, 'no_adaptive_candidate_select'):
        cfg.adaptive_candidate_select = False
    cfg.adaptive_candidate_margin = getattr(cfg, 'adaptive_candidate_margin', 0.003)
    if args.adaptive_candidate_margin is not None:
        cfg.adaptive_candidate_margin = args.adaptive_candidate_margin
    cfg.logit_guard = getattr(cfg, 'logit_guard', True)
    if hasattr(args, 'logit_guard'):
        cfg.logit_guard = True
    elif hasattr(args, 'no_logit_guard'):
        cfg.logit_guard = False
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")

    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = 'cuda:0'
    device = torch.device(args.device)

    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }

    seed_all(args.seed)

    logging.info('Building model ...')
    try:
        model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(model_zoo[args.model]))
    except:
        model = timm.create_model(model_zoo[args.model], pretrained=True)

    model.to(device)
    model.eval()

    # Keep a full-precision copy for Fisher-guided MLP reconstruction
    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()

    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})

    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    # ================================================================
    # Stage 0: Fisher-guided MLP Reconstruction (optional unified FIM pipeline)
    # ================================================================
    shared_raw_pred_softmaxs = None  # Will be shared across stages
    block_residual_stats = None

    if args.reconstruct_mlp:
        # Replace GELU with ReLU in the model's MLP blocks (keep full_model with GELU)
        for name, module in model.named_modules():
            if name.split('.')[-1] == 'mlp':
                module.act = nn.ReLU()

        if args.load_reconstruct_checkpoint is not None:
            logging.info(f"Restoring checkpoint from '{args.load_reconstruct_checkpoint}'")
            ckpt = torch.load(args.load_reconstruct_checkpoint)
            result = model.load_state_dict(ckpt, strict=False)
            logging.info(str(result))
            model.to(device)
            model.eval()
            if args.test_reconstruct_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        elif args.load_calibrate_checkpoint is None:
            logging.info('Building MLP calibrator ...')
            calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)

            logging.info('{} - Start Fisher-guided MLP reconstruction ...'.format(get_cur_time()))
            mlp_reconstructor = MLPReconstructor(
                model, full_model, calib_loader,
                metric=cfg.recon_metric,
                temp=cfg.temp,
                k=cfg.k,
                p1=cfg.p1,
                p2=cfg.p2
            )
            shared_raw_pred_softmaxs = mlp_reconstructor.reconstruct_model(pct=cfg.pct)
            logging.info("{} - Fisher-guided MLP reconstruction finished.".format(get_cur_time()))

            # Save reconstructed checkpoint
            save_path = os.path.join(root_path, '{}_reconstructed.pth'.format(args.model))
            logging.info(f"Saving checkpoint to {save_path}")
            torch.save(model.state_dict(), save_path)

            logging.info('Validating after MLP reconstruction ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)

            # Release full_model to save GPU memory (raw_pred_softmaxs is already cached)
            del full_model
            torch.cuda.empty_cache()

    # ================================================================
    # Wrap quant modules (after MR so that reconstructed weights are used)
    # ================================================================
    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantization modules (reparam: {}, recon: {}) ...'.format(reparam, args.reconstruct_mlp))
    model = wrap_modules_in_net(model, cfg, reparam=reparam, recon=args.reconstruct_mlp)
    model.to(device)
    model.eval()

    # ================================================================
    # Stage 1: Calibration (with optional Fisher weighting)
    # ================================================================
    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
            if args.test_calibrate_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
            if (cfg.adaptive_k or cfg.adaptive_p) and args.optimize:
                residual_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
                residual_calibrator = QuantCalibrator(
                    model, residual_loader,
                    calib_metric=cfg.calib_metric,
                    temperature=cfg.temp
                )
                block_residual_stats = residual_calibrator.compute_block_residual_stats()
                if args.diagnose_residual_only:
                    logging.info('Residual diagnostics finished; exiting before block reconstruction.')
                    return
        else:
            logging.info("{} - start {} guided calibration".format(get_cur_time(), cfg.calib_metric))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(
                model, calib_loader,
                calib_metric=cfg.calib_metric,
                temperature=cfg.temp
            )
            # Pass shared raw_pred_softmaxs if available (unified Fisher framework)
            quant_calibrator.batching_quant_calib(raw_pred_softmaxs=shared_raw_pred_softmaxs)
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - {} guided calibration finished.".format(get_cur_time(), cfg.calib_metric))
            calibrate_ckpt_path = save_model(model, args, cfg, mode='calibrate')
            logging.info('Validating after calibration ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
            if args.optimize:
                logging.info(
                    "Reloading calibrated checkpoint before block reconstruction: {}".format(
                        calibrate_ckpt_path
                    )
                )
                model = load_model(model, args, device, mode='calibrate', ckpt_path=calibrate_ckpt_path)
            if (cfg.adaptive_k or cfg.adaptive_p) and args.optimize:
                residual_calibrator = QuantCalibrator(
                    model, calib_loader,
                    calib_metric=cfg.calib_metric,
                    temperature=cfg.temp
                )
                block_residual_stats = residual_calibrator.compute_block_residual_stats()
                if args.diagnose_residual_only:
                    logging.info('Residual diagnostics finished; exiting before block reconstruction.')
                    return

    # ================================================================
    # Stage 2: Block Reconstruction (DPLR-FIM AdaRound)
    # ================================================================
    if args.optimize:
        logging.info('Building optim loader ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logit_guard_loader = None
        if cfg.logit_guard and args.logit_guard_size > 0:
            logging.info(
                'Building held-out logit guard loader (size {}, seed {}) ...'.format(
                    args.logit_guard_size,
                    args.seed + args.logit_guard_seed_offset,
                )
            )
            logit_guard_loader = g.calib_loader(
                num=args.logit_guard_size,
                batch_size=cfg.optim_batch_size,
                seed=args.seed + args.logit_guard_seed_offset,
            )
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        block_reconstructor = BlockReconstructor(
            model, cfg.optim_batch_size, calib_loader,
            metric=cfg.optim_metric, temp=cfg.temp,
            k=cfg.k, dis_mode=cfg.dis_mode, p1=cfg.p1, p2=cfg.p2,
            adaptive_k=cfg.adaptive_k, adaptive_p=cfg.adaptive_p,
            block_residual_stats=block_residual_stats,
            logit_guard=cfg.logit_guard,
            logit_guard_batches=args.logit_guard_batches,
            logit_guard_loader=logit_guard_loader,
            adaptive_candidate_select=cfg.adaptive_candidate_select,
            adaptive_candidate_margin=cfg.adaptive_candidate_margin,
        )
        # Share Fisher base if available (unified FIM framework).
        # NOTE: Skip sharing when Fisher calibration was used. The Fisher-guided
        # calibration already optimizes important channels, leaving little headroom
        # for Fisher-DPLR block recon on those same channels (diminishing returns).
        # Letting block_recon compute its own self-consistency softmaxs avoids this.
        if shared_raw_pred_softmaxs is not None and cfg.calib_metric in ['mse', 'mae']:
            block_reconstructor.set_shared_raw_pred_softmaxs(shared_raw_pred_softmaxs)
        block_reconstructor.reconstruct_model(
            quant_act=True,
            mode=cfg.optim_mode,
            drop_prob=cfg.drop_prob,
            keep_gpu=cfg.keep_gpu,
            block_start=args.recon_block_start,
            block_end=args.recon_block_end,
            skip_patch_embed=args.skip_patch_embed,
            skip_head=args.skip_head,
        )
        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        if not args.no_logit_bias_correction:
            logging.info('Building logit bias correction loaders ...')
            bias_loader = g.calib_loader(
                num=args.bias_correction_size,
                batch_size=cfg.optim_batch_size,
                seed=args.seed + args.bias_correction_seed_offset,
            )
            bias_guard_loader = g.calib_loader(
                num=args.bias_correction_guard_size,
                batch_size=cfg.optim_batch_size,
                seed=args.seed + args.bias_correction_seed_offset + 1,
            )
            teacher_model_for_bias = full_model if 'full_model' in locals() else None
            guarded_head_logit_bias_correction(
                model,
                teacher_model_for_bias,
                bias_loader,
                bias_guard_loader,
                criterion,
                device,
                args.bias_correction_max_abs,
            )
        save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building optim loader ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info("{} - finished the process.".format(get_cur_time()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
