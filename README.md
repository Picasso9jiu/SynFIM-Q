# SynFIM-Q: Fisher-Calibrated Adaptive PTQ for Vision Transformers

SynFIM-Q is a PyTorch implementation for post-training quantization (PTQ) of Vision Transformers. The current main branch focuses on improving DeiT-Tiny W4A4 PTQ by combining two effective Fisher-based stages:

- **Fisher-Calib (C)**: Fisher-weighted scale/zero-point calibration.
- **Adaptive Fisher-DPLR AdaRound (E)**: Fisher-DPLR block reconstruction with residual-aware adaptive `k/p` and a hybrid keep/revert guard.

The project is built on top of the engineering ideas from APHQ-ViT and FIMA-Q, but the current effective contribution is not a direct stack of all modules. We found that naive stacking can reduce final ImageNet accuracy, even when each module works well alone. SynFIM-Q therefore adds an adaptive and guarded reconstruction strategy so that useful Fisher-DPLR updates are retained while harmful or overfitted updates are reverted.

## Highlights

- **Fisher-weighted calibration**: moves Fisher sensitivity into the calibration stage, so quantization parameters protect dimensions that matter more to the final loss.
- **Residual-aware adaptive `k/p`**: adjusts the Fisher low-rank rank `k` and the low-rank/diagonal weights `p1/p2` according to block residual statistics.
- **Hybrid reconstruction guard**: combines local MSE, final reconstruction loss, and full-model logits/confidence signals to decide whether each block reconstruction should be kept or reverted.
- **C+E synergy**: after guard correction, Fisher-Calib and Adaptive Fisher-DPLR improve together on DeiT-Tiny W4A4 instead of cancelling each other.

## Main Result

Dataset: ImageNet validation set

Model: DeiT-Tiny

Full precision reference: about 72.21% Top-1

### W4A4 DeiT-Tiny

| Setting | Calibration | Block Reconstruction | Guard | Top-1 | Top-5 | Loss |
|---|---|---|---|---:|---:|---:|
| Baseline PTQ | MSE | Fisher-DPLR | MSE guard | about 66.8 | - | - |
| C only | Fisher-Calib | Fisher-DPLR | no adaptive k/p | 67.198 | 88.258 | 1.518 |
| Previous C+E best | Fisher-Calib | Adaptive Fisher-DPLR | MSE guard | 67.176 | 88.194 | 1.477 |
| Failed C+E variant | Fisher-Calib | Adaptive Fisher-DPLR | over-sensitive logits guard | 66.646 | 87.936 | 1.495 |
| **SynFIM-Q C+E5** | **Fisher-Calib** | **Adaptive Fisher-DPLR** | **hybrid guard** | **67.364** | **88.530** | **1.474** |

The best current W4A4 result is from log `20260608_0813_C&E5`.

Compared with C only, C+E5 improves:

- Top-1: `+0.166`
- Top-5: `+0.272`
- Loss: `1.518 -> 1.474`

This result changes the earlier conclusion: C and E can be stacked effectively, but only when the block reconstruction stage uses a guard that avoids both mid-layer under-retention and late-layer calibration overfitting.

## What Changed in C+E5

Earlier C+E experiments showed that validation-on-calibration accuracy was not a reliable proxy for final ImageNet accuracy. Some blocks improved calibration logits but hurt final accuracy; other blocks had tiny calibration Top-1 fluctuations but still improved reconstruction and final performance.

C+E5 uses the following policy:

- Keep useful mid-layer updates even when calibration Top-1 has small noise.
- Revert clearly harmful updates when CE or confidence signals degrade.
- Prevent late blocks (`blocks.9` to `blocks.11`) from bypassing MSE/loss stability just because calibration logits improve.
- Keep the classifier head update when both local MSE and logits improve.

In the best run, `patch_embed`, `blocks.0` to `blocks.9`, and `head` were kept, while `blocks.10` and `blocks.11` were reverted.

## Method

### 1. Fisher-Calib

Standard PTQ calibration searches quantization scale and zero-point by minimizing output MSE. Fisher-Calib reweights this search with Fisher sensitivity, so calibration prioritizes output dimensions that are more important to the final objective.

In this codebase, Fisher-Calib is enabled with:

```bash
--calib-metric fisher_diag
```

### 2. Adaptive Fisher-DPLR AdaRound

FIMA-Q uses a Fisher-DPLR approximation during block reconstruction. SynFIM-Q extends this stage with residual-aware adaptive parameters:

- `k`: rank of the low-rank Fisher component.
- `p1`: weight of the low-rank Fisher term.
- `p2`: weight of the diagonal Fisher term.

The implementation uses block residual statistics to decide whether a block should use stronger low-rank modeling. In the current best W4A4 setup, `p2` remains conservative because previous experiments showed that aggressive `p2` increases can hurt final accuracy.

### 3. Hybrid Guard

After each block reconstruction, the model decides whether to keep or revert the update.

The guard checks:

- local block MSE before/after reconstruction;
- final reconstruction loss for sensitive late blocks;
- full-model logits on the calibration set, including CE, Top-1, true-class probability, margin, and prediction flips.

The guard is intentionally not a pure logits rule. Calibration-set logits are noisy and can overfit, so logits improvements do not automatically override local stability for late blocks.

## Quick Start

### Environment

```bash
pip install torch torchvision timm
```

Prepare ImageNet validation data, for example:

```text
D:/AI/IaS-ViT-main/dataset/imagenet/val/
```

### Run the Current W4A4 C+E Pipeline

From an existing Fisher-Calib checkpoint:

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/20260530_1051_C/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda
```

From scratch:

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda
```

### Reproduce C Only

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-logit-guard
```

## Useful Flags

| Flag | Meaning |
|---|---|
| `--calib-metric fisher_diag` | Enable Fisher-weighted calibration. |
| `--adaptive-k` / `--no-adaptive-k` | Enable or disable adaptive Fisher rank. |
| `--adaptive-p` / `--no-adaptive-p` | Enable or disable adaptive `p1/p2`. |
| `--no-logit-guard` | Disable the full-model logits guard. |
| `--logit-guard-batches N` | Evaluate logits guard on only the first `N` calibration batches. |
| `--recon-block-start N` | Start block reconstruction from block `N`. |
| `--recon-block-end N` | End block reconstruction at block `N`. |
| `--skip-patch-embed` | Skip patch embedding reconstruction. |
| `--skip-head` | Skip classifier head reconstruction. |
| `--diagnose-residual-only` | Only diagnose residual Fisher statistics, without running reconstruction. |

## Code Structure

```text
SynFIM-Q/
|-- test_quant.py                 # Main entry point
|-- configs/
|   |-- 3bit/
|   |-- 4bit/
|   |   |-- best.py
|   |   `-- fim_unified.py        # Current SynFIM-Q config
|   `-- 6bit/
|-- utils/
|   |-- calibrator.py             # Calibration and Fisher-Calib logic
|   |-- block_recon.py            # Fisher-DPLR AdaRound, adaptive k/p, hybrid guard
|   |-- mlp_recon.py              # Optional MLP reconstruction
|   |-- datasets.py
|   |-- wrap_net.py
|   `-- test_utils.py
|-- quantizers/
|   |-- uniform.py
|   |-- adaround.py
|   `-- logarithm.py
`-- quant_layers/
    |-- linear.py
    |-- conv.py
    `-- matmul.py
```

## Implementation Notes

Recent fixes and additions include:

- fixed residual Fisher diagnosis so gradients are captured during quantized forward passes;
- fixed asymmetric AdaRound hard-value reconstruction with `zero_point`;
- added per-block final loss logging;
- added selective block reconstruction controls;
- added full-model logits/confidence guard;
- changed logits guard from over-sensitive single-metric rejection to a hybrid policy that keeps useful mid-layer updates and prevents late-layer overfitting.

## Relation to APHQ-ViT and FIMA-Q

SynFIM-Q is based on the observation that APHQ-ViT and FIMA-Q are complementary but not automatically compatible when directly stacked.

- Compared with APHQ-ViT, this project focuses less on using MR as the main W4A4 contribution and instead uses Fisher-Calib plus adaptive Fisher-DPLR reconstruction as the current effective pipeline.
- Compared with FIMA-Q, SynFIM-Q adds Fisher-weighted calibration, residual-aware adaptive `k/p`, and a guard mechanism that decides whether each reconstructed block should be retained.

The current W4A4 result shows that the main issue is not whether Fisher information is useful, but where and how strongly it should be injected across PTQ stages.

## Citation

If you use this repository, please also cite the related works that this project builds on:

```bibtex
@inproceedings{wu2025fimaq,
  title={FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation},
  author={Wu, Zhuguanyu and Wang, Shihe and Zhang, Jiayi and Chen, Jiaxin and Wang, Yunhong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

See [LICENSE](LICENSE).
