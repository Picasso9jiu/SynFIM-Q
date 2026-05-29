# SynFIM-Q: Synergized Fisher Information Matrix for Vision Transformer PTQ

Official PyTorch implementation of **SynFIM-Q** — a unified Fisher Information Matrix-guided Post-Training Quantization framework for Vision Transformers.

> **SynFIM-Q** synergizes two CVPR 2025 works (APHQ-ViT and FIMA-Q) into a single framework where the **same Fisher Information Matrix** guides all three quantization stages: **MLP Reconstruction → Calibration → Block Reconstruction (AdaRound)**.

## Key Contributions

1. **Fisher-guided MLP Reconstruction (MR)** — Replaces the perturbation Hessian in APHQ-ViT with DPLR-FIM (Diagonal + Probabilistic Low-Rank Fisher), providing a more principled importance measure for MLP weight optimization. GELU activations are clamped and replaced with ReLU for quantization-friendliness.

2. **Fisher-weighted Calibration** — Extends the scale/zero-point search with Fisher importance weights. A single KL-divergence backward pass computes per-module Fisher gradients, avoiding wasteful per-module forward passes.

3. **Shared Fisher Backbone** — The `raw_pred_softmaxs` (full model outputs on calibration set) are computed once during MR and reused across all three stages, saving compute and ensuring consistency.

4. **Unified FIM Pipeline** — First framework to unify Fisher estimation across all PTQ stages for Vision Transformers.

### Architecture: SynFIM-Q Pipeline

```
   ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
   │  Stage 0:    │    │  Stage 1:     │    │  Stage 2:        │
   │  Fisher-MR   │───▶│  Fisher-Calib │───▶│  Fisher-AdaRound │
   │              │    │               │    │  (DPLR-FIM)      │
   │  • GELU→ReLU │    │  • Fisher-grad│    │  • AdaRound opt  │
   │  • fc1,fc2,  │    │    weighted   │    │  • QDrop reg     │
   │    norm2 opt │    │    MSE search │    │  • DPLR loss     │
   └──────┬───────┘    └──────┬────────┘    └────────┬─────────┘
          │                   │                      │
          └───────────────────┴──────────────────────┘
                  shared raw_pred_softmaxs
```

## Getting Started

- Clone this repo:

```bash
git clone https://github.com/Picasso9jiu/SynFIM-Q.git
cd SynFIM-Q
```

- Install PyTorch and [timm](https://github.com/huggingface/pytorch-image-models/tree/main):

```bash
pip install torch torchvision timm
```

- Pretrained ViT checkpoints can be obtained via timm or directly downloaded:

```bash
wget https://github.com/GoatWu/AdaLog/releases/download/v1.0/deit_tiny_patch16_224.bin
mkdir -p ./checkpoints/vit_raw/
mv deit_tiny_patch16_224.bin ./checkpoints/vit_raw/
```

## Dataset

ImageNet (ILSVRC2012) validation set. Default path:
```
D:/AI/IaS-ViT-main/dataset/imagenet/val/
```
Or pass `--dataset /path/to/imagenet` to override.

## Quick Start

### Full SynFIM-Q Pipeline (4-bit DeiT-Tiny)

```bash
# Stage 0 + 1 + 2: Fisher-guided MR → Fisher Calib → Fisher DPLR AdaRound
python test_quant.py \
  --model deit_tiny \
  --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 \
  --reconstruct-mlp \
  --calibrate \
  --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

### Ablation Study Commands

To isolate the contribution of each Fisher-guided stage:

| Experiment | MR | Calibration | AdaRound | Description |
|:----------:|:--:|:----------:|:--------:|:-----------:|
| A (Baseline) | ✗ | MSE | Fisher-DPLR | FIMA-Q equivalent |
| B (+Fisher-MR) | Fisher-Diag | MSE | Fisher-DPLR | Test MR gain |
| C (+Fisher-Calib) | ✗ | Fisher-Diag | Fisher-DPLR | Test Calib gain |
| D (Full SynFIM) | Fisher-Diag | Fisher-Diag | Fisher-DPLR | Full unified gain |

```bash
# A: Baseline (no MR, MSE calib, Fisher-DPLR AdaRound)
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric mse --optim-metric fisher_dplr \
  --calibrate --optimize --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# B: +Fisher-MR only
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric mse --calibrate --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# C: +Fisher-Calib only
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric fisher_diag --calibrate --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# D: Full SynFIM-Q (all three stages Fisher-guided)
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric fisher_diag --calibrate --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

## Evaluation Commands

Example: Calibrate + optimize a model from scratch.

```bash
python test_quant.py --model vit_small --config ./configs/4bit/fim_unified.py \
  --dataset ~/data/ILSVRC/Data/CLS-LOC --calibrate --optimize \
  --optim-metric fisher_dplr
```

Example: Load a calibrated checkpoint, then run optimization.

```bash
python test_quant.py --model vit_small --config ./configs/4bit/fim_unified.py \
  --dataset ~/data/ILSVRC/Data/CLS-LOC \
  --load-calibrate-checkpoint ./checkpoints/quant_result/vit_small_w4_a4_calibsize_128_mse.pth \
  --optimize --optim-metric fisher_dplr
```

Example: Load an optimized checkpoint and test.

```bash
python test_quant.py --model vit_small --config ./configs/4bit/fim_unified.py \
  --dataset ~/data/ILSVRC/Data/CLS-LOC \
  --load-optimize-checkpoint ./checkpoints/quant_result/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_recon_qdrop.pth \
  --test-optimize-checkpoint
```

## Configuration

Edit `configs/4bit/fim_unified.py` or create your own:

```python
class Config:
    def __init__(self):
        # Calibration
        self.calib_size = 128
        self.calib_batch_size = 32
        self.calib_metric = 'mse'       # 'mse' | 'fisher_diag'
        # Quantization bits
        self.w_bit = 4
        self.a_bit = 4
        # Block Reconstruction
        self.optim_size = 1024
        self.optim_batch_size = 32
        self.optim_metric = 'fisher_dplr'  # 'fisher_dplr' | 'fisher_diag' | 'mse' | 'mae'
        self.temp = 20
        # MLP Reconstruction (SynFIM-Q)
        self.recon_metric = 'fisher_diag'  # 'fisher_diag' | 'fisher_dplr' | 'mse' | 'mae'
        self.pct = 0.9999                  # GELU clamping percentile
        # Fisher parameters (shared across all stages)
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        # QDrop
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5
```

## Supported Models

- **ViT**: Tiny, Small, Base, Large (patch16_224)
- **DeiT**: Tiny, Small, Base (patch16_224)
- **Swin**: Tiny, Small, Base (patch4_window7_224), Base (patch4_window12_384)

All pretrained weights are obtained via `timm` or from `./checkpoints/vit_raw/`.

## Code Structure

```
SynFIM-Q/
├── test_quant.py              # Main entry point (3-stage pipeline)
├── configs/
│   ├── 3bit/                  # 3-bit quantization configs
│   ├── 4bit/                  # 4-bit quantization configs
│   │   ├── best.py            # Standard FIMA-Q config
│   │   └── fim_unified.py     # ★ SynFIM-Q unified config
│   └── 6bit/                  # 6-bit quantization configs
├── utils/
│   ├── mlp_recon.py           # ★ Fisher-guided MLP Reconstruction (new)
│   ├── calibrator.py          # QuantCalibrator (Fisher-extended)
│   ├── block_recon.py         # BlockReconstructor + LossFunction (DPLR-FIM)
│   ├── datasets.py            # ImageNet data loader
│   ├── wrap_net.py            # Model wrapping (Linear/Conv/MatMul → Quant)
│   └── test_utils.py          # Accuracy helpers
├── quantizers/
│   ├── uniform.py             # UniformQuantizer (per-tensor/channel)
│   ├── adaround.py            # AdaRoundQuantizer (learned rounding)
│   └── logarithm.py           # Log2 quantizer
└── quant_layers/
    ├── linear.py              # QuantLinear variants
    ├── conv.py                # QuantConv2d
    └── matmul.py              # QuantMatMul (Q@K, Attn@V)
```

## Design Decisions

### Why Fisher-guided MLP Reconstruction?

APHQ-ViT uses a perturbation-based Hessian (two forward/backward passes with ±1e-6 perturbation) to estimate output importance. Our Fisher-based approach:
- Uses a single backward pass with KL divergence loss, matching the Fisher computation used in downstream stages
- Naturally integrates with DPLR-FIM: diagonal Fisher for per-channel weighting

### Why Shared Fisher Base?

Computing `raw_pred_softmaxs` once and reusing across MR → Calibration → BlockRecon:
- Saves ~2 full-model forward passes per stage
- Ensures Fisher gradients are computed against the same target distribution

### Why Fisher-weighted Calibration?

Standard calibration minimizes MSE between raw and quantized outputs. Our Fisher-weighted variant weights errors by the gradient magnitude at each output channel, prioritizing channels that matter more for the final loss.

## Citation

If you find SynFIM-Q useful, please consider citing:

```bibtex
@inproceedings{wu2025fimaq,
  title={FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation},
  author={Wu, Zhuguanyu and Wang, Shihe and Zhang, Jiayi and Chen, Jiaxin and Wang, Yunhong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.

---

*Maintained by [Picasso9jiu](https://github.com/Picasso9jiu)*
