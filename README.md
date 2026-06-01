# SynFIM-Q: Synergized Fisher Information Matrix for Vision Transformer PTQ

**SynFIM-Q** 的官方 PyTorch 实现——统一的 Fisher Information Matrix 引导的 Vision Transformer 后训练量化（PTQ）框架。

> SynFIM-Q 融合了两篇 CVPR 2025 工作（APHQ-ViT 和 FIMA-Q），将**同一套 Fisher Information Matrix**贯穿三个量化阶段：**MLP 重建 → 校准 → 块重建（AdaRound）**。

## 核心贡献

1. **Fisher-guided MLP Reconstruction (MR)** — 用 DPLR-FIM（Diagonal + Probabilistic Low-Rank Fisher）替代 APHQ-ViT 中的扰动 Hessian，为 MLP 权重优化提供更合理的输出重要性度量。对 GELU 激活进行截断并用 ReLU 替换。

2. **Fisher-weighted Calibration** — 将 scale/zero-point 搜索扩展为 Fisher 重要性加权。一次 KL 散度反向传播即可计算所有模块的 Fisher 梯度，避免了逐模块的前向传递。

3. **自适应 Fisher 优化（Adaptive k/p）** — 分层动态 Fisher 秩（深层 block 用更高 rank）和自适应 p1/p2 权重（根据激活方差调整低秩项 vs 对角项权重）。

4. **统一 DPLR-FIM 框架** — 首个将 Fisher 估计统一应用于 ViT PTQ 所有阶段的框架。

### 架构：SynFIM-Q Pipeline

```
   ┌──────────────┐    ┌───────────────┐    ┌──────────────────────────┐
   │  Stage 0:    │    │  Stage 1:     │    │  Stage 2:                │
   │  Fisher-MR   │───▶│  Fisher-Calib │───▶│  Fisher-AdaRound         │
   │              │    │               │    │  + Adaptive k/p          │
   │  • GELU→ReLU │    │  • Fisher-grad│    │  • AdaRound opt          │
   │  • fc1,fc2,  │    │    weighted   │    │  • QDrop reg             │
   │    norm2 opt │    │    MSE search │    │  • DPLR loss             │
   └──────────────┘    └───────────────┘    └──────────────────────────┘
```

## 实验结果

### 4-bit DeiT-Tiny (W4A4)

FP 基线: 72.21% Top-1

| 实验 | MR | 校准 | Adaptive k/p | Top-1 | Δ Baseline |
|:----:|:--:|:---:|:------------:|:-----:|:----------:|
| A (Baseline) | ✗ | MSE | ✗ | ~66.8% | — |
| B (+Fisher-MR) | ✅ | MSE | ✗ | 66.93% | +0.13% |
| **C (+Fisher-Calib)** | ✗ | **Fisher** | ✗ | **67.20%** | **+0.40%** |
| D (Full SynFIM) | ✅ | Fisher | ✗ | 66.55% | -0.25% |
| **E (+Adaptive)** | ✗ | MSE | ✅ | **67.14%** | **+0.34%** |
| C+E (Fisher+Adaptive) | ✗ | Fisher | ✅ | 66.95% | +0.15% |

**关键发现**：
- **单独使用效果显著**：Fisher 校准 (+0.40%) 和 Adaptive k/p (+0.34%) 各自优于 Baseline
- **叠加无增益**：4-bit 下多个 Fisher 优化叠加存在 diminishing returns，因为量化误差本身有限，优化头寸被第一个 Fisher 阶段消耗
- **C（Fisher 校准）为 4-bit 最优方案**，无需 MR 和 Adaptive 即可达到 67.20%

### 3-bit DeiT-Tiny (W3A3)

待测试。预期 3-bit 下量化误差更大，Fisher 优化叠加将展现累加增益。

## 环境准备

- 克隆仓库：

```bash
git clone https://github.com/Picasso9jiu/SynFIM-Q.git
cd SynFIM-Q
```

- 安装 PyTorch 和 [timm](https://github.com/huggingface/pytorch-image-models/tree/main)：

```bash
pip install torch torchvision timm
```

- 预训练 ViT 权重可通过 timm 直接加载，或下载到本地：

```bash
wget https://github.com/GoatWu/AdaLog/releases/download/v1.0/deit_tiny_patch16_224.bin
mkdir -p ./checkpoints/vit_raw/
mv deit_tiny_patch16_224.bin ./checkpoints/vit_raw/
```

## 数据集

ImageNet (ILSVRC2012) 验证集。默认路径：
```
D:/AI/IaS-ViT-main/dataset/imagenet/val/
```
可通过 `--dataset /path/to/imagenet` 覆盖。

## 快速开始

### 4-bit DeiT-Tiny 完整流程

```bash
# Fisher 校准 + Fisher-DPLR AdaRound（4-bit 推荐配置）
python test_quant.py \
  --model deit_tiny \
  --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 \
  --calib-metric fisher_diag \
  --calibrate --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

### 4-bit 消融实验

```bash
# A: Baseline（FIMA-Q 等价：无 MR，MSE 校准，Fisher-DPLR AdaRound）
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric mse --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# B: +Fisher-MR only
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric mse --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# C: +Fisher-Calib only（4-bit 最优）
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric fisher_diag --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# E: Baseline + Adaptive k/p
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric mse --calibrate --optimize \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

### 3-bit 消融实验

```bash
# A3: Baseline
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --calib-metric mse --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# B3: +Fisher-MR only
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric mse --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# C3: +Fisher-Calib only
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --calib-metric fisher_diag --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# D3: Full SynFIM（MR + Fisher 校准 + Fisher-DPLR）
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric fisher_diag --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# E3: Baseline + Adaptive k/p（k=8）
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --calib-metric mse --calibrate --optimize \
  --k 8 --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# F3: Full SynFIM-Q（所有优化 + Adaptive k/p, k=8）
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric fisher_diag --calibrate --optimize \
  --k 8 --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

## 断点续跑

```bash
# 加载校准 checkpoint，跳过校准直接跑 Block Reconstruction
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 --calib-metric fisher_diag \
  --load-calibrate-checkpoint ./checkpoints/quant_result/xxx/xxx.pth \
  --optimize --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"

# 加载 Block Reconstruction checkpoint，直接测试
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 \
  --load-optimize-checkpoint ./checkpoints/quant_result/xxx/xxx.pth \
  --test-optimize-checkpoint --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

## 自适应 Fisher 优化

### 1. 分层动态秩 (`adaptive_k=True`，默认启用)

根据 block 深度分配不同的 Fisher 秩（k）：

- **深层**（patch_embed, blocks 0-2）：`k_base + 3`（更丰富的 Fisher，最大 12）
- **中层**（blocks 3-8）：`k_base`（默认值）
- **Head**：`k_base - 2`（最小 1）

3-bit 建议使用 `--k 8` 将各层上移。

### 2. 自适应 p1/p2 权重 (`adaptive_p=True`，默认启用)

根据每层激活标准差动态调整低秩项 vs 对角项权重：

- **高方差**（std > 2.5）：`p1 × 1.3, p2 × 0.7` — 低秩项更重要
- **正常**（1.0 < std ≤ 2.5）：`p1, p2` — 默认
- **低方差**（std ≤ 1.0）：`p1 × 0.7, p2 × 1.3` — 对角项更稳定

控制参数：

```bash
--no-adaptive-k     # 禁用分层动态秩（全局使用 k）
--no-adaptive-p     # 禁用自适应 p1/p2（全局使用 p1/p2）
--k 8               # 调整基础 Fisher 秩
```

## 配置文件

编辑 `configs/4bit/fim_unified.py` 或创建新文件：

```python
class Config:
    def __init__(self):
        # 校准
        self.calib_size = 128
        self.calib_batch_size = 32
        self.calib_metric = 'mse'       # 'mse' | 'fisher_diag'
        # 量化位宽
        self.w_bit = 4
        self.a_bit = 4
        # Block Reconstruction
        self.optim_size = 1024
        self.optim_batch_size = 32
        self.optim_metric = 'fisher_dplr'
        self.temp = 20
        # MLP Reconstruction
        self.recon_metric = 'fisher_diag'
        self.pct = 0.9999
        # Fisher 参数（跨阶段共享）
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        # 自适应优化
        self.adaptive_k = True
        self.adaptive_p = True
        # QDrop
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5
```

## 支持的模型

- **ViT**: Tiny, Small, Base, Large (patch16_224)
- **DeiT**: Tiny, Small, Base (patch16_224)
- **Swin**: Tiny, Small, Base (patch4_window7_224), Base (patch4_window12_384)

所有预训练权重通过 `timm` 获取或从 `./checkpoints/vit_raw/` 加载。

## 代码结构

```
SynFIM-Q/
├── test_quant.py              # 主入口（3 阶段 pipeline）
├── configs/
│   ├── 3bit/                  # 3-bit 量化配置
│   ├── 4bit/                  # 4-bit 量化配置
│   │   ├── best.py            # 标准 FIMA-Q 配置
│   │   └── fim_unified.py     # ★ SynFIM-Q 统一配置
│   └── 6bit/                  # 6-bit 量化配置
├── utils/
│   ├── mlp_recon.py           # ★ Fisher 引导的 MLP 重建
│   ├── calibrator.py          # QuantCalibrator（Fisher 扩展）
│   ├── block_recon.py         # BlockReconstructor + LossFunction（DPLR-FIM + Adaptive）
│   ├── datasets.py            # ImageNet 数据加载器
│   ├── wrap_net.py            # 模型包装（Linear/Conv/MatMul → Quant）
│   └── test_utils.py          # 精度辅助函数
├── quantizers/
│   ├── uniform.py             # UniformQuantizer
│   ├── adaround.py            # AdaRoundQuantizer
│   └── logarithm.py           # Log2 quantizer
└── quant_layers/
    ├── linear.py              # QuantLinear 变体
    ├── conv.py                # QuantConv2d
    └── matmul.py              # QuantMatMul（Q@K, Attn@V）
```

## 设计说明

### 为什么用 Fisher 引导 MLP 重建？

APHQ-ViT 使用基于扰动（±1e-6）的 Hessian 来估计输出重要性。我们的 Fisher 方法：
- 使用单次反向传播 + KL 散度 loss，与下游阶段的计算一致
- 天然集成 DPLR-FIM：对角 Fisher 用于逐通道加权

### 为什么用 Fisher 加权校准？

标准校准最小化原始输出和量化输出之间的 MSE。我们的 Fisher 加权变体用梯度幅度加权误差，优先保护对最终 loss 更重要的通道。

### 为什么 4-bit 下叠加无增益？

4-bit 量化误差本身有限。Fisher 校准已将最重要的通道优化到接近 FP，后续 Fisher-DPLR 优化在这些通道上的梯度接近零，AdaRound 无法进一步改进。3-bit 下误差更大，预期叠加增益将体现。

## 引用

```bibtex
@inproceedings{wu2025fimaq,
  title={FIMA-Q: Post-Training Quantization for Vision Transformers by
         Fisher Information Matrix Approximation},
  author={Wu, Zhuguanyu and Wang, Shihe and Zhang, Jiayi and Chen, Jiaxin
          and Wang, Yunhong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

详见 [LICENSE](LICENSE) 文件。

---

*维护者：[Picasso9jiu](https://github.com/Picasso9jiu)*
