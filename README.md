# SynFIM-Q: Synergized Fisher Information Matrix for Vision Transformer PTQ

**SynFIM-Q** 的官方 PyTorch 实现——统一的 Fisher Information Matrix 引导的 Vision Transformer 后训练量化（PTQ）框架。

> SynFIM-Q 融合了两篇 CVPR 2025 工作（APHQ-ViT 和 FIMA-Q），将**同一套 Fisher Information Matrix**贯穿三个量化阶段：**MLP 重建 → 校准 → 块重建（AdaRound）**。

## 核心贡献

1. **Fisher-guided MLP Reconstruction (MR)** — 用 DPLR-FIM（Diagonal + Probabilistic Low-Rank Fisher）替代 APHQ-ViT 中的扰动 Hessian，为 MLP 权重优化提供更合理的输出重要性度量。对 GELU 激活进行截断并用 ReLU 替换。

2. **Fisher-weighted Calibration** — 将 scale/zero-point 搜索扩展为 Fisher 重要性加权。一次 KL 散度反向传播即可计算所有模块的 Fisher 梯度，避免了逐模块的前向传递。

3. **自适应 Fisher 优化（Adaptive k/p）** — 分层动态 Fisher 秩（深层 block 用更高 rank）和自适应 p1/p2 权重（根据激活方差调整低秩项 vs 对角项权重）。

4. **Bit-width-aware Fisher 注入策略** — 系统分析 Fisher 信息在 ViT PTQ 多阶段中的协同与竞争，并根据量化位宽选择更合适的 Fisher 注入阶段，而不是盲目叠加所有模块。

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

FP 基线: 72.21% Top-1

PTQ Baseline: 55.55% Top-1

| 实验 | MR | 校准 | Adaptive k/p | k | Top-1 | Δ Baseline | 关键结论 |
|:----:|:--:|:---:|:------------:|:-:|:-----:|:----------:|:--------|
| Baseline | ✗ | MSE | ✗ | 5 | 55.55% | — | 标准 3-bit PTQ 对照 |
| B3 (+Fisher-MR) | ✅ | MSE | ✗ | 5 | 56.82% | +1.27% | MR 在 3-bit 下提供稳定增益 |
| C3 (+Fisher-Calib) | ✗ | Fisher | ✗ | 5 | 56.40% | +0.85% | Fisher 校准单独有效，但弱于 MR |
| D3 (MR + Fisher-Calib) | ✅ | Fisher | ✗ | 5 | 56.68% | +1.13% | 校准后精度提升，但后续重建存在收益抵消 |
| E3 (+Adaptive) | ✗ | MSE | ✅ | 8 | 55.99% | +0.44% | Adaptive 单独不如 MR |
| B+adaptive_k | ✅ | MSE | k only | 8 | 56.51% | +0.96% | 仅动态 k 不如完整 Adaptive k/p |
| **B+E (推荐)** | ✅ | MSE | ✅ | 8 | **56.92%** | **+1.37%** | **3-bit 当前最佳：MR 与 Adaptive 互补** |

**关键发现**：
- **MR 是 3-bit 的核心增益来源**：B3 相比 C3/E3 更稳定，说明低比特下先修正 MLP 非线性误差非常重要。
- **Adaptive k/p 与 MR 互补**：B+E 在 B3 基础上进一步提升到 56.92%；仅启用 adaptive_k 的 B+adaptive_k 为 56.51%，说明动态秩和自适应 p1/p2 需要配合使用。
- **Fisher-Calib 的收益主要体现在校准阶段**：D3 与 D3-no-share 的校准后精度高于 B3，但最终 Top-1 没有超过 B3/B+E，表明 Fisher-Calib 与后续 Fisher-DPLR AdaRound 仍存在阶段竞争。

### 推荐配置

SynFIM-Q 不默认把所有 Fisher 模块全部叠加，而是根据量化误差强度选择 Fisher 注入阶段：

| 位宽 | 推荐策略 | 命令关键参数 | 说明 |
|:---:|:--------|:------------|:-----|
| W4A4 | Fisher-Calib | `--calib-metric fisher_diag --no-adaptive-k --no-adaptive-p` | 4-bit 误差较小，Fisher 校准是最有效的轻量注入点 |
| W3A3 | Fisher-MR + Adaptive Fisher-DPLR | `--reconstruct-mlp --calib-metric mse --k 8` | 3-bit 误差更大，MR 与 Adaptive k/p 更互补 |

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

### 4-bit DeiT-Tiny 推荐流程

```bash
# Fisher 校准 + Fisher-DPLR AdaRound（4-bit 推荐配置：禁用 Adaptive k/p）
python test_quant.py \
  --model deit_tiny \
  --config ./configs/4bit/fim_unified.py \
  --w_bit 4 --a_bit 4 \
  --calib-metric fisher_diag \
  --calibrate --optimize \
  --no-adaptive-k --no-adaptive-p \
  --dataset "D:/AI/IaS-ViT-main/dataset/imagenet"
```

### 3-bit DeiT-Tiny 推荐流程

```bash
# Fisher-MR + MSE 校准 + Adaptive Fisher-DPLR（3-bit 推荐配置）
python test_quant.py \
  --model deit_tiny \
  --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 \
  --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric mse \
  --calibrate --optimize \
  --k 8 \
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

# B+E: Fisher-MR + Adaptive k/p（3-bit 当前推荐）
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py \
  --w_bit 3 --a_bit 3 --reconstruct-mlp --recon-metric fisher_diag \
  --calib-metric mse --calibrate --optimize \
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

实验证明，在 3-bit DeiT-Tiny 上仅启用 adaptive_k 不如同时启用 adaptive_k 与 adaptive_p；完整 Adaptive k/p 与 Fisher-MR 组合达到当前最佳 56.92% Top-1。

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

### 为什么不简单叠加所有 Fisher 模块？

SynFIM-Q 的设计目标不是把 MR、Fisher-Calib 和 Fisher-DPLR AdaRound 全部无条件打开，而是研究 Fisher 信息在不同 PTQ 阶段的作用边界。实验表明，Fisher 引导会优先优化对最终 loss 更敏感的通道；当一个阶段已经充分修正这些通道后，后续阶段继续使用强 Fisher 约束可能出现收益饱和甚至目标竞争。

因此，SynFIM-Q 采用 **bit-width-aware Fisher injection**：
- **W4A4**：量化误差较小，Fisher-Calib 是最有效的轻量注入点。
- **W3A3**：量化误差更大，MR 先修正 MLP 非线性误差，再由 Adaptive Fisher-DPLR 优化 rounding，二者更互补。

这种阶段选择比“全模块叠加”更稳定，也更符合不同位宽下误差来源不同的事实。

### 为什么用 Fisher 引导 MLP 重建？

APHQ-ViT 使用基于扰动（±1e-6）的 Hessian 来估计输出重要性。我们的 Fisher 方法：
- 使用 KL 散度反向传播估计输出对最终预测分布的敏感性，与下游 Fisher-DPLR AdaRound 的目标保持一致。
- 用 Fisher 重要性替代固定扰动 Hessian，使 MLP 的 `fc1/fc2/norm2` 优化更关注影响最终分类结果的输出维度。
- 对 GELU 激活进行截断并用 ReLU 近似，降低低比特量化下非线性激活的极端值影响。

在 W3A3 上，MR 是最稳定的增益来源，说明低比特 ViT PTQ 中 MLP 非线性误差是主要瓶颈之一。

### 为什么用 Fisher 加权校准？

标准校准最小化原始输出和量化输出之间的 MSE，对所有输出维度近似等权处理。Fisher-Calib 用梯度幅度加权 scale/zero-point 搜索中的输出误差，优先保护对最终 loss 更重要的通道。

这一策略在 W4A4 上最有效：4-bit 量化误差相对有限，校准阶段已经能把关键通道调整到较优位置，因此后续继续叠加 MR 或强 Fisher-DPLR 的收益空间较小。

在 W3A3 上，Fisher-Calib 也能提高校准阶段精度，但与后续 Fisher-DPLR AdaRound 的优化目标存在部分重叠，最终 Top-1 不如 MR + Adaptive k/p 稳定。

### 为什么需要 Adaptive k/p？

Fisher-DPLR 由低秩项和对角项共同描述输出敏感性。固定 rank 和固定 `p1/p2` 在所有层上使用同一强度，无法适应 ViT 不同 block 的激活分布差异。

Adaptive k/p 做了两件事：
- **Adaptive k**：早期 block 和 patch embedding 使用更高 Fisher rank，head 使用更低 rank，降低无效低秩估计。
- **Adaptive p1/p2**：根据 block 输出标准差调整低秩项与对角项权重，高方差层更依赖低秩相关性，低方差层更依赖稳定的对角加权。

3-bit 实验显示，仅启用 adaptive_k 的 B+adaptive_k 为 56.51%，低于完整 B+E 的 56.92%，说明动态 rank 和自适应 `p1/p2` 需要协同使用。

### 为什么 4-bit 下叠加无增益，而 3-bit 更依赖 MR + Adaptive？

4-bit 和 3-bit 的主要误差来源不同，因此最优 Fisher 注入阶段也不同。

4-bit 量化误差本身有限。Fisher 校准已将最重要的通道优化到接近 FP，后续 Fisher-DPLR 优化在这些通道上的梯度接近零，AdaRound 难以进一步改进。

3-bit 下量化误差更大，MLP 非线性替换和低比特 rounding 误差同时放大。实验证明 MR 是最稳定的低比特修正项，而 Adaptive k/p 能在 MR 后继续改善 Fisher-DPLR AdaRound；相比之下，Fisher-Calib 虽能提高校准阶段精度，但与后续 Fisher-DPLR 存在一定阶段竞争。

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
