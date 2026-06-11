# SynFIM-Q：面向 Vision Transformer 的 Fisher 感知自适应 PTQ

SynFIM-Q 是一个基于 PyTorch 的 Vision Transformer 后训练量化（Post-Training Quantization, PTQ）工程。当前版本主要针对 DeiT-Tiny 在 ImageNet 上的 W4A4 量化进行优化。

本工程参考了 APHQ-ViT 与 FIMA-Q 的实现思路，但当前有效主线不是简单叠加所有模块，而是以 **Fisher-Calib checkpoint** 为统一起点，在块重构阶段加入 **自适应候选选择式 Fisher-DPLR AdaRound**。核心思想是：每个 block 同时构造固定 `k/p` 候选和 adaptive `k/p` 候选，再用 MSE 与 logits guard 判断最终保留哪一个。

## 核心结果

- 模型：DeiT-Tiny
- 数据集：ImageNet validation
- 量化设置：W4A4

| 方法 | 起点 | 块重构策略 | Top-1 | Top-5 | Loss | 日志 |
|---|---|---|---:|---:|---:|---|
| 实验 C | Fisher-Calib | 固定 `k=5, p1=p2=1` | 67.198 | 88.258 | 1.518 | `20260530_1051_C` |
| 当前代码 fixed 对照 | 同一 Fisher-Calib checkpoint | 固定 `k=5, p1=p2=1` | 67.304 | 88.328 | 1.472 | `20260610_1552` |
| 旧版直接 adaptive | 同一 Fisher-Calib checkpoint | 全层直接 adaptive `k/p` | 67.282 | - | - | `20260610_1900` |
| **SynFIM-Q** | 同一 Fisher-Calib checkpoint | **fixed/adaptive 候选选择** | **67.414** | 88.302 | 1.476 | `20260610_2256` |

主要结论：

- 相比历史实验 C，SynFIM-Q 提升 `+0.216 Top-1`。
- 在同代码、同 checkpoint 的严格对照下，adaptive 候选选择相比 fixed `k/p` 提升 `+0.110 Top-1`。
- 直接全层 adaptive 并不稳定，必须通过候选选择和 guard 控制每个 block 是否采用 adaptive 更新。

## 方法概述

### 1. Fisher-Calib

普通 PTQ 校准通常根据输出 MSE 搜索量化 scale 和 zero-point。Fisher-Calib 在校准阶段引入 Fisher 敏感度，使校准更关注对最终任务损失更重要的通道或维度。

启用方式：

```bash
--calib-metric fisher_diag
```

### 2. 固定 `k/p` 的 Fisher-DPLR 块重构

实验 C 的主体流程是：

```text
Fisher-Calib checkpoint
    -> 固定 k=5, p1=1, p2=1 的 Fisher-DPLR AdaRound 块重构
```

这一路线是当前 adaptive 方法的固定参数对照组。

### 3. 自适应候选选择式 Fisher-DPLR

早期实验说明，单独优化 `k/p` 时，校准集指标可能变好，但最终 ImageNet 精度不一定提升。因此当前版本没有强制所有 block 使用 adaptive 参数，而是采用候选选择：

```text
对每个 block：
    1. 构造 fixed 候选：k=5, p1=1, p2=1
    2. 根据 residual 统计构造 adaptive 候选
    3. 两个候选从同一 block 状态开始训练
    4. 训练前恢复相同随机状态，保证比较公平
    5. 用局部 MSE、CE、true-class probability、logit margin 等指标评分
    6. adaptive 必须超过 fixed 一个 margin 才会被采用
    7. 如果两个候选都触发明显坏更新，则回退到重构前状态
```

在当前最优日志 `20260610_2256` 中，最终选择如下：

```text
patch_embed: adaptive
blocks.0: fixed/adaptive 相同
blocks.1: fixed
blocks.2: fixed
blocks.3: adaptive
blocks.4: fixed
blocks.5: fixed
blocks.6: fixed
blocks.7: fixed/adaptive 相同
blocks.8: adaptive
blocks.9: fixed
blocks.10: adaptive
blocks.11: revert
head: fixed/adaptive 相同
```

这说明 adaptive `k/p` 不是全层越强越好，而是要在合适的 block 上使用。

## 与 APHQ-ViT 和 FIMA-Q 的关系

APHQ-ViT 和 FIMA-Q 都证明了 Fisher 信息对 ViT PTQ 有价值，但直接叠加不同阶段的 Fisher 优化并不一定带来更高最终精度。本工程当前有效改进主要体现在：

- 将 Fisher-Calib 作为稳定校准基础；
- 将 FIMA-Q 的 Fisher-DPLR 块重构扩展为 fixed/adaptive 双候选选择；
- 引入残差统计驱动的 adaptive `k/p`；
- 使用 MSE + logits guard 判断每个 block 的 adaptive 更新是否真正有效；
- 避免全层盲目增强低秩 Fisher 项导致叠加失效。

因此，SynFIM-Q 的重点不是“更多地使用 Fisher”，而是“在合适的 block、以合适的强度使用 Fisher”。

## 复现实验

### 环境

```bash
pip install torch torchvision timm
```

ImageNet 路径示例：

```text
D:/AI/IaS-ViT-main/dataset/imagenet/
```

目录下需要包含：

```text
train/
val/
```

### 第一步：生成 Fisher-Calib checkpoint

仓库不上传 `checkpoints/` 目录，因此复现实验前需要先在本地生成 Fisher-Calib checkpoint。以 4bit DeiT-Tiny 为例：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --w_bit 4 --a_bit 4 --calib-metric fisher_diag --val-batch-size 64 --num-workers 0 --device cuda
```

运行结束后，会在如下目录保存 checkpoint：

```text
checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth
```

后续 fixed 对照和 SynFIM-Q 都应该加载这个同一个 Fisher-Calib checkpoint。README 中的 `20260530_1051_C` 是本文实验使用的历史日志目录，复现时请替换为你本地实际生成的 `<timestamp>`。

### 第二步：固定 `k/p` 对照

加载第一步生成的 Fisher-Calib checkpoint，关闭 adaptive `k/p` 和候选选择：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-bias-correction
```

### 第三步：运行当前 SynFIM-Q

加载同一个 Fisher-Calib checkpoint，使用默认 adaptive 候选选择：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

### 迁移到 3bit / 6bit

自适应候选选择已经接入 `test_quant.py`，因此 3bit、6bit 不需要改脚本，只需要更换 config 和 bit 参数。例如在 3bit 上启用当前候选选择逻辑：

```bash
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_3bit_fisher_calib_checkpoint.pth> --optimize --w_bit 3 --a_bit 3 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

在 6bit 上启用：

```bash
python test_quant.py --model deit_tiny --config ./configs/6bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_6bit_fisher_calib_checkpoint.pth> --optimize --w_bit 6 --a_bit 6 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

如果只想跑固定 `k/p` 对照，在相同命令中改为：

```bash
--no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select
```

如果想跑“直接 adaptive、但不做 fixed/adaptive 候选选择”，使用：

```bash
--adaptive-k --adaptive-p --no-adaptive-candidate-select
```

## 关键参数

| 参数 | 说明 |
|---|---|
| `--calib-metric fisher_diag` | 启用 Fisher-Calib。 |
| `--adaptive-k` / `--no-adaptive-k` | 启用或关闭自适应 Fisher rank。 |
| `--adaptive-p` / `--no-adaptive-p` | 启用或关闭自适应 `p1/p2`。 |
| `--adaptive-candidate-select` | 启用 fixed/adaptive 候选选择。 |
| `--no-adaptive-candidate-select` | 关闭 fixed/adaptive 候选选择。 |
| `--adaptive-candidate-margin` | adaptive 候选超过 fixed 候选所需的分数 margin，默认 `0.003`。 |
| `--logit-guard` | 启用 logits guard。 |
| `--no-logit-guard` | 关闭 logits guard。 |
| `--logit-guard-batches N` | 只用前 `N` 个校准 batch 做 logits guard。 |
| `--logit-guard-size N` | 使用额外 held-out 校准样本做 logits guard。当前主结果使用默认 `0`。 |
| `--no-logit-bias-correction` | 关闭分类头 logit bias correction。当前主结果关闭该项。 |
| `--recon-block-start N` | 从第 `N` 个 Transformer block 开始重构。 |
| `--recon-block-end N` | 重构到第 `N` 个 Transformer block。 |
| `--skip-patch-embed` | 跳过 patch embedding 重构。 |
| `--skip-head` | 跳过分类头重构。 |
| `--diagnose-residual-only` | 只诊断 residual/Fisher 统计，不运行重构。 |

## 代码结构

```text
SynFIM-Q/
|-- test_quant.py                 # 主入口
|-- configs/
|   |-- 3bit/
|   |-- 4bit/
|   |   |-- best.py
|   |   `-- fim_unified.py        # 当前 SynFIM-Q 配置
|   `-- 6bit/
|-- utils/
|   |-- calibrator.py             # 校准与 Fisher-Calib
|   |-- block_recon.py            # Fisher-DPLR AdaRound、自适应 k/p、候选选择和 guard
|   |-- mlp_recon.py              # 可选 MLP reconstruction
|   |-- datasets.py               # 数据加载与确定性校准集
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

## 当前实现要点

- 校准集加载使用确定性随机种子，减少实验漂移。
- 块重构记录每个 block 的最终 loss、MSE、logits guard 指标和候选选择结果。
- fixed/adaptive 候选训练前会恢复相同 RNG 状态，保证对比公平。
- adaptive `k/p` 从 residual 统计生成，固定 `k/p` 作为稳定锚点。
- adaptive 候选必须超过 fixed 候选指定 margin 才会被采用。
- 对明显损害 CE、置信度或局部 MSE 的 late block 更新执行回退。

## 注意事项

- 当前主结果以 Top-1 为主要指标；Top-5 和 Loss 不是所有历史实验中的最高值。
- 候选选择会让运行时间增加，因为部分 block 会训练 fixed 和 adaptive 两个候选。
- 为了保证对比严谨，fixed 对照和 SynFIM-Q 应加载同一个 Fisher-Calib checkpoint。
- 当前主结果使用 `--no-logit-bias-correction`，因为此前实验中该模块收益不稳定。

## 许可证

见 [LICENSE](LICENSE)。
