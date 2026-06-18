# SynFIM-Q：面向 Vision Transformer 的 Fisher 感知自适应 PTQ

SynFIM-Q 是一个基于 PyTorch 的 Vision Transformer 后训练量化（Post-Training Quantization, PTQ）工程。当前版本以 DeiT-Tiny 在 ImageNet 上的 W4A4/W3A3 低比特量化为主要实验场景，目标是在不重新训练模型的前提下，降低权重和激活量化误差对最终分类精度的影响。

本工程以 FIMA-Q 的 Fisher-DPLR 块重构框架为主要基线，在固定 `k/p` Fisher 重构路径之上，引入 **SynFIM-Q：Fisher 感知的 block-wise adaptive `k/p` 候选选择机制**。方法不改变原始 PTQ 的无训练设定，而是在校准和块重构两个阶段增强 Fisher 信息的使用方式：先通过 Fisher-Calib 得到更稳定的量化起点，再在每个 block 的重构过程中根据 residual 统计、层位置和 logits 一致性动态选择 Fisher-DPLR 参数。

核心思想是把原本全局固定的 Fisher-DPLR 超参数选择转化为受约束的 block 级决策问题。对于每个待重构 block，SynFIM-Q 会从同一初始状态出发构造固定 `k/p` 候选和 adaptive `k/p` 候选，分别完成重构训练，并结合局部 MSE、交叉熵、true-class probability、logit margin 等信号判断候选是否真正改善了全模型行为。这样既保留了 FIMA-Q 固定 Fisher 重构的稳定性，又允许不同 block 根据自身误差传播特征使用不同强度的低秩 Fisher 修正。

## 当前结果

- 模型：DeiT-Tiny
- 数据集：ImageNet validation
- 校准样本数：128
- 块重构样本数：1024
- 主指标：Top-1 accuracy

### W4A4 结果

| 方法 | 起点 | `k/p` 策略 | Top-1 | 相对实验 1 |
|---|---|---|---:|---:|
| 实验 1 / Baseline | MSE-Calib | 固定 `k/p` | 66.840 | - |
| 实验 2 | Fisher-Calib | 固定 `k/p` | 67.198 | +0.358 |
| 实验 3 / **SynFIM-Q** | Fisher-Calib | adaptive `k/p` | **67.414** | **+0.574** |

### W3A3 当前结果

| 方法 | 起点 | `k/p` 策略 | Top-1 | 相对实验 1 |
|---|---|---|---:|---:|
| 实验 1 / Baseline | MSE-Calib | 固定 `k/p` | 55.550 | - |
| 实验 2 | Fisher-Calib | 固定 `k/p` | 56.320 | +0.770 |
| 实验 3 / **SynFIM-Q** | Fisher-Calib | adaptive `k/p` | **56.476** | **+0.926** |

主要结论：

- W4A4 下，实验 3 相比实验 1 / Baseline 提升 `+0.574 Top-1`，相比实验 2 提升 `+0.216 Top-1`。
- W3A3 下，实验 3 相比实验 2 提升 `+0.156 Top-1`，说明 Fisher-Calib 起点上继续加入 adaptive `k/p` 仍有增益。
- 当前消融采用递进式设计：实验 1 作为 FIMA-Q baseline，实验 2 在实验 1 的基础上引入 Fisher-Calib，实验 3 在实验 2 的 Fisher-Calib 起点上继续叠加 adaptive `k/p` 候选选择。
- 当前更有效的做法不是全层统一增强 Fisher 项，而是根据 block 级统计和分类一致性自适应决定 Fisher-DPLR 参数，尤其要避免早期 block 的噪声候选误放行。

## 方法动机

ViT PTQ 的困难在于量化误差会在 attention、MLP 和 residual 分支之间逐层传播。单纯优化局部输出 MSE，往往无法保证最终分类 logits 仍然保持一致；而使用 Fisher 信息时，如果所有 block 都采用同一组固定低秩近似和损失权重，也难以适配不同层的误差传播特征。

FIMA-Q 证明了 Fisher Information Matrix（FIM）可以刻画参数扰动对任务损失的敏感性，并通过 Fisher-DPLR AdaRound 改善 ViT PTQ 的块重构质量。但在固定 `k=5, p1=1, p2=1` 的设置下，所有 block 使用同一强度的低秩项和对角项，存在两个限制：

- 对 residual 更明显或误差更集中的 block，固定参数可能不足以利用 Fisher 低秩信息；
- 对 logits 更敏感的中后层 block，过强的 Fisher 修正又可能改善局部 MSE 但损害最终分类一致性。

SynFIM-Q 针对上述限制，将 Fisher 信息从“固定全局超参数”扩展为 **block-wise adaptive decision**：校准阶段用 Fisher-Calib 提供更稳定起点，重构阶段根据 block 统计生成候选 `k/p`，再通过固定候选、adaptive 候选和 logits/MSE guard 的比较决定是否保留自适应更新。

## 核心设计

### 1. Fisher-Calib

普通 PTQ 校准通常根据输出 MSE 或 MAE 搜索量化 scale。Fisher-Calib 在校准阶段引入 Fisher 敏感度，使校准更关注对最终任务损失更重要的通道、token 或权重维度。

启用方式：

```bash
--calib-metric fisher_diag
```

在当前主实验中，Fisher-Calib 先生成一个校准后的 checkpoint，后续实验 2 和实验 3 都加载同一个 checkpoint，从而保证比较公平。

### 2. Fisher-DPLR 块重构

块重构阶段使用 Fisher-DPLR 形式近似 Fisher Information Matrix，将重构损失拆分为低秩项和对角项：

```text
Loss_rec = p1 * Loss_low-rank + p2 * Loss_diag
```

其中：

- `k` 控制低秩 Fisher 近似的 rank。
- `p1` 控制低秩项权重。
- `p2` 控制对角项权重。

实验 2 使用固定参数：

```text
k = 5, p1 = 1.0, p2 = 1.0
```

这一路线是当前 adaptive 方法的固定参数基准。

### 3. Adaptive `k/p`

SynFIM-Q 在实验 2 的 Fisher-Calib 起点上，对 Fisher-DPLR 的 `k/p` 进行自适应调整。当前实现会根据每个 block 的 residual 统计、相对误差信号和层位置，为不同 block 生成 adaptive 候选参数。

直观理解：

- residual 较明显的 block 可以适当提高低秩 Fisher 修正强度；
- 对输出分布更敏感的 block 需要限制 `p2` 或回退到固定参数；
- 中后层 block 对最终分类 logits 更敏感，因此不能只看局部 MSE。

因此，adaptive `k/p` 本身不是一个简单的全局超参数，而是 block-wise 的动态策略。当前主线不强调 adaptive `k/p` 脱离 Fisher-Calib 单独使用，而是验证它在 Fisher-Calib 稳定起点上的进一步增益。

在 3bit 实验中，当前还加入了 `safe_plus` profile：在固定参数候选和普通 adaptive 候选之外，只对少数中后层（如 `blocks.5/6/8/10`）额外构造 `strong_adaptive` 候选。该候选主要轻微增强 `k` 或 `p1`，不全局抬高 `p2`，并且仍然必须通过 logits/MSE guard 才能被保留。这样可以避免早期 block 过度自适应，同时给中后层保留更大的搜索空间。

### 4. Adaptive 候选选择

早期实验发现，直接让所有 block 使用 adaptive `k/p` 并不稳定。当前版本改为候选选择：

```text
对每个 block：
    1. 保存重构前 block 状态；
    2. 构造固定参数候选：k=5, p1=1, p2=1；
    3. 构造 adaptive 候选：根据 residual 统计生成 k/p；
    4. 两个候选都从同一初始状态开始训练；
    5. 每个候选训练前恢复相同 RNG 状态；
    6. 分别计算局部 MSE、CE、true-class probability、logit margin 等指标；
    7. adaptive 必须超过固定参数候选一个 margin 才会被采用；
    8. 如果候选明显破坏 logits 或局部重构质量，则回退到重构前状态。
```

这种设计的意义在于：adaptive 分支负责探索更优的 Fisher-DPLR 参数，固定参数分支负责提供稳定参照，guard 负责避免局部指标改善但最终分类退化。在消融实验中，实验 2 到实验 3 的提升用于衡量 adaptive `k/p` 候选选择在 Fisher-Calib 起点上的贡献。

### 5. Logits Guard

局部 MSE 只描述当前 block 输出是否接近全精度输出，但最终分类精度取决于全模型 logits。当前实现加入 logits guard，记录并比较：

- calibration/guard 子集上的 Top-1；
- cross entropy；
- true-class probability；
- true-class logit margin；
- 预测翻转情况。

对明显损害后层分类一致性的候选，会执行回退；对局部 MSE 下降且 logits 明显改善的候选，则允许保留。这是当前实验 3 在 W4A4 和 W3A3 下均能超过实验 2 的关键。

## 与 FIMA-Q 及相关工作的关系

本工程的主要基线是 FIMA-Q，而不是 APHQ-ViT 与 FIMA-Q 的模块拼接。当前主结果没有引入 APHQ-ViT 的 MR 路径；APHQ-ViT 更适合作为 ViT PTQ 中层重构和激活建模方向的相关工作背景。SynFIM-Q 的方法改进集中在 FIMA-Q 的 Fisher-DPLR 重构框架内部：

- 实验 1 / Baseline 对齐 FIMA-Q 官方 baseline：MSE-Calib + Fisher-DPLR AdaRound + fixed `k=5, p1=1, p2=1`；
- 使用 Fisher-Calib 作为校准阶段的稳定基础；
- 保留 Fisher-DPLR AdaRound 作为块重构主体；
- 在 Fisher-DPLR 中加入 residual-aware adaptive `k/p` 参数生成；
- 使用固定参数与自适应参数的候选比较，避免全层统一 adaptive 带来的不稳定；
- 使用 MSE + logits guard 共同判断每个 block 的 adaptive 更新是否应保留。

相较于 FIMA-Q 的固定全局 Fisher 超参数，SynFIM-Q 的主要改进点可以概括为：**将 Fisher-DPLR 的 `k/p` 从静态配置扩展为具有候选比较、分类一致性约束和低比特 profile 的 block-wise Fisher 自适应策略。**

需要注意的是，FIMA-Q 原始 baseline 不包含 adaptive `k/p`、fixed/adaptive 候选选择、logits guard 或 logit bias correction。因此复现实验 1 时，应显式关闭这些新增模块。

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

仓库不上传 `checkpoints/` 目录。所有涉及 `--load-calibrate-checkpoint` 的命令，都需要先在本地生成对应的校准 checkpoint。

## 实验命令

下面命令均以 4bit DeiT-Tiny 为例。为了保证实验 2 和实验 3 对比严谨，建议使用同一个 Fisher-Calib checkpoint。

### 1. 实验 1 / Baseline：MSE-Calib + 固定 `k/p`

先生成普通 MSE-Calib checkpoint：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --w_bit 4 --a_bit 4 --calib-metric mse --val-batch-size 64 --num-workers 0 --device cuda
```

运行结束后，会在如下目录保存 checkpoint：

```text
checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_mse.pth
```

然后加载该 checkpoint，关闭 adaptive `k/p` 和候选选择：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_mse.pth --optimize --w_bit 4 --a_bit 4 --calib-metric mse --optim-metric fisher_dplr --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-guard --no-logit-bias-correction
```

这一路径用于严格复现 FIMA-Q baseline。由于当前工程额外实现了 logits guard 和 bias correction，baseline 命令中需要显式关闭它们。

### 2. 生成 Fisher-Calib checkpoint

该步骤对应实验 2 和实验 3 的共同起点：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --w_bit 4 --a_bit 4 --calib-metric fisher_diag --val-batch-size 64 --num-workers 0 --device cuda
```

运行结束后，会在如下目录保存 checkpoint：

```text
checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth
```

上述 `<timestamp>` 是本地运行时自动生成的 checkpoint 目录占位符，复现时请替换为你本地实际生成的目录名。

### 3. 实验 2：Fisher-Calib + 固定 `k/p`

加载 Fisher-Calib checkpoint，关闭 adaptive `k/p` 和候选选择：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-guard --no-logit-bias-correction
```

该命令用于复现实验 2。它保留 Fisher-Calib 和 Fisher-DPLR AdaRound，但不启用 adaptive 参数。

### 4. 实验 3 / SynFIM-Q：Fisher-Calib + Adaptive `k/p`

加载与实验 2 相同的 Fisher-Calib checkpoint，启用当前最优的 adaptive 候选选择逻辑：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

这是当前 README 中 SynFIM-Q 主结果使用的实验设置，对应实验 3。

### 5. 迁移到 3bit / 6bit

自适应候选选择已经接入 `test_quant.py`，因此 3bit、6bit 不需要额外脚本，只需要更换 config、bit 参数和对应 checkpoint。

3bit 实验 1 / FIMA-Q baseline 示例：

```bash
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --optimize --w_bit 3 --a_bit 3 --calib-metric mse --optim-metric fisher_dplr --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-guard --no-logit-bias-correction
```

3bit 实验 2 / Fisher-Calib + fixed `k/p` 示例：

```bash
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_3bit_fisher_calib_checkpoint.pth> --optimize --w_bit 3 --a_bit 3 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

说明：该设置仍然是固定 `k/p`，`logit_guard` 只用于低比特重构过程中的更新回退，与表格中的 3bit 实验 2 结果保持一致。

3bit 实验 3 / SynFIM-Q 示例：

```bash
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_3bit_fisher_calib_checkpoint.pth> --optimize --w_bit 3 --a_bit 3 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --adaptive-3bit-select-profile safe_plus --logit-guard --no-logit-bias-correction
```

6bit 示例：

```bash
python test_quant.py --model deit_tiny --config ./configs/6bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_6bit_fisher_calib_checkpoint.pth> --optimize --w_bit 6 --a_bit 6 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `--calibrate` | 只执行校准并保存 calibrated checkpoint。 |
| `--load-calibrate-checkpoint` | 加载已有 calibrated checkpoint，用于后续块重构。 |
| `--optimize` | 执行块重构 / AdaRound 优化。 |
| `--calib-metric fisher_diag` | 启用 Fisher-Calib。 |
| `--calib-metric mse` | 使用普通 MSE 校准，适合实验 1 / Baseline。 |
| `--optim-metric fisher_dplr` | 使用 Fisher-DPLR 重构损失。 |
| `--k` | 全局 Fisher 低秩 rank，默认主实验为 `5`。 |
| `--p1` / `--p2` | 低秩项和对角项的全局权重。 |
| `--adaptive-k` / `--no-adaptive-k` | 启用或关闭自适应 Fisher rank。 |
| `--adaptive-p` / `--no-adaptive-p` | 启用或关闭自适应 `p1/p2`。 |
| `--adaptive-candidate-select` | 启用固定参数 / 自适应参数候选选择。 |
| `--no-adaptive-candidate-select` | 关闭候选选择，直接按当前 `k/p` 逻辑重构。 |
| `--adaptive-candidate-margin` | adaptive 候选超过固定参数候选所需的分数 margin，默认 `0.003`。 |
| `--adaptive-3bit-select-profile` | 3bit 专用 profile，可选 `safe`、`safe_plus`、`balanced`、`std_prior`；当前 3bit 实验 3 使用 `safe_plus`。 |
| `--logit-guard` / `--no-logit-guard` | 启用或关闭 logits guard。 |
| `--logit-guard-batches N` | 只用前 `N` 个校准 batch 做 logits guard。 |
| `--logit-guard-size N` | 使用额外 held-out 校准样本做 logits guard；当前主结果使用默认 `0`。 |
| `--no-logit-bias-correction` | 关闭分类头 logit bias correction；当前主结果关闭该项。 |
| `--recon-block-start N` | 从第 `N` 个 Transformer block 开始重构。 |
| `--recon-block-end N` | 重构到第 `N` 个 Transformer block。 |
| `--skip-patch-embed` | 跳过 patch embedding 重构。 |
| `--skip-head` | 跳过分类头重构。 |
| `--diagnose-residual-only` | 只诊断 residual/Fisher 统计，不运行块重构。 |

## 日志解读

块重构阶段会输出每个模块的最终摘要，例如：

```text
Block blocks.3 final loss: total=..., rec=..., round=..., k=..., p1=..., p2=..., guard=..., mse_before=..., mse_after=..., logit_before=(...), logit_after=(...)
```

其中：

- `total/rec/round` 是 AdaRound 优化过程中的总损失、重构损失和 round loss。
- `k/p1/p2` 是该 block 实际使用的 Fisher-DPLR 参数。
- `guard` 表示该 block 的候选选择或回退结果。
- `mse_before/mse_after` 用于观察局部重构误差变化。
- `logit_before/logit_after` 用于观察全模型分类一致性变化。

如果出现校准集精度上升但最终 ImageNet 精度不升，通常说明局部重构指标与全局分类指标存在偏差，需要结合 logits guard 和最终验证集结果判断。

## 代码结构

```text
SynFIM-Q/
|-- test_quant.py                 # 主入口，支持校准、加载 checkpoint、块重构和自适应开关
|-- configs/
|   |-- 3bit/
|   |-- 4bit/
|   |   |-- best.py               # 普通 4bit 对照配置
|   |   `-- fim_unified.py        # 当前 SynFIM-Q 默认配置
|   `-- 6bit/
|-- utils/
|   |-- calibrator.py             # 校准与 Fisher-Calib
|   |-- block_recon.py            # Fisher-DPLR AdaRound、自适应 k/p、候选选择和 guard
|   |-- mlp_recon.py              # 可选 MLP reconstruction，当前主结果未使用
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
- 固定参数 / 自适应参数候选训练前会恢复相同 block 状态和 RNG 状态，保证对比公平。
- adaptive `k/p` 从 residual 统计生成，固定 `k/p` 作为稳定锚点。
- adaptive 候选必须超过固定参数候选指定 margin 才会被采用。
- 3bit `safe_plus` 会在少数中后层增加 `strong_adaptive` 候选，用于实验 3；`std_prior` 保留为 3bit MSE-Calib 场景下的可选 profile，不属于当前主结果。
- 对明显损害 CE、置信度、logit margin 或 late-block 分类一致性的更新执行回退。
- 块重构会记录每个 block 的最终 loss、MSE、logits guard 指标和候选选择结果，方便后续消融分析。

## 注意事项

- 当前主结果以 Top-1 为主要指标。
- 实验 1 是 FIMA-Q baseline 复现，必须显式关闭 adaptive `k/p`、candidate-select、logits guard 和 logit bias correction；尤其 3bit 配置文件中当前默认打开了 adaptive `k/p`，不能省略关闭开关。
- 候选选择会增加运行时间，因为部分 block 会分别训练固定参数和自适应参数两个候选。
- 为了保证对比严谨，实验 2 和实验 3 应加载同一个 Fisher-Calib checkpoint。
- 当前消融重点是递进式比较：实验 1 -> 实验 2 衡量 Fisher-Calib 的贡献，实验 2 -> 实验 3 衡量 adaptive `k/p` 候选选择在 Fisher-Calib 起点上的贡献。
- 当前主结果使用 `--no-logit-bias-correction`，因为此前实验中该模块收益不稳定。
- `checkpoints/` 已在 `.gitignore` 中忽略，仓库不会上传实验 checkpoint。

## 许可证

见 [LICENSE](LICENSE)。
