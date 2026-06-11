# SynFIM-Q：面向 Vision Transformer 的 Fisher 感知自适应 PTQ

SynFIM-Q 是一个基于 PyTorch 的 Vision Transformer 后训练量化（Post-Training Quantization, PTQ）工程。当前版本主要围绕 DeiT-Tiny 在 ImageNet 上的 W4A4 量化展开，目标是在不重新训练模型的前提下，降低 ViT 量化误差对最终分类精度的影响。

本工程参考了 APHQ-ViT 和 FIMA-Q 的实现思路，但当前有效主线不是简单叠加所有模块，而是以 **Fisher-Calib checkpoint** 为统一起点，在块重构阶段引入 **Fisher 感知的自适应 `k/p` 候选选择机制**。核心思想是：对于每个待重构模块，同时构造固定 `k/p` 候选和 adaptive `k/p` 候选，并使用局部重构误差与全模型 logits 一致性共同判断最终保留哪一个候选。

简单来说，SynFIM-Q 不是让所有 block 都更强地使用 Fisher 信息，而是尝试回答一个更细的问题：**哪些 block 适合使用更强的低秩 Fisher 修正，哪些 block 应该保留稳定的固定参数重构。**

## 当前结果

- 模型：DeiT-Tiny
- 数据集：ImageNet validation
- 量化设置：W4A4
- 校准样本数：128
- 块重构样本数：1024

| 方法 | 起点 | Top-1 | Top-5 | 相对 Baseline |
|---|---|---:|---:|---:|
| 实验 A / Baseline | MSE-Calib | 66.800 | - | - |
| 实验 B | Fisher-Calib | 67.198 | 88.258 | +0.398 |
| 实验 C | MSE-Calib | - | - | - |
| 实验 D / **SynFIM-Q** | Fisher-Calib | **67.414** | 88.302 | **+0.614** |

主要结论：

- 相比实验 A / Baseline，SynFIM-Q 提升 `+0.614 Top-1`。
- 相比实验 B，SynFIM-Q 进一步提升 `+0.216 Top-1`。
- 实验结果说明，Fisher-Calib 可以提供更稳定的校准起点，而 adaptive `k/p` 可以在块重构阶段继续降低任务相关量化误差。
- 实验 C 用于单独评估 adaptive `k/p` 模块，不叠加实验 B 的 Fisher-Calib，目前结果待补充。
- 当前更有效的做法不是全层统一增强 Fisher 项，而是根据 block 级统计和分类一致性自适应决定 Fisher-DPLR 参数。

## 方法动机

ViT PTQ 的困难在于量化误差会在 attention、MLP 和 residual 分支之间逐层传播。单纯优化局部输出 MSE，往往无法保证最终分类 logits 仍然保持一致；而单纯增强 Fisher 项，又可能让个别 block 的局部重构过拟合校准样本，最终导致验证集精度下降。

APHQ-ViT 和 FIMA-Q 都说明 Fisher 信息对 ViT PTQ 是有效的：

- APHQ-ViT 更强调层内重构和激活分布建模。
- FIMA-Q 更强调利用 Fisher Information Matrix（FIM）刻画参数扰动对任务损失的敏感性。

SynFIM-Q 当前采用的思路是：保留 Fisher-Calib 作为稳定校准基础，在块重构阶段进一步判断 Fisher-DPLR 中低秩项和对角项的使用强度。也就是说，本工程的重点不是机械叠加 APHQ-ViT 和 FIMA-Q 的所有模块，而是将 Fisher 信息用于 **block-wise adaptive decision**。

## 核心设计

### 1. Fisher-Calib

普通 PTQ 校准通常根据输出 MSE 或 MAE 搜索量化 scale。Fisher-Calib 在校准阶段引入 Fisher 敏感度，使校准更关注对最终任务损失更重要的通道、token 或权重维度。

启用方式：

```bash
--calib-metric fisher_diag
```

在当前主实验中，Fisher-Calib 先生成一个校准后的 checkpoint，后续实验 B 和实验 D 都加载同一个 checkpoint，从而保证比较公平。

### 2. Fisher-DPLR 块重构

块重构阶段使用 Fisher-DPLR 形式近似 Fisher Information Matrix，将重构损失拆分为低秩项和对角项：

```text
Loss_rec = p1 * Loss_low-rank + p2 * Loss_diag
```

其中：

- `k` 控制低秩 Fisher 近似的 rank。
- `p1` 控制低秩项权重。
- `p2` 控制对角项权重。

实验 B 使用固定参数：

```text
k = 5, p1 = 1.0, p2 = 1.0
```

这一路线是当前 adaptive 方法的固定参数基准。

### 3. Adaptive `k/p`

实验 C 对 Fisher-DPLR 的 `k/p` 进行自适应调整。当前实现会根据每个 block 的 residual 统计、相对误差信号和层位置，为不同 block 生成 adaptive 候选参数。

直观理解：

- residual 较明显的 block 可以适当提高低秩 Fisher 修正强度；
- 对输出分布更敏感的 block 需要限制 `p2` 或回退到固定参数；
- 中后层 block 对最终分类 logits 更敏感，因此不能只看局部 MSE。

因此，adaptive `k/p` 本身不是一个简单的全局超参数，而是 block-wise 的动态策略。

### 4. Fixed/Adaptive 候选选择

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

这种设计的意义在于：adaptive 分支负责探索更优的 Fisher-DPLR 参数，固定参数分支负责提供稳定参照，guard 负责避免局部指标改善但最终分类退化。

### 5. Logits Guard

局部 MSE 只描述当前 block 输出是否接近全精度输出，但最终分类精度取决于全模型 logits。当前实现加入 logits guard，记录并比较：

- calibration/guard 子集上的 Top-1；
- cross entropy；
- true-class probability；
- true-class logit margin；
- 预测翻转情况。

对明显损害后层分类一致性的候选，会执行回退；对局部 MSE 下降且 logits 明显改善的候选，则允许保留。这是当前实验 D 能稳定超过实验 B 的关键。

## 与 APHQ-ViT 和 FIMA-Q 的关系

本工程并不是把 APHQ-ViT 和 FIMA-Q 直接拼接，而是在复现实验和消融分析后，保留当前最稳定的有效主线：

- 继承 APHQ-ViT/FIMA-Q 的 ViT PTQ 工程框架；
- 使用 Fisher-Calib 作为校准阶段的稳定基础；
- 以 Fisher-DPLR AdaRound 作为块重构主体；
- 在块重构中加入 residual-aware adaptive `k/p`；
- 使用固定参数与自适应参数的候选比较，避免全层 adaptive 失效；
- 使用 MSE + logits guard 共同判断每个 block 的更新是否应保留。

相较于原始实现，SynFIM-Q 的主要改进点可以概括为：**从固定全局 Fisher 超参数，扩展为具有候选比较和分类一致性约束的 block-wise Fisher 自适应策略。**

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

下面命令均以 4bit DeiT-Tiny 为例。为了保证实验 B 和实验 D 对比严谨，建议使用同一个 Fisher-Calib checkpoint。

### 1. 实验 A / Baseline：MSE-Calib + 固定 `k/p`

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
python test_quant.py --model deit_tiny --config ./configs/4bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_mse.pth --optimize --w_bit 4 --a_bit 4 --calib-metric mse --optim-metric fisher_dplr --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-bias-correction
```

### 2. 生成 Fisher-Calib checkpoint

该步骤对应实验 B 和实验 D 的共同起点：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --w_bit 4 --a_bit 4 --calib-metric fisher_diag --val-batch-size 64 --num-workers 0 --device cuda
```

运行结束后，会在如下目录保存 checkpoint：

```text
checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth
```

上述 `<timestamp>` 是本地运行时自动生成的 checkpoint 目录占位符，复现时请替换为你本地实际生成的目录名。

### 3. 实验 B：Fisher-Calib + 固定 `k/p`

加载 Fisher-Calib checkpoint，关闭 adaptive `k/p` 和候选选择：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --no-adaptive-k --no-adaptive-p --no-adaptive-candidate-select --no-logit-bias-correction
```

该命令用于复现实验 B。它保留 Fisher-Calib 和 Fisher-DPLR AdaRound，但不启用 adaptive 参数。

### 4. 实验 C：MSE-Calib + Adaptive `k/p`

实验 C 用于评估 adaptive `k/p` 模块本身，不叠加实验 B 的 Fisher-Calib。该实验与实验 D 使用相同的 adaptive 候选选择逻辑，差别只在校准阶段使用 MSE-Calib。

可以用一条命令先生成 MSE-Calib checkpoint，再自动加载该 checkpoint 继续运行 adaptive `k/p`：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --calibrate --optimize --w_bit 4 --a_bit 4 --calib-metric mse --optim-metric fisher_dplr --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

说明：

- 这条命令用于 adaptive `k/p` 消融，只考察 adaptive `k/p` 在块重构阶段的作用。
- 当前实验 C 结果还未补充，README 表格中暂时保留为空。
- 实验 C 和实验 D 都启用 `--adaptive-candidate-select`，因此二者的 adaptive `k/p` 选择逻辑保持一致。

### 5. 实验 D / SynFIM-Q：Fisher-Calib + Adaptive `k/p`

加载与实验 B 相同的 Fisher-Calib checkpoint，启用当前最优的 adaptive 候选选择逻辑：

```bash
python test_quant.py --model deit_tiny --config ./configs/4bit/fim_unified.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint checkpoints/quant_result/<timestamp>/deit_tiny_w4_a4_calibsize_128_fisher_diag.pth --optimize --w_bit 4 --a_bit 4 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
```

这是当前 README 中 SynFIM-Q 主结果使用的实验设置，对应实验 D。

### 6. 迁移到 3bit / 6bit

自适应候选选择已经接入 `test_quant.py`，因此 3bit、6bit 不需要额外脚本，只需要更换 config、bit 参数和对应 checkpoint。

3bit 示例：

```bash
python test_quant.py --model deit_tiny --config ./configs/3bit/best.py --dataset D:/AI/IaS-ViT-main/dataset/imagenet --load-calibrate-checkpoint <your_3bit_fisher_calib_checkpoint.pth> --optimize --w_bit 3 --a_bit 3 --calib-metric fisher_diag --optim-size 1024 --val-batch-size 64 --num-workers 0 --device cuda --adaptive-k --adaptive-p --adaptive-candidate-select --logit-guard --no-logit-bias-correction
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
| `--calib-metric mse` | 使用普通 MSE 校准，适合实验 A 和实验 C。 |
| `--optim-metric fisher_dplr` | 使用 Fisher-DPLR 重构损失。 |
| `--k` | 全局 Fisher 低秩 rank，默认主实验为 `5`。 |
| `--p1` / `--p2` | 低秩项和对角项的全局权重。 |
| `--adaptive-k` / `--no-adaptive-k` | 启用或关闭自适应 Fisher rank。 |
| `--adaptive-p` / `--no-adaptive-p` | 启用或关闭自适应 `p1/p2`。 |
| `--adaptive-candidate-select` | 启用固定参数 / 自适应参数候选选择。 |
| `--no-adaptive-candidate-select` | 关闭候选选择，直接按当前 `k/p` 逻辑重构。 |
| `--adaptive-candidate-margin` | adaptive 候选超过固定参数候选所需的分数 margin，默认 `0.003`。 |
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
- 固定参数 / 自适应参数候选训练前会恢复相同 block 状态和 RNG 状态，保证对比公平。
- adaptive `k/p` 从 residual 统计生成，固定 `k/p` 作为稳定锚点。
- adaptive 候选必须超过固定参数候选指定 margin 才会被采用。
- 对明显损害 CE、置信度、logit margin 或 late-block 分类一致性的更新执行回退。
- 块重构会记录每个 block 的最终 loss、MSE、logits guard 指标和候选选择结果，方便后续消融分析。

## 注意事项

- 当前主结果以 Top-1 为主要指标；Top-5 作为辅助参考。
- 候选选择会增加运行时间，因为部分 block 会分别训练固定参数和自适应参数两个候选。
- 为了保证对比严谨，实验 B 和实验 D 应加载同一个 Fisher-Calib checkpoint。
- 实验 C 建议加载 MSE-Calib checkpoint，避免 Fisher-Calib 对 adaptive `k/p` 消融产生混淆。
- 当前主结果使用 `--no-logit-bias-correction`，因为此前实验中该模块收益不稳定。
- `checkpoints/` 已在 `.gitignore` 中忽略，仓库不会上传实验 checkpoint。

## 许可证

见 [LICENSE](LICENSE)。
