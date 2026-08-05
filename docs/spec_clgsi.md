# CLGSI 实现规格（写代码之前定稿）

**CLGSI — A Multimodal Sentiment Analysis Framework based on Contrastive Learning Guided by Sentiment Intensity**（Findings of NAACL 2024, pp.2099-2110），作者实现 [AZYoung233/CLGSI](https://github.com/AZYoung233/CLGSI)，**MIT**。

顺序纪律照前四个执行（已四次奏效）：**先读作者训练器定协议 → 写本规格 → 再写模型 → 等价测试 → 冒烟 → 10 seed**。

## 为什么是它：与 ConFEDE 构成一对可检验的问题

CLGSI 的核心是**按情感强度差异挑选正负样本对并据此加权** —— 而 **ConFEDE 实现了完全相同的想法却从未启用它**（`cont_NTXentLoss` 只在调用 `update_label` 后才加权，而三份定义无一处调用；见 `storyline.md` 第 18 节）。

**两篇放在一起能问一个具体问题：这个加权到底有没有用。** 这比再往对照表加一行有价值，也正对导师指定的方向（`decisions.md` 2026-08-04）。

## 核实结果（2026-08-05，读正文）

数据集 MOSI / MOSEI / SIMS，`feature_dims (768, 5, 20)`、`seq_lens (50,50,50)`、`KeyEval: MAE` —— **与本仓库 aligned 口径一致**，不需要额外下载。代码为 MIT，且派生自 MMSA（`AMIO`、`trains/multiTask`、`config_regression`），移植路径与前四个相同。

## 加权机制（`models/contrastive_loss.py`）

标签先映射到 [-1,1]：`(y + 3) / 3 - 1`，对 MOSI 即 `y/3`。令 `d = |y_i - y_j|`，`dividing_line = 0.4`、`gain = 1.5`、`temperature = 0.03`：

| 对 | 条件 | 权重 |
|---|---|---|
| 正 | `d ≤ 0.4` | `-tanh(d - 0.8) · 1.5` |
| 负 | `d > 0.4` | `tanh(d) · 1.5` |

损失分两支：**跨模态**（`input1 @ input2.T`）与**模态内**（`input1 @ input1.T`、`input2 @ input2.T`），两支都用同一套标签监督掩码。

### 只在代码里的两处

1. **负样本结果被硬编码乘 0.8**（`get_negative_pair` 的 `return neg_result*0.8`）。论文未提；这等于把负项的温度改了一档。
2. **注释掉的消融就在旁边**：`# self_supervised_mask[...] = 1` —— 即"不加权"的版本。作者试过，论文没报。**这恰好是我们要问的那个对照，作者代码里已经留好了开关。**

## 训练协议（`trains/multiTask/CLGSI.py`）

| 项 | 作者实现 | 本仓库 |
|---|---|---|
| 优化器 | `AdamW`，**五个参数组** | 由模型的 `param_groups` 提供 |
| lr | bert 5e-5 / audio 5e-3 / video 1e-3 / other 1e-2 | 同 |
| weight decay | bert 0.01（no-decay 组 0）/ audio 0.01 / video 0.001 / other 0.001 | 同 |
| 调度器 | `get_cosine_schedule_with_warmup`，**按步**，`num_training_steps = len(train) × warm_up_epochs(75)`，warmup 为其 10% | **本仓库缺这一档，需新增**（见下） |
| 批大小 | 64 | `--batch-size 64` |
| 梯度累积 | `update_epochs: 1`（即不累积） | `--accumulate-steps 1` |
| 早停 | `early_stop: 8` | `--patience 8` |
| 选轮 | `KeyEval: MAE`（验证集） | 与约定 2 一致 |

**BERT 有独立学习率 5e-5**——与 DPDF-LQ / DLF / DMD（单组）和 ConFEDE（冻结）都不同。**这三种情况已经各出现一次，说明「BERT 怎么训」必须每篇单独读，不存在可套用的惯例。**

### 调度器：必须新增一档，不得用现有的近似

本仓库现有 `warmup_cosine` 是**按 epoch** 计、warmup 占 10%、余弦退火覆盖 90%。CLGSI 是**按步**计，且总步数取 `len(train) × 75` 而**与实际训练轮数无关**——早停在第 8 轮无改善时触发，通常远早于 75 轮，**所以学习率实际只走完余弦曲线的前一小段**。

用 `warmup_cosine` 近似会得到一条完全不同的曲线。**近似就是猜协议，DPDF-LQ 上这条值 2.7 SE。** 按 `warmup_linear` 的先例新增 `warmup_cosine_steps`：**新选项没有任何既有组使用，所有已存数字逐比特不变**，以 `check_repro` 哈希确认。

## 待核实（实现时必须查清，不得想当然）

1. **五个参数组的划分依据**——按名字前缀还是模块？`bert_params_decay` / `no_decay` 的判定字符串要逐字核对（no-decay 通常是 `bias` 与 `LayerNorm.weight`，但不得假定）。
2. `H: 3.0`、`gamma: 0.95`、`skip_net_reduction: 2`、`fusion_filter_nums: 16` 各自在哪里生效，是否有声明未用的。
3. 对比损失的两支如何合成总损失、与主回归损失的权重比。
4. `label_map` 用的是真实标签还是某种伪标签（它派生自 MMSA 的 multiTask 目录，Self-MM 那一支是有伪标签的）。

## 验收计划

等价测试（权重复制，**点名比较全部输出头与对比学习用的投影**，不只比预测——理由见 `investigations.md#dlf-task-heads`）→ 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照 → 判定 → storyline / 台账 / experiments / model_table 四处文档。

**参照侧预警**：`requirement.txt` 需先看；前四篇里 DMD 与 ConFEDE 的 release 都需要兼容补丁才能在当前 PyTorch 上启动。若需改数值的补丁，按 `investigations.md#dmd-reference-patches` 的分档规矩降级标注参照来源。
