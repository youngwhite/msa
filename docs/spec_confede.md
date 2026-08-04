# ConFEDE 基础设施方案（写代码之前定稿）

**ConFEDE — Contrastive Feature Decomposition for Multimodal Sentiment Analysis**（ACL 2023），作者实现 [Haoyu-ha/ConFEDE](https://github.com/Haoyu-ha/ConFEDE)。

前四个近年方法（DPDF-LQ / DLF / DMD）**训练循环一行没改**。ConFEDE 是第一个改不改要先想清楚的——它需要三样此前没有的东西。本文先把这三样落地成方案，**再动手写模型**（顺序纪律见 `spec_dmd.md`，已三次奏效）。

## 一、它比前面几个多要什么

| 需求 | 作者实现在哪 | 前面几个模型是否需要 |
|---|---|---|
| **每步第二个 batch**：每个锚样本额外取 6 个伙伴 | `dataloader/MOSI.py:sample()` | 否 |
| **训练中周期性重建相似度矩阵**：整个训练集前向一遍，重算 n×n 余弦、排序、重建候选池 | `TVA_fusion_train.py:12` + `dataloader/MOSI.py:update_matrix()` | 否 |
| **NT-Xent，且负样本按标签距离加权** | `util/metrics.py:cont_NTXentLoss` | 否 |

伙伴的取法：每个锚样本从三个池子各随机取 2 个——`ss`（同标签、最相似）、`dd`（异标签、最不相似）、`sd`（异标签、相似）。**batch 16 → 每步实际过 16 + 96 = 112 个样本。**

## 二、两阶段结构已确认，沿用既定的方案 C

`main.py` 依次做文本、视觉、音频三个单模态预训练；`main-fusion.py` 才是融合。融合模型 `load_model(load_pretrain=True)` 分别读三个编码器的 `state_dict`（`TVA_fusion.py:213-217`）。

**方案 C（已裁定）**：单模态预训练做成一次性产物。

- 新增 `scripts/pretrain_confede.py`，产出三个编码器权重到 `outputs/confede_pretrain/`（**不入库**，与 `best.pt` 同例；产物可由脚本重建，且逐比特可复现）
- 融合阶段读盘，**不重复预训练**
- 三个单模态阶段各自的超参：batch 128、lr 1e-4、epoch 200/100/100

**为什么不做双组**（已裁定）：预训练是达成融合阶段的手段，不是要汇报的结果。把它做成实验组会让表里多出三行没人比较的数字。

## 三、训练循环仍然不改——用现有钩子

本仓库 `Trainer` 已有的钩子恰好够用：

| ConFEDE 要做的事 | 挂在哪个现有钩子 |
|---|---|
| 每个奇数 epoch 重建相似度矩阵 | `on_train_epoch_start(epoch)` |
| 每步取伙伴 batch | `forward(batch)` 内部，用 `batch["index"]` |
| 只在训练时取伙伴，验证/测试不取 | `self.training` |

**关键点：`batch["index"]` 本仓库已经在提供**（`check_data.py` 打印的第一个键就是它）。作者实现也正是用样本下标去查候选池，所以这条路径与作者一致，不是我们的发明。

模型持有训练集张量的引用以便 gather 伙伴。**代价已量过**：文本 1284×50×768 float32 ≈ 197MB，视觉 ≈ 5MB，音频 ≈ 1MB。可接受，且不额外拷贝一份（引用而非复制）。

> **不改训练循环这条不是洁癖。** 训练循环是全部 20 个组共用的，改它意味着此前所有数字的可比性都要重新论证。宁可让模型内部复杂，也不动共用路径。

## 四、依赖：加 `pytorch-metric-learning`

作者的 `cont_NTXentLoss` **继承** `pytorch_metric_learning.losses.NTXentLoss`，只重写 `_compute_loss`，把负样本对乘上 `|label[a2] - label[n]| / 2`。

已裁定「NT-Xent 按作者实现」。要做到，两条路：

1. **装 `pytorch-metric-learning`，照作者的方式继承**——与作者代码路径完全相同，且等价测试可以直接对比损失值
2. 自己重写 `NTXentLoss` + `convert_to_pairs`——多一份可能出错的代码，且**违背参照层级**（作者代码 > 自实现）

**取 1。** 该库 MIT、纯 Python、约 1MB，磁盘代价可忽略（当前余量 5G）。**版本按作者的 `requirements.txt` 钉死在 `0.9.99`**，理由同 `transformers` 钉 5.14.1：跨版本改行为的库不能浮动。

## 五、读代码已经查出的三件事（论文都没写）

### 1. 融合阶段全程冻结 BERT

`TVA_fusion.py:227` 的默认 `train_module = [False, False, True, True]`——**文本编码器两部分都不训练**。解冻发生在 `if epoch == finetune_epoch`，而配置里 `finetune_epoch = 200`、`epoch = 25`（`config.py:80-86`，同一个类）。

**25 < 200，这个分支永远不触发。BERT 在整个融合阶段是冻的。**

这与 DPDF-LQ / DLF / DMD 完全相反（那三个都微调 BERT）。**按本仓库惯例给 BERT 设 lr×0.1 会是彻底错的方向——它根本不该有梯度。** DPDF-LQ 上「BERT 学习率想当然」值 2.7 SE，这次的坑更深。

### 2. `ds` 候选池建了不用

`__pre_sample` 建四个池 `ss` / `sd` / `ds` / `dd`，而 `sample()` 只从 `ss`、`dd`、`sd` 取。**`ds` 是死代码**——与 DLF 的四条通路、DMD 的 `proj_cosine_*` 同一类。

### 3. `train_bool` 长度对不上

解冻时传的是 5 元列表，默认是 4 元；代码只用到 `[0:2]` 和 `[3]`，所以不报错，第 5 个元素无意义。不影响数值，记录备查。

## 六、代价估算

- **每步 112 个样本过编码器**（16 锚 + 96 伙伴）。BERT 冻结 ⇒ 无反向，代价主要在前向
- **每奇数 epoch 一次全训练集前向**（1284 样本）+ n² 的 Python 双重循环建池（1284² ≈ 165 万次）。25 epoch ⇒ 13 次
- 融合阶段 25 epoch × 81 步 = 2025 步

**先量再跑**：实现后先跑单 epoch 计时，若单 seed 超过 40 分钟，10 seed 就要排到后台并重新安排磁盘余量（参照跑批已证明单 seed 可吃掉 22GB 临时空间）。

## 七、验收计划

与前四个一致，外加一项：

1. **等价测试**（权重复制）——**必须点名比较全部输出头与对比学习用的投影**，不能只比预测。理由见 `investigations.md#dlf-task-heads`：DLF 漏掉四项损失而预测比对照样通过
2. **损失单独比对**：伙伴采样有随机性，故等价测试固定 `random.seed` 后比对 NT-Xent 项本身，而不只是端到端损失
3. 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照 → 判据 → storyline 与台账

**参照侧预警**：作者 `requirements.txt` 钉 `torch==1.8.1`、`transformers==4.5.1`、`pytorch-pretrained-bert`。DMD 已经证明这个年代的 release 在当前 PyTorch 上可能根本起不来（3-D `CosineEmbeddingLoss`）。**若 ConFEDE 也需要改数值的补丁，按 `investigations.md#dmd-reference-patches` 的分档规矩处理：进梯度或选轮的补丁一律降级标注参照来源。**

## 八、实现时必须查清（不得想当然）

1. 单模态预训练的**产物契约**：三个 `state_dict` 的键名与融合模型的子模块是否严格对应（`load_state_dict` 非 strict 会静默吞掉不匹配）
2. `update_matrix` 用的是**模型特征**，而首次建矩阵用的是**原始特征**——首个 epoch 的池子来源与之后不同，需确认我们的实现也保持这个差别
3. `cont_NTXentLoss.update_label` 的调用时机决定负样本加权是否生效；**若忘记调用，损失静默退化为无加权 NT-Xent**
4. 伙伴 batch 是否参与梯度（作者实现里 `sample2` 走同样的编码器，故参与）
5. 三个单模态阶段的选轮指标（`check={'MAE':10000}`）与本仓库约定 2 是否一致
