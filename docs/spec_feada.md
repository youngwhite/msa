# FeaDA 实现规格（写代码之前定稿）

**FeaDA**（IJCNLP-AACL 2025），作者实现 [PowerLittleYin/FeaDA-main](https://github.com/PowerLittleYin/FeaDA-main)。**仓库无 LICENSE 文件**，即默认保留所有权利——阅读、重实现、本地跑作参照是通行学术做法，但**台账不得记作 MIT**。

## 核实结果（三项全过，2026-08-06）

| 项 | 结果 |
|---|---|
| 数据集 | CMU-MOSI ✅ |
| 作者代码 | 仓库有实质代码（48 个 .py），非空壳 ✅ |
| **特征口径** | **`unaligned_50.pkl` + `bert-base-uncased`**，与本表 22 组一致 ✅ |

依赖只有常规项加 `pytorch_metric_learning`（为 ConFEDE 已装）与 matplotlib（参照跑批用桩）。**无外部产物依赖**——这正是它排在 KuDA（权重只在百度网盘）与 TF-Mamba（需编译 `mamba_ssm`）之前的原因。

## 它是 ConFEDE 的后续工作

配置命名逐项相同（`uni_fea_encoder` / `sds_heat` / `const_heat` / `visionPretrain`），训练循环里同样有 `sample2 = train_data.dataset.sample(idx)` 的伙伴采样。**这与 DLF 之于 DMD 是同一种关系**，本项目已第二次遇到。

**据此可以问一个具体问题**（与强度加权那次同类）：FeaDA 相对 ConFEDE 的增量，在 10 seed 下是否可检出？两者已在同一协议、同一特征上跑过，ConFEDE 是 MAE 0.7358 ± 0.0185。

**但不得据此复用 `confede.py` 的内部实现**——ConFEDE 已验收入库，共享代码会让改 FeaDA 有改动它数字的风险。宁可重复（同 `spec_dmd.md` 的隔离决定）。

## 训练协议（`train/TVA_train.py`）

| 项 | 作者实现 | 本仓库 |
|---|---|---|
| 优化器 | **`Adam`**（不是 AdamW），四个参数组 | 由模型 `param_groups` 提供 |
| lr | text 5e-5 / audio 1e-3 / vision 1e-3 / other 1e-3 | 同 |
| weight decay | 四组均 1e-3 | 同 |
| 批大小 | 32 | `--batch-size 32` |
| 梯度累积 | `update_epochs: 4` | `--accumulate-steps 4` |
| 轮数 | 25 | `--epochs 25` |
| 调度器 | **无** | `--lr-schedule none` |
| 选轮 | **验证集 loss**（`result_loss < best_loss`），非 MAE | 需确认该 loss 的定义，见待核实 1 |

**参数分组按模块名而非前缀**：text 组是 BERT 相关，vision 组是 `proj_v` + `vision_with_text` + **`promptv_m`**，audio 组同理加 `prompta_m`，other 组是名字含 `_decoder` 的参数。**注意 other 组只收 `_decoder`**——不在四组里的参数**根本不进优化器**，实现时必须逐个核对有无遗漏（见待核实 2）。

## 单模态预训练是必需的

`models/model.py:181-184` 在构造时读 `encoder_path/<seed>/` 下的音频与视觉编码器权重。**只有音频与视觉两支**（`train/` 下只有 `Atrain.py` 与 `Vtrain.py`，没有文本预训练），各 25 轮、lr 1e-3、decay 1e-3。

**不得跳过。** ConFEDE 上跳过预训练的代价是 MAE 1.1549 对 0.7358，而那个错数字在表里看起来完全正常（见 `storyline.md` 第 18 节）。本仓库已有 `scripts/pretrain_confede.py` 可作范本，但**另写 `pretrain_feada.py`**，理由同上（隔离）。

## 待核实（实现时必须查清，不得想当然）

1. **选轮用的 `result_loss` 是什么**——若含对比项等训练期损失，则与本仓库约定 2 的「按验证集主指标选轮」不同口径，须显式记录并决定如何对齐。
2. **四个参数组是否覆盖全部参数**。`model_params_other` 只收名字含 `_decoder` 的；若有参数落在四组之外，它们在作者实现里**不被更新**，这属于「声明了但不训练」，必须原样复现而不是补上。
3. `promptv_m` / `prompta_m` / `p2a` 三个提示向量的形状与用法（`p_len = 3`）。
4. `sds_heat` / `const_heat`（均 0.5）分别喂给哪个损失。
5. 伙伴采样的池子构造是否与 ConFEDE 相同（同标签相似/异标签不相似/异标签相似各取 2），还是有改动。

## 验收计划

等价测试（权重复制，**点名比较全部输出头与对比学习用的表征**，并**单独验对比损失**——CLGSI 那次证明只比前向不够）→ 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照 → 判定 → storyline / 台账 / experiments / model_table 四处文档。

**等价测试的输入形状必须取自真实数据管道**（音频 375、视觉 500 的 unaligned 长度），不得取配置里对齐后的数值——CLGSI 那次正是这样让测试覆盖了训练不走的路径。
