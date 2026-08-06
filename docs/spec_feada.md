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

## 已核实（2026-08-06）

### 1. 选轮口径：验证集**预测损失**，符合约定 2

`eval` 取的是模型返回值的第 3 位。对照训练处的解包
（`pred, loss, loss_nce, pred_loss, sup_const_loss`），该位是 **`pred_loss`**，即
`MSELoss(pred, label)`——只含主任务、不含对比项，且只在验证集上算。**与约定 2 同口径。**

**但有一处末批超权**：`loss += _loss.item() * 32` 用的是写死的 32 而非实际批大小，再除以批数。
最后一个不满批被当成整批计。这正是本项目 `#select-reduction` 查过的口径，实现时以
`--select-reduction batch` 对齐，不得"顺手改对"。

### 2. 七个模块参与前向，却从不被更新——**本项目至今最强的一例**

四个参数组是：`proj_t`+`text_encoder` / `proj_v`+`vision_with_text`+`promptv_m` /
`proj_a`+`audio_with_text`+`prompta_m` / **名字含 `_decoder` 的**（即 `TVA_decoder`、`mono_decoder`）。

**落在四组之外、但在 `forward` 里被调用的：**

| 模块 | 用处 | 行号 |
|---|---|---|
| `T_simi_proj` / `T_dissimi_proj` | 文本的相似/相异投影 | 227 |
| `V_simi_proj` / `V_dissimi_proj` | 视觉同上 | 228 |
| `A_simi_proj` / `A_dissimi_proj` | 音频同上 | 230 |
| `p2a` | `Linear(384, 768)`，投影相似表征 | 293-294 |

**这七个模块永远停在随机初始化。** 它们有梯度，但优化器里没有它们，`step()` 永远不动它们。

**与音视频编码器的情况不同**：那两个是 `set_froze()` 显式冻结的（第 186-187 行），
意图明确、有据可查。**这七个是被参数分组遗漏的**——代码里没有任何一处说要冻结它们。

**而对比学习正是靠这些投影头把表征投到对比空间的。** 也就是说，
**FeaDA 的对比损失作用在一组随机投影上。**

按约定 5 原样复现（本仓库的 `param_groups` 也不收它们），并在 storyline 记录。
**不得"修正"它**——那样就不是这篇论文了。

### 3-5. 其余

- `promptv_m` / `prompta_m` 是**加性提示**：`proj_v(vision).permute(1,0,2) + promptv_m.unsqueeze(1)`，
  形状 `(seq_len, 768)`（视觉 500、音频 375），**不是 `p_len=3` 那种前缀提示**。`p_len` 另有用处，实现时再核。
- 总损失：`pred_loss + 0.02·sup_const_loss + 0.03·mono_task_loss + 0.09·(loss_v + loss_a)`，
  其中 `loss_v`/`loss_a` 用的是 **`KLDivLoss(reduction='batchmean')`**——这是 FeaDA 相对 ConFEDE 的增量所在。
- 伙伴采样：与 ConFEDE 同一套 `dataset.sample(idx)`，具体池子构造实现时逐行核。

## 我自己违反了本规格（2026-08-06，必须记下来）

本规格的「验收计划」一节写着：等价测试要**单独验对比损失**，理由是 CLGSI 那次证明只比前向不够。

**然后我写了一个只比前向输出（六个视图 + 两个门 + 预测）而不验对比损失的测试，并且在模型里根本没实现对比损失。**

作者的总损失是四项：

```
pred_loss + 0.02·sup_const_loss + 0.03·mono_task_loss + 0.09·(loss_v + loss_a)
                ↑ 我没实现
```

`sup_const_loss` 需要伙伴采样（`sample2 = train_data.dataset.sample(idx)`，与 ConFEDE 同一套），
我在移植时**整条路径都没接**。等价测试全部通过（7.451e-09），10 seed 也跑完了（MAE 0.7449），
**而那个数字是一个缺了一项损失的模型的数字，已丢弃。**

**暴露它的不是任何检查，是作者代码参照跑批的一次 CUDA OOM**——它的显存占用比我们高得多，
因为伙伴采样把每步的有效批量乘了 7。**我去查为什么它更耗显存，才发现我少算了一项。**

**这与 DLF 那次是同一个失效形态**（`investigations.md#dlf-task-heads`），而且更糟：
DLF 那次我没预料到，这次**规格里写明了要防的就是这个**。

> **写下防范措施不等于执行了它。** 规格里的「必须」，要有一个会失败的检查来兑现，
> 否则它只是一句写在文档里的话。

**补救**：等价测试增加对比损失的逐项比对（对上作者的 `cont_NTXentLoss`），
模型补上伙伴采样与 `sup_const_loss`，然后重跑 10 seed。

## 验收计划

等价测试（权重复制，**点名比较全部输出头与对比学习用的表征**，并**单独验对比损失**——CLGSI 那次证明只比前向不够）→ 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照 → 判定 → storyline / 台账 / experiments / model_table 四处文档。

**等价测试的输入形状必须取自真实数据管道**（音频 375、视觉 500 的 unaligned 长度），不得取配置里对齐后的数值——CLGSI 那次正是这样让测试覆盖了训练不走的路径。
