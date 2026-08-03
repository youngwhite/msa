# ConFEDE 与 DPDF-LQ 实现规格（草案，待确认后再写代码）

同 `spec_mctn_mfm.md` 的做法：**结构与目标函数取自论文**，超参优先取论文、论文未给才取作者实现，官方代码只作验证仪器。两者数据口径都已核实与本仓库同源（见下）。

---

## ConFEDE — Contrastive Feature Decomposition (ACL 2023, pp.7617-7630)

### 论文定义的结构

1. **编码**：文本用 **BERT 的 `[CLS]`**；视觉与听觉各用**一个独立的 Transformer 编码器**
2. **分解**：每个模态经两个投影器分成**相似特征**与**相异特征** → `Ts/Td, Vs/Vd, As/Ad`
   - 投影器 = LayerNorm → Linear + Tanh → Dropout
3. **融合与预测**：六个特征拼接后过 MLP(ReLU)

### 目标函数

```
L_all = L_pred + λ_uni · L_uni + λ_cl · L_cl
```

- **`L_pred`**：`ŷ = MLP([Ts;Vs;As;Td;Vd;Ad])`，对多模态标签取 **MSE**
- **`L_uni`**：六个特征各自过**权重共享**的 MLP，得 `û`；目标向量是 `u = [y_m, y_m, y_m, y_t, y_v, y_a]`——**相似特征去预测多模态标签，相异特征去预测各自的单模态标签**
- **`L_cl`**：NT-Xent，含**样本内对比**与**样本间对比**

### 移植时的第一个坑：MOSI 没有单模态标签

`L_uni` 需要 `y_t / y_v / y_a`，而 **MOSI 不提供逐模态标注**（这正是 `mlf_dnn` / `mtfn` 等只有 SIMS 配置的原因）。

**论文脚注 1 自己交代了做法**：*"we trained our models on the MOSI and MOSEI datasets **without unimodal labels**. Instead, we used their **multimodal labels** for compatibility."* 即在 MOSI 上 `y_t = y_v = y_a = y_m`。

**必须照此实现并在 docstring 写明**——否则 `L_uni` 的一半项无定义，而这是个只在脚注里出现的信息。

### 第二个坑：它的基线就是那张不可达的表

同一条脚注给出的对照来源是 `github.com/thuiar/MMSA/blob/master/results/result-stat.md`——**本项目已证 MMSA 自己的代码在 9/11 个模型上都达不到那张表**（`#mmsa-all-eleven`）。ALMT 亦然。

**含义**：ConFEDE 论文里"outperforms all baselines"的**相对提升幅度需要在同口径下重新测量**，这不是说它错了。这正是本项目能提供而原论文提供不了的东西。

### 待检验的主张

> 把每个模态分解成**相似/相异**两部分，并用对比关系（以文本为中心）来学，比不分解更好。

**检验方式是复现它的消融，不是它的头条数字**：去掉 `L_cl`、去掉分解（只留一路特征）后，方向与量级是否与论文一致。

### 数据与超参

- **数据口径**：README 明写 **"The dataset is available for download at: GitHub Repository - MMSA"**——**与本仓库完全同源**，可直接进对照表
- 许可证 **MIT**；提供 checkpoint（Google Drive），**可做权重复制的数值等价测试**
- `λ_uni`、`λ_cl` 及各层维度取作者实现，逐项记出处

---

## DPDF-LQ — Dual-Path Dynamic Fusion with Learnable Query (EMNLP 2025 主会)

### 论文定义的结构

**双路并行**，最后动态门控融合：

- **全局路**：各模态序列首位置一个**可学习全局 token**（ViT 式）→ 取出 `h_m = H_m[0,:]` → 经 **DGLQA 编码器**（Dynamic Global Learnable Query Attention + 通道注意力）→ 再经**文本引导的交叉注意力**精炼
- **局部路**：以**音视频特征作为 query** 的交叉注意力，抽细粒度特征
- **动态门控**：整合两路

投影：`X_m = FC(X_m)` → `H_m = Trans(X_m)`

### 目标函数

**只有一项 MSE**（式 36）：

```
L = (1/N) Σ ||y_n − ŷ_n||²
```

论文特意强调"简单的优化目标使 DPDF-LQ 比多目标方法更易训练"——**没有辅助损失、没有两阶段训练**，是这批近年方法里移植成本最低的。

### 超参（论文 Table 3 直接给出，MOSI 列）

| 项 | 值 |
|---|---|
| Learnable query length | 8 |
| Learnable query dimension | 128 |
| DGLQA depth | 3 |
| GPath CA depth | 2 |
| LPath CA depth | 2 |
| Hidden dimension | 256 |
| Learning rate | 1e−4 |
| Weight decay | 1e−4 |
| Batch size | 64 |

**这在近年论文里是少见的完整**——按调研台账的判据，它不触发 `reimpl-partial`。

### 数据口径

论文写明特征来自 **BERT（文本）/ OpenFace（视觉）/ Librosa（听觉）**，README 写 **"Download the aligned versions of CMU-MOSI and CMU-MOSEI"** 且"默认配置用 BERT-base **以与基线公平比较**"。与本仓库的 `aligned_50.pkl` 同源。

许可证 **MIT**，`best_cpk/` 提供预训练权重 → **可做数值等价测试**。

### 值得单记的一点

论文说它做了 **五次独立随机种子运行**并报平均（附录 Figure 6 与 ALMT 对比）。**这在本领域是少数派**——绝大多数论文报单值。这意味着：

1. 它的数字比多数论文更可信
2. **但 5 seed 仍不足以支撑本项目判据要求的精度**（我们用 10 seed，且 MOSI 上 LF-LSTM 的 seed 间 MAE 标准差是 0.039）
3. 它与 ALMT 的对比是"5 seed 均值 vs 5 seed 均值"，**没有报方差**，所以那个对比的显著性仍未知——而我们能算

### 待检验的主张

> 双路（全局 + 局部）加动态门控，优于单路。

同样检验消融方向，而非头条数字。

---

## 两者共同的实施要点

1. **都先做数值等价测试**（两者都提供 checkpoint），确认骨架无误，再谈数字
2. **都进 `reproduce_all.sh`**，10 seed（42-51），参照用它们自己的官方代码在本机跑 10 seed——不用它们论文里的数字（那是 5 seed 或单值，且对照基线是那张不可达的表）
3. **ConFEDE 的 `L_uni` 在 MOSI 上退化**这一点必须进 storyline，否则读者会以为我们漏实现了

## 已确认（2026-08-02）

- **不建双组**：两者数据与本仓库同源，不存在 MCTN/MFM 那种"参照偏离论文"的问题
- **NT-Xent 按作者实现取**

## 读作者代码后补入的两条（ConFEDE）

**一、它的 NT-Xent 是被改过的。** `util/metrics.py` 的 `cont_NTXentLoss` 继承 `pytorch_metric_learning.NTXentLoss`，但重写了 `_compute_loss`：

```python
weight = abs(self.label[a2] - self.label[n])     # 锚点与负样本的标签距离
neg_pairs = neg_pairs * weight / 2               # 负样本按标签距离加权
```

即**标签越接近的负样本被压得越轻**——这是把回归标签的连续性注入对比损失，**论文正文没有明说**。温度取 `sds_heat = const_heat = 0.5`（不是 NT-Xent 常见的 0.07）。按约定 5，这属于"参照实现的行为与正文不一致"，必须记录并照行为实现。

**二、它是分阶段训练的，不是端到端。** 作者代码分四步：`Ttrain.py` / `Vtrain.py` / `Atrain.py` 各自预训练单模态编码器（epoch 200 / 100 / 100，bs 128，lr 1e-4），再由 `main-fusion.py` 训练融合（epoch 25，**bs 16**）。

**本仓库的单一 trainer 不支持多阶段。** 这比 MMIM 当初那个"每 epoch 先跑一遍 MMILB"更重——那是 epoch 内的两段，这是四个独立训练过程。三条路：

- **A. 扩展 trainer 支持阶段化训练**（新增 `stages` 契约）。忠实，但属训练循环的结构性改动，影响所有模型
- **B. 端到端训练**，把分阶段作为已知协议差异记录。成本最低，但**不是论文/作者跑的东西**，数字含义会变
- **C. 把单模态预训练做成一次性产物**（类似特征提取），融合阶段读它。介于两者之间

**这一条需要拍板后才能动 ConFEDE。** DPDF-LQ 不受影响——单损失、单阶段。

### 其余已核实的 ConFEDE 超参（作者 `MOSI/config.py`）

`raw_data_path = data/MOSI/Processed/**unaligned_50.pkl**`（**unaligned，不是 aligned**）、`text_fea_dim 768 / vision_fea_dim 20 / audio_fea_dim 5`、`video_seq_len 500 / audio_seq_len 375`、`proj_fea_dim 256`、三路 dropout 0.5、`vision_nhead = audio_nhead = 8`、`lr 1e-4`。**维度与序列长度与本仓库的 unaligned pkl 逐项一致。**

作者用的 seed 是 `[1, 12, 123, 1234, 12345]`（5 个）；我们用注册的 42-51（10 个）。
