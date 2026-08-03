# DLF 实现规格（草案，待确认后再写代码）

**DLF — Disentangled-Language-Focused Multimodal Sentiment Analysis**（AAAI-25），作者实现 [pwang322/DLF](https://github.com/pwang322/DLF)，**MIT**。

同前几个的做法：结构与协议取自作者实现，超参取自其 `config/config.json`，等价测试作为验收仪器。**这一次在写模型之前就把协议对齐了**——DPDF-LQ 上正是协议没对齐导致返工三次。

## 数据口径：完全同源，不需要额外下载

`config/config.json` 的 mosi 条目与 MMSA 逐项一致：

```
featurePath: MOSI/Processed/aligned_50.pkl
feature_dims: [768, 5, 20]      train_samples: 1284      KeyEval: Loss
```

`data_loader.py` 读的键是 `text_bert` / `vision` / `audio` / `raw_text`，并有 `need_data_aligned` 开关——**DLF 是从 MMSA 框架派生的**。README 里那个 Google Drive 链接（与 DMD 共用）不必下载，本仓库现有的 `aligned_50.pkl` 即可。

## 训练协议（`trains/singleTask/DLF.py`）

| 项 | 作者实现 | 本仓库参数 |
|---|---|---|
| 优化器 | `optim.Adam(model[0].parameters(), lr=1e-4)`，**单一参数组** | `--optimizer adam --lr 1e-4` |
| 调度器 | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` | `--lr-schedule plateau --lr-schedule-factor 0.5 --lr-schedule-patience 5` |
| 梯度裁剪 | `clip_grad_value_(params, 0.6)` | `--grad-clip 0.6 --clip-mode value` |
| 梯度累积 | `update_epochs: 10` | `--accumulate-steps 10` |
| 早停 | `early_stop: 10` | `--patience 10` |
| 批大小 / 权重衰减 | 16 / 0.005 | `--batch-size 16 --weight-decay 0.005` |

**每一项本仓库都已支持，不需要改训练循环**——因为 trainer 本就是照 MMSA 的行为建的，而 DLF 派生自 MMSA。

**BERT 不单独设学习率。** `params = model[0].parameters()` 是一个组，BERT 与其余同为 1e-4。**不得套用本仓库其它微调 BERT 模型那套 `lr*0.1` 的惯例**——DPDF-LQ 上这条值 2.7 SE。

## 结构：MulT 骨架 + 解耦 + 重构

模型超参：`dst_feature_dim_nheads: [50, 10]`、`nlevels: 2`、`conv1d_kernel_size_{l,a,v}: 5`、`attn_dropout 0.3`、`attn_dropout_a 0.2`、`attn_dropout_v 0.0`、`text_dropout 0.5`、`output_dropout 0.5`、`embed_dropout 0.2`。

前向的六个阶段：

1. **投影**：BERT →（转置）→ `Conv1d(kernel=5, padding=0)` 把三模态投到 50 维。**无 padding，所以序列长度从 50 变成 46**
2. **解耦**：每个模态过一个**模态专属**编码器 `encoder_s_*`，再过**共享**编码器 `encoder_c`（三模态复用同一个实例）
3. **重构**：`decoder_* = Conv1d(2·d → d, kernel=1)` 吃 `[s_x; c_x]`，重构回该模态；重构结果**再过一次 `encoder_s_*`** 得到 `s_*_r`
4. **增强（低阶）**：三个 `c_*` 各自出一个 logits（残差 MLP）
5. **LFA（高阶）**：`s_l` 自注意力，`s_a`/`s_v` 分别以 `s_l` 为 query 做跨模态注意力（`trans_l_with_a(s_l, s_a, s_a)`）——**这是"语言聚焦"的实处：音视频都被投到语言的时间轴上**
6. **融合**：四路（l/v/a 高阶 + c 融合）各过 `sigmoid` 投影后拼接

**可复用的积木**：`src/msa/models/bert.py` 的 `BertTextEncoder`、`src/msa/models/transformers.py` 的 `TransformerEncoder`（与 DLF 用的是 MMSA 同一份代码）。

## 损失：五项

```
L = L_task + 0.1 · [ L_s→sr + L_recon + 0.1 · (L_sim + L_ort) ]
```

- **`L_task`** 主回归损失
- **`L_recon`** 重构损失（MSE）
- **`L_s→sr`** 重构后再编码应回到原 specific 表征（MSE，三模态相加）
- **`L_ort`** `CosineEmbeddingLoss(target=-1)`，逼 `s_x` 与 `c_x` 正交，三模态相加
- **`L_sim`** 三个 `c_*_sim` 上的 triplet margin（`HingeLoss`），按样本标签构造 id：同一样本的三个模态共享一个 id

## 待核实（实现时必须查清，不得想当然）

1. **`num = 50` 的语义**。`L_ort` 里 `output['s_l'].reshape(-1, num)` 的 `num` 在 mosi 是 50、mosei 是 10。50 与 `dst_feature_dim` 相同，但 reshape 的分组方式直接决定余弦在哪个轴上算——**分错轴，正交损失就变成另一个东西**。
2. **`HingeLoss` 的具体定义**（margin、正负对构造）在 `utils` 里，需逐行核对。
3. **`encoder_c` 三模态共享同一实例**——参数共享，等价测试时参数顺序会与"每模态一个"不同。
4. Conv1d 无 padding 导致长度 50→46，后续所有 `view(batch, -1)` 的维度都依赖这个数，改 kernel 会连锁。

## 验收计划

与 DPDF-LQ 相同：等价测试（权重复制，预期 `0.000e+00`）→ 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照（`scripts/author_reference.py` 加一个条目）→ 判据 → storyline 与台账。

**顺序纪律**（DPDF-LQ 的三次返工换来的）：先定协议 → 再写模型 → 等价测试 → **改完模型必重跑冒烟** → 才起 10 seed。
