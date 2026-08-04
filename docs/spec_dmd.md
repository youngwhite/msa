# DMD 实现规格（写代码之前定稿）

**DMD — Decoupled Multimodal Distillation for Emotion Recognition**（CVPR 2023 highlight），作者实现 [mdswyz/DMD](https://github.com/mdswyz/DMD)，**MIT**。

顺序纪律照 DLF 那次执行（三次返工换来的）：**先读作者训练器定协议 → 写本规格 → 再写模型 → 等价测试 → 冒烟 → 10 seed**。DLF 上这么做，等价测试零计算差异。

## 台账遗留问题，已核实

`survey.md` 待核项「DMD 使用谁家的预处理特征（README 未写）」——**答案：与本仓库同源**。其 `config/config.json` 的 mosi 条目是 `MOSI/Processed/aligned_50.pkl`、`feature_dims [768, 5, 20]`、`train_samples 1284`、`KeyEval Loss`，逐项与 MMSA 一致。**不需要下载 README 里那个 Google Drive 包。**

## DLF 与 DMD 的关系：DLF 是 DMD 砍掉四条通路

这条关系有独立价值，**它把 DLF 那两处「论文没写」的发现坐实成可验证的来源**：

| | DMD（CVPR 2023） | DLF（AAAI 2025） |
|---|---|---|
| 六条跨模态注意力 | **全部调用** | 只调用 `trans_l_with_a`、`trans_l_with_v`；另外四条构造了不接线 |
| `proj_cosine_*` | 使用 | 构造了不使用 |
| 任务头 | **8 个**，权重全 1 | 5 个，语言头权重 **3** |
| 低阶（homo）分支 | 图蒸馏 | 退化为普通 logits |
| `nlevels` | 4 | 2 |

**「语言聚焦」在实现层面 = DMD 去掉四条通路 + 语言头加权。** 两篇论文都没写这件事。

## 训练协议（`trains/singleTask/DMD.py`）

| 项 | 作者实现 | 本仓库参数 |
|---|---|---|
| 优化器 | `Adam(model[0]+model[1]+model[2] 参数, lr=1e-4)`，**单一参数组** | `--optimizer adam --lr 1e-4` |
| 调度器 | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` | `--lr-schedule plateau --lr-schedule-factor 0.5 --lr-schedule-patience 5` |
| 梯度裁剪 | `clip_grad_value_(三个模块全部参数, 0.6)` | `--grad-clip 0.6 --clip-mode value` |
| 梯度累积 | `update_epochs: 10` | `--accumulate-steps 10` |
| 早停 | `early_stop: 10` | `--patience 10` |
| 批大小 / 权重衰减 | 16 / 0.005 | `--batch-size 16 --weight-decay 0.005` |
| `nlevels` | **4**（DLF 是 2） | 模型内超参 |

**BERT 不单独设学习率**（同 DLF）：三个模块的参数拼成一个组，BERT 与其余同为 1e-4。**不得套用本仓库 `lr*0.1` 的惯例**——DPDF-LQ 上这条值 2.7 SE。

**两个蒸馏核是被优化的模块，不是损失函数。** 它们有参数（`W_logit`、`W_repr`、`W_edge`）且进同一个优化器。本仓库的 `MSAModel` 契约允许模型自带子模块，故三者合成一个 `nn.Module` 即可，**不需要改训练循环**。

## 损失：八个任务头 + 两路图蒸馏 + 四项解耦

```
L = L_task + 0.05·(homo: L_logit + L_reg)
           + 0.05·(hetero: L_logit + L_repr + L_reg)
           + 0.1·(L_s→sr + L_recon + 0.1·(L_sim + L_ort))

L_task = 全部 8 个头等权相加：
         output_logit + {l,v,a}_homo + {l,v,a}_hetero + logits_c
```

**DLF 的教训直接适用**：等价测试必须点名比较**全部八个头**，不能只比最终预测——其中六个不进最终预测。见 `investigations.md#dlf-task-heads`。

### 两处只在代码里的不对称（论文未写）

1. **homo 支路丢掉了表征蒸馏项。** 训练器只取 `0.05*(loss_logit + loss_reg)`，而 `distillation_loss` 照常算出 `loss_repr_homo` 并**丢弃**。hetero 支路则三项全用。
2. **两个核算表征距离的函数不同**：hetero 用 `min_cosine`，homo 用 `distance_metric`——**而 homo 的结果正好是被丢弃的那一项**。也就是说两个核的唯一实质差异，落在唯一不生效的地方。

### 图先验不同

`gd_prior` 是 6 元向量（3×2 个有序对）：homo 用 `softmax([0,0,1,0,1,0], 0.25)`，hetero 用 `softmax([0,0,1,0,1,1], 0.25)`。`alpha = 1/8`，`gd_size` homo 64 / hetero 32，`w_losses = [1, 10]`，`metric = 'l1'`。

## 待核实项（实现时必须查清，不得想当然）

1. **`forward` 与 `distillation_loss` 的跳过条件写法不一致。** `forward` 里 `for i in self.from_idx: if i == j`（比较模态编号）；`distillation_loss` 里 `for i, idx in enumerate(self.from_idx): if i == j`（比较**枚举位置**）。**本配置下 `from_idx == [0,1,2]`，位置恰等于编号，两者取到同一组对，因此当前无差异。**属于脆弱写法而非现行 bug——记录在此，若将来改 `from_idx` 会立刻分岔。
2. ~~`edges_origin` 是否有下游消费者~~ **已核实：没有。** 两个核都返回它，训练器都接收（`edges_origin_homo`、`edges_origin_hetero`），此后全文再无引用——**又一处死输出**。我们的实现不产出它。
3. `nlevels=4` 但 `trans_*_mem` 固定 `layers=3`——两个深度不同，不要合并。
4. Conv1d 无 padding，长度 50→46，所有 `view(batch, -1)` 依赖此数（同 DLF）。

## 复用与隔离

复用 `src/msa/models/transformers.py` 的通用积木（与 DMD/DLF 同源）。**不复用 `dlf.py` 的内部实现**：DLF 已通过等价测试并入库，共享代码会让改 DMD 有改动 DLF 数字的风险。宁可重复，也不让已验收的模型依赖未完成的模型。

**位置编码必须显式 `position_embedding=True`**——本仓库 MulT 丢位置编码正是照抄 MMSA 默认关闭的开关（约定 5）。

## 验收计划

等价测试（权重复制，**八个头全部比较**）→ 冒烟 → 10 seed（42-51）→ 作者代码 10 seed 参照（`author_reference.py` 加一个条目）→ 判据 → storyline 与台账。

**参照侧已确认**：DMD 同样每 epoch 存一次 checkpoint（`trains/singleTask/DMD.py:212`，存到 `./pt/<epoch>.pth`）且从不删除——与 DLF 同一个坑。`author_reference.py` 的清理线程直接接上（`"reap": "pt"`），不必再踩一次。
