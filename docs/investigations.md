# 调查记录

**这个文件存在的唯一目的：不要把同一件事查第二遍。**

每条记录写清楚：问题是什么、试过哪些假设、**哪些被排除了**、结论是什么。被排除的假设和结论一样重要——它们是下一个人不必再走的路。

**结论被推翻时不要删除原文，追加「修正」小节。** 错误的裁定和正确的裁定同样值得留存：它记录了当时的推理为什么不成立。

---

## <a id="protocol-faithful"></a>复现组必须用忠实移植，不能带自己的改动（2026-07-26）

### 这是本项目到目前为止最重要的一条方法论教训

原先的验收组用的是**我们改进过的默认值**（按真实长度取 LSTM 末状态、按有效帧做掩码均值池化），而不是 MMSA 的原始行为（读 `h[-1]`、按 padded 宽度求均值）。

后果：**"复现是否到位"和"我们的改动是否有益"两件事被混在一起**，判定结果无法解释。

实测（10 seed，MOSI unaligned）：

| 模型 | 配置 | MAE | Acc-2(non0) | 判定 |
|---|---|---|---|---|
| TFN | 我们的默认（掩码） | 0.9546 (+0.8 SE) | 0.7779 (+4.0 SE) | 未复现 |
| TFN | **忠实移植** | **0.9511 (+0.5 SE)** | **0.7855 (+1.0 SE)** | **复现成功** |
| LMF | 我们的默认（掩码） | 0.9628 (+2.4 SE) | 0.7803 (+4.3 SE) | 未复现 |
| LMF | **忠实移植** | **0.9513 (+0.1 SE)** | **0.7854 (+1.6 SE)** | **复现成功** |

**两个模型都是同一个原因。** 掩码池化对 TFN 影响小、对 LMF 影响大（MAE 差 0.012），但两者的 Acc-2 都因此掉了 0.5-1 个百分点。

### 现行协议

- **验收组 = 忠实移植**：`--model-arg use_lengths=False --model-arg mask_pooling=False`，尽可能逐行复刻参照实现
- **我们的改动 = 单独的消融组**（`*_ablation_masked`），按自身价值评估，不与复现混谈
- `check_acceptance.py --all` 只判定验收组（命名为 `<model>_<dataset>`），消融组不与 MMSA 对比

### 移植新模型时的固定核对清单

每个模型都要逐项过一遍，前四项在 TFN/LMF 上都出现过真差异：

1. **梯度裁剪**——MMSA 的 trainer 一律不裁剪
2. **轮数上限**——MMSA 用 `while True` + patience，无上限
3. **padding / 池化行为**——用 `use_lengths=False, mask_pooling=False` 对齐
4. **optimizer 的参数分组**——MMSA 多处用 `list(model.parameters())[a:b]` 按位置切片，极易切错（见下方 LMF 条目）
5. `use_bert` 取值、数据是 aligned 还是 unaligned
6. 配置里声明但 forward 从未使用的层（见下方 LMF 条目）

---

## <a id="grad-accumulation"></a>梯度累积：影响 MulT / MISA / Self-MM 三个模型（2026-07-26）

### 现象

Self-MM 的 10 seed 验收**七个指标全部低约 5 SE**。如此均匀的落后不像某个模块实现错误（那通常只打击特定指标），更像训练动力学层面的差异。

### 结论

MMSA 的 `update_epochs` 是**梯度累积**：优化器每 N 个 batch 才 step 一次。三个模型都用：**MulT 8、Self-MM 4、MISA 2**。我们原先每个 batch 都 step，等效批大小与更新次数都不同。

原作者的配置文件里有一行注释直接印证：`# the batch_size of each epoch is update_epochs * batch_size`。

已加入 `TrainConfig.accumulate_steps`。实现细节按 MMSA 的实际行为：窗口内梯度**求和不平均**（平均会改变等效步长），epoch 末尾不足一窗口的梯度**丢弃**。

**教训**：移植时 trainer 的循环结构和模型定义同等重要。核对清单已加入这一项。

---

## <a id="self-mm-features"></a>Self-MM：伪标签的特征空间用错（2026-07-26）

对照**作者自己的实现**（`thuiar/Self-MM`，非 MMSA）发现：伪标签机制里的 `Feature_f/t/a/v` 是**各分支第一层线性 + ReLU 之后**的状态（fusion 128 / text 32 / audio 16 / vision 32 维），不是编码器的原始输出（816 / 768 / 16 / 32 维）。

我原先用的是原始输出。类中心因此落在完全不同且宽得多的空间里，而"到正负类中心的相对距离"正是 Self-MM 生成伪标签的唯一依据——**这是它的核心机制，不是细节**。

修正后单 seed 5 epoch 即达 MAE 0.753 / Acc-2 0.845 / Corr 0.796，而修正前 10 seed 均值为 0.756 / 0.835。

**教训**：只对照 MMSA 不够。MMSA 是二次实现，作者的原始发布才是第一手依据。已把原始仓库的获取方式写进本文件末尾。

---

## <a id="ef-lstm-collapse"></a>EF-LSTM 有 20% 的 seed 训练崩溃，且这让验收判据变松（2026-07-26）

10 个 seed 中 **seed 46 和 48 完全崩溃**：MAE 1.46（≈ 恒定预测）、Acc-2 0.422（= 多数类比例）。其余 8 个正常（MAE 0.95–1.03，Acc-2 0.76–0.80）。

崩溃把标准差从约 0.03 抬到 0.21，SE 随之放大约 7 倍。而验收判据用的是"落后多少个 SE"——**于是不稳定的模型反而更容易通过**。EF-LSTM 因此被判"通过"，但那不是它的优点，是判据的缺陷。

`check_acceptance.py` 现在会点名崩溃的 seed 并写明"宽的离散度让这个检验更弱，而不是让模型更好"。**只报告不剔除**——剔除 seed 正是判据明令禁止的。

**待办**：判据可以补一个稳健性维度（例如同时看中位数，或对崩溃 seed 数设上限）。改动判据要按 `docs/decisions.md` 的模板记一条决策，且**不得在已有结果之后调整以迎合结果**。

---

## <a id="tfn-acc2"></a>TFN 的 Acc-2 低于 MMSA（2026-07-26）— 已解决

### 现象

10 seed 验收：MAE 与 Acc-7/Acc-5 打平，Acc-2(non0) 低 4.3 SE、Corr 低 4.6 SE。问题在预测符号而非幅度。

### 已排除的假设

| # | 假设 | 如何验证 | 结论 |
|---|---|---|---|
| 1 | 指标实现有别 | 用 MMSA 的 `metricsTop.py` 跑我们的预测 | **排除**。逐位一致 |
| 2 | 梯度裁剪 | MMSA 不裁剪，我们默认 1.0；10 seed 对照 | **真差异，已采纳不裁剪**。MAE 0.9585 → 0.9546，但 Acc-2 仍低 |
| 3 | 轮数上限 | MMSA 无上限，我们 40 轮 | **真差异，已改 200 轮**。10 seed 中仅 1 个触顶 |
| 4 | 文本编码器不同 | TFN 未设 `use_bert`，用预抽取 768 维特征 | **排除**。与我们一致 |
| 5 | 超参不同 | 逐项比对 `config_regression.json` | **排除**。我们用的就是它的值 |
| 6 | 预测有系统性偏移 | 统计预测符号分布 | **排除**。预测为正 45.0% vs 真实 42.2%，漏正/误正大致平衡 |
| 7 | **padding 掩码与池化方式** | 改用忠实移植后重跑 10 seed | **这就是原因**。Acc-2 0.7779 → 0.7855（+4.0 SE → +1.0 SE），MAE 0.9546 → 0.9511 |

### 结论

**TFN 复现成功**（忠实移植，10 seed）：MAE +0.5 SE、Acc-7 −1.4 SE（更优）、Acc-2 +1.0 SE（1-2 SE 区间，标记待查但不阻塞）。

Corr 仍低 4.3 SE，是唯一残留项。Corr 不是主指标，暂记为待查。

### <a id="tfn-retraction"></a>修正（2026-07-26 当日）：先前的归因是错的

**曾经的裁定**：把 Acc-2 差距归因于「MMSA 按测试集挑超参，其数字带乐观偏差」，并写入 `acceptance_status.json` 作为 `gap_explained`。

**为什么被推翻**：LMF 出现同样的差距，改用忠实移植后差距消失——回头再测 TFN，同样消失。**差距是我们自己的改动造成的，与参照值的选择偏差无关。**

关于 MMSA 按测试集调参这个观察本身**仍然是事实**（代码位置见下条），但它**不解释这个差距**，不得再被援引来解释类似现象。

**教训（已写入 `docs/decisions.md`）**：裁定是一个关于因果的断言。在记录任何归因之前，先排除我们自己相对参照实现的偏离——那是最可能的原因，也是最容易验证的。

---

## <a id="mmsa-test-selection"></a>MMSA 的超参按测试集挑选（2026-07-26）— 事实记录，非归因工具

`MMSA/src/MMSA/run.py`（commit a94e65d）第 267-270 行，调参分支按**测试集**评估候选配置：

```python
elif is_tune:
    # use valid set to tune hyper parameters
    # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
    results = trainer.do_test(model, dataloader['test'], mode="TEST")
```

用验证集的那行被注释掉了。而 `config_regression.json` 中 TFN/MOSI 的每项超参，都恰好是 `config_tune.json` 搜索空间里的一个取值。

**这意味着**其表格数字很可能是"按测试集挑出的超参在测试集上的成绩"，理论上带乐观偏差。

**但请注意**：TFN 与 LMF 在忠实移植下都复现成功了，说明这个偏差在这两个模型上**实际影响不大**。**不要用这条来解释复现差距**——先查自己的实现（见上一条的修正）。

**待办**：把这个偏差量化。用我们自己的实现做一次同规模超参搜索，比较"按验证集选"与"按测试集选"的测试成绩之差。这能把定性观察变成一个数字。

---

## <a id="lmf-optimizer"></a>LMF：MMSA 的 optimizer 按位置切参数，切错了（2026-07-26）

`MMSA/src/MMSA/trains/singleTask/LMF.py`：

```python
optimizer = optim.Adam([{"params": list(model.parameters())[:3], "lr": factor_lr},
                        {"params": list(model.parameters())[5:], "lr": learning_rate}],
                        weight_decay=weight_decay)
```

意图显然是给三个 fusion factor 单独的学习率。但 `parameters()` 的顺序由 `__init__` 中的模块注册顺序决定，前三个是 **audio_subnet 的 BatchNorm weight、bias 和 linear_1.weight**，不是 factor。

更严重的是 **索引 3、4（`linear_1.bias`、`linear_2.weight`）落在两个组之外，从初始化起就没有被更新过**。实测：该切片只覆盖 27 个参数中的 25 个。

MOSI 上 `factor_lr == learning_rate`，所以两组学习率其实相同，唯一实际后果就是那两个被冻结的张量。

**实测这是 bug 而非隐藏特性**：复刻该行为后 MAE 0.9806，全部参数参与训练则为 0.9628（10 seed）。

我们的实现让全部参数参与训练，并通过 `MSAModel.param_groups` 把 factor 真正路由到 `factor_lr`。`--model-arg freeze_mmsa_quirk=True` 可复刻原行为。

**同类风险**：TFN 的 trainer 用 `list(model.parameters())[2:]` 跳过两个 `requires_grad=False` 的 Parameter。那里恰好是对的，但同样脆弱——构造顺序一变就会静默丢参数。我们用 buffer 存常量，从根上避免。

---

## <a id="lmf-dropout"></a>LMF：配置声明了 post-fusion dropout，forward 里从未调用（2026-07-26）

MMSA 的 LMF 在 `__init__` 中按配置创建 `self.post_fusion_dropout = nn.Dropout(p=0.3)`，但 `forward` 中**没有任何一处调用它**——声明后被遗忘。

我们最初照配置施加了这个 dropout，代价是 **MAE 从 0.9628 恶化到 0.9906**（10 seed）。现默认 `post_fusion_dropout=0.0`，即对齐 MMSA 的**实际行为**而非其**书面配置**。

**教训**：移植时以 forward 的实际执行路径为准，不能只看配置文件。配置里的键可能是历史遗留。


---

## 原论文实现的获取方式

对照 MMSA 不够——它是二次实现。原始发布在这些仓库（scratchpad 里的克隆不随会话保留，需要时重新拉）：

```bash
for r in declare-lab/MISA thuiar/Self-MM yaohungt/Multimodal-Transformer \
         Justin1904/Low-rank-Multimodal-Fusion pliang279/MFN; do
  git clone --depth 1 https://github.com/$r.git
done
```

`A2Zadeh/TensorFusionNetwork` 已失效（克隆报 404）。

**已核对并记录的配置分歧**（原作者 vs MMSA，均以 MMSA 为准，因为验收对标的是 MMSA 的表）：

| 模型 | 原作者 | MMSA |
|---|---|---|
| Self-MM (MOSI) | bs 32、audio lr 1e-3、video lr 1e-4、LSTM 32/64 | bs 16、均 5e-3、LSTM 16/32 |
| MISA | diff_weight 0.3、sim_weight 1.0 | 0.1、0.3 |

MISA 的结构与原实现逐项一致（融合层顺序、transformer `nhead=2 / num_layers=1`、`sp_weight` 默认 0 即该损失确实未启用）。
