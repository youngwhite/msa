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
3. **梯度累积 `update_epochs`**——MulT 8 / Self-MM 4 / MISA 2（见 `#grad-accumulation`）
4. **padding / 池化行为**——用 `use_lengths=False, mask_pooling=False` 对齐
5. **optimizer 的参数分组**——MMSA 多处用 `list(model.parameters())[a:b]` 按位置切片，极易切错（见 `#lmf-optimizer`）
6. `use_bert` 取值、数据是 aligned 还是 unaligned
7. **声明了但不生效的东西**，三种形态都要查（见 `#misa-sp-weight`）：
   - 创建了但 forward 不调用（LMF 的 post_fusion_dropout）
   - 整个函数是死代码（MMSA data_loader 的 `__truncate`）
   - **forward 调用了，但输出不进损失**（MISA 的 sp_discriminator）——最难发现，必须追踪输出去向
8. **参照实现是否关掉了论文里的结构件**——MMSA 给 MulT 的位置编码加了默认关闭的开关（见 `#mult-position`）。查法：把原作者的 `__init__` 与 MMSA 的逐行对读，重点看新增的布尔开关及其默认值
9. **发现参照实现有 bug 时，回原作者确认是继承的还是二次实现引入的**——两者处理方式相反（见 `#lmf-optimizer-correction`）

第 1-6 项都在实际模型上命中过真差异；第 7-9 项来自 2026-07-27 回头核对原作者实现的那一轮。

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

### <a id="lmf-optimizer-correction"></a>修正（2026-07-27）：被冻结的是哪两个参数，原文写错了

**曾经的说法**：`[:3]` 取到的是 audio_subnet 的 BatchNorm weight/bias 和 linear_1.weight，不是 factor；落在缺口里的是 `linear_1.bias` 与 `linear_2.weight`。

**这两句都是错的。** 实测参数顺序（`named_parameters()` 先产出模块自身的 `nn.Parameter`，再递归子模块）：

```
[0] audio_factor   [1] vision_factor   [2] text_factor
[3] fusion_weights [4] fusion_bias
[5] audio_subnet.norm.weight ...
```

所以 `[:3]` **正确**选中了三个 factor，落进缺口的是 **`fusion_weights` 和 `fusion_bias`**。"27 个参数覆盖 25 个"这个计数不变，错的只是那两个的身份。

**更正后问题变严重了**，不是变轻：被冻结的是把 rank 个分量线性组合起来的 (1, rank) 向量，以及输出偏置——而 `fusion_bias` 初始化为 **0**。MOSI 的训练标签均值非零，把回归输出的偏置永久钉死在 0 是实质性损伤，远不止"两个随机线性权重没训练"。这也更好地解释了实测的 MAE 0.9806 vs 0.9628。

**并且这个 bug 是 MMSA 独有的。** 原作者 `train_mosi.py:106-107` 写的是：

```python
factors = list(model.parameters())[:3]
other   = list(model.parameters())[3:]      # 注意是 3，不是 5
```

**连续切片、无缺口、全覆盖**，且在原作者的参数顺序下 `[:3]` 恰好就是三个 factor——原作者的写法虽然脆弱但是对的。MMSA 把 `[3:]` 改成 `[5:]` 时引入了缺口。

**教训（新增一条，与既有的不同）**：发现参照实现有 bug 时，**必须回原作者确认这个 bug 是继承来的还是二次实现引入的**。两者的处理完全不同——继承来的 bug 可能是原始方法的一部分（见下条 dropout），二次实现引入的则应当修正。我们此前把两类混为一谈。

---

## <a id="lmf-dropout"></a>LMF：配置声明了 post-fusion dropout，forward 里从未调用（2026-07-26）

MMSA 的 LMF 在 `__init__` 中按配置创建 `self.post_fusion_dropout = nn.Dropout(p=0.3)`，但 `forward` 中**没有任何一处调用它**——声明后被遗忘。

我们最初照配置施加了这个 dropout，代价是 **MAE 从 0.9628 恶化到 0.9906**（10 seed）。现默认 `post_fusion_dropout=0.0`，即对齐 MMSA 的**实际行为**而非其**书面配置**。

**教训**：移植时以 forward 的实际执行路径为准，不能只看配置文件。配置里的键可能是历史遗留。

### <a id="lmf-dropout-correction"></a>修正（2026-07-27）：这个 bug 是从原作者继承的，不是 MMSA 的

原文把它记成"MMSA 的 LMF"。查原作者发布版（`Justin1904/Low-rank-Multimodal-Fusion`，`model.py:120`）：

```python
self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)   # 创建
```

而 `forward`（135-170 行）里**同样一次都没有调用它**。MMSA 只是原样继承。

数值结论不变（我们默认 0.0 仍然正确），但归因必须改：这不是二次实现的失误，而是**原始发布就有的**。与上一条 optimizer 切片形成对照——同一个模型上，一个 bug 是继承的、一个是 MMSA 引入的，处理方式应当不同。


---

## <a id="mult-position"></a>MulT：MMSA 关掉了位置编码，我们跟着关了（2026-07-27）

原作者 `yaohungt/Multimodal-Transformer`，`modules/transformer.py:30`——**无条件**创建：

```python
self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
```

MMSA 的同一文件加了开关，默认关闭：

```python
def __init__(self, ..., position_embedding=False):
    if position_embedding: self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
    else:                  self.embed_positions = None
```

全仓库 grep `position_embedding=True` 只有 `missingTask/TFR_NET/alignment.py` 传过，**MulT 一次都没有**。

**后果**：MMSA 的 MulT 跑在一个没有任何位置信息的 transformer 上。注意力对输入顺序完全置换不变，模型看到的是三袋无序的帧。我们的移植照抄了 MMSA，所以**当前 `outputs/mult_mosi` 里那个东西不是论文里的 MulT**。

这是"忠实复现二次实现 = 忠实复现它偏离论文的地方"的最清晰案例，也是本轮回头核对原作者代码的直接收获。

**处理方式**（按 LMF 冻结 quirk 的既有模式）：加开关，两个版本都跑。`忠实 MMSA` 版用于验收对标（同协议才可比），`忠实论文` 版用于 storyline 的技术演进叙事。**不要用后者去和 MMSA 的表比**——那会重演 `#protocol-faithful` 那条教训。

---

## <a id="misa-sp-weight"></a>MISA：`sp_weight` 声明为 1.0，但那项损失从未进入损失函数（2026-07-27）

`config_regression.json` 里 MISA/MOSI 有 `"sp_weight": 1.0`。模型也确实构造了 `sp_discriminator`，并且**在 forward 里被调用**（`models/singleTask/MISA.py:228-231`）：

```python
self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
...
self.shared_or_private_s = self.sp_discriminator((utt_shared_t + utt_shared_v + utt_shared_a)/3.0)
```

但 `sp_weight` 在 trainer 与模型里 grep **零命中**——这四个输出算完即丢弃。该层拿不到梯度，是一个永不训练的死层。

**这是"声明但不生效"模式的第三个实例，而且形态升级了**：

| 实例 | 形态 |
|---|---|
| LMF post_fusion_dropout | 创建了，forward 不调用 |
| MMSA data_loader `__truncate` | 整个函数是死代码 |
| **MISA sp_discriminator** | **forward 调用了，但输出不进损失** |

第三种最难发现：grep 层名会命中，看 forward 也在跑，只有追踪"输出去了哪里"才能识破。

**对我们的影响：无数值差异。** 我们不实现该层也不实现该损失，与 MMSA 的实际行为一致（无梯度的参数在 Adam 与 `clip_grad_value_` 下都被跳过）。记录在此仅为供后来者省一次排查。

---

## <a id="mult-weight-decay"></a>MulT：配置里的 weight_decay 从未到达优化器（2026-07-27）

### 现象

MulT 是缺口最大的一个：七项指标全部落后，MAE +10.3 SE（绝对差 0.061）。

### 已排除的假设

| # | 假设 | 如何验证 | 结论 |
|---|---|---|---|
| 1 | 超参不同 | 逐项比对 `dst_feature_dim_nheads [50,10]`、nlevels 4、conv1d kernel 5、六个 dropout | **排除**，全部一致 |
| 2 | 梯度累积语义 | 比对 MMSA 的 `update_epochs` 循环：求和不平均、尾部残余丢弃 | **排除**，行为一致（连其循环外那句死代码的效果都一致） |
| 3 | mem 网络层数 | MMSA `get_network('l_mem', layers=3)` 实际是 `max(nlevels=4, 3)` = **4 层不是 3 层** | **排除**，我们也是 `max(layers, 3)` |
| 4 | 缺少位置编码 | MMSA 同样没有（见 `#mult-position`），却报出 0.8799 | **排除**。那是"偏离论文"，不解释"偏离 MMSA" |
| 5 | 我们用 `nn.MultiheadAttention` 换掉了 MMSA 手写的注意力 | **数值等价测试**：两边权重对拷、同输入比对输出 | **排除**。self-attention 与 cross-modal 的最大差异均为 **4.8e-7**（float32 噪声） |
| 6 | 欠拟合 / 早停过早 | 看训练轨迹：train loss 1.73→0.47，valid MAE 1.41→0.92，45 轮后早停 | **排除**，模型在正常收敛 |
| 7 | **weight_decay** | 见下 | **真差异** |

### 结论

`MMSA/src/MMSA/trains/singleTask/MULT.py:21`：

```python
optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
```

**没有 `weight_decay` 参数。** 而 `config_regression.json` 的 mult/mosi 写着 `"weight_decay": 0.005`——该键被读进 `args`，但 trainer 从不使用它。**MMSA 的 MulT 实际以 weight_decay=0 训练**，我们按配置用了 0.005，等于施加了一份对方从不施加的 L2 正则。

### 这是"声明但不生效"的第四种形态

| 实例 | 形态 |
|---|---|
| LMF post_fusion_dropout | 创建了，forward 不调用 |
| MMSA data_loader `__truncate` | 整个函数是死代码 |
| MISA sp_discriminator | forward 调用了，但输出不进损失 |
| **MulT weight_decay** | **配置键读进了 args，trainer 不传给优化器** |

### 只有 MulT 被咬到

逐个核对全部 trainer 的优化器构造，与配置里的 weight_decay 交叉比对：

| 模型 | 配置声明 | trainer 传递？ | 是否受影响 |
|---|---|---|---|
| EF-LSTM / LF-DNN / MFN / Graph-MFN / LMF | 0.005–0.01 | ✅ | 否 |
| MISA | 0.0 | ❌ | 否（本来就是 0） |
| TFN | 未声明 | ❌ | 否（我们也用 0） |
| **MulT** | **0.005** | ❌ | **是** |

### 处理

`reproduce_all.sh` 的 `mult_mosi` 改为 `--weight-decay 0`。**这个修正独立于它能否缩小差距**——对齐参照实现的实际行为而非书面配置，是 LMF dropout 那次已确立的原则。改动先提交、再跑完整 10 seed，不做"先试探再决定"，以免把诊断和结果混为一谈。

**待验证**：重跑后缺口是否收敛。若仍未复现，第 7 项也要移进"已排除"，继续查。

---

## <a id="provenance-timing"></a>result.json 记录的 commit 可能是从未执行过的代码（2026-07-27）

### 现象

`graph_mfn_mosi` 的 10 个 seed 记录了**三个不同的 commit**（`4a8268e` / `e669d22` / `5c299a0`），其中 `misa_mosi/seed47` 还记着 `dirty=true`。

### 成因

`repro.py` 的 `git_revision()` 由 `collect_env()` 在**每个 seed 落盘时**调用。而真正被执行的代码，是**进程启动那一刻载入内存的那份**。时间线：

| 时刻 | 事件 |
|---|---|
| 14:21:52 | 提交 `4a8268e` |
| 14:26:23 | 训练进程启动，载入 `4a8268e` 的代码 |
| 14:41:41 | seed42 落盘 → 记 `4a8268e` ✅ |
| 14:45:23 | 提交 `e669d22`（进程仍在跑） |
| 14:52:01 | seed43 落盘 → 记 `e669d22` ❌ 但跑的仍是 `4a8268e` |
| 16:54:56 | 提交 `5c299a0` |
| 16:55:19 | seed49 落盘 → 记 `5c299a0` ❌ |

**10 个 seed 执行的都是 `4a8268e`，但只有 1 个记对了。**

### 为什么这条最严重

本项目的核心主张是"每个数字都能追溯到产生它的代码"。`result.json` 写着 `commit X, dirty=false` 本应意味着 checkout X 就能拿回那份代码——现在它不保证，而 `verify_runs.py` 信任这个字段，查不出来。

### 本次的具体判定

`graph_mfn_mosi` 这组**结果有效**：`git diff --stat 4a8268e 5c299a0 -- <graph_mfn 的全部依赖文件>` 为空（那三次提交只动了 `self_mm.py`、`check_acceptance.py` 和文档），执行的代码路径逐比特相同。这是运气好，不是流程对。

### 修复方向

进程启动时采集一次 git 状态并缓存，作为 `env.git` 的权威值；落盘时再采一次，**两者不一致则额外记录**，由 `verify_runs.py` 报警。这样 CLAUDE.md 第 3 条"后台跑实验时不要改代码"就从一条自觉约定变成了机器可检测的条件。

**附带隐患**：`num_workers>0` 且 spawn 启动时，worker 进程会**重新 import** 一次 msa。若运行期间磁盘上的代码变了，worker 与主进程可能执行不同版本。本项目默认 `num_workers=0`，暂不受影响，但换配置前要想到这一点。

---

## <a id="systematic-bias"></a>逐模型判据看不见的系统性偏置（2026-07-27）

把配置已核实的六组（ef_lstm / graph_mfn / lf_dnn / lmf / mfn / tfn）对 MMSA 的偏差按 SE 列出（正 = 比 MMSA 差）：

| 组 | MAE | Acc-2(non0) |
|---|---|---|
| ef_lstm | +1.9 | +1.6 |
| graph_mfn | +0.3 | **−0.6** |
| lf_dnn | +0.5 | +0.2 |
| lmf | +0.1 | +1.6 |
| mfn | +1.3 | +0.3 |
| tfn | +0.5 | +1.0 |

**11/12 项偏差。** 若差距纯属复现噪声、方向应各半，出现 11/12 或更极端的概率约 **3×10⁻³**（符号检验）。

**问题在判据本身**：每个模型各差 0.5–1.5 SE，逐个都落在"通过"或"通过但标记"区间，聚合起来却是一个 p≈0.003 的系统性信号。**逐模型的验收判据在结构上无法看见它。**

候选成因（均未证实，不得当结论引用）：

1. MMSA 按测试集挑超参（见 `#mmsa-test-selection`）——会在所有模型上产生一致的乐观偏差，与观察到的形态吻合
2. 选模型的 Loss 口径：MMSA 按 batch 平均（末批不满被过度加权），我们按样本加权
3. MMSA 表未声明 seed 集合（其默认为 1111-1115），我们用 42-51

**可做的检验**：`tfn_mosi_mmsaseeds`（seeds 1111-1115）已有数据，可先算 seed 集合的贡献。

**待办**：给验收补一个跨模型的聚合统计量。单模型判据看不见的东西，需要一个总体检验来兜底。改判据前按 `docs/decisions.md` 的模板记决策，**且不得在已有结果之后调整以迎合结果**。

---

## <a id="reading-live-outputs"></a>在后台批次仍在写入时读 `outputs/`，会据此下错结论（2026-07-27）

### 我踩的过程

会话开始时读 `outputs/misa_mosi` 等三组，看到 `epochs=30`、无 `accumulate_steps`、commit 横跨四个版本，据此判定"**三组整组作废，必须重跑**"，并围绕这个判定安排了后续计划。

**这个判定是错的。** 后台批次当时正**依次重跑**这三组。数小时后再看，同样的目录已是：commit 全部 `5c299a0`、`dirty=false`、`accumulate_steps` 分别 2/4/8、`epochs=200`——正确配置，10 seed 齐全。我读到的是被覆盖前的旧内容。

### 为什么没能及时发现

`pgrep -af "train.py|reproduce_all"` 只显示**当时正在执行的那一条命令**。上一轮会话启动的是一个逐组调用的循环，外层循环不出现在 `pgrep` 结果里，所以看到的是"只在跑 graph_mfn"，实际排队等着的还有四组。

### 曾经的误判（已推翻）

一度记为"`reproduce_all.sh` 按前缀匹配组名，连带跑了没点名的组"。**这是错的**：其组选择是精确匹配

```bash
if [ -n "$WANTED" ] && [ "$WANTED" != "$group" ]; then continue; fi
```

多出来的那些组来自外层循环，不是脚本的匹配行为。链条实际顺序 graph_mfn → misa → self_mm → mult → graph_mfn 冻结消融，与 `docs/roadmap.md` 当时的记载完全一致——**我如果先信任交接文档，就不会误判**。

### 规避

1. 依据 `outputs/` 下的内容下任何结论前，先确认**没有进程正在写它**——查目录 mtime 是否还在推进，别只看 `pgrep`
2. `pgrep` 看不到外层循环。要确认"还有什么排着队"，读交接文档里记的批次计划
3. 交接文档写了的东西，与现场观察冲突时，**先怀疑观察的时机**，而不是先怀疑文档

### 附带事实

`graph_mfn_mosi_ablation_frozen` 确实只有 7/10 个 seed 且无 `summary.json`——该组在 22:28 后被中断，原因未知。它是消融不是验收目标，需要时重跑即可。

---

## 原论文实现的获取方式

对照 MMSA 不够——它是二次实现。原始发布在这些仓库（scratchpad 里的克隆不随会话保留，需要时重新拉）：

```bash
for r in declare-lab/MISA thuiar/Self-MM yaohungt/Multimodal-Transformer \
         Justin1904/Low-rank-Multimodal-Fusion pliang279/MFN; do
  git clone --depth 1 https://github.com/$r.git
done
```

**取不到原始出处的**（2026-07-27 复核）：

| 模型 | 情况 |
|---|---|
| TFN | `A2Zadeh/TensorFusionNetwork` 克隆报 404。`Justin1904/TensorFusionNetworks` 可克隆，但作者是 LMF 的一作、**属二次实现**，证据等级与 MMSA 相同，不能当原始出处用 |
| Graph-MFN | 出处在 `A2Zadeh/CMU-MultimodalSDK`，克隆要求认证，未取到 |
| EF-LSTM / LF-DNN | **本就没有原始出处**——它们是 MMSA 自定义的基线，不对应任何论文。这两个模型的"原论文核对"不适用，不是遗漏 |

**已核对并记录的分歧**（原作者 vs MMSA，验收一律以 MMSA 为准，因为对标的是 MMSA 的表）：

| 模型 | 原作者 | MMSA | 性质 |
|---|---|---|---|
| Self-MM (MOSI) | bs 32、audio lr 1e-3、video lr 1e-4、LSTM 32/64 | bs 16、均 5e-3、LSTM 16/32 | 超参分歧 |
| MISA | diff_weight 0.3、sim_weight 1.0 | 0.1、0.3 | 超参分歧 |
| **MulT** | 位置编码**始终启用** | 默认关闭且从未开启 | **结构性偏离**，见 `#mult-position` |
| **LMF** | `[:3]`/`[3:]` 连续切片，全覆盖 | `[:3]`/`[5:]`，两个参数从不训练 | **MMSA 引入的 bug**，见 `#lmf-optimizer-correction` |
| LMF | post_fusion_dropout 声明后不调用 | 同样不调用 | 从原作者**继承**的 bug |

逐项核对已完成的：MISA（结构与原实现一致：融合层顺序、transformer `nhead=2 / num_layers=1`）、Self-MM、LMF、MulT。

**尚未核对**：MFN（`pliang279/MFN` 已克隆，未比对）。
