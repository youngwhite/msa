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
10. **参照实现可运行时，先跑它，再读它。** 读代码能证明两份实现一致，**不能证明一个数字可达**。TFN 上九轮排查耗时数天没有结论，让 MMSA 自己跑一遍在依赖装好后十几分钟就解决了（见 `#tfn-corr-resolved`）

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

### 处理方式与预先登记的比较协议（2026-07-28，在跑之前定死）

已实现 `position_embedding` 开关，**默认 False**（验收组保持忠实 MMSA；实测默认路径预测哈希 `6b7a13160dd4225c` 与改动前逐比特一致）。消融组 `mult_mosi_posenc` 除该开关外与 `mult_mosi` **完全同参**。

**为什么只改一个变量。** 查原文后发现，MMSA 与原作者的超参几乎处处不同：

| | 原文 (Table 5, MOSI) | MMSA (mosi) |
|---|---|---|
| batch size | 128 | 16 |
| lr | 1e-3 | 2e-3 |
| hidden | 40 | 50 |
| conv kernel L/V/A | (1或3)/3/3 | 5/5/5 |
| 文本 embedding dropout | 0.2 | 0.5 |
| 跨模态 attention dropout | 0.2 | 0.3 |
| 输出 dropout | 0.1 | 0.5 |
| grad clip | 0.8 | 0.6 |
| 位置编码 | **始终启用** | **从未启用** |
| 一致的只有 | 块数 4、头数 10 | |

于是"忠实原文"有三种做法，可达性完全不同：

| 方案 | 能回答什么 | 可行性 |
|---|---|---|
| A. MMSA 超参 + 位置编码 | **位置编码值多少分**（单变量） | ✅ 采用 |
| B. 原文超参 + 位置编码 | 无法归因（同时改十余个变量） | ✗ |
| C. 真正复现原文 | 需要原文的 GloVe/Facet/COVAREP 特征 | ✗ **不可达** |

**预先登记的判读规则**：`mult_mosi_posenc` 不进 `check_acceptance`；**唯一合法的比较对象是我们自己的 `mult_mosi`**；不与 MMSA 的表比（不同模型），也不与原文的表比（三个模态的特征全不同）。结果变好变坏都如实记录。

原文数字与超参已录入 `docs/paper_reference_mosi.json`，**仅作背景，不是达标目标**。

### 结果（2026-07-28）：位置编码在这批特征上不值分

10 seed，与 `mult_mosi` 除该开关外完全同参：

| 指标 | 无位置编码 | 有位置编码 | 配对 SE |
|---|---|---|---|
| MAE ↓ | 0.8974 ± 0.0140 | 0.9034 ± 0.0122 | +1.0（略差） |
| Acc-2(non0) | 0.8082 ± 0.0080 | 0.8078 ± 0.0091 | −0.1 |
| Acc-2(has0) | 0.7936 ± 0.0077 | 0.7934 ± 0.0073 | −0.0 |
| Acc-7 | 0.3665 ± 0.0093 | 0.3684 ± 0.0097 | +0.4（略好） |
| Acc-5 | 0.4315 ± 0.0141 | 0.4321 ± 0.0106 | +0.1（略好） |
| Corr | 0.6917 ± 0.0123 | 0.6958 ± 0.0066 | +0.9（略好） |

**六项全部落在 ±1 SE 内，方向还不一致**（MAE 略差、Corr 略好）。结论：在 MMSA 的这批特征与超参下，MulT 的位置编码**没有可测量的价值**。

三点解读：

1. **MMSA 关掉它虽然偏离论文，但代价接近零。** 这不为它开脱——一个开关的默认值把架构改掉了却没人记录，仍然是问题；但它确实不解释我们相对 MMSA 的缺口（本来也不该，见上）。
2. **这是又一个"架构件在这批特征上不兑现"的例子**，与 MFN 的跨时间记忆（喂真实序列反而更差）、MISA 的音视频被截掉一半属同一类。第 0 节的模态消融已经标定了原因：音频与视觉在这批 2018 年特征上几乎不携带信息，那么让模型更精细地建模它们的时序，自然也换不来收益。
3. **零结果照样入库**。预先登记时写明"变好变坏都如实记录"，这就是兑现。

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

### 重跑结果：确认是主因，但不是全部

10 seed，`--weight-decay 0`：

| 指标 | 修正前 | 修正后 | MMSA | SE 倍数 |
|---|---|---|---|---|
| MAE ↓ | 0.9406 | **0.9037** | 0.8799 | +10.3 → **+4.4** |
| Acc-2(non0) | 0.7979 | **0.8079** | 0.8098 | +5.4 → **+0.6 ✅** |
| Acc-2(has0) | 0.7841 | 0.7936 | 0.7971 | +7.4 → +1.8 |
| Acc-7 | 0.3485 | 0.3631 | 0.3691 | +6.6 → +1.1 |
| Acc-5 | 0.4045 | 0.4185 | 0.4268 | +7.8 → +1.2 |
| Corr | 0.6750 | 0.6904 | 0.7022 | +9.2 → **+5.2** |

**判定仍是未复现**（MAE 是主指标，+4.4 SE 超过门槛），但四个分类类指标已基本对齐。残差集中在 MAE 与 Corr，见下一条。

---

## <a id="mult-out-proj-init"></a>MulT：`nn.MultiheadAttention` 的 out_proj 初始化比参照实现窄 1.73 倍（2026-07-27）

### 怎么找到的

修掉 weight_decay 后 MulT 仍差 MAE +4.4 SE / Corr +5.2 SE。先验证了"预测被压向均值"的假设——**不成立**：

| 组 | pred std / true std | 回归斜率 |
|---|---|---|
| mult | 0.883 | 0.779 |
| self_mm | 0.899 | 0.881 |
| misa | 0.863 | 0.897 |
| tfn | 0.794 | 0.829 |

MulT 的预测离散度在各模型里偏高而非偏低，斜率低是相关性低的**结果**（slope = corr × σ_y/σ_p），不是输出标定问题。模型整体精度略低。

### 结论

我们用 `nn.MultiheadAttention` 替换了 MMSA 手写的注意力，并有 `scripts/check_mult_encoder.py` 证明前向数值等价（差异 4.8e-7）。**但那个测试是把权重对拷之后比对的——它验证前向计算，验证不了初始化。**

fairseq/MMSA 的 `reset_parameters`（`multihead_attention.py:39-44`）：

```python
nn.init.xavier_uniform_(self.in_proj_weight)
nn.init.xavier_uniform_(self.out_proj.weight)     # <- 这一行
```

PyTorch 的 `nn.MultiheadAttention._reset_parameters` 只 xavier 了 `in_proj_weight`，**`out_proj.weight` 保留 `nn.Linear` 默认的 `kaiming_uniform_(a=√5)`**。实测（embed_dim=50）：

| | 界 | std |
|---|---|---|
| MMSA (xavier_uniform) | 0.2449 | 0.1429 |
| 我们 (kaiming 默认) | 0.1414 | 0.0818 |

**窄 1.732 倍**，即 `sqrt((50+50)/(6·50/…))` 的比值。`in_proj_weight` 两边一致，只有 out_proj 不同。MulT 有 6 个跨模态 encoder + 3 个 self-attention 栈、每个 4 层，out_proj 的尺度直接决定残差分支初始的贡献大小。

### 影响范围

只有 MulT。`src/msa/models/transformers.py` 仅被 `mult.py` 使用；MISA 两边都用 `nn.TransformerEncoderLayer`，默认初始化相同。

### 教训

**等价测试要覆盖它声称覆盖的全部内容。** 我们的 docstring 写的是"数学相同，参数初始化细节有别"——初始化差异是被**写下来了的**，却没人评估它的量级，于是它以"已知无害"的身份存活了下来。写下一个差异不等于排除了它。

### 重跑结果：方向对，但量级远小于预期

10 seed，两处修正累计：

| 指标 | 原始 | 修 wd 后 | 再修 init | MMSA | 最终 SE | 判定 |
|---|---|---|---|---|---|---|
| MAE ↓ | 0.9406 | 0.9037 | **0.8974** | 0.8799 | +4.0 | 未达标 |
| Acc-2(non0) | 0.7979 | 0.8079 | 0.8082 | 0.8098 | +0.6 | ✅ |
| Acc-2(has0) | 0.7841 | 0.7936 | 0.7936 | 0.7971 | +1.4 | 标记 |
| Acc-7 | 0.3485 | 0.3631 | 0.3665 | 0.3691 | +0.9 | ✅ |
| Acc-5 | 0.4045 | 0.4185 | **0.4315** | 0.4268 | **−1.0** | ✅ 更优 |
| Corr | 0.6750 | 0.6904 | 0.6917 | 0.7022 | +2.7 | 未达标 |

**如实记录：`weight_decay` 是主因，初始化只贡献了很小的一步**（MAE 0.9037→0.8974，约占总收敛量的 15%）。我先前把它当作剩余缺口的主要嫌疑，那个判断偏乐观。它确实是真实差异、修正正确，但**不要因为一个差异真实存在就预期它能解释缺口**——这两件事需要分别验证。唯一明显受益的是 Acc-5，从落后 1.2 SE 变为优于 MMSA 1.0 SE。

**MulT 仍未复现**：MAE +4.0 SE、Corr +2.7 SE。形态在两轮修正后保持稳定——**四个分类类指标已对齐，只有幅度/相关性类落后**，说明还有一处影响预测精度而非决策边界的差异未找到。

---

## <a id="self-dirty"></a>入库结果 + 落盘时采集 = 每次重跑都把自己标记为脏（2026-07-27）

### 现象

MulT 重跑的 10 个 seed 里 9 个记了 `dirty=true`，而运行期间没有任何人碰过文件。

### 成因

`edf34ff` 把结果入库后，`outputs/<组>/seed*/result.json` 成了受版本控制的文件。于是：

1. seed42 落盘，此时树干净 → 记 `dirty=false`
2. **它写下的 `result.json` 本身让工作树变脏了**
3. seed43 起全部记 `dirty=true`

实测运行期间 `git status --porcelain` 的全部条目都在 `outputs/mult_mosi/` 之下，没有一个源码文件。

### 为什么必须修

按约定"进入文档的数字必须来自干净工作树"，**这批完全合法的结果会被自己的闸门判为不可用**。而且这不是偶发——自结果入库那天起，**任何一次重跑都会如此**，`verify_runs.py` 会稳定地报出大批假警告，直到没人再认真看它。

### 修法

`dirty` 只反映**源码**状态，排除 `outputs/`：

```python
changed = [line[3:] for line in status.splitlines() if line.strip()]
source_changed = [p for p in changed if not p.lstrip('"').startswith("outputs/")]
```

产物变动不改变"这份代码是什么"，与既有的"不计入未跟踪文件"是同一条理由。

### 这批结果已单独验证

重跑 `mult_mosi/seed43`（记着 `dirty=true` 的那批之一），预测哈希 `f17c0972639685b2` 与已存**逐比特一致**。证明它们确实来自 `581a600` 的代码，假 dirty 标记是产物写入造成的假象。

### 顺带：一个自我匹配的等待循环（后来发现共发生三次）

用 `until ! pgrep -f "run-group tmp_mult_verify"; do sleep 30; done` 等待训练结束——**这个循环自己的命令行里含有那个字符串**，`pgrep` 一直匹配到它自己，永不退出。训练早已结束，等待器空转了一小时。

**修正（2026-07-29）**：当时记成偶发，其实**在那之前已经中过两次**，只是没被发现——清理进程时才看到两个僵尸等待器分别空转了 **2 天 9 小时**和 **2 天 0 小时**：

```
until ! pgrep -f "train.py --model graph_mfn" ...   # 2d09h
until ! pgrep -f "train.py --model mult" ...        # 2d00h
```

两次的训练都早已正常结束，我是靠手工轮询拿到结果并继续的，所以**没有影响任何结论**，但等待器从此再没退出过。这类"静默空转"不报错、不占 GPU，只有主动列进程才看得见。

**规避**（两条都用）：
1. **不要用 `pgrep` 等自己**——匹配产物文件更可靠：`until [ -f outputs/<组>/summary.json ]; do sleep 60; done`
2. 非要匹配进程时排除自身：`pgrep -f "[t]rain.py"` 的括号技巧，或匹配可执行文件路径而非完整命令行

**教训**：一个不会失败、只会静默不退出的等待器，比一个会报错的更难发现。

---

## <a id="misa-audited"></a>MISA：逐项核对完毕，未发现实现差异（2026-07-27）

### 缺口的量级

Acc-2(non0) 落后 2.4 SE，绝对差 0.0063——该指标在 656 个非零样本上计算，**等于 4 个测试样本**。同时 MAE 优 2.3 SE、Acc-7 优 3.7 SE、Acc-5 优 3.7 SE。

按 |真值| 分区看符号错误率，**各区间均匀略高于 Self-MM，没有可定位的失效模式**：

| \|真值\| | n | MISA 符号错误率 | Self-MM |
|---|---|---|---|
| 0–0.5 | 96 | 0.625 | 0.603 |
| 0.5–1 | 101 | 0.273 | 0.234 |
| 1–2 | 272 | 0.135 | 0.126 |
| 2–4 | 217 | 0.082 | 0.069 |

（零点附近 60%+ 的错误率是三个模型共有的——|真值|<0.5 时符号本就接近随机。）

### 已逐项核对且一致的项目

对照 MMSA 与原作者 `declare-lab/MISA`，**下列各项均无差异**，记录于此以免重查：

| 项目 | 细节 |
|---|---|
| 任务损失 | `nn.MSELoss`（不是别处的 L1） |
| diff 损失 | **6 项**（3 个 private-shared + 3 个 private 两两），**不除以 6** |
| CMD 损失 | 3 项**除以 3**，5 阶矩 |
| 重构损失 | 3 项**除以 3**，目标是 projected 表征 |
| 总损失 | `cls + diff_w·diff + sim_w·sim + recon_w·recon` |
| `DiffLoss` 细节 | 零均值 → L2 归一化（**范数上有 `.detach()`**、`+1e-6`）→ `(a.t()@b).pow(2).mean()`（**mean 不是 sum**） |
| 结构 | 投影 `Linear→ReLU→LayerNorm`；private/shared `Linear→Sigmoid`；fusion `Linear→Dropout→ReLU→Linear` |
| transformer 输入 | 堆叠顺序 `(pri_t, pri_v, pri_a, sha_t, sha_v, sha_a)` |
| 优化器 | 单组 Adam、lr 1e-4、无 weight decay（配置也是 0） |
| 训练 | 按值裁剪 0.8、梯度累积 2、无轮数上限 |
| `sp_weight` | 死配置，不实现是正确的（见 `#misa-sp-weight`） |

### 顺带发现：MISA 用文本长度截断音视频

`MISA.py:199` 的 `lengths = mask_len`（取自 BERT attention mask）被**同时**传给 `vrnn` 与 `arnn` 的 `pack_padded_sequence`。实测 MOSI unaligned 训练集：

| 模态 | 平均真实长度 |
|---|---|
| text | 14.8 |
| audio | 38.8 |
| vision | 42.6 |

**音频只被读进 43.6%，视觉只被读进 40.8%。** 我们忠实复现了这一行为（复现就该复现它实际跑的东西），但它与 MFN 那条"音视频被压成常量"属同一家族：**参照实现里多个模型的多模态部分都被削弱了**，这是 storyline 需要讲的事实。

### 判定

**维持"未复现"。** 查了一圈没找到差异**不是**归因证据——按 `acceptance_status.json` 的规矩，裁定需要证据支撑。此处只记录"已核对哪些项目、缺口的绝对量级是 4 个样本"，供下一个人不必重查，不写入 `gap_explained`。

**尚未核对**：BiLSTM 双向末状态的展平顺序（MMSA 用 `cat(dim=2).permute(1,0,2).view(batch,-1)`）。线性层能吸收输入置换，量级上不像成因，但没测过。

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

### 已修复（2026-07-27）

`repro.py` 现在在**进程启动时**采集一次并缓存（由 `msa/__init__.py` 在 import 时钉住 `snapshot_git_revision()`），作为 `env.git` 的权威值；落盘时再采一次，两者不一致则额外记入 `env.git.at_save`，由 `verify_runs.py` 报出"运行期间仓库动过"。CLAUDE.md 第 3 条"后台跑实验时不要改代码"因此从自觉约定变成了机器可检测的条件。

同一次改动还修了 `dirty` 把产物算进去的问题，见 `#self-dirty`。

**验收**：改的是 `repro.py`（`set_seed` 也在其中），故按约定 6 验证数值未受影响——LF-LSTM 默认路径预测哈希仍为 `172967c7dced83b2`，`check_invariants` 全绿。

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

**已排除：seed 集合（2026-07-27）。** `tfn_mosi_mmsaseeds` 用 MMSA 默认的 1111-1115 跑 TFN，与我们注册的 42-51 对比：

| seed 集合 | MAE | Acc-2(non0) | Acc-7 |
|---|---|---|---|
| 42-51（我们） | 0.9511 | 0.7855 | 0.3528 |
| 1111-1115（MMSA 默认） | 0.9542 | 0.7848 | 0.3557 |
| MMSA 报告 | 0.9473 | 0.7908 | 0.3446 |

换用 MMSA 自己的 seed，我们的 MAE 与 Acc-2 **反而更差**。方向与假设相反，**排除**。

**剩余候选**：MMSA 按测试集挑超参（`#mmsa-test-selection`）、选模型 Loss 的 batch 平均 vs 样本加权。前者需要自做一次同规模超参搜索才能量化，是当前最大的未验证项。

**注意**：这个偏置的一部分正在被逐个消解——MulT 的 `weight_decay`（`#mult-weight-decay`）与 out_proj 初始化（`#mult-out-proj-init`）都是我们这边的真实错误。**在逐模型的排查穷尽之前，不应把残差归因于 MMSA。** 这是 `#tfn-retraction` 那条教训的直接应用。

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

---

## <a id="selection-bias-measured"></a>按测试集挑超参能拿到多少便宜：0.026 MAE / 2.3 个百分点（2026-07-28）

### 做法

系统性偏置（`#systematic-bias`）唯一未验证的候选，是 MMSA 的调参分支按**测试集**评估候选配置（`#mmsa-test-selection`）。用我们自己的实现测这个机制能制造多大虚高，所以得出的数字是我们的，不是对 MMSA 的指控。

`scripts/selection_bias.py`，协议在跑之前登记（commit `cc6e9b2`）：LMF，40 组配置抽自 MMSA 自己的 `config_tune.json`（11,520 种组合），每组 3 个 seed，全部同样训练，**不丢弃任何一组**。在同一批运行上选两次。

### 结果

| 口径 | 指标 | 按验证集选→测试 | 按测试集选→测试 | **乐观偏差** |
|---|---|---|---|---|
| K=3（主，保守） | MAE ↓ | 0.9290 | 0.9032 | **+0.0258** |
| | Acc-2(non0) | 0.7790 | 0.8023 | **+2.34 pt** |
| K=1（接近 MMSA 协议） | MAE ↓ | 0.9809 | 0.9071 | **+0.0738** |
| | Acc-2(non0) | 0.7622 | 0.8110 | **+4.88 pt** |

对照我们的残差：MulT MAE 0.0175、Self-MM MAE 0.0062、MISA Acc-2 0.63 pt。**保守估计的选择偏差比每一个都大。**

### 能证明的与不能证明的

**能证明**：在 MMSA 自己的搜索空间上，按测试集选择可制造 0.026 MAE / 2.3 pt 的虚高，**量级超过我们全部残差**。因此 **MMSA 的表在判据要求的精度（0.5–1.5 SE ≈ 0.004–0.017 MAE）上不能被当作无偏参照**。

**不能证明**：MMSA 的任何一个具体数字被如此虚高。**并且有一条反证**——我们搜出的测试最优配置是 MAE **0.9032**，而 MMSA 为 LMF 选定的配置只到 0.9504。若它真在这个空间上按测试集狠选，不该停在那里。至少 LMF 上它没有充分利用这个机制。

按 `#tfn-retraction` 的教训，**不得写"我们的残差由 MMSA 的测试集选择解释"**。正确表述只有上面那句关于量级的。

### 真正的方法论结论

**判据要求的一致性精度，细于参照物自身的不确定度。** 这不替我们开脱，它指出判据设计的结构性问题：拿一个带未知选择偏差的单值当靶心，却要求 1 SE 以内命中。

附带数据：valid 与 test 的 MAE 系统性落差 **+0.058**（验证集仅 229 条）；按验证集选出的配置在测试集上排 **3/40**——验证集选择本身不差，差的是拿测试集选。

**复现命令**：`python scripts/selection_bias.py --configs 40 --seeds 42 43 44`（约 80 分钟；配置抽样用固定种子，配置集合本身可复现）。它不是 `train.py` 组，故不在 `reproduce_all.sh` 的组列表里。

---

## <a id="corr-gaps"></a>TFN 与 EF-LSTM 的 Corr 缺口（2026-07-29）

判据在 2026-07-28 改为 MAE+Corr 后新浮现的两项。此前 Corr 不是主指标，TFN 那 4.3 SE 一直挂着"残留待查"没人追。

### TFN：缺口 0.0131，小于参照物自身的不确定度

10 个 seed **无一**达到 MMSA 的 0.6733（范围 0.6355–0.6710，最好 0.6710，σ=0.0095）。分布很紧，是稳定落后而非噪声。

两个量级参照：

| | 数值 |
|---|---|
| TFN Corr 缺口 | **0.0131** |
| Corr 的测试集选择偏差（实测，K=3 保守） | +0.0113 |
| 同上（K=1，更接近 MMSA 协议） | +0.0257 |
| 在我们的 seed 分布下，单次抽样 ≥0.6733 的概率 | **8.5%** |

第二项从 `#selection-bias-measured` 的**同一批运行**免费算出（那次保存了全部指标）。第三项的前提是 MMSA 报的是单值、无方差。

**能说的**：缺口落在"测试集选择 + 单值抽样"能产生的量级之内。**不能说的**：缺口由此解释——按 `#tfn-retraction` 的教训，没有直接证据就不做归因。

### 已排除的假设（累计九项，2026-07-29/30）

| # | 假设 | 如何验证 | 结论 |
|---|---|---|---|
| 1-7 | 见 `#tfn-acc2`（指标实现、梯度裁剪、轮数上限、文本编码器、超参、预测偏移、padding/池化） | | 全部排除，第 7 项是 Acc-2 缺口的真因 |
| 8 | **按 valid MAE 选 epoch 牺牲了 Corr** | 逐 epoch 比对：按 MAE 选比按 Corr 选平均损失 **0.0081** valid Corr，量级接近缺口 0.0131 | **机制成立，但不是差异**——MMSA 的 `KeyEval` 同为 `Loss`(=MAE)，两边付同样代价 |
| 9 | **sigmoid 有界输出饱和压缩排序** | 预测超过 \|2.9\| 的占比仅 **0.03%**、超过 \|2.5\| 仅 0.54% | **排除**，边界没压缩任何东西 |

**实现已三方核对一致**：我们 / MMSA / `Justin1904/TensorFusionNetworks`（二次实现）的输出路径逐行相同——`post_fusion_dropout → layer_1+relu → layer_2+relu → layer_3 → sigmoid → ×6−3`，外积构造与拼 1 的顺序也一致，`output_range=6 / shift=−3` 都存为不可训练常量。

### 结论：未找到实现原因

**排查已穷尽现有证据，缺口成因不明。** 能说的三件事：

1. 缺口 **0.0131** 是噪声地板（0.0110）的 **1.2 倍**——超出，但是全表最窄的真实失败
2. 它落在实测的 Corr 测试集选择偏差范围内（**0.0113–0.0257**，见 `#selection-bias-measured`）
3. MMSA 报的是单值。我们 10 个 seed 全部低于它（0.6355–0.6710，σ=0.0095），但从这个分布抽一次达到 0.6733 的概率是 **8.5%**

**判定维持"未复现"，不写入 `gap_explained`**——上面三条都是量级参照，不是因果证据。这正是 `#tfn-retraction` 立下的规矩。

**唯一能了结此事的证据取不到**：TFN 原作者仓库 `A2Zadeh/TensorFusionNetwork` 克隆报 404。若日后能获取，应优先核对它与 MMSA 在 post-fusion 部分的差异。

### EF-LSTM：一半是崩溃，一半是残余

**崩溃率实测（30 个 seed）**：验收组 42-51 的 2/10 是低估。补跑 20 个 seed（52-71，`ef_lstm_mosi_collapse_rate`）后：

| | |
|---|---|
| 崩溃 seed | 46, 48, 52, 58, 59, 60, 71 |
| **崩溃率** | **7/30 = 23%**，Wilson 95% CI **[12%, 41%]** |
| 正常 23 个 | MAE 0.9702 / Corr 0.6475 / Acc-2 0.7821 |
| 崩溃 7 个 | MAE 1.4671 / Corr 0.2354 / **Acc-2 0.4223（= 多数类比例）** |

**崩溃形态**：`train_loss` 从第 1 轮起就卡在 1.32 不动（正常 seed 是 1.325→0.562），早停在第 10–13 轮触发。**不是逐渐退化，是从初始化起就没开始学。**

**成因不是移植错误。** 与 MMSA 逐行核对无差异：BatchNorm 加在**时间轴**（`BatchNorm1d(seq_len)` 作用于 (batch, seq, feature)）、`h[-1]` 读取、dropout 位置、4 层 LSTM 层间 dropout 0.5、trainer 同样不裁剪梯度、lr 1e-3 / wd 5e-3 / bs 32 全部一致。**这是这个架构+配置本身的性质**：4 层 LSTM、层间 dropout 0.5、1284 条训练样本。

**Corr 缺口的分解**：崩溃贡献 0.0204；即便（仅为诊断）剔除全部崩溃 seed，正常 seed 的 Corr 0.6475 仍低于 MMSA 的 0.6690 约 **0.0215**——与 TFN 的残余同量级，适用同样的选择偏差参照。

### 这条记录本身是一个超越参照框架的产出

**MMSA 的表是单值无方差，结构上无法显示"这个基线有 23% 的概率训不出来"。** 我们能给出这个数，正是因为协议要求多 seed。它也让 `#ef-lstm-collapse` 那条更尖锐：EF-LSTM 的验收判定实质上取决于抽到几个崩溃 seed——42-51 抽到 2 个，若抽到 4 个，σ 会更宽、SE 更大、判据反而更松。

**待办**：判据的稳健性维度（中位数，或崩溃数上限）仍未实现。改判据前按 `decisions.md` 模板记决策，**不得在已有结果之后调整以迎合结果**。

---

## <a id="self-matching-patterns"></a>进程模式自我匹配：同一个错我连犯四次（2026-07-29）

不是模型问题，是操作问题，但它浪费的时间比多数模型 bug 都多，且**每一次都以"看起来正常"的方式失败**。

| # | 写法 | 后果 |
|---|---|---|
| 1 | `until ! pgrep -f "train.py --model graph_mfn"; do sleep 20; done` | 循环自身命令行含该串 → 永不退出，**空转 2 天 9 小时** |
| 2 | `until ! pgrep -f "train.py --model mult"; ...` | 同上，**空转 2 天** |
| 3 | `until ! pgrep -f "run-group tmp_mult_verify"; ...` | 同上，空转 1 小时 |
| 4 | `pkill -f "train.py --model ef_lstm"` | **匹配到我自己的 shell**，把自己杀了（exit 144）；同一条命令里后续的 `rm` 与 `nohup` 全部未执行，留下孤儿进程继续写 `outputs/` |

第 4 种最危险：前三种只是空转，第 4 种**造成了实际破坏**——清理未做、重启未做，而孤儿进程还在往结果目录里写。

### 第五种形态：把"任何进程都会产生的东西"当完成信号

等待并行训练结束时用了 `until [ -f outputs/<组>/summary.json ]`。但 `train_parallel.py` 的每个**子进程**都会写一份只含自己那个 seed 的 `summary.json`——**这个行为是我自己写在该文件 docstring 里的**。于是第一个 seed 结束时信号就触发了，我据此算出并汇报了一个建立在 4/20 数据上的崩溃率（21%），随后才发现跑了 12/20。真实值是 23%，方向没错，但那次汇报当时没有依据。

**与 MulT 那次同型**：docstring 里写着"参数初始化细节有别"，却从没估过它的量级。**把一件事记下来，不等于在用它。**

### 规避（三条都用）

1. **等待信号必须是只有父进程会产生的东西**——`train_parallel.py` 的 `rebuilt ...` 那行，或进程本身退出
2. **不要用 `pgrep`/`pkill -f` 等自己或杀自己**。要按模式杀时，先 `pgrep` 列出 PID、逐个核对 `ps -o cmd=` 再 `kill <pid>`
3. **超过约 90 秒的命令一律后台**。工具层有 2 分钟上限，`timeout 900` 不解决问题——它只会让父进程被杀而子进程变孤儿（第 4 次就是这么来的）

---

## <a id="cenet-vs-paper"></a>CENET：MMSA 的版本与论文的是两个不同的模型（2026-07-29）

移植 CENET（Wang et al., TMM 2022）时逐行对照原作者发布（`Say2L/CENet`）与 MMSA。**核心机制不同，不是细节差异。**

| | 原作者 | MMSA |
|---|---|---|
| 主干 | **SentiLARE**（RoBERTa 系，带词性/情感知识的预训练） | `bert-base-uncased` |
| CE 模块的输入 | **把模态量化成 16 个离散标签**，`nn.Embedding` 查表 | 原始连续特征过 MLP（`Linear→ReLU→Linear`） |
| MOSI 特征维度 | audio 74 / vision 27 | audio 5 / vision 20 |
| 注入层 | 编码器第 1 层之前（`ROBERTA_INJECTION_INDEX = 1`） | 同 ✓ |
| SelfAttention | `softmax(scores * 8)` | **逐字节相同** ✓ |

原作者的 CE `forward` 接收 `visual`/`acoustic` 两个参数却**从不使用**，只用 `visual_ids`/`acoustic_ids`：

```python
def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
    visual_ = self.visual_embedding(visual_ids)      # 只有 ids 被用到
    acoustic_ = self.acoustic_embedding(acoustic_ids)
```

MMSA 反过来：用连续特征，忽略 ids。**论文的机制建立在"模态被量化成 16 类语义标签"上，MMSA 的建立在"原始连续特征"上——这是把机制换掉了，不是把它实现得略有不同。**

这是 `#mult-position` 的放大版：MulT 丢的是一个结构件，CENET 换的是跨模态增强的输入表示。

**处理**：验收组忠实移植 MMSA（同协议才可比，其表报的就是这个版本）。storyline 的 CENET 一节**不得写成"我们复现了论文的 CENET"**。忠实论文版需要复原原作者的量化步骤（怎么把特征分成 16 类），未确认可行，暂不做。

### `softmax(scores * 8)` 是继承的

标准缩放点积注意力除以 `sqrt(d_k) ≈ 27.7`，这里是**乘以 8**，相差约 222 倍。原作者的 `CEmodule.py` 与 MMSA 的该类**逐字节相同**，所以按清单第 9 条判定为**继承**，予以复现。（原作者还有一个带 `* math.sqrt(text_dim)` 的 `Attention` 类，CE 里没用到，是死代码。）

### 实现上没有重抄 487 行 BERT

MMSA 为了在层间插入 CE，重实现了 `BertLayer`/`BertEncoder`/`BertOutput`/`BertIntermediate` 共 487 行，且仍依赖已废弃的 `pytorch_transformers`。我们改为迭代 HuggingFace `bert.encoder.layer`，约 30 行。

**等价性已实测**：手动迭代 12 层与原生 `BertModel.forward` 的输出**最大差异 0.0**。

**移植中踩到的坑**：`transformers >= 5` 的 `BertLayer.forward` **直接返回 Tensor**，不再返回 tuple。照 MMSA 的老写法取 `[0]` 不会报错，而是**静默取到第一个样本**，形状 (B,L,H) 变成 (L,H)，错误在几帧之外才以 batch 维不匹配的形式暴露。已加 `torch.is_tensor` 判断并注释。

---

## <a id="tfn-corr-resolved"></a>TFN 的 Corr 缺口：跑了 MMSA 自己的代码才解决（2026-07-30）

### 为什么之前查不出来

`#corr-gaps` 记录了九项假设全部排除、三份实现（我们 / MMSA / `Justin1904/TensorFusionNetworks`）逐行一致，而缺口仍在。**问题出在方法上：九轮排查全是在"读"MMSA 的代码，从没"跑"过它。**

### 做法

在独立 venv 里跑 MMSA 自己的 `MMSA_run('tfn','mosi')`，**MMSA 的算法代码一行未改**。两处算法之外的让步：`torch.cuda.set_device` 被 monkeypatch 成空操作（磁盘不足以装 CUDA 轮子，只能用 CPU 版 torch），数据路径经其公开的 `config={'featurePath': ...}` 覆盖。`pytorch_transformers` 打桩——只影响我们不跑的 CENET。

为消除设备混淆，**我们的实现也在 CPU 上跑同样的 seed**。

### 结果（CPU，同数据，seed 42-46）

| | MAE | Corr |
|---|---|---|
| **MMSA 自己的代码** | 0.9640 ± 0.0358 | **0.6456 ± 0.0136** |
| **我们的代码** | 0.9454 ± 0.0315 | **0.6595 ± 0.0159** |
| *MMSA 发表的表* | *0.9473* | *0.6733* |

| 逐 seed Corr | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| MMSA | 0.6360 | 0.6487 | 0.6568 | 0.6592 | 0.6275 |
| 我们 | 0.6629 | 0.6431 | 0.6541 | 0.6851 | 0.6524 |

### 三条结论

1. **两份实现在同一设备上等价甚至我们略优**（Corr +0.0139、MAE −0.0186），与"三方逐行核对一致"相符。
2. **MMSA 自己的代码复现不出它自己发表的 Corr**：差 **−0.0277**，是我们那 0.0138 的**两倍**。它的 5 个 seed **无一**达到 0.6733（最高 0.6592）。
3. **不是设备造成的**：我们的实现 CUDA 上 Corr 0.6602（10 seed）、CPU 上 0.6595（5 seed），**设备只移动 0.0007**。

### 限定（必须与结论同读）

- 5 个 seed，低于本项目 10 seed 的协议
- CPU；MMSA 的表大概率来自 CUDA。上面第 3 条只证明**我们的实现**对设备不敏感，未直接证明它的实现也不敏感
- **它的表用的 seed 集合未知**（其默认为 1111-1115，非 42-46）

因此正确措辞是：**MMSA 的代码在我们能控制的条件下复现不出它自己的表**——而不是"它的表是错的"。

### 方法论教训

**读代码能证明两份实现一致，不能证明一个数字可达。** 九轮排查耗时数天，而让参照实现自己跑一遍在依赖装好后只用了十几分钟。**参照实现可运行时，先跑它，再读它。** 这条已加入核对清单。

---

## <a id="ef-lstm-corr-resolved"></a>EF-LSTM 的 Corr 缺口：同样跑 MMSA 才解决（2026-07-30）

用 `#tfn-corr-resolved` 那套办法跑 MMSA 自己的 EF-LSTM，seed 集合与我们的验收组完全一致（42-51）。

### 结果（MMSA 为 CPU，我们为 CUDA，同数据同 seed）

| | 崩溃数 | Corr | MAE | 仅未崩 seed 的 Corr |
|---|---|---|---|---|
| **MMSA 自己的代码** | **1/10**（seed 45） | 0.6358 ± 0.0230 | 1.0236 ± 0.1570 | 0.6424 (n=9) |
| **我们的代码** | 2/10（46, 48） | 0.6281 ± 0.0459 | 1.0717 ± 0.2095 | **0.6485 (n=8)** |
| *MMSA 发表的表* | *—* | *0.6690* | *0.9488* | *—* |

### 三条结论

1. **MMSA 的实现也会崩。** 崩溃是这个配置的固有性质（4 层 LSTM、层间 dropout 0.5、1284 条训练样本），与移植无关。1/10 与 2/10 在约 20% 崩溃率下属同一分布，`#corr-gaps` 里 30 seed 实测的 23% 得到独立佐证。
2. **它自己的代码从未达到它发表的数字。** Corr 最高 0.6558，均值差 **0.0332**；MAE 差 **0.075**。表里的 0.9488 实际上要求一次崩溃都没有。
3. **只看未崩的 seed，我们（0.6485）优于它（0.6424）。**

### 这同时解释了那半个残留

`#corr-gaps` 记录过：即便（仅为诊断）剔除崩溃 seed，我们仍差表值 0.0215。现在清楚了——**MMSA 自己的代码在未崩 seed 上也只有 0.6424，差表值 0.0266**，比我们差得更多。那 0.0215 不是我们的问题。

### 与 TFN 同型，但不要外推

两个模型都指向"发表值无法由发表代码在可控条件下达到"。但这**只对已实测的两个模型成立**。其余九个模型若要做同类归因，**必须各自跑一遍 MMSA**——环境与驱动脚本（`run_mmsa.py <模型> <seeds...>`）已就绪，成本很低。

**限定同 `#tfn-corr-resolved`**：CPU vs 其表大概率的 CUDA；其表的 seed 集合未声明。措辞是「在我们能控制的条件下不可达」，不是「它的表是错的」。

---

## <a id="mmsa-five-models"></a>把 MMSA 跑起来比五个模型：两个结论，一个被证伪的猜想（2026-07-30）

`#tfn-corr-resolved` 与 `#ef-lstm-corr-resolved` 之后，用同一套办法（`run_mmsa.py <模型> <seeds>`）再跑三个冻结特征模型。**BERT 微调的四个（MISA/Self-MM/CENET/TETFN）跑不了**——CPU 上每 seed 数小时，而磁盘剩 3.3 GB 装不下 CUDA 轮子。Graph-MFN 与 MulT 因耗时暂缓。

### 结论一：我们的实现与 MMSA 的实现，在同设备同 seed 下无法区分

seed 42-46，CPU，同一份 pkl：

| 模型 | 指标 | MMSA 的代码 | 我们的代码 | 差 | t |
|---|---|---|---|---|---|
| lf_dnn | MAE | 0.9448 | 0.9567 | +0.0119 | +0.62 |
| lf_dnn | Corr | 0.6701 | 0.6599 | −0.0102 | −1.53 |
| lmf | MAE | 0.9423 | 0.9501 | +0.0078 | +0.52 |
| lmf | Corr | 0.6653 | 0.6617 | −0.0036 | −0.70 |
| mfn | MAE | 0.9409 | 0.9484 | +0.0075 | +0.28 |
| mfn | Corr | 0.6543 | 0.6570 | +0.0027 | +0.23 |
| tfn | MAE | 0.9640 | 0.9454 | −0.0186 | −0.87 |
| tfn | Corr | 0.6456 | 0.6595 | +0.0139 | +1.49 |

**八项全部不显著**，差异方向也不一致（两个模型我们略优，两个它略优）。

**但必须同时说明检验功效**：n=5，最小可检出差约 **2.78×SE ≈ 0.04–0.05 MAE**。所以"不显著"只排除了**大**差异，排不掉 0.01 量级的小差异。正确表述是：**观察到的差异绝对值都很小（MAE ≤0.019、Corr ≤0.014），且与零无法区分；本检验只能排除大于约 0.05 MAE 的差异。**

### 结论二：MMSA 的表能否由它自己的代码达到，是逐模型的

| 模型 | 它的代码 Corr | 它的表 | |
|---|---|---|---|
| TFN | 0.6456 | 0.6733 | **代码达不到**（−0.0277） |
| EF-LSTM | 0.6358 | 0.6690 | **代码达不到**（−0.0332） |
| MFN | 0.6543 | 0.6702 | **代码达不到**（−0.0159） |
| **LF-DNN** | **0.6701** | 0.6584 | **代码超过表**（+0.0117） |
| **LMF** | **0.6653** | 0.6510 | **代码超过表**（+0.0143） |

### 被证伪的猜想

`#systematic-bias` 曾把"MMSA 按测试集挑超参"列为系统性偏置的首要候选。**若成立，五个模型应当一致地"表优于代码"——实际上有两个反过来。**

因此：**"MMSA 的表系统性偏乐观"不成立。** `#selection-bias-measured` 测出的 0.026 MAE / 0.011–0.026 Corr 仍然有效，但它只说明**那个机制能制造多大偏差**，不说明它在每个模型上都被用足了——现在有了直接反证。

`#tfn-corr-resolved` 与 `#ef-lstm-corr-resolved` 两条裁定**仍然成立**，因为它们是逐模型实测且都标了 `does_not_generalise`。这次的反例把那条限定从"谨慎措辞"变成了"有证据的必要限定"。

### 对"我们比 MMSA 差吗"这个问题的回答

**在能控制的条件下（同设备、同数据、同 seed），我们的实现与参照实现无法区分。** 相对其**表格**的差距，有三个模型是它自己的代码也达不到的。

**未覆盖**：Graph-MFN、MulT、MISA、Self-MM、CENET、TETFN 六个模型没有跑过 MMSA，**对它们不得做同类陈述**。尤其是微调 BERT 的四个——那正是数值最好、最受关注的一批。
