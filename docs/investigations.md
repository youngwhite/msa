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
11. **重写（而非转抄）的模型，必须做权重复制的数值等价测试，并加进 `check_all.sh`。** 两边各建一个模型，按注册顺序逐个复制权重，喂同一份输入比对输出。**参数张量的数量是第一道筛子**——数量对不上，形状和输出都不用比。ALMT 就是这样抓出三处缺失（`token_len=None` 的分支仍然建位置嵌入、融合层的两个位置嵌入、融合层的 CLS token），而此前九轮读代码一处都没发现，且那个错误模型的指标**全面优于参照**、验收判为通过（见 `#almt-better-than-reference`）。已有的两个：`check_mult_encoder.py`、`check_almt_equivalence.py`

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

逐项核对已完成的：MISA（结构与原实现一致：融合层顺序、transformer `nhead=2 / num_layers=1`）、Self-MM、LMF、MulT、**MFN（2026-07-31，见下）**。

### MFN 对原作者实现的核对：无发现（2026-07-31）

`pliang279/MFN` 的 `test_mosi.py`，逐行比对我们的 `src/msa/models/mfn.py`：

| 环节 | 原作 | 我们 | |
|---|---|---|---|
| 注意力窗口 | `cStar = cat[prev_cs, new_cs]`（更新前后的 c 拼接） | 同 | ✓ |
| 注意力 | `softmax(att1_fc2(drop(relu(att1_fc1(cStar)))), dim=1) * cStar` | 同 | ✓ |
| 提案 | `tanh(att2_fc2(drop(relu(att2_fc1(attended)))))`，输入是 **attended** 而非 cStar | 同 | ✓ |
| 门控输入 | `cat[attended, mem]` | 同 | ✓ |
| 记忆更新 | `mem = γ₁·mem + γ₂·cHat`，两个 γ 各自 sigmoid | 同 | ✓ |
| 输出 | `cat[last_h_l, last_h_a, last_h_v, last_mem]` → fc1 → relu → dropout → fc2 | 同 | ✓ |
| 损失 | `nn.L1Loss()` | 同（MMSA 也是） | ✓ |
| 模型选择 | **验证集 loss**（`if valid_loss <= best_valid: torch.save`） | 验证集 MAE | ✓ |

**三方（原作 / MMSA / 我们）在结构与损失上完全一致。** 差异只有两处，都不构成实现分歧：原作跑满 `num_epochs` 无早停（MMSA 与我们 patience=8），原作的 `ReduceLROnPlateau(patience=100)` 在其轮数下不可能触发。超参不可比——原作是在 CMU-SDK 的 GloVe 特征上做随机搜索得到的。

**结论：MFN 对表的缺口（MAE +1.4 SE / Corr +2.3 SE）不来自结构或协议差异。** 这条核对的价值是排除，不是发现。

**顺带值得记一笔**：MFN 原作**用验证集选模型**，与 `#originals-select-on-test` 记的 MMIM / ALMT / BERT-MAG 三个原作相反。所以"原作用测试集选"不是这条线的普遍规律，而是分模型的事实，每个都得自己去看。

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

**已实现（2026-07-31）**：判据的稳健性维度落地为 **SE 分母封顶** `min(σ_模型, NOISE_FLOOR)`——离散度可让检验更严，不可比数据集基线噪声更松。EF-LSTM 的 MAE 因此由"标记 +1.9 SE"变为"未达标 +10.0 SE"，那 3.2 倍地板的缺口不再被两个崩溃 seed 撑大的 σ 藏住。理由与全部影响见 `docs/decisions.md` 2026-07-31。**这是一次收紧。**

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

### 第六、七种形态：等待器自己造成破坏

| # | 写法 | 后果 |
|---|---|---|
| 6 | 把 `nohup ... &` 与前台等待循环写在**同一条命令**里 | 前台 2 分钟超时（exit 143）连带杀掉整个进程组，**已跑到一半的 TFN 训练被中断** |
| 7 | 等待器末尾放了 `ls`/判断语句 | 文件不存在时 `ls` 返回非零，**整个等待被标记为失败**，而被等的训练其实正常 |

**规则：等待器只等，不做别的。** 不打印、不判断、不清理——它的退出码必须只反映被等待的任务。启动长任务用 `setsid` 彻底脱离会话，等待另起一条命令。

**同一类失误（对命令边界判断错误）在本会话出现七次。** 单独列出不是为了自责，而是因为它造成的损失（两天空转、一次训练中断、一次错误汇报）超过了多数模型 bug。

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

---

## <a id="mmsa-all-eleven"></a>把 MMSA 跑遍十一个模型：对"我们比它差"这个前提的最终回答（2026-07-30）

### 起因

整个项目的验收都在拿我们的数字比 MMSA **表格里的数字**，并反复出现"落后 1–4 SE"。前提一直是：那些数字代表 MMSA 实现的真实水平。**这个前提从未被检验过**，直到 `#tfn-corr-resolved` 第一次去跑它的代码。

### 做法

`run_mmsa.py <模型> <seeds>`，seed 42-46，与我们验收组的前五个完全一致，同一份 pkl，**同为 CUDA**（用主 venv 已有的 torch，未装第二份）。取数一律以 MMSA 自己写的 CSV 为准——`self_mm` 等 multiTask trainer 的日志里 `TEST-` 行只有 loss、没有指标，只解析日志会漏。四个模型交叉核对过 CSV 与日志一致。

**MMSA 的算法代码一行未改。** 全部让步及其性质：

| 让步 | 触及算法？ |
|---|---|
| `torch.cuda.set_device` 空操作（仅无 CUDA 时） | 否，设备管道 |
| `ReduceLROnPlateau` 吞掉新版 torch 已移除的 `verbose` | 否，日志开关 |
| `featurePath` 经其公开 `config` 参数覆盖 | 否，数据路径 |
| 主 venv 的 CUDA torch + 符号链接补 5 个小包 | 否，环境组装 |
| CENET 装**真的** `pytorch_transformers` | 否 |

**CENET 那次值得单记**：初次失败是 `'CENET' object has no attribute 'all_tied_weights_keys'`——我的桩把 `PreTrainedModel` 重定向到了现代 `transformers`，契约已变。**当时可以硬加该属性让它跑起来，但那会掩盖真实的 API 不匹配、产出不可信的数字**，所以改为安装原始依赖。为凑齐一个模型而放进可疑数据，会污染整批结论。

### 结果（seed 42-46，CUDA，同数据）

| 模型 | 它的代码 MAE | 我们 MAE | 表 MAE | 它 Corr | 我们 Corr | 表 Corr | 代码−表 (Corr) | 我们−它 (Corr, t) |
|---|---|---|---|---|---|---|---|---|
| ef_lstm | 1.0236 | 1.0712 | 0.9488 | 0.6358 | 0.6215 | 0.6690 | −0.0332 | −0.0143 (t=−0.53) |
| lf_dnn | 0.9448 | 0.9513 | 0.9548 | 0.6701 | 0.6587 | 0.6584 | **+0.0117** | −0.0114 (t=−1.43) |
| tfn | 0.9566 | 0.9545 | 0.9473 | 0.6603 | 0.6559 | 0.6733 | −0.0130 | −0.0044 (t=−0.45) |
| lmf | 0.9423 | 0.9515 | 0.9504 | 0.6653 | 0.6689 | 0.6510 | **+0.0143** | +0.0036 (t=+0.54) |
| mfn | 0.9409 | 0.9400 | 0.9268 | 0.6543 | 0.6570 | 0.6702 | −0.0159 | +0.0027 (t=+0.17) |
| graph_mfn | 0.9814 | 0.9680 | 0.9557 | 0.6452 | 0.6533 | 0.6486 | −0.0034 | +0.0081 (t=+1.09) |
| mult | 0.9270 | 0.8974 | 0.8799 | 0.6861 | 0.6872 | 0.7022 | −0.0161 | +0.0011 (t=+0.25) |
| misa | 0.7883 | 0.7539 | 0.7765 | 0.7657 | 0.7756 | 0.7781 | −0.0124 | +0.0099 (t=+1.54) |
| self_mm | 0.7108 | 0.7129 | 0.7080 | 0.7915 | 0.7943 | 0.7963 | −0.0048 | +0.0028 (t=+0.89) |
| cenet | 0.7638 | 0.7296 | 0.7254 | 0.7808 | 0.7926 | 0.7953 | −0.0145 | **+0.0118 (t=+2.28)** |
| tetfn | 0.7349 | 0.7308 | 0.7084 | 0.7892 | 0.7932 | 0.7984 | −0.0092 | +0.0040 (t=+1.85) |

### 结论一：MMSA 的表在 9/11 个模型上不可由它自己的代码达到

只有 LF-DNN 与 LMF 例外（代码超过表 +0.012 / +0.014）。**其余九个模型，它的代码差自己发表的 Corr 0.003–0.033。**

因此**本项目此前所有"落后 MMSA 若干 SE"的表述，比较的是一个多数情况下连参照实现自己都达不到的目标。**

### 结论二：我们的实现不比它的差

同设备同 seed 下：**我们更优 6、它更优 2、混合 3**。多数 t 值小于 1.5，即两份实现基本不可区分——这正是忠实移植应有的样子。

最显著的三处都在我们这边：**CENET t=+2.28**、TETFN t=+1.85、MISA t=+1.54。CENET 上我们的 MAE 0.7296 对它的 0.7638，**比它自己的代码更接近它自己的表（0.7254）**。

它更优的两个：LF-DNN（t=−1.43）与 EF-LSTM（t=−0.53）。**EF-LSTM 那条不构成实现差异的证据**——5 个 seed 里我们崩在 46、它崩在 45，崩溃位置不同即可主导结果（完整 30 seed 分析见 `#corr-gaps`）。

### 必须与结论同读的限定

- **n=5**，且是我们注册的 10 个 seed 的前五个，非其表所用的集合（其默认 1111-1115，未声明）
- 检验功效低：n=5 时最小可检出差约 2.8×SE，**"不显著"只排除大差异**
- 措辞是「**其发表值在我们能控制的条件下不可由其代码达到**」，**不是**「它的表是错的」

### 这也终结了一个猜想

`#systematic-bias` 曾把"MMSA 按测试集挑超参"列为首要成因。若成立，十一个模型应当一致地"表优于代码"——**实际有两个反过来**。`#selection-bias-measured` 测出的偏差量级仍然有效，但它只说明**机制能制造多大偏差**，不说明每个模型都用足了。

### 方法论：这条应当排在核对清单第一位

**参照实现可运行时，先跑它，再读它。** 读代码能证明两份实现一致，**不能证明一个数字可达**。TFN 上九轮排查、三方逐行核对、耗时数天无果；跑一遍十几分钟就清楚了，而且顺带解决了其余十个模型的同类疑问。

## <a id="originals-select-on-test"></a>两个原作实现都用测试集选模型（2026-07-31）

移植 BERT-MAG 与 MMIM 时按第 5 条回原作核对，**两个都在测试集上做模型选择**。这不是推测，是代码。

### MAG-BERT（Rahman et al., ACL 2020，`WasifurRahman/BERT_multimodal_transformer`）

`multimodal_driver.py` 的训练循环里**没有 `torch.save`**——它从不保存检查点。每个 epoch：

```python
valid_loss = eval_epoch(...)                         # 497 行前
test_acc, test_mae, test_corr, test_f = test_score_model(model, test_data_loader)
...
"best_valid_loss": min(valid_losses),                # 只打印，不用于任何选择
"best_test_acc":  max(test_accuracies),              # ← 报告值
```

**报告的是逐 epoch 测试集准确率的最大值。** 验证损失算了、记了，然后被丢掉。

### MMIM（Han et al., EMNLP 2021，`declare-lab/Multimodal-Infomax`）

`solver.py` 存检查点的条件是嵌套的：

```python
if val_loss < best_valid:                # 外层：验证集
    ...
    elif test_loss < best_mae:           # 内层：测试集 ← 真正的门
        best_epoch = epoch
        best_results = results           # results 来自 evaluate(test=True)
        save_model(...)
```

外层看验证集，但**内层用测试损失决定存不存**，且报告的 `best_results` 取自测试集评测。

### 为什么这条重要

我们此前实测过这种乐观偏差的量级（`#selection-bias-measured`）：3 seed 下 **0.026 MAE / 2.3 点 Acc-2**，1 seed 下 0.074 / 4.9。**这与文献里大多数"改进"的幅度同量级。**

直接后果：

1. **这两篇论文的表格数字不能与验证集选择的复现结果比。** 差距不代表复现失败，代表协议不同。凡把原文数字当复现目标的，都在追一个用不同规则得到的数。
2. **MMSA 在这一点上比原作严格。** 它的 `KeyEval` 走 `dataloader['valid']`，没有任何测试集参与选择。这与 `#systematic-bias` 里"MMSA 的表偏乐观"的旧猜想方向相反——那个猜想已被 `#mmsa-all-eleven` 推翻，这里再添一条反证。
3. **我们的第 2 条约定（模型选择只看验证集）不是保守，是这个领域里的少数派。**

### 未做的事

没有量化"若改用测试集选择，我们的数字会涨多少"。可以做（跑一遍 `--select-on` 走测试集），但**那会产出一个我们不该报告的数字**，且乐观偏差的量级已经单独测过。留作说明，不留作实验。

## <a id="bert-mag"></a>BERT-MAG 移植：一次通过，以及参照从哪里来（2026-07-31）

### MMSA 的公开表没有这个模型

`results/result-stat.md` 里查不到 MAG-BERT、MMIM、ALMT 中的任何一个。**不是遗漏，是它从未登记。** 所以"对齐 MMSA 公开指标"这条路对这三个不存在。

替代方案：跑 MMSA 自己的代码取参照，seeds 42-51，同一份 MOSI pickle、同一个 CUDA torch、它自己的默认超参。结果与出处记在 `docs/mmsa_code_runs_mosi.json`。

**这比公开表更强**：表是无方差单值，自跑值带 σ，比较从"点 vs 分布"变成"分布 vs 分布"。判据相应改为两样本 SE，见 `docs/decisions.md` 2026-07-31——**规则在这三个模型一次都没跑之前定死**。

### 实现：重写而非转抄

MMSA 那份 `BERT_MAG.py` 在 transformers 5 下**根本 import 不了**——它继承的 `BertPreTrainedModel` 契约已不存在。所以计算逻辑照它逐行读（读得了，只是跑不了），管道用现代写法。

事后与原作者 `modeling.py` 的 `class MAG` 对照，**逐行一致**：

```
W_hv/W_ha  -> gate_v/gate_a          W_v/W_a -> project_v/project_a
eps=1e-6,  where(hm_norm==0, 1, ·),  alpha=min(em_norm/(hm_norm+eps)*beta_shift, 1)
return dropout(LayerNorm(alpha * h_m + text_embedding))
```

注入点也一致：嵌入层之后、编码器之前，只此一次。

### 唯一需要判断的地方

transformers 5 里 `BertLayer.forward` 返回**裸张量**而非元组。照旧写法取 `[0]` 会**静默取到第一个样本**——形状仍然合法，结果全错。已在 `#cenet-vs-paper` 记过一次，这里用 `torch.is_tensor` 守卫再次绕开。这个坑不报错，只出坏数字。

### 结果（seeds 42-51）

| 指标 | 我们 | MMSA 自跑 | 差 | SE |
|---|---|---|---|---|
| MAE | 0.7368 ± 0.0182 | 0.7380 ± 0.0226 | **-0.0012** | -0.1 |
| Corr | 0.7885 ± 0.0066 | 0.7891 ± 0.0065 | +0.0006 | +0.2 |
| Acc-2 (non0) | 0.8410 ± 0.0088 | 0.8428 | +0.0018 | +0.5 |
| Acc-7 | 0.4436 ± 0.0186 | 0.4430 | **-0.0006** | -0.1 |

七项指标全部通过，**一次通过，无需排查**。这是移植的第十二个模型，也是第一个不用回头找差异的。

原因不难说清：MAG 只有 35 行、没有训练循环上的花样（无梯度累积、无多优化器、无伪标签），且 MMSA 对它的移植本身就忠实。**复现难度与模型的训练协议复杂度相关，与模型的结构复杂度关系不大**——BERT-MAG 有 1.1 亿参数却一次过，TFN 只有几百万却查了九轮。

## <a id="mmim"></a>MMIM 移植：通过（标记），以及一处系统性协议差异（2026-07-31）

### 结果（seeds 42-51）

| 指标 | 我们 | MMSA 自跑 | 差 | SE | 判定 |
|---|---|---|---|---|---|
| MAE | 0.7452 ± 0.0225 | 0.7339 ± 0.0226 | +0.0113 | +1.1 | 标记 |
| Corr | 0.7776 ± 0.0112 | 0.7789 ± 0.0112 | +0.0013 | +0.3 | 通过 |
| Acc-2 (non0) | 0.8311 ± 0.0138 | 0.8392 | +0.0081 | +1.6 | 标记 |
| Acc-7 | 0.4494 ± 0.0185 | 0.4551 | +0.0057 | +0.8 | 通过 |

**判定：已复现**，但四个指标一致偏低 1-2 SE。

### 把这个差距放到正确的尺度上

**MAE 差 0.0113，MOSI 的噪声地板是 0.0387。** 差距不到重跑一次会产生的 seed 波动的三分之一。它被标记只是因为两侧各 10 个 seed 把 SE 压得很紧——这正是两样本 SE 判据比单值参照更严的地方。

### 排查过的、不是原因的

1. **xavier 初始化的 O(n²) 嵌套**：每次调用都重新采样、覆盖前一次，最终分布与干净单次相同，只有 RNG 流位置不同。
2. **`_RNNEncoder` 的 dropout**：MMSA 把同一个 `dropout` 同时给 LSTM 和其后的 `nn.Dropout`，而 `n_layer=1` 时该值是 0.0。我们一致。
3. **LSTM 的 `batch_first`**：MMSA 建 LSTM 时写 `batch_first=False` 却用 `batch_first=True` 打包。**对 PackedSequence 输入，LSTM 的该标志不起作用**，两种写法等价。
4. **序列长度**：逐元素比对过，我们的 `audio_lengths`/`vision_lengths` 与原始 pickle 完全相同，MMSA 读的是同一份。
5. **`Fusion` 里的死代码**：`fusion = self.linear_2(y_1)` 算完丢弃，`y_2` 又算一遍。Linear 不消耗 RNG，无影响。（继承自原作 `SubNet`。）

### 一处真实的系统性差异，适用于我们移植的**每一个**模型

MMSA 的验证集选择量是：

```python
eval_loss = eval_loss / len(dataloader)      # 逐 batch 平均，不是逐样本
eval_results["Loss"] = round(eval_loss, 4)   # 先四舍五入到 1e-4 再比较
```

两个后果：

1. **逐 batch 平均超权了最后一个不满批。** 验证集 229 个样本、batch 32 → 最后一批只有 5 个，却与 32 个的批等权，**超权 6.4 倍**。
2. **`round(·, 4)` 让小于 1e-4 的改善不可见**，而改善判据是 `<= best - 1e-6`。MMSA 的早停因此比我们略钝。

我们的 `--select-on mae` 是按样本平均、不取整。

**为什么不改成它那样**：我们的量是更正确的那个；改动会波及全部 13 个模型并破坏 LF-LSTM 的哈希 `172967c7dced83b2`；而且为了抹平一个**低于噪声地板**的差距去迁就一个更差的指标，本质上就是朝目标调参——正是本项目明令禁止的。记录为已知差异。

## <a id="deleted-a-live-run"></a>把一个正在跑的实验判成"失败"并删掉了它的输出（2026-07-31）

### 发生了什么

启动 ALMT 的 10 seed 验收后，用这条命令抓 PID 以便等待：

```bash
setsid nohup env JOBS=2 bash scripts/reproduce_all.sh almt_mosi > log 2>&1 &
PID=$(pgrep -f "reproduce_all.sh almt_mosi" | head -1)     # <- 拿到 742811
```

`742811` 不是 `reproduce_all.sh`，而是 `setsid`/`env` 这条链上的一个中间进程，**它转眼就退出了**。真正的 `reproduce_all.sh` 是 `742815`。

后果是连锁的，一步比一步严重：

1. 等待器盯着一个秒退的进程，**立刻报告"运行结束"**；
2. 我去看 `outputs/almt_mosi/`，只有 `best.pt`、没有 `result.json`，**判定"运行失败"**；
3. 于是 **`rm -rf outputs/almt_mosi`——删掉了一个正在运行的实验的输出目录**；
4. 又起了个前台 smoke test 想看报错，**抢掉显存，把还在跑的两个进程挤到 CUDA OOM**。

一直到 OOM 的报错信息里出现 `Process 742878 has 6.92 GiB memory in use. Process 742879 has 6.92 GiB` 才发现：那两个进程一直活着。

**没有丢失已确认的结果**——那个组当时尚未产出任何 `result.json`。但那次运行的状态已不可信，只能停掉重跑。

### 这属于哪一类

这是 `#self-matching-patterns` 里"命令边界"那一类的第八次，也是**代价最大的一次**。之前七次的教训被总结成"等待器只做等待"，我遵守了——**但那条规则漏了前半句**：

> 等待器只做等待。**但在等之前，先确认盯的是不是正确的进程。**

前七次的失败模式是"等待器多做了事"（附带 `ls`、附带 `pkill`、模式串匹配到自己）。这次是"等待器什么都没多做，只是从一开始就盯错了对象"。**一个盯错对象的等待器，比一个会自匹配的等待器更危险**——后者会一直不返回、看得出来有问题，前者会立刻返回，看起来一切正常。

### 现在的做法

```bash
setsid nohup bash -c 'echo $$ > pidfile; exec env JOBS=2 bash scripts/reproduce_all.sh <group>' \
  > log 2>&1 < /dev/null &
```

`exec` 用目标进程**替换**这个 shell，所以记录下来的 `$$` 就是 `reproduce_all.sh` 本身，中间进程不存在。启动后必须回读验证一次：

```bash
tr '\0' ' ' < /proc/$(cat pidfile)/cmdline      # 必须看到 reproduce_all.sh <group>
```

### 比 PID 更重要的那条规则

`#reading-live-outputs` 记的是"别读运行中的 `outputs/` 就下结论"——那次我把正在被覆写的结果误判成陈旧。这次是同一个根因的**破坏性版本**：不只是读错，是**删掉了**。

所以那条规则要升级：

> **在判定一个实验失败、并对它的输出做任何破坏性操作之前，必须先确认没有进程正在写它。**
> 判据不是"目录里缺 `result.json`"——一个正常运行的实验，绝大部分时间里都缺 `result.json`。
> 判据是 `pgrep`/`nvidia-smi --query-compute-apps` 确认无进程占用，**且**日志里有明确的失败信息。

这次两个判据一个都不满足：日志停在启动横幅、没有任何 traceback，**"日志没有报错"本身就该是"它还在跑"的证据，而不是"它悄悄死了"的证据**。真被 SIGKILL 的进程确实不留 traceback，但那时 `pgrep` 会是空的——这一步只要做了，整件事就不会发生。

### 一个附带的观察

显存也该算进"破坏性操作"。GPU 只有 15.5 GiB，两个 ALMT 进程占 14.2 GiB，**余量 1.3 GiB**。此时再起任何一个用 GPU 的诊断命令，都会把正在跑的实验挤爆。要在实验运行期间做 GPU 诊断，先看 `nvidia-smi` 的余量，或者干脆改用 CPU。

## <a id="almt-better-than-reference"></a>比参照"更好"是个警报，不是好消息（2026-07-31）

### 起因

ALMT 第一版移植跑完 10 seed，**七项指标全部优于 MMSA 自跑的参照**：

| 指标 | 我们（错误版本） | MMSA 自跑 | 差 |
|---|---|---|---|
| MAE | 0.7392 | 0.7540 | **-1.9 SE**（更好） |
| Corr | 0.7890 | 0.7810 | **-3.2 SE**（更好） |

我们的验收判据只在"更差"的方向上判失败，所以它给出的是"通过"。**但 Corr 领先 3.2 SE 是不正常的**——我们与参照跑的应该是同一个模型、同一份数据、同样的超参。领先这么多，只可能是模型不一样。

### 做了什么

写权重复制的等价测试（`scripts/check_almt_equivalence.py`）：两边各建一个模型，按注册顺序逐个复制权重，喂同一份输入，比对输出。BERT 两边打桩，因为它按构造就相同。

第一次运行就暴露了问题：**MMSA 183 个参数张量，我们只有 179 个。**

### 找到三处缺陷，读代码全都漏掉了

1. **`l_encoder` 的位置嵌入丢了。** MMSA 用同一个 `Transformer` 类构造它，只是 `token_len=None`——**但那条分支仍然建 `pos_embedding` 并加上去**。我看到 `token_len=None` 就以为它退化成了裸编码器。后果：文本流在被 AHL 层查询之前，**失去了唯一的顺序信息**。

2. **融合层的两个位置嵌入丢了。** `CrossTransformer` 给 source 和 target 各加一个。

3. **最严重的一处：融合层的 CLS token 丢了。** `CrossTransformer` 会给两条流各前置一个**共享的** CLS token，末尾的 `[:, 0]` 读的就是**那个 CLS**。我的实现没有它，`[:, 0]` 读到的是超模态的第一个 token——**回归头读的根本不是同一个量。**

修好之后：**183 vs 183，最大绝对差 `0.000e+00`**，逐比特一致。

用修正后的模型重跑 10 seed，"过好"的信号随之消失：

| 指标 | 错误版本 | **修正版本** | MMSA 自跑 |
|---|---|---|---|
| MAE | 0.7392（-1.9 SE） | **0.7404（-1.7 SE）** | 0.7540 |
| Corr | 0.7890（**-3.2 SE**） | **0.7853（-1.3 SE）** | 0.7810 |

七项全部通过。**残余的领先只能来自训练协议，不可能来自模型**——模型已被证明逐比特相同。方向也与已知的选择量差异吻合（我们按样本平均且不取整，MMSA 逐 batch 平均并四舍五入到 1e-4；更细的粒度能选到更好的 epoch）。

### 教训

**"比参照更好"必须当作实现有差异来排查，与"比参照更差"同等对待。** 我们的判据是单向的（只罚更差），这在防止朝目标调参上是对的，但它意味着**一个反向的错误可以静悄悄地通过验收**。这次那批数字已经写不进文档就被作废了——但如果领先只有 0.5 SE 而不是 3.2 SE，它很可能就混进去了。

**读代码定位不了这类缺陷。** 三处缺失里有两处的形态是"某个类在某个参数取特定值时仍然做了一件事"——`token_len=None` 依然建位置嵌入，`CrossTransformer` 的构造参数名里根本看不出有 CLS。这与 `#tetfn-inert` 的"声明了但不生效"正好相反：**这次是"没声明但在生效"**。

**核对清单新增第 11 项**：凡是重写（而非转抄）的模型，必须做权重复制的数值等价测试，并把它加进 `check_all.sh`。参数张量的**数量**是第一道筛子——数量对不上，形状和输出就不用比了。这条测试便宜（几秒）、判据明确（最大绝对差），而且它发现的这三处缺陷，此前九轮读代码一处都没发现。

## <a id="select-reduction"></a>验证选择口径值多少：TFN 上 0.016 MAE，MFN 上几乎为零（2026-07-31）

### 背景

MMSA 选 epoch 用 `eval_loss / len(dataloader)` 再 `round(·, 4)`——**逐 batch 平均**，末批不满被超权；我们逐样本平均、不取整。这个差异自 MMIM 移植时就记在案（`#mmim` 末节），**但量级从未测过**，而它适用于我们移植的每一个模型，是 `#systematic-bias` 唯一还没排除的可测候选。

`--select-reduction mmsa` 复现它的量（默认路径不变，LF-LSTM 哈希仍是 `172967c7dced83b2`）。两组消融，各 10 seed，除选择口径外与验收组完全相同。

### 结果

配对检验（同实现、同 seed、同 RNG 流，**这里配对前提成立**，与跨实现的情况相反）：

| | MAE 平均差 | t (df=9) | 逐 seed | Corr 平均差 | t |
|---|---|---|---|---|---|
| TFN (bs 32) | **−0.0161（更好）** | **−2.53** | 7 负 / 1 正 / 2 零 | +0.0014 | +0.52 |
| MFN (bs 128) | +0.0037 | +1.00 | 1 正 / **9 零** | −0.0015 | −1.00 |

### 机制：末批超权，幅度由 batch size 决定

一开始猜的是"round 让微小改善不可见 → 早停更早"。**猜反了**：MMSA 口径让 TFN 选到**更晚**的 epoch（平均 11.5 → 16.8，跑到 19.5 → 24.8）。

真正起作用的是超权。验证集 229 条：

末批实得权重是 `1/批数`，应得 `末批样本数/229`，两者之比即超权倍数：

| | 批数 | 末批 | 实得 | 应得 | 超权 |
|---|---|---|---|---|---|
| TFN, bs 32 | 8 | 5 条 | 0.125 | 0.0218 | **5.7×** |
| MFN, bs 128 | 2 | 101 条 | 0.500 | 0.441 | 1.13× |

MFN 的验证集只切成两批，逐 batch 与逐样本几乎是同一个量——**9/10 个 seed 选到完全相同的 epoch**，这正是"效应应当消失"的地方，它消失了。

### 对系统性偏置的意义

**这个候选被排除为统一解释**：方向不一致（TFN 更好、MFN 更差），且在大 batch 的模型上根本不起作用。它是一个真实的、可达 0.016 MAE 的效应，但**只在小 batch 模型上**。

顺带解释了一部分 TFN 的缺口：TFN 对表差 0.0038 MAE，而这一个协议差异就值 0.0161——**缺口小于单个已知协议差异的量级**。

### 为什么不采用它

采用能让 TFN 更接近表。**但这是拿一个更差的量去换分数**：末批超权是缺陷不是特性，而 MFN 上它还使结果变差。**朝目标调协议与朝目标调超参是同一件事**，本项目禁止。记录为已量化的已知差异，判据不变。

**给下一个人的可检验预测**：超权倍数**不是 batch size 的单调函数**——它由余数 `229 mod bs` 决定。bs 16 与 bs 32 的末批都是 5 条，但批数不同（15 对 8），所以超权是 **3.1× 对 5.7×**；bs 128 的余数是 101，超权几乎消失。

若机制正确，效应大小应跟着这个倍数走，而不是跟着 batch size 走：MISA / Self-MM / MulT / text_bert 用 bs 16（3.1×），应弱于 TFN 而强于 MFN。

### 预测的验证结果（2026-08-01）：方向成立，精度有限

选 MISA 与 Self-MM 验证——它们的训练协议与 TFN/MFN 同构（200 轮上限、patience 8、无 lr 调度器）。**MulT 与 text_bert 被排除**：前者用同一个量做 plateau 调度，换归约会同时改变两件事；后者只跑 12 轮，选择空间太小。

检验量在看结果之前定死为**改变了所选 epoch 的 seed 比例**：机制只预测"选择会变"，不预测变好还是变坏（TFN 变好、MFN 变坏，方向本就随机），而 MAE 的变化量还取决于验证曲线形状。

| 模型 | batch | 超权 | **epoch 变动** | MAE 平均差 | 配对 t |
|---|---|---|---|---|---|
| MFN | 128 | 1.13× | 1/10 | +0.0037 | +1.00 |
| **Self-MM** | **16** | **3.05×** | **2/10** | −0.0022 | −1.13 |
| **MISA** | **16** | **3.05×** | **4/10** | +0.0017 | +0.31 |
| TFN | 32 | 5.73× | 8/10 | −0.0161 | −2.53 |

**方向成立**：两个 bs 16 的模型都落在 MFN 与 TFN 之间，四个点按超权排序单调（1 → 2, 4 → 8）。

**但精度有限，必须一起读**：

1. **同样 3.05× 的两个模型相差一倍**（2/10 对 4/10）。超权解释趋势，不解释模型间差异——验证曲线的平坦程度显然也在起作用。
2. **n=4，不能做显著性声明。** Spearman ρ=0.95 看着漂亮，但 n=4 时只有 ρ=1.0 才够 p<0.05。这是四个点的一致方向，不是一个被检验的定律。
3. MAE 的效应量**不单调**（MISA 的 |ΔMAE| 比 MFN 还小）——这正是事先不拿它当检验量的原因，如果当时用了它，同一批数据会得出"预测失败"的结论。**检验量必须在看数据前定死**，这条本身比预测的结论更值得记住。

## <a id="table-predates-config"></a>那张表比现在的超参早两年半——但这不解释缺口（2026-07-31）

### 起因

跨模型聚合判据把 `#mmsa-all-eleven` 的观察变成了一个数：**MMSA 自己的代码对自己发表的表，MAE Z=+6.49、Corr Z=+9.11**，与我们对表的落后同量级（+7.52 / +6.96）。

两份互不相干的实现、同时、同方向差同一张表。可疑的就不再是实现。

### 查到的事实

`git log -- results/result-stat.md`：九个模型的行写于 **2021-05-06**（`9a87fa0`），CENET 与 TETFN 两行是 **2023-12-20** 追加的。而 `models/`、`trains/`、`config/` 自 2021-05-06 起有 **37 次提交**。

逐模型比 2021-05-06 的 `config_regression.py` 与今天的 `config_regression.json`（MOSI 段）：

| 模型 | 变了什么 |
|---|---|
| TFN | text_out 128→32、post_fusion_dim 32→64、hidden 中间维 16→32、dropout 全变、lr 5e-4→1e-3 |
| LMF | batch 32→64、rank 4→3、hidden (256,…)→(128,16,128)、weight_decay 1e-4→5e-3 |
| MFN | hidden (128,…)→(256,32,256)、四个子网 dropout 全变、batch 64→128、lr 1e-3→2e-3 |
| Graph-MFN | memsize 128→300、hidden (64,…)→(256,32,256)、inner_node_dim 32→64、lr 1e-3→2e-3 |
| EF-LSTM | hidden 256→128、**层数 2→4**、dropout 0.3→0.5、batch 128→32、weight_decay 1e-4→5e-3 |
| LF-DNN | hidden (64,…)→(128,16,128)、post_fusion_dim 16→128、dropout 全变 |
| MulT | **batch 4→16**、**conv 核 1→5**、lr 1e-3→2e-3、patience 20→5、六个 dropout 全变 |
| MISA | batch 64→16、diff 0.3→0.1、sim 0.8→0.3、recon 0.8→1.0、**sp_weight 0→1** |
| **Self-MM** | **完全一致** |
| CENET / TETFN | **完全一致**（表行 2023-12-20 写入，配置自那以后未动） |

指标实现自 2021 起无实质变化（只有 dtype 与新数据集分支），**指标可比**。

### 直接检验：用 2021 的超参跑 TFN

假说是"表对应当年的超参，我们复现的是今天的"。这个假说可以直接测——`tfn_mosi_2021config`，10 seed，除超参外与验收组完全相同：

| | 当前配置 | **2021 配置** | MMSA 表 |
|---|---|---|---|
| MAE ↓ | 0.9511 ± 0.0249 | **0.9588 ± 0.0349** | 0.9473 |
| Corr ↑ | 0.6602 ± 0.0095 | **0.6578 ± 0.0163** | 0.6733 |
| Acc-2 (non0) | 0.7855 | 0.7816 | 0.7908 |

**假说被证伪。** 2021 的超参不但没更接近表，两个主指标都更差。表值既非今天的配置可达，也非当年的配置可达。

### 还剩什么没排除

- **表的产生方式没有记载。** `result-stat.md` 不声明 seed 集合、运行次数，也不说是否取自调参搜索（MMSA 有 `-t` 调参模式）。`#selection-bias-measured` 测过：按测试集挑超参能拿到 0.026 MAE / 2.3 个百分点——**足以覆盖这里全部缺口**，但没有证据说它就是这么产生的，不得当结论用。
- **特征文件版本未知。** 表用的 MOSI pkl 是否与我们这份逐比特相同，无法查证（不在仓库里）。
- **配置变更解释不了三个模型**：Self-MM、CENET、TETFN 的配置**从未变过**，而它们对表的 MAE z 是 +2.72 / +2.83 / +4.96。

### 结论与它的用法

这条调查**没有**找到缺口的成因，它把边界划清楚了：那张表的可达性无法从仓库本身确认，**两份独立实现同方向、同量级地差它**，而最直接的候选解释（超参代际）已被实测排除。

**因此"对表的系统性落后"不能读作我们的实现缺陷**，也不能读作"表是错的"——只能读作：**该目标的产生条件未公开，且不可由今天的仓库复现。** 判定所依赖的参照应当是能复现的那个（我们自己跑的 MMSA 代码），对表的比较继续报告，但不作为实现质量的证据。

**方法论**：这条差点变成一个漂亮的结论。commit 里已经写好了"表对应旧超参"的叙述，是那次 10-seed 实测把它拦下来的。**可检验的假说必须真的去测**——尤其是在它已经足够动听的时候。

### 补充：EF-LSTM 上，配置代际解释了大部分缺口（2026-07-31）

TFN 的 2021 配置更差，于是这条线本来要收在"配置代际不解释缺口"。**EF-LSTM 给出了相反的结果**，而它恰恰是聚合检验里权重最大的那一项（对表 MAE z=+10.04、Corr z=+11.75）。

2021 的 EF-LSTM/MOSI 是 **2 层** LSTM、hidden 256、dropout 0.3、batch 128、weight_decay 1e-4；今天是 **4 层**、hidden 128、dropout 0.5、batch 32、weight_decay 5e-3。

| | 当前配置（4 层，验收组） | 2021 配置（2 层） | MMSA 表 |
|---|---|---|---|
| MAE ↓ | 1.0717 ± 0.2095 | **0.9658 ± 0.0371** | 0.9488 |
| Corr ↑ | 0.6281 ± 0.0459 | **0.6510 ± 0.0180** | 0.6690 |
| Acc-2 (non0) | 0.7102 ± 0.1521 | **0.7800 ± 0.0126** | 0.7848 |
| **崩溃 seed** | **2/10** | **0/10** | — |

缺口从 0.1229 MAE 缩到 0.0170，标准差从 0.2095 收到 0.0371。

**这同时回答了 `#ef-lstm-collapse` 留下的开放问题**：23% 的崩溃率是**四层堆叠引入的**，不是"该架构+配置本身的性质"这么笼统的东西。表所对应的那个两层配置，十个 seed 一个都不崩。

**判定不变，理由也没变**：验收组复现的是参照今天给出的配置，这是对的——我们对标的是今天能跑起来的那份代码。**不得**把验收组换成 2021 配置：那是看到结果之后去挑一个更好看的配置，正是本项目禁止的。

**它改变的是归因**：EF-LSTM 对表的缺口，主要不是实现问题，而是**被检验对象与参照值处在不同的配置代际**。`acceptance_status.json` 里那条裁定（"崩溃率是两份实现共有的配置性质"）仍然成立——MMSA 今天的代码同样崩——但要补上一句：**那个配置不是表所对应的配置**。

**与 TFN 合起来读**：配置代际的影响方向因模型而异（TFN 更差、EF-LSTM 好得多），所以它不是一个能统一解释系统性偏置的因素；但在个别模型上它可以主导整个缺口。**逐模型看，不要外推。**

## <a id="aggregate-verdict"></a>跨模型聚合判定：对可复现的基准，我们超过了它（2026-07-31）

`#systematic-bias`（2026-07-27）留下的待办——"给验收补一个跨模型的聚合统计量"——现在做完了。判据先立（[decisions](decisions.md) 2026-07-31，写在看任何数字之前），参照补到与我们同精度（14 个模型 × 10 seed，`docs/mmsa_code_runs_mosi.json`，140 次运行），然后测量。

### 三个视角，全部 n=10、同一套 SE 口径

| 被检验方 | 参照 | MAE Z | Corr Z | 判定 |
|---|---|---|---|---|
| **我们** | **MMSA 的代码** | **−2.21**（p=0.027） | **−3.17**（p=0.0015） | **超过** |
| 我们 | MMSA 的公开表 | +7.52 | +6.96 | 系统性落后 |
| MMSA 的代码 | MMSA 的公开表 | **+8.54** | **+11.28** | 系统性落后（**比我们更多**） |

第三行是关键的对照。此前 `#mmsa-all-eleven` 只能说"它的代码达不到它的表"，因为参照只有 n=5，SE 更宽、z 不可与我们的 n=10 直接比。补齐之后三行同口径：**MMSA 自己的实现对那张表的落后，比我们的还大。**

### 逐模型（我们 vs 它的代码）

MAE 上我们更差的只有 3 个：EF-LSTM（+2.78，见下）、MMIM（+1.09）、LF-DNN（+0.69）；更好的 11 个里 CENET（−2.67）、MulT（−2.16）、Graph-MFN（−1.77）最显著。Corr 同样是 3/14，最显著的是 TETFN（−3.58）与 CENET（−2.70）。

### 必须与判定同读的三条

1. **Stouffer 假设各模型独立，而它们不独立**——同一个 686 条测试集、同一批特征、同一套协议。这会低估方差、放大 |Z|，**对"超过"这个方向同样是放大**。
2. **更稳健的那个统计量没有过线**：符号检验 3/14，双侧 **p=0.057**。所以严格的表述是"**至少打平，方向一致地偏向我们**"，"超过"是判据按其定义给出的判定，不是一个稳健到不需要限定的结论。
3. **参照是 MMSA 的实现**，凡它偏离原论文之处，这份参照一并继承（CLAUDE.md 第 5 条）。"超过 MMSA 的实现"不等于"复现了原论文"。

### 那张表怎么办

两份独立实现同方向、同量级地差它，而它不声明 seed、运行次数、是否来自调参搜索，特征文件版本也无法查证；最直接的候选解释（超参代际）在 TFN 上被实测证伪、在 EF-LSTM 上却解释了大部分缺口（`#table-predates-config`）。

**结论：对表的落后不作为实现质量的证据**，继续报告，但判定以能复现的那个参照为准。

### 这条待办可以关掉了

`#systematic-bias` 记的现象（"11/12 项偏差，逐模型判据看不见"）是真的，成因不是我们的实现：**换一个能复现的参照，同一批模型的偏差方向就整体反了过来。**

---

## <a id="cross-machine-hash"></a>换机器后 LF-LSTM 的锚点哈希对不上：是机器，不是代码（2026-08-01）

迁到新实例后（GPU 从 RTX 5070 Ti 换成 RTX 5080，CPU 是 AMD Ryzen 7 7700），约定 6 的锚点哈希 `172967c7dced83b2` 对不上。这条记录把原因钉死，**并且给出以后再遇到同类情况的判定方法**。

### 现象

默认路径 `--model lf_lstm --seeds 42 --device cuda`（即 `reproduce_all.sh` 里 `lf_lstm_mosi_cuda` 那条）在本机得 `199f629bfb036c93`。CPU 路径同样对不上：得 `969d5a4579736f8f`，记录值是 `0201e2133a68f4e9`。

### 两个候选解释

1. **机器变了**——GPU 型号不同，归约顺序/kernel 选择随之不同
2. **代码变了**——committed 的两次运行都产于 commit `8ea438f1`，而 HEAD 是 `24c8004`。这中间 `src/msa/` 有 26 个 commit，其中 `repro.py`（含 `set_seed`）和 `trainer.py` 都有实质改动

单看哈希无法区分。**"换了机器所以对不上"是个太容易接受的解释，必须先把代码这条排掉**——约定 6 的全部价值就在于它能抓出无意的数值改动，而换机器恰好提供了一个把它糊弄过去的借口。

### 判定方法：2×2 交叉

用 `git worktree` 把 `8ea438f1` 签出到工作区外，用 `PYTHONPATH` 覆盖 editable 安装（`pip install -e .` 指向 `/workspace/msa/src`，不覆盖会跑到 HEAD 的代码），在**同一台机器**上跑新旧两份代码 × CUDA/CPU 两个设备：

```bash
git worktree add --detach /tmp/msa-8ea438f 8ea438f1
ln -s /workspace/msa/datasets /tmp/msa-8ea438f/datasets     # 数据集按 PROJECT_ROOT 相对定位
cd /tmp/msa-8ea438f && PYTHONPATH=/tmp/msa-8ea438f/src /workspace/msa/.venv/bin/python \
    scripts/train.py --model lf_lstm --seeds 42 --device cuda --run-group old_code_cuda
# 验证确实跑的是旧代码：python -c "import msa; print(msa.__file__)"
```

| 代码 | 设备 | 机器 | 预测哈希 | best_epoch | test MAE |
|---|---|---|---|---|---|
| HEAD `24c8004` | CUDA | 本机 5080 | `199f629bfb036c93` | 23 | 0.9318 |
| **`8ea438f1`** | CUDA | 本机 5080 | **`199f629bfb036c93`** | 23 | 0.9318 |
| HEAD `24c8004` | CPU (8 线程) | 本机 7700 | `969d5a4579736f8f` | 4 | 0.9967 |
| **`8ea438f1`** | CPU (8 线程) | 本机 7700 | **`969d5a4579736f8f`** | 4 | 0.9967 |
| `8ea438f1`（记录值） | CUDA | 旧机 5070 Ti | `172967c7dced83b2` | 13 | 0.9543 |
| `8ea438f1`（记录值） | CPU (8 线程) | 旧机 | `0201e2133a68f4e9` | 11 | 0.9800 |

### 结论

**同机器上，HEAD 与 `8ea438f1` 在两个设备上都逐比特相同。约定 6 成立——`8ea438f1 → 24c8004` 之间数值行为未变。** 差异全部来自机器。

两个设备各自独立给出同一结论，这一点重要：如果只在 CUDA 上做这个对照，仍无法排除"代码改动恰好只在 GPU 上显形"。

（`investigations.md#612` 记过一次改 `repro.py` 后验哈希未变。这次的对照把那个结论从当时的那个 commit 一路延长到了 HEAD，中间的 25 个 commit 一并覆盖。）

### 意外发现：CPU 也不跨机器复现

`migration.md` 与 `README.md#复现性` 原先都称"CPU 钉住线程数即可跨机比对"。**实测不成立**：

- 同代码（`8ea438f1`）、同 torch `2.11.0+cu128`、同 numpy `2.5.1`、同 `--num-threads 8`、同 `deterministic=True`
- 两台机器的 CPU 哈希不同

**最可能的解释是 CPU 指令集分派**——torch 的 CPU kernel 按运行时检测到的 ISA 选择向量化实现，本机是 Ryzen 7 7700（有 `avx512_vnni` / `avx512_bf16` / `avx512_vbmi2`）。**未进一步验证**：旧机器的 CPU 型号没有记录进 `result.json`（`env` 只存 `platform` 与 `cpu_threads`），无从对照。

**留给下一个人的开口**：若要确证，可用 `ATEN_CPU_CAPABILITY=default` 强制两边降级到非向量化实现，看是否收敛到同一哈希。

> **2026-08-01 补：`env` 已经开始记了。** 新增两个字段：`cpu`（型号，Linux 读 `/proc/cpuinfo`、macOS 读 `sysctl`）与 `cpu_capability`（`torch.backends.cpu.get_cpu_capability()`，即 torch **实际**分派到的向量化 kernel 集合——型号只是间接暗示它，这个才是假设本身讲的东西）。本机记录为 `AMD Ryzen 7 7700 8-Core Processor` / `AVX512`。
>
> **这救不了这次的排查**：旧机器的两个值都没有记录，也已无法取得，所以上面那条假设**仍然是未验证状态，且永远不会被验证**。补这个字段是为了下一次换机器时不必再写这段话。

**这是本条留下的通用教训**：`env` 少记一个决定结果的东西，代价要到想复查的那天才付，而那天往往已经晚了。transformers 版本是同一个教训的另一个实例（`decisions.md` 2026-08-01）。

### 差异的量级：early stopping 会放大它

不是"末位几个 bit"级别的出入：

| | 旧机 5070 Ti | 本机 5080 | 差 |
|---|---|---|---|
| best_epoch | 13 | 23 | 收敛轨迹完全不同 |
| test MAE | 0.9543 | 0.9318 | 0.0225 |
| test Corr | 0.6477 | 0.6541 | 0.0064 |

机制是**微小数值差被模型选择放大**：逐 epoch 的 valid MAE 差在小数点后若干位，但一旦某一轮的排序被翻转，选出的就是另一个 checkpoint，测试集预测随之整体改变。

0.0225 MAE 在 LF-LSTM 的 seed 间标准差（0.039）之内，所以**不构成结论层面的问题**；但它说明"换机器只会有细微出入"这个说法在单 seed 上是不成立的——这正是约定 1（不接受单 seed 数字）保护的东西。

### 对约定 6 的影响

**锚点哈希绑定机器。** `172967c7dced83b2` 在本机失效，且无法通过任何代码改动恢复。要继续把它当验收锚点用，得在本机重立基线并**在文档里标注它绑定哪台机器**。

**不要**因为哈希对不上就去改代码——那是朝目标调参的一种形态。判定顺序永远是：先用 2×2 排除代码，再谈机器。

### 已排除，不要再查

- **代码变更**——2×2 交叉已排除，两设备独立同向
- **库版本**——torch / numpy 两机完全相同（都在 `result.json` 的 `env` 里）
- **线程数**——CPU 两边都是 8；CUDA 那次记录是 12 vs 本机默认 8，但 CUDA 的差异在 CPU 对照里独立复现了，与线程数无关
- **非确定性**——本机 `check_repro.py` 在 lf_lstm 与 tfn 上连跑两次均逐比特一致，本机内部是确定的

### 连带发现：几道闸门在新机器上没有真正生效

排查过程中顺带确认的，与本条同源：

1. **`check_all.sh` 不含 `check_reproduction.py`**。它跑的是 `check_repro.py --epochs 1`（同机连跑两次是否一致），不是"重训预测 vs git 里 committed 的"。所以**九道闸门全绿不代表旧数字在本机可重现**——它们读的是 committed 的 `result.json`，没有重新训练。这次的哈希失配正是全绿之后才发现的。
2. **`check_reproduction.py` 本身也不重新训练**。它只比对工作树里现成的 `result.json` 与 git 版本；工作树干净时必然通过。真正会重训的是 `reproduce_all.sh`，它跑完才调用这个脚本。
3. **`check_almt_equivalence.py` 在无 MMSA 检出时 SKIP 但记 PASS**（`check_all.sh` 里有注释说明是有意为之，为了让 fresh clone 走绿）。代价是**约定 5 里那道最关键的检查在新机器上静默失效**——它当初正是抓出"错误 ALMT 指标全面优于参照却验收通过"的那一道（`#almt-better-than-reference`）。新机器上要恢复它，必须先把 `/workspace/MMSA` 弄回来。

前两条合起来的含义：**`reproduce_all.sh` 在本机会红**，`docs/experiments.md` 的数字不会逐位重现。按 `migration.md` 这是硬件事实不是回归，但在本机重跑之前，那些数字在本机是**未经重训验证**的状态。

### 后续（2026-08-01）：第 3 条已修复，并且参照本身也绑定机器

`/workspace/MMSA` 与 `/workspace/mmsa_env` 已重建，重建步骤固化为 `scripts/setup_mmsa_reference.sh`（幂等，以等价检查收尾且不看退出码看结论——SKIP 时退出码也是 0）。**`check_almt_equivalence.py` 现在真的在比对**：183 个参数张量逐一形状匹配并复制，两侧输出 max abs diff `0.000e+00`。

重建时发现该脚本失效有**两个**原因，只修一个不够：除了 MMSA 检出不在，`SHIM` 还硬编码成某次会话的临时目录（`/tmp/claude-0/.../scratchpad/mmsa_shim`）。那个路径连在旧机器上都活不过会话结束，所以这道闸门**并不是从换机器才开始静默失效的**。现改为读 `MMSA_SHIM`，默认 `/workspace/mmsa_env/shim`，与 `mmsa_reference.py` 同一个约定。

**顺带得到一个与本条直接相关的事实：MMSA 自己的代码同样不跨机器复现。** 用重建后的环境跑 `mmsa_reference.py run tfn 42`（同一份 pkl、同 seed、同为 CUDA）：

| | 旧机 5070 Ti（`mmsa_code_runs_mosi.json` 记录值） | 本机 5080 | 差 |
|---|---|---|---|
| test MAE | 0.9305 | 0.9441 | 0.0136 |
| test Corr | 0.6789 | 0.6576 | 0.0213 |

量级与我们自己那次（0.0225 MAE）同级。**含义是：`docs/mmsa_code_runs_mosi.json` 与 `docs/experiments.md` 处在完全相同的处境**——都是旧机器的数字。因此聚合判据的两侧目前是**同机器配对**的，仍然自洽；但**只重跑其中一侧**（无论是我们的还是参照的）就会把两台机器混进同一个分布里，比两侧都不重跑更糟。这条应当与「本机要不要重立数值基线」那个待决项一起考虑：要重立就两侧一起重立。


## <a id="contrastive-screen-noise"></a>对比损失初筛第一轮：保留数正好等于噪声期望（2026-09-04）

### 现象

TFN（`--unaligned --lr 1e-3 --weight-decay 0`）+ 14 个对比损失各作为辅助项，λ=0.1，seeds 42-46，判据「任一指标超过对照组 1 个标准误」。对照组纯 L1：验证 MAE 0.8709 ± 0.0212、Corr 0.7160 ± 0.0146（SE 0.0095 / 0.0065）。

保留 4/14：supcon、hcl、simsiam、triplet。

### 为什么这 4 个不是结果

单侧 1-SE 阈值在零假设下的通过概率是 0.159。两个指标任一通过即保留，两指标近似独立时每候选约 0.29，**14 个候选的期望保留数是 4.1**。实际保留 4。

效应量（以对照组标准误为单位）：

| 候选 | Δmae | Δcorr |
|---|---|---|
| hcl | +0.19σ | +1.24σ |
| simsiam | −0.04σ | +1.13σ |
| triplet | −0.81σ | +1.06σ |
| supcon | +1.03σ | +0.39σ |

**没有一项超过 1.3σ，且每一项都是一个指标勉强过线、另一个为负。** `triplet` 是最清楚的例子：Corr +1.06σ 伴随 MAE −0.81σ——那不是改进，是旋转。

换用带 BH-FDR 校正的判据（`decisions.md` 2026-09-04）重算同一批数据：**28 个检验中 0 个在 q=0.10 下通过。**

### 唯一方向一致的信号是反向的

```
cpc   Δmae −2.13σ    p=0.916
rnc   Δmae −1.82σ    p=0.900
```

**「cpc 与 rnc 有害」的证据强于任何一项「有益」的证据。** 这两个恰好是这批里机制最讲究的两个——学出来的双线性打分、按连续标签排序的对比。至于为什么是它们，本轮没有查，不要当成已解释。

### 不要读成「对比学习在 MOSI 上无效」

λ=0.1 是拍的，只试了一个值。所有候选都可能被这个值压得太轻（无影响）或太重（干扰任务）。在单一 λ 下宣布无效与宣布「保留 4 个」同样不成立。**分开这两种可能的唯一实验是 λ 扫描**，已排入下一轮（λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0}）。

### 顺带确认：配对检验前提在这里也不成立

对照与候选共用 seeds 42-46，看似可以配对。实测逐 seed 相关性**中位数 −0.00，范围 −0.91…+0.86**。与第 1 阶段跨实现那次结论相同（`roadmap.md`「撤回改用配对检验」）。判据用不配对 Welch，相关性作为诊断量逐轮输出。

### 一个闸门的漏洞（已修）

`hcl` 在初筛第七组崩了：`torch.tensor(-1/temperature)` 建在 CPU 上，`clamp_min` 对 CUDA 输入报错。**`check_losses.py` 本就是为拦住这类问题写的，却没拦住——因为它所有张量都建在 CPU 上，从未碰过训练真正使用的设备。** 现已改为按 trainer 的方式解析设备并在其上运行，且打印所用设备，避免一次绿灯被读成覆盖了它从未见过的设备。


## <a id="screen-underpowered"></a>对比损失初筛的判据没有检出能力，且这在跑之前就可算（2026-09-04）

### 结论先说

λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0} × 14 候选，seeds 42-46，**140 个检验在 BH q=0.10 下拒绝 0 个**。

**但这个「0」不支持「对比学习无效」这个结论**，因为这个设计检不出它要找的量级：

```
对照组 MAE sd = 0.0212, n=5  →  差值标准误 0.0134, df=8

              α          t_crit    最小可检出效应(80% power)
无校正       0.05          1.86        0.0369 MAE
BH 校正    0.10/140        4.76        0.0759 MAE

实测最大效应 = 0.0174 MAE
参照尺度：TFN(0.8709) → ALMT(0.7404)，2017-2023 全部架构进步 = 0.1305 MAE
```

一个候选要在 BH 判据下通过，需要带来超过**整条技术演进线一半以上**的改进。

### 两个必须分清的诊断

第一版判据（1-SE 阈值）的问题是**没有多重比较校正**，保留数恰好等于噪声期望（见 [`#contrastive-screen-noise`](#contrastive-screen-noise)）。第二版（BH + 效应量地板）修了这个，但**功效问题第一版就有，不是校正造成的**——无校正的 MDE 已经是 0.0369，仍是实测效应的两倍多。

所以「放松阈值」不是解法。**只有加 seed 能解，而 n=5 的选择从一开始就不够。** 这个 MDE 只依赖 n 与 sd、不依赖任何结果，完全可以在跑之前算出来。没算是判据设计的疏漏，记在这里。

### 一个统计限定

140 个检验**共用同一个 5-seed 对照组**，彼此不独立，有效检验数远小于 140。BH 在正相依下仍然有效，但功效更低。所以「0/140」里的 140 是虚高的。

### 修正：撤回 `cpc` / `rnc`「有害」的归因

[`#contrastive-screen-noise`](#contrastive-screen-noise) 记过「cpc 与 rnc 有害的证据强于任何一项有益的证据」（λ=0.1 上 −2.13σ / −1.82σ MAE）。**λ 扫描不支持这个说法。**

`rnc` 在 λ=0.03 上是整个扫描里 p 最小的一格（Δmae **+0.0174**, p=0.090），而在 λ=0.1 上是最差的两个之一。同一个损失在相邻两个 λ 上从最差跳到最好——这是 λ 网格上的噪声形状，不是有结构的响应。那条归因是看单个 λ 得出的，现予撤回。

### 绑死一切的是数据集，不是判据

TFN 在 MOSI 上的 seed 间 MAE 标准差是 0.021，所以**这个骨干 + 这个数据集上任何 5-seed 的比较，MDE 都是 0.037**——包括后面自适应权重那个比较。要看到 0.02 量级的效应需要约 20 个 seed。

这与 `roadmap.md` 早已写下的判断是同一件事：**MOSI 测试集仅 686 条、判别力不足，主数据集应该是 MOSEI。** 这一轮等于用实验重新发现了项目自己记过的结论。

### 由此改变的做法

初筛就此冻结，不再加实验——它没能挑出候选，原因是样本量而非候选。组合项改为**按明说的规则选，不假装是显著性挑的**，20 个 seed 花在任务书真正问的问题上（自适应权重 vs 固定权重）。设计见 `decisions.md` 2026-09-04 两条。


## <a id="seed-batch-variation"></a>同一配置跨 seed 批次的极差是 0.0103 MAE，与要测的效应同量级（2026-09-04）

### 现象

纯 L1 的 TFN（`--unaligned --lr 1e-3 --weight-decay 0`），完全相同的配置与代码，三批 seed：

| seed 批次 | n | valid MAE |
|---|---|---|
| 42-46 | 5 | 0.8709 |
| 100-119 | 20 | **0.8803** |
| 120-139 | 20 | 0.8700 |

**极差 0.0103 MAE。** 这不是 seed 间标准差（那是 0.021-0.029），这是**20 个 seed 的均值之间**的差。

### 为什么这条最重要

第 2 阶段里所有候选的效应量都在 0.01-0.02 MAE。**这意味着任何单批 n=20 的比较，其批次噪声与要测的效应同量级。**

它直接制造了一个假线索：五臂实验里「四项组合优于纯 L1」+0.0131 MAE（p=0.045），在第三批 seed 上变成 −0.0015。回看即知——组合在两批上是 0.8672 / 0.8715（差 0.004），而纯 L1 是 0.8803 / 0.8700（差 0.010）。**差距来自对照臂的批次波动，不是来自组合。**

### 由此得到的操作规则

- **单批 n=20 上、0.01 量级的效应，不足以支撑结论。** 需要第二批独立 seed 重复，这次的预注册确认正是这样抓住它的。
- **不同实验之间不要跨 seed 批次比数字。** 已入库结果分属 42-51 / 42-46 / 100-119 / 120-139 四批，跨批次比较会把 0.01 的批次差读成效应。
- 若必须给单一最佳估计，合并批次（n=40）后 MDE 约 0.0144 MAE——仍然大于本阶段的全部观测效应。

### 与 roadmap 早已写下的判断是同一件事

`roadmap.md` 定主数据集用 MOSEI 的理由是「MOSI 测试集仅 686 条，判别力不足」。第 2 阶段用 500 多次训练把这条重新发现了一遍：**MOSI 上分辨不出 0.01-0.02 量级的差异**，无论判据怎么设计。


## <a id="contrastive-below-floor"></a>对比学习的效应在 MOSI 上低于 20-seed 实验的测量地板（2026-09-05）

### 这条改写了第 2 阶段全部结论的措辞

第 2 阶段在 TFN 上得到的负面结果（十四个候选、四项组合、三种自适应权重）都留着同一个未排除的竞争解释：**本协议与数据集能否测出一个已知有效的对比损失？** 现在有答案了：**不能。**

MMIM 的 InfoNCE + MI 项（原论文有消融支持，已在本仓库复现验收），开关开/关，n=20：

| 指标 | Δ (on − off) | p | d | MDE |
|---|---|---|---|---|
| mae | −0.0042 | 0.809 | −0.28 | 0.0120 |
| corr | −0.0010 | 0.668 | −0.14 | 0.0060 |

**已排除的替代解释：**

- **开关失效**——两臂 20/20 个 seed 预测哈希全不同，`model_kwargs` 记录了 `contrast=False`。
- **功效不足于这一轮**——MMIM 的 seed 标准差 0.0139（TFN 是 0.021），MDE 0.0120 是本阶段最紧的检出条件。
- **轮数上限截断**——训练轮数中位数 20/22，无 run 触到 40。

### 由此得到的结论改写

> **2026-09-05 补**：本节首次得出时诊断跑在 aligned 数据上（配置错误，见 [`#mmim-diagnostic-aligned`](#mmim-diagnostic-aligned)）。正确配置（unaligned）下重跑，裁定未变（Δmae −0.0046, p=0.861），故本节结论原样成立，依据已换为正确配置的 `mmim_diag_mosi_unaligned_*`。

**第 2 阶段在 MOSI 上的全部负面结果，措辞应为「不确定」而非「无效」。** 具体地：

- 「十四个对比损失均无收益」→ **在一个连已知有效对比项都测不出的设置里，测不出它们。**
- 「四项组合不优于纯 L1」→ 同上。
- **例外：自适应权重那一条仍然成立。** 它比较的是权重规则而非损失项，其对照臂（固定均等）与实验臂用的是同一组损失，MDE 0.0133 覆盖得住，且已排除「策略未执行」。它说的是「在这个组合上，调权不如不调」，这个结论不依赖组合本身是否有效。

### 与第 1 阶段的一条发现连起来

第 1 阶段记过：LF-LSTM 在 MOSI 上的 seed 间 MAE 标准差 0.039，**宽于文献里大多数「改进」**。本条是同一现象在辅助损失上的版本——**MOSI 上一个 20-seed 实验的测量地板约 0.012 MAE，而这一类方法的效应量在其之下。**

这不构成「文献里的对比学习结论是错的」这一断言：本项目没有测量各论文报告的效应量，也没有复现它们的完整协议。它构成的断言是：**在 MOSI 上、以本项目的协议、用 20 个 seed，这一类效应不可分辨。** 论文若以单次运行或未声明 seed 汇报，其报告的增益可能落在这条噪声带内——这是一个需要另做实验才能回答的问题，不是本条的结论。

### 下一步只能是数据集

不是换损失，不是换骨干（MMIM 已经是最强的一个），不是加 seed（要把 MDE 降到 0.005 量级需要约 115 个 seed/臂）。**是 MOSEI。**


## <a id="mosei-seed-noise"></a>MOSEI 的 seed 方差比 MOSI 小 7.9 倍，这把第 2 阶段的预算算式翻了过来（2026-09-05）

### 测量

LF-LSTM 基线、seeds 42-46、仓库默认配置，与 MOSI 的 `NOISE_FLOOR` 同一定义（测试集各指标的样本标准差，ddof=1）。

**MAE 的 seed 标准差：MOSI 0.0387 → MOSEI 0.0049。** 远好于按样本量平方根的推测（13 倍数据 → 约 3.6 倍），实际是 7.9 倍。

### 由此得到的检出能力

```
最小可检出效应 (MAE, 80% power, 单侧 α=0.05)
        MOSI      MOSEI
n= 5   0.0672    0.0085
n=10   0.0449    0.0057
n=20   0.0310    0.0039
```

（表中 MOSI 的值按 LF-LSTM 的 sd 算；第 2 阶段实测在 TFN 上是 n=20 → 0.0133，因为 TFN 的 sd 比 LF-LSTM 小。）

**MOSEI 上 5 个 seed 的检出能力已优于 MOSI 上 20 个 seed。** 第 2 阶段观测到的效应量范围是 0.0004-0.0174 MAE，n=20 的 MOSEI 地板 0.0039 把其中大部分覆盖了。

而且**单 seed 只慢 3.1 倍**（69s vs 22s），不是数据量的 13 倍。两件事合起来：**在 MOSEI 上重跑第 2 阶段，比在 MOSI 上做同样的事更便宜且更有分辨力。**

### 一个反常：`acc2_has0` 反而更差

它是唯一变差的指标（0.0274 vs 0.0101，其余全部变好 1.7-7.9 倍）。MOSEI 的中性（label 恰好为 0）样本比例远高于 MOSI，而 `acc2_has0` 把零标签折进负类，所以该指标在 MOSEI 上本身不稳——**原因属于指标而非数据量**。

按实测值填入 `NOISE_FLOOR_BY_DATASET`，未做平滑。地板是**放宽**判据的，把一个实测更宽的方差改小会让判据变严且不再对应真实噪声；反之也不能因为它"看起来不该这么大"就调它。它不是主判据指标（主判据是 mae/corr）。

### 顺带：MOSI 的 `NOISE_FLOOR` 有两个值与已入库的组对不上

按 `lf_lstm_mosi_cuda`（seeds 42-46）重新推导 MOSI 的七个值，**五个精确一致，两个不一致**：

| 指标 | 已入库组实测 | `NOISE_FLOOR` 登记值 |
|---|---|---|
| f1_non0 | 0.0120 | 0.0122 |
| acc5 | 0.0316 | 0.0299 |

`NOISE_FLOOR` 是「在任何验收运行存在之前」登记的（`check_acceptance.py` 注释），其后未按已入库的组重新推导。两个都不是主判据指标，mae/corr 精确一致。

**未改动。** 改 `NOISE_FLOOR` 就是改验收判据，不能作为一次顺手的清理来做——若要改，应先记决策再动手，且需说明为什么在 14 组已通过验收之后改动它是安全的。记录于此以免下一个人以为它是精确复现出来的。


## <a id="mmim-diagnostic-aligned"></a>协议诊断跑在了错误的数据设置上（2026-09-05）— 已作废重跑

### 错误本身

`scripts/mmim_diagnostic.py` 的第一版**没有传 `--unaligned`**，于是整个诊断跑在 aligned 数据上。

- MMSA 对 MMIM 的配置是 `need_data_aligned: false`
- 第 1 阶段验收 MMIM 的 `mmim_mosi` 组用的是 `--unaligned`
- **`CLAUDE.md` 约定 5 的核对清单里明确列着 `aligned/unaligned`**

逐项比对其余超参（lr、weight-decay、batch-size、grad-clip、patience），**只有这一项不同**；`--epochs 40` 与已验收组的 200 在 MOSI 上无差别（0/20 撞上限）。

### 为什么这不是小事

MMIM 的整体表现在两种设置下接近（测试 MAE 0.7452 unaligned 对 0.7393 aligned，在噪声内），所以不是灾难性的。但诊断问的是"**一个已知有效的对比项**能否被本协议测出"，而 MMIM 的 CPC 与互信息项作用在三模态融合表征上——aligned 时 audio/vision 是词级对齐的 50 帧，unaligned 时是 375/500 帧带真实长度。**这个差异恰好落在那些对比项作用的地方。**

因此第一版诊断测的是「MMIM 在 aligned 数据上的对比项」，不是「MMIM 如已发表」。由它推出的「第 2 阶段全部改写为不确定」，依据被削弱，需要用正确设置重新验证。

### 处理

- **aligned 那两组（`mmim_contrast_on` / `mmim_contrast_off`）保留原样**，不删不改名。它们诚实记录了跑过什么，`cli.run_group` 与目录名一致，将来还可以用来回答「aligned 与 unaligned 下诊断结论是否相同」。
- 正确的诊断改用把数据设置写进组名的命名：**`mmim_diag_{dataset}_{setting}_{arm}`**。一次配置写错的重跑会落到不同目录，而不是静默覆盖一个正确的结果。
- `report` 现在会**逐个 seed 核对 `result.json` 里的 `aligned` 字段**是否与本诊断声明的设置一致，不一致就拒绝出裁定。
- MOSEI 那次未完成的 aligned 运行（6/20）已删除——配置错误且不完整，留着只会被误用。

### 重跑结果（2026-09-05 当日）：裁定未变

正确配置（`--unaligned`）下重跑，**仍然测不出**，且两组数字高度一致：

| | Δmae | p | Δcorr | p | MDE |
|---|---|---|---|---|---|
| unaligned（正确） | −0.0046 | 0.861 | −0.0022 | 0.817 | 0.0105 |
| aligned（作废） | −0.0042 | 0.809 | −0.0010 | 0.668 | 0.0120 |

**所以这个配置错误必须修，但它没有改变结论。** 两件事都要说：修是必要的（诊断问的正是"已知有效的对比项能否被测出"，而对比项恰好作用在受该设置影响的融合表征上，不修就无法宣称测的是"MMIM 如已发表"）；而修完之后 [`#contrastive-below-floor`](#contrastive-below-floor) 的结论原样成立，现在建立在正确配置上。

保留 aligned 那两组的价值在此兑现：它让"数据设置是否改变诊断裁定"成为一个可以从已入库数据回答的问题，答案是不改变。

### 顺带记两条流程上的失误

1. **我在提交代码之前就把实验起来了。** 约定 3 要求进文档的数字来自干净工作树，那批 run 会带 `dirty=true`。已停止并清除输出，改为先提交再跑。
2. **`pkill -f` 的模式匹配到了发起它的那条命令本身**，把自己也杀了。停止后台训练要按 PID，不要按命令行模式。


## <a id="f32-sidecar"></a>特征读取改为 float32 内存映射，峰值 22.9GB → 0.5GB（2026-09-05）

### 问题

pickle 把音视频存成 float64，而所有消费者要的是 float32，于是 `MMSADataset` 同时持有两份：MOSEI unaligned 反序列化后 12.6GB，加上由它构建的 7.9GB 张量，**实测峰值 22.9GB**。本机 `free` 只有约 33GB（`available` 116GB），这个峰值把四次长任务喂给了内存看门狗——λ 扫描一次、MOSEI 诊断三次。

### 中间走过的一次弯路

第一次修法是在 `build_dataloaders` 末尾清掉 pickle 缓存。常驻降了，但 **`train.py` 每个 seed 调一次 `build_dataloaders`**，所以"一次峰值 + 全程常驻"变成了"**每个 seed 一次峰值**"。续跑只推进 1 个 seed 就被杀，正是这个改动的结果——**峰值频率涨了 20 倍**。

第二步把缓存挪到构造好的张量上（`_build_dataset`，按 (dataset, split, aligned) 缓存），峰值回到一次。仍然被杀。

### 根治

`src/msa/features.py`：把数据集真正需要的数组（已清洗 NaN/Inf、已转 float32、连续）写成 `.npy` 旁路文件，之后**内存映射**读取。一个 batch 只换入它索引到的那些行，其余留在可回收的 page cache 里。

```
                       峰值 RSS    稳态 RSS   加载耗时
改动前                  22.9 GB     8.5 GB      14s
构建旁路那一次           16.4 GB     0.6 GB      18s
之后每次（训练时）        0.5 GB     0.6 GB       1s
```

**构建过程本身也是分块的**，这是 `_copy_in_chunks` 存在的理由：朴素写法会同时持有源数组与目标，峰值和它要消除的那个一样大。按 4096 行填充目标后，构建峰值只剩 pickle 那一份（实测 16.4GB）。

### 两个设计决定

**失效判断用钉死的哈希，不重新计算。** `DatasetSpec` 已经记了每个文件的 sha256，`check_data --verify-files` 是核对它的地方；每次加载都去哈希 13.65GB 比加载本身还贵。所以旁路记录源文件的**钉死哈希** + 大小，新数据版本会改变钉死哈希从而触发重建；文件被暗中替换而 spec 未变则由 `check_data` 抓——那本来就是它的职责。mtime 有意排除：跨机器拷贝会改 mtime 而字节不变。

**跨 seed 共享数据集对象是安全的，这一点查过而非假设。** `MMSADataset` 构造后只读（只有 `__len__`/`__getitem__`/`feature_dims`）；唯一在训练中改写自己训练目标的模型（Self-MM）把它们存在模块 buffer 里。每个 seed 独有的是 sampler，仍在 `build_dataloaders` 里新建，故 batch 顺序仍只取决于 seed。

`torch.from_numpy` 对只读映射会告警"非可写"——那正是这里想要的性质，故按信息定向抑制并注明原因。

### 验证

约定 4 逐项过：不变量通过；**TFN 预测哈希仍是 `ead6d52481abff6c`**；十一道闸门全绿。`check_repro` 在这里是强验证——它在同一进程内连训两次，第二次正好是张量缓存生效的情形。

**代价：磁盘。** MOSEI 旁路 8.0GB、MOSI 0.3GB，余量从 25G 降到 **15G**。旁路不入库、按需重建，所以换机器时不必搬。


## <a id="mosei-contrast-harmful"></a>MOSEI 上协议有分辨力，而 MMIM 的对比项显著有害（2026-09-06）

### 这条改变了第 2 阶段的处境

MOSI 上诊断测不出，于是关于损失项的结论只能写「不确定」。MOSEI 上**测得出**——

| | Δmae | \|d\| | MDE | 双侧 p |
|---|---|---|---|---|
| MOSI | −0.0046 | 0.35 | 0.0105 | 0.279 |
| **MOSEI** | **−0.0159** | **1.39** | 0.0092 | **8.7e-05** |

**所以在 MOSEI 上，结果不再受分辨力限制。** 这意味着两件事：

1. **第 2 阶段那批「不确定」在 MOSEI 上是可以变成结论的。** 十四候选的初筛、四项组合、自适应权重，在 MOSEI 上重跑都有意义——而且更便宜（[`#mosei-seed-noise`](#mosei-seed-noise)：5 seed 优于 MOSI 的 20 seed）。
2. **「MOSI 上测不出」是关于 MOSI 分辨力的陈述**，现在有了对照才站得住：同一个检验、同一个模型、同一套判据，换个数据集就测出了 |d|=1.39。

### 效应方向与论文主张相反

对比项让 MMIM 在 MOSEI 上**变差**：MAE +0.0159、Corr −0.0195，两个指标双侧 p 都在 1e-4 量级。

**这不是可以直接当结论的。** 三条限定必须先消除：

- **本仓库的 MMIM 移植只在 MOSI 上验证过。** 第 1 阶段的验收、与 MMSA 代码的逐项对照、参照值，全部是 MOSI 的。这个移植在 MOSEI 上是否忠实**没有任何检验**——约定 5 要求的对照在 MOSEI 上一次都没做过。
- **MMSA 自己的代码没在 MOSEI 上跑过。** 参照环境已在本机重建（`.mmsa-reference/`），所以这件事做得到。
- 超参不是问题：MMSA 对 MMIM 的配置**在 MOSI 与 MOSEI 上逐字段相同**（已核对 `config_regression.json`），所以不存在"拿 MOSI 超参套 MOSEI"。

正确表述：**在本项目的 MMIM 实现与协议下，MMSA 配置的对比项在 MOSEI 上显著有害。** 不是「MMIM 的对比学习无效」。

### 报告逻辑的缺陷：单侧检验会把"有害"打印成"测不出"

预注册的检验是单侧、只问"是否变好"。MOSEI 上它返回 **p=1.0000**，脚本据此打印 `NOT DETECTED`——而实际效应是 |d|=1.39。**单侧 p 等于 1 恰恰是反方向压倒性显著的信号，却被当成了"没有效应"。**

这是判据设计与诊断目的之间的错位：判据问"有没有帮助"，而诊断问的是"**协议能不能分辨这一类效应**"——一个显著有害的效应对后者是肯定回答。

已加双侧检验并列报告，**加的是方向而不是更松的阈值**：预注册的单侧比较原样保留并照常显示。MOSI 重跑裁定不变（双侧 p=0.279 / 0.366）。

### 下一步（按优先级）

1. **在 MOSEI 上跑 MMSA 自己的 MMIM 代码**，与我们的两臂对照。这一步同时回答"移植是否忠实"和"有害是否是 MMSA 实现本身的性质"。参照环境已就绪。
2. 若移植确认忠实，则这条有害性本身是一个值得追的发现——它与原论文的消融结论相反，需要回到原作者实现确认是继承的还是二次实现引入的（约定 5 的要求）。
3. 十四候选初筛与自适应权重实验在 MOSEI 上重跑。


## <a id="mmim-mosei-vs-mmsa"></a>MOSEI 上与 MMSA 对照：三个发现，其中一个推翻了上一条的前提（2026-09-06）

按 [`#mosei-contrast-harmful`](#mosei-contrast-harmful) 定的第一优先级，在 MOSEI 上跑 MMSA 自己的 MMIM 作对照（seeds 100-109，n=10，与我们的臂同一批 seed）。

### 发现 1：MMSA 的 `contrast=False` 路径会崩溃

```
trains/singleTask/MMIM.py:186
UnboundLocalError: cannot access local variable 'train_loss_mmilb'
```

`train_loss_mmilb` 只在 `if self.args.contrast:` 分支内赋值（第 181 行），而第 186 行的日志语句无条件引用它。全文件只出现这两次，**且使用处是纯日志 f-string，不参与任何计算**。

结论：**MMSA 这条路径从未被执行过。** 这是它自身的 bug，与本项目此前记录的多处同类。

### 发现 2：两边的「关掉对比」不是同一个目标函数

MMSA（`trains/singleTask/MMIM.py:114-121`）：

```python
if self.args.contrast:
    loss = y_loss + alpha * nce - beta * lld
else:
    loss = y_loss

if i_batch > self.args.mem_size:
    loss -= self.args.beta * results['H']     # ← 在 contrast 判断之外
```

**熵项不受开关控制。** 我们的移植（`models/mmim.py`）在 `contrast=False` 时直接 early return，**不含熵项**。

所以「MMSA 关掉对比」与「我们关掉对比」优化的是不同的目标。这处偏离在第 1 阶段不可见——验收只跑过 `contrast=True`。

### 发现 3（最重要）：`contrast=True` 时我们与 MMSA 相差约 1 个标准差，且方向与 MOSI 相反

| | 我们 | MMSA 代码 | 差 | 双侧 p | d |
|---|---|---|---|---|---|
| **MOSEI** test MAE | 0.5774 ± 0.0139 | 0.5912 ± 0.0122 | **−0.0138** | 0.030 | −1.05 |
| **MOSEI** test Corr | 0.7258 ± 0.0095 | 0.7099 ± 0.0159 | **+0.0160** | 0.016 | +1.22 |
| MOSI test MAE（第 1 阶段） | 0.7452 ± 0.0225 | 0.7339 ± 0.0226 | +0.0113 | — | — |

**MOSI 上我们更差，MOSEI 上我们更好。方向翻转。**

已记录的那处系统性协议差异（[`#mmim`](#mmim) 末节：MMSA 的验证损失逐 batch 平均、且 `round(·,4)` 后比较）**解释不了这个**，因为它在 MOSEI 上应当更小：

| | valid 样本数 | 最后一批 | 超权倍数 | 占全部批次 |
|---|---|---|---|---|
| MOSI | 229 | 5 | **6.4×** | 1/8 |
| MOSEI | 1871 | 15 | 2.1× | 1/59 |

**已知差异在 MOSEI 上小得多，我们却反而变好。** 按约定「比参照『更好』要与『更差』同等力度排查」，这是一个未解释的差异，不能当作"我们实现得更好"收下。

### 由此产生的后果：上一条的前提被推翻

[`#mosei-contrast-harmful`](#mosei-contrast-harmful) 把「对比项在 MOSEI 上有害」列为待确认的发现，并把「跑 MMSA 对照」列为消除限定的第一步。这一步跑完的结果是：**限定没有被消除，反而更强了。**

- 我们的移植在 MOSEI 上**未被确认忠实**——与参照相差 |d|≈1.0-1.2，且已知协议差异解释不了。
- 这个差距**与要研究的效应同量级**（有害效应 |d|=1.39）。
- MMSA 侧的 on/off 对照**目前跑不了**：off 路径崩溃，且即使修好，它的 off 含熵项而我们的不含，两者不可比。

所以「对比项在 MOSEI 上有害」仍然是**关于本项目 MMIM 实现的陈述**，且现在多了一条：该实现在 MOSEI 上与参照有未解释的差异。

### 下一步的选项（需要决策，不要默认执行）

1. **查清发现 3 的来源。** 这是最有价值的一步——它同时影响有害性结论的可信度。可查方向：验证选择量在 MOSEI 上的实际影响（可直接测）、`update_epochs` 梯度累积、早停在 11 轮 vs MMSA 轮数上的差异。
2. **修 MMSA 的日志 bug 并跑 off 臂**（改动仅一行、只影响日志），同时**另跑一个把熵项也关掉的变体**以对齐我们的语义。两个变体都跑才说得清。
3. 若发现 3 查明为协议差异而非实现 bug，则有害性结论可在"本项目协议下"的限定内成立。


## <a id="mmim-mosei-selection-test"></a>检验：选择量差异能否解释 MOSEI 上与参照的 1 个标准差（2026-09-06）

### 已排除的（都很便宜，先做的）

| 候选 | 结论 |
|---|---|
| `update_epochs=2`（梯度累积） | **不适用**——它在 MMSA 配置里，而 MMSA 的 MMIM trainer 从不读它（grep 为空，移植注释早已记录） |
| 选中轮次差异 | **不是**——MMSA 总轮数中位数 12 / best 4，我们 11 / 3 |
| NaN/Inf 清洗差异 | **不是**——两个数据集的 unaligned 音视频**都没有非有限值**（0/604M、0/285M）。顺带修正：`_clean` 的注释说「MMSA 的音视频带 NaN/Inf」，在这两份数据上不成立 |
| 数据预处理 | **一致**——MMSA 只把 audio 的 `-inf` 置零（该数据上是空操作），长度字段读的是同一份 `audio_lengths` |
| 三个学习率 | **一致**——MMSA 的 main 1e-3 / bert 5e-5 / mmilb 1e-3、decay 1e-4，我们的 `param_groups` 逐项对应 |

### 待检验的假设

差异在**方向**上随数据集变（MOSI 我们差、MOSEI 我们好），但**量级都是约 1 个标准差**。已记录的选择量差异恰好是这种形状：MMSA 的验证损失是**逐 batch 平均**且 `round(·,4)` 后比较，早停因此**更钝**。

> **钝的选择在小验证集上可能反而更稳**（MOSI valid 仅 229 条，锐利选择容易过拟合验证集），**在大验证集上则吃亏**（MOSEI valid 1871 条）。

`--select-reduction mmsa` 正是 `round(sum(batch_maes)/len(batch_maes), 4)`，可直接复现 MMSA 的选择量。

### 预测（在运行之前定死）

我们的 MMIM 在 MOSEI 上、contrast=True、seeds 100-109，改用 `--select-reduction mmsa` 重跑：

- 当前（sample 选择）test MAE **0.5774**，MMSA 是 **0.5912**，差 −0.0138。
- **若选择量是原因**：MAE 应向 0.5912 移动，即变差约 0.014，残余差距应小于 0.005。
- **若不是**：MAE 移动应小于 0.005，残余差距仍在 0.010 以上。

组名 `mmim_diag_mosei_unaligned_on_mmsasel`。**这是诊断，不改变任何既有判据**。

### 结果：假设被否定，且否得很干净

| | test MAE | test Corr | best_epoch 中位数 |
|---|---|---|---|
| MMSA 代码 | 0.5912 ± 0.0122 | 0.7099 ± 0.0159 | — |
| 我们（sample 选择） | 0.5774 ± 0.0139 | 0.7258 ± 0.0095 | 3 |
| 我们（**mmsa 选择**） | **0.5774 ± 0.0139** | **0.7258 ± 0.0095** | 3 |

**移动 `+0.0000`**，远低于预测的 +0.014 阈值；残余差距仍是 −0.0138（p=0.030）。

两组结果不是"接近"，而是**逐比特相同**：`cli.select_reduction` 记录为 `mmsa`（开关确实生效），而 **10/10 个 seed 在两种规则下选中同一个 epoch**，预测哈希逐个一致。

原因与前面的算术吻合：MOSEI 的 valid 有 1871 条 / 59 个 batch，逐 batch 平均已约等于逐样本平均，`round(·,4)` 也基本不改变 argmin。**这处已记录的协议差异在 MOSEI 上是惰性的。**

### 又排除一项：熵项的批计数器与记忆库作用域

MMSA 的 `i_batch` 与 `mem_pos_*` 列表都在 `train_others` 内、**每个 epoch 重建**。我们的 `on_train_epoch_start` 调 `_reset_memory()`，同时把 `_batch` 归零。**一致**，非原因。

### 已排除清单（截至 2026-09-06）

`update_epochs` / 选中轮次 / NaN-Inf 清洗 / 数据预处理 / 三个学习率 / **验证选择量** / **熵项批计数器与记忆库作用域**。

**MOSEI 上 |d|≈1.1 的差距仍无解释。**

### 下一步应当是数值等价测试，而不是继续跑训练

约定 5 要求「重写而非转抄的模型，必须做权重复制的数值等价测试」。ALMT 有 `check_almt_equivalence.py`，**MMIM 没有**。

该测试能把问题一刀切开：复制权重后两侧前向输出若一致，则模型定义相同、差异只能来自训练协议；若不一致，则差异在模型本身。**它不需要训练，只需要一次前向**——比再跑任何对照臂都便宜且更有判别力。这正是 ALMT 那次抓出"指标全面优于参照却是错的"的工具。


## <a id="mmim-equivalence"></a>MMIM 数值等价测试：模型定义完全一致，差异在训练协议（2026-09-06）

### 结果

`scripts/check_mmim_equivalence.py`（已接入 `check_all.sh`）：按显式模块映射复制 **52 个参数张量**后，喂同一批输入：

| 输出 | 最大绝对差 |
|---|---|
| `M`（预测） | **0.000e+00** |
| `nce`（InfoNCE / CPC） | **0.000e+00** |
| `lld`（互信息下界） | **0.000e+00** |

**EQUIVALENT。** 不只是预测一致——**两个对比项本身也逐比特一致**，而它们正是 MOSEI 那个问题的对象。

### 为什么必须按名映射而不能按位置

MMSA 的注册顺序与我们不同两处：它先 `visual_enc` 后 `acoustic_enc`（我们相反），且把 `fusion_prj` 放在三个 CPC 头之后（我们放在之前）。ALMT 那次的位置盲拷在这里是错的。

按位置拷**恰好**会被形状检查抓住（audio 74 维、vision 35 维），但那是偶然——若两者维度相同就完全抓不住。测试因此逐模块按名复制，并核对"复制的张量数 == 两侧 BERT 之外的参数总数"，否则部分复制后的"一致"毫无意义。

### 这个结果如何改变前面的结论

[`#mmim-mosei-vs-mmsa`](#mmim-mosei-vs-mmsa) 记的是：我们在 MOSEI 上比 MMSA 好约 1 个标准差，原因不明，而这个差距与要研究的效应同量级，所以「对比项有害」不能归给 MMIM。

**等价测试把这一层限定拆掉了大半：**

- **模型定义相同**，所以 |d|≈1.1 的差距只可能来自训练协议，不是移植错误。第 1 阶段 MMIM 的验收不需要重新审视。
- **更要紧的是：on/off 对照本来就是协议内受控的。** 两个臂用的是同一套协议、同一个（现已证明与 MMSA 一致的）模型，唯一变量是 `contrast` 开关。**一处与 MMSA 的协议差异会影响绝对数值，但不会影响同协议内两臂之差。**

所以「MOSEI 上对比项使结果显著变差」现在可以说成：**MMIM 已发表架构的对比项，在本项目的训练协议下、在 MOSEI 上显著有害**（|d|=1.39，双侧 p=8.7e-05）。这比之前的表述强得多——不再有"也许我们的模型写错了"这一层。

### 仍然保留的限定

- **与 MMSA 之间的协议差异尚未定位。** 已排除：`update_epochs`、选中轮次、NaN/Inf 清洗、数据预处理、三个学习率、验证选择量、熵项计数器与记忆库作用域、**以及现在的模型定义**。余下候选都在训练循环层面（mmilb 辅助优化轮次、`ReduceLROnPlateau` 的 `when`、梯度裁剪次序、`drop_last` 等）。
- 因此**不能断言在 MMSA 自己的协议下也会出现同样的有害性**。MMSA 侧的 on/off 对照仍跑不了：其 `contrast=False` 路径崩溃，且修好后语义也不同（熵项不受开关控制）。


## <a id="mosei-screen-control-precision"></a>MOSEI 初筛：那个「几乎全体略微变差」的模式，是真的还是对照抽样的假象（2026-09-07）

### 观察

MOSEI 上 14 候选、λ=0.1、seeds 42-46：**BH 28 个检验拒绝 0 个**（对照 MAE 0.5318 ± 0.0040，n=5 的 MDE 0.0069，实测最大效应 0.0052）。这与 MOSI 那轮的「0」性质不同——**这里是效应确实小，不是看不见**。

但方向高度一致：

| | 比对照**差**的候选 | 符号检验双侧 p |
|---|---|---|
| MAE | **12/14** | 0.0129 |
| Corr | **13/14** | 0.0018 |

平均 Δmae = **−0.0022**。与 MMIM 自带对比项在 MOSEI 上 |d|=1.39 的显著有害**方向一致**。

### 为什么这个符号模式还不能当结论

**14 个候选共用同一个 5-seed 对照**，彼此通过对照的抽样误差相关。对照的 SE 是 0.0018，若它恰好抽到一个偏低（偏好）的样本，**全部 14 个 Δ 会一起偏负**。符号检验的独立性前提不成立，上面两个 p 不能按字面读。

### 诊断（预测先写死）

把对照扩到 **20 个 seed（42-61）**，组名 `screen_mosei_control_n20`，其余配置完全相同。SE 从 0.0018 降到约 0.0009。约 37 分钟。

- **若 20-seed 对照的 MAE 与 5-seed 的相差不到 0.001** → 对照不是侥幸，那个负向模式是真的：**辅助对比损失在 MOSEI 上系统性地轻微有害**。
- **若 20-seed 对照的 MAE 比 5-seed 高（更差）约 0.002** → 5-seed 对照抽到了好样本，模式是假象，正确表述回到「14 个候选与对照没有可检出差异」。

**这是诊断，不改动预注册的初筛判据。** 原 5-seed 对照组保留不动，新组另立，两者都入库。


## <a id="mosei-hyperparams-differ"></a>MMSA 对 14 个模型中的 13 个在 MOSEI 上用了不同超参（2026-09-09）

### 现象

为规划「架构演进在 MOSEI 上是否成立」而对比 MMSA 的 `config_regression.json`，逐模型 diff `datasetParams.mosi` 与 `datasetParams.mosei`：

| 模型 | 差异项数 | 模型 | 差异项数 |
|---|---|---|---|
| tetfn | **16** | graph_mfn | 8 |
| mult | **11** | lf_dnn | 5 |
| mfn | 9 | misa | 5 |
| self_mm | 8 | lmf | 4 |
| tfn | 6 | almt | 3 |
| ef_lstm | 6 | bert_mag | 2 |
| | | cenet | 1 |
| | | **mmim** | **0（完全相同）** |

差异不是小数点级的：MulT 的 `batch_size` 16→4、`learning_rate` 0.002→0.0005；TFN 的 `batch_size` 32→128、`learning_rate` 0.001→0.005、`text_out` 32→128、`post_fusion_dim` 64→16。TETFN 甚至把 `train_samples` 写进配置（1284→16326），根本不可跨数据集复用。

### 两个后果

**MMIM 侧的全部工作不受影响**——它是唯一两个数据集配置完全相同的模型，我们用的正是它。第 2 阶段所有 MMIM 结论成立。

**TFN 在 MOSEI 上的数字配置不对。** `screen_mosei_control*` 用的是 MOSI 的超参（lr 1e-3、默认隐层维度）。这个数字被用在了给老师的材料里，作为「损失工程的收益只是追平一个 2017 年模型」的对照，**该对照因此需要修正后重新评估**。

顺带核对清楚的第三点：第 1 阶段验收 TFN 用的是 `use_lengths=False mask_pooling=False`（**逐比特忠实**版），而 MOSEI 上跑的是默认值（仓库的改进版）。这一项**不是错误**——回答「简单架构能到什么水平」时应当用我们最好的 TFN，而非忠实复现 MMSA 缺陷的那一版。修正运行沿用默认值，只换超参。

`need_normalized: true` 出现在 tfn / lmf / lf_dnn / mfn 四个模型上。它**不是特征标准化，而是把音视频在时间维压成 1 帧**——本仓库早已记录（`storyline.md` 讨论 MFN 那节），且 TFN 的 `mask_pooling` 标志正是针对它的、有文档的有意偏离。**非新问题。**

### 对「架构对比」实验的影响

原估算「14 模型 × 5 seed ≈ 27 小时」只算了算力。**实际还需先把 13 个模型的 MOSEI 超参逐一移植到我们的 CLI 与 `model-arg` 上**——这是约定 5 要求的工作量，且移植错误会静默产出一张错的排序表。规划时必须把这部分算进去。
