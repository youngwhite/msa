# 技术决策记录

**写给下一个人：这里记的是"为什么"，不是"是什么"。代码能回答后者，只有这里能回答前者。**

新增条目用下面的模板。**「已否决」和「复查条件」两栏不能省**——它们正是防止同一个方案被反复重新裁定的东西。

```
### [日期] 一句话结论
背景：什么问题触发了这个决定
选项：A / B / C
裁定：选了哪个
依据：证据（实测数字、代码位置），不是直觉
已否决：为什么不选其它的
复查条件：什么情况下这条结论需要重新评估
```

若某条结论后来被推翻，不要删除，在末尾追加「后续修正」。

## 索引

| 日期 | 决策 |
|---|---|
| 2026-07-25 | PyTorch 用 cu128 轮子；src layout；两套 Acc-2 口径；用 regression_labels |
| 2026-07-25 | 设备解析集中化；采样器私有 generator；CPU 线程数是复现性的一部分 |
| 2026-07-26 | 序列编码按真实长度取末状态；aligned 下三模态共用文本长度 |
| 2026-07-26 | 单一 trainer + 模型注册表；模型契约收窄为三件事 |
| 2026-07-26 | 汇总用样本标准差；指标对齐 MMSA 实现；运行记录写入 git commit |
| 2026-07-26 | 验收标准：10 seed + 标准误分级；参照 MMSA 表而非原论文 |
| 2026-07-26 | TFN 采纳不裁剪梯度与放开轮数上限；Acc-2 差距归因于参照值偏差 |


## 2026-07-25 基础环境

**PyTorch 用 cu128 轮子。** GPU 是 RTX 5070 Ti（Blackwell, sm_120），PyPI 默认的 CUDA 版本没有 sm_120 kernel，会在第一次 CUDA 运算时报 "no kernel image is available"。改用 `--index-url https://download.pytorch.org/whl/cu128`，实测 torch 2.11.0+cu128 正常。

**src layout + editable install。** `pip install -e .` 让 `scripts/` 直接 `import msa`，不用在每个脚本头部塞 `sys.path.insert`。

**特征 NaN 直接置零。** MMSA 的 audio/vision 特征在部分数据集上带 NaN/Inf。本次 MOSI 加载后检查全部 finite，`_clean()` 作为对 MOSEI 等数据集的防御保留。

**评测指标保留两套 Acc-2 口径。** 文献里 MOSI 的二分类准确率有两种算法：`non0`（丢掉 label==0 的中性样本，Zadeh 等）和 `has0`（非负 vs 负，全样本，Yu 等），数值能差 1-2 个点。两个都算，比较时不会对错基线。

**统一使用 `regression_labels` 作为监督信号，不用 `annotations` 字符串。** aligned 与 unaligned 两个 pkl 的 id 顺序、回归标签完全一致，但 `03bSnISJMiM$_$11`（label -0.5）在 aligned 里标为 Neutral、在 unaligned 里标为 Negative——annotations 字段有边界不一致，只作展示用。

**模型选择以 valid MAE 为准，early stopping patience=8。** 与 MMSA 官方基线一致，避免用 test 调参。

## 2026-07-25 设备兼容与复现性

**设备解析集中在 `msa/device.py`，其余模块不出现 `torch.cuda`。** `--device auto` 按 cuda > mps > cpu 选择；显式指定但不可用时抛错而非退回 CPU——静默降级会让计时和显存数据失去意义。`pin_memory` / `non_blocking` 只在 CUDA 上开启，MPS 和 CPU 上是无意义的（MPS 没有 pinned host memory 概念）。

**特征在 Dataset 里就 cast 成 float32。** MMSA 的 audio/vision 存的是 float64，而 MPS 后端不支持 float64，放到 device 上才转换会直接报错。

**打乱用采样器私有的 generator，不用 DataLoader 的 generator。** 踩到的坑：DataLoader 会从自己的 generator 里抽 worker 的 base_seed，抽取次数随 `num_workers` 和 `persistent_workers` 变化，于是 `num_workers=0` 与 `=2` 的打乱顺序不同、结果对不上。改成 `RandomSampler(generator=...)` 后，batch 顺序只是 seed 的函数，实测 0/2/4 workers 哈希一致。

**CPU 线程数是复现性的一部分。** CPU 归约按线程切分，1/4/12 线程给出三份不同结果。`--num-threads` 调 `torch.set_num_threads()` 并覆盖 `OMP_NUM_THREADS`，跨机器对比时必须显式传。

**跨设备类型不追求数值一致。** CUDA / MPS / CPU 的归约顺序与快速数学实现不同，强求一致意味着放弃全部硬件加速。保证的是"同设备内逐比特可复现"，并把设备写进 run_name 与 `result.json` 的 `env`，避免把不同设备的数字混进同一张表。

**每次运行落盘预测哈希。** `result.json` 里的 `test_pred_sha256_16` 是判断两次实验是否真的等价的最短路径——指标相同不代表预测相同。

## 2026-07-26 代码审查

**序列编码按真实长度取末状态，而不是 `h[-1]`。** aligned_50 里 70.4% 的步是 padding，且 BERT 对 `[PAD]` 输出的嵌入非零，audio/vision 虽是零填充但零输入同样会推动 LSTM 状态演化。改为用 `attention_mask` 长度 gather 最后一个真实步的输出。**这不是精度优化**：5 seed 消融显示掩码版反而略差（MAE 0.971 vs 0.954，约 0.5 个标准差，unaligned 上同样持平）。留它的理由是语义正确、对 padding 长度不变；旧行为保留为 `--ignore-lengths` 供对照。数据见 `docs/experiments.md`。

**aligned 模式下三个模态共用文本长度。** 已全量核对：aligned_50 中每个样本在 `attention_mask` 长度之后的 audio/vision 帧严格为零，说明它们与文本共享同一条词级时间轴。unaligned 模式则各用自己的 `audio_lengths` / `vision_lengths`。

**pkl 的 `classification_labels` 不用。** 它是 `sign(regression)+1` 的 3 分类（0/1/2 = 负/恰好零/正），与我们按四舍五入推导的 7 分类 `label_7` 不是一回事，名字容易误导。监督信号统一用 `regression_labels`。

**特征缓存按 resolve 后的路径做键。** `lru_cache` 直接以入参为键时，同一个文件用 `Path` 和 `str` 各传一次就会缓存两份 ~400MB 的 pickle。

**不变量检查固化成 `scripts/check_invariants.py`。** 指标实现、标签口径、padding 约定、掩码正确性、评测顺序、缓存行为、平凡预测下界——改动 `data`/`metrics`/模型后跑一次，退出码非零即回归。

## 2026-07-26 单一 trainer + 模型注册表

**一个训练循环服务所有模型。** MMSA 的 `trains/singleTask/` 是 14 个文件、2325 行几乎重复的 trainer（`TFN.py` 与 `LMF.py` 名字归一后仅差 14 行）——这正是各模型评测协议静默漂移的来源。这里模型只通过 `MSAModel` 契约定制行为，其余（选择指标、early stopping、checkpoint、provenance）全部共享，因此天然可比。

**模型契约刻意收窄成三件事**：`forward(batch) -> {"M": ...}`（键 `M` 必需）、`compute_loss`（默认 L1，多任务模型覆写）、`param_groups`（微调文本编码器时要更小的学习率）。trainer 对模型内部一无所知，加模型不需要碰 trainer。

**注册表用惰性导入。** `registry` 在模块层导入 `models` 会与"模型 import registry 来注册自己"形成循环，所以 `issubclass` 检查放在装饰器内部，`available_models()` 触发一次 `import msa.models`。

**重构以预测哈希验收。** LF-LSTM 默认路径重构前后哈希均为 `172967c7dced83b2`，5 seed 汇总也与重构前手工循环的数字逐位一致（MAE 0.9708±0.0346）——证明这次重构没有改变任何数值。这是重构该有的验收方式。

**多 seed 是 CLI 的默认形态。** `--seeds` 接受多个值并直接输出 mean/std/min/max；只给一个 seed 时会打印警告。理由见 `docs/roadmap.md` 动机第 1 条。

**输出目录改为 `outputs/<model>_<dataset>_<device>/seed<N>/` + 同级 `summary.json`。** 第 1 阶段要跑 6 个模型 × 5 seed × 多数据集，扁平命名撑不住。

## 2026-07-26 复现口径加固与可审计性

**汇总用样本标准差（ddof=1），不是 numpy 默认的 ddof=0。** 我们报的是"再跑几个 seed 会有多大波动"的估计，ddof=0 在 n=5 时低估 11%。本项目的核心论点正是拿这个波动去比对已发表的差距，所以低估等于把论点做强——必须用无偏估计。已按新口径重算全部记录（均值不变，仅 std 变大）。

**指标实现改为逐行对齐 MMSA 的 `metricsTop.py`。** Acc-7/Acc-5 改为「先 clip 后 round」的原顺序（与原先的「先 round 后 clip」在 20 万个点含全部 .5 边界上完全等价，已验证），便于两份实现逐行对读。新增 Acc-5，使可对比的列更多。

**`corr` 在任一侧恒定时返回 0 而非 NaN。** 原先只守卫了预测一侧；真值恒定同样会让 `np.corrcoef` 返回 NaN，而一个 NaN 会污染整组 seed 的均值。

**运行记录里写入 git commit 与 dirty 标记。** 没有这个，一个数字出自哪份代码就无法还原。`scripts/verify_runs.py` 把 dirty 列为警告而非错误——用未提交代码做实验是正常的，但**进入文档的数字必须来自干净工作树**。

**新增 `scripts/verify_runs.py`：从落盘预测重算全部指标、校验哈希、重新推导 summary。** 它能抓住两类事故：产物损坏，以及"改了指标或汇总口径却没重跑实验"。后者是复现性事故里最隐蔽的一种。

**`--select-on` 与 `TrainConfig` 参数在构造时即校验。** 拼错的指标名过去要等到第一个 epoch 结束才炸在 KeyError 上。

**训练损失按样本数加权。** 最后一个 batch 通常不满，按 batch 平均会过度加权它。MMSA 用于选模型的 `Loss` 正是按 batch 平均的——这是我们与其协议之间一处细微但真实的差异，记录在此以免日后误判为复现失败。

**`fit()` 在没有任何 epoch 产生可用分数时明确报错**，而不是让 `torch.load` 抛出令人费解的文件缺失。

## [2026-07-26] 验收标准用标准误分级，参照 MMSA 表而非原论文

背景：要求"复现不能比 MMSA / 原文差太多"，但"差太多"需要一个可执行的界线，且 MMSA 只报单个数字、没有方差。

选项：A 严格不低于报告值 / B 差距 ≤ 1 SE 视为通过 / C 按 SE 倍数分级（≤1 通过，1-2 标记待查，>2 未复现）

裁定：C，判定由 `scripts/check_acceptance.py` 执行。参照物用 MMSA 的 `result-stat.md`，原论文超参查阅记录但不作达标目标。

依据：我们自己 seed 的离散度是唯一诚实的标尺。TFN 的 MAE 差 0.0005（0.02 SE）去追毫无意义，而 Acc-2 差 4 SE 则确实指向问题——分级判据能区分这两种情况。参照原论文不可行：TFN(2017) 等用的是 CMU-SDK 的 GloVe 特征、部分还用了不同划分，比的是特征不是模型。

已否决：A —— 会诱导挑 seed 和事后调参凑数，而这正是本项目路线图批评的做法；B —— 单一阈值无法区分"值得看一眼"和"真有问题"。

复查条件：换到 MOSEI（训练集大 12 倍，σ 会明显变小）后，SE 阈值对应的绝对差距会收紧，届时需确认阈值是否仍合理。

## [2026-07-26] TFN 采纳不裁剪梯度、放开轮数上限

背景：TFN 10 seed 验收判定 Acc-2 低 4.3 SE，排查实现差异。

选项：A 保持我们的默认（裁剪 1.0、上限 40 轮）/ B 对齐 MMSA 的 trainer 行为

裁定：B。验收组改为 `--grad-clip 0 --epochs 200`。

依据：MMSA 的 trainer 完全不做梯度裁剪，且用 `while True` + patience 无轮数上限。改为不裁剪后 MAE 由 0.9585 → 0.9546（+1.0 SE → +0.8 SE，转为通过）；轮数上限在 10 个 seed 中仅影响 seed 50，但同属实现差异，一并消除。

已否决：A —— 复现阶段的目标是消除实现差异，我们的默认值再合理也不该带进参照对比。注意这两项只在**复现组**对齐；自建模型仍用我们的默认。

复查条件：若某个后续模型在不裁剪下发散，则该模型需单独记录并保留裁剪，不适用本条。

## [2026-07-26] TFN 的 Acc-2 差距归因于参照值的测试集选择偏差，不再追平

背景：排除七项假设后，TFN 的 Acc-2/Corr 仍低约 4 SE，而 MAE/Acc-7/Acc-5 打平。

选项：A 继续查实现 / B 在测试集上调参追平 / C 归因于参照值本身并记录

裁定：C。裁定写入 `docs/acceptance_status.json`，`check_acceptance.py` 按 `gap_explained` 处理。

依据：`MMSA/src/MMSA/run.py` 第 267-270 行，调参分支用 `do_test(dataloader['test'])` 评估候选配置，用验证集的那行被注释掉；而 `config_regression.json` 中 TFN 的每项超参都落在 `config_tune.json` 的搜索空间内。故其数字是"按测试集选出的超参在测试集上的成绩"。完整排查见 `docs/investigations.md#tfn-acc2`。

已否决：B —— 那等于把对方的方法论缺陷一起继承，与本项目的全部前提冲突；A —— 七项假设已逐一排除，继续查的边际收益低于把偏差量化（已列入待办）。

复查条件：若后续某个模型出现同样模式（Acc-2/Corr 低而 MAE/Acc-7 打平），**不得直接套用本条**，必须给出该模型自己的排查记录后再引用。
