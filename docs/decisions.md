# 技术决策记录

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
