# MSA — 多模态情感分析

基于 MMSA 预处理特征的多模态情感分析研究代码。当前实现了 CMU-MOSI 的完整训练/评测链路、LF-LSTM 基线与 TFN 复现。代码在 CUDA / Apple Silicon (MPS) / CPU 上通用，且同设备同 seed 下逐比特可复现。

## 环境

Python ≥ 3.10。按平台装 PyTorch，其余依赖由 `pip install -e .` 带入。

```bash
python -m venv .venv

# NVIDIA GPU（本机 RTX 5070 Ti 是 Blackwell sm_120，必须 cu128 及以上，PyPI 默认轮子不含 sm_120 kernel）
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128

# macOS Apple Silicon（默认轮子即带 MPS）/ 纯 CPU
.venv/bin/pip install torch

.venv/bin/pip install -e .
```

## 数据

`datasets/CMU-MOSI/`（已下载，不入库）：

```
Processed/aligned_50.pkl     词级对齐，text/audio/vision 均为 50 帧
Processed/unaligned_50.pkl   未对齐，audio 375 帧 / vision 500 帧，另带真实长度
label.csv                    2199 条 video_id/clip_id/text/label/mode
```

特征维度：text 768（冻结 BERT）、audio 5、vision 20。划分 train/valid/test = 1284/229/686。

## 用法

```bash
# 数据体检：结构、划分、标签分布、NaN、与 label.csv 交叉核对
python scripts/check_data.py --dataset mosi [--unaligned]

# 训练：结果写到 outputs/<model>_<dataset>_<device>/seed<N>/，另有 summary.json
python scripts/train.py --list-models
python scripts/train.py --model lf_lstm --seeds 42 43 44 45 46        # --device 默认 auto
python scripts/train.py --model lf_lstm --device mps --num-workers 2   # Apple Silicon
python scripts/train.py --model lf_lstm --device cpu --num-threads 8

# 模型超参走 --model-arg（可重复），值按 Python 字面量解析
python scripts/train.py --model lf_lstm --model-arg modalities=t --model-arg dropout=0.3

# 不变量回归检查：指标实现、标签口径、padding 约定、掩码正确性、平凡预测下界
python scripts/check_invariants.py

# 复现性自检：同一设备上连跑两次，比对权重与预测的哈希
python scripts/check_repro.py --device auto

# 审计已落盘的全部结果：从预测重算指标、校验哈希与多 seed 汇总
python scripts/verify_runs.py
```

改动 `msa.data` / `msa.metrics` / 模型之后，请先跑 `check_invariants.py`（失败即退出码非零）。

`--device auto` 的优先级是 cuda > mps > cpu；显式指定但不可用时直接报错，不会静默退回 CPU（否则计时数据毫无意义）。

## 复现性

同一台机器、同一设备类型、同一 seed → 权重与预测逐比特一致。已实测：CUDA 与 CPU 均通过 `check_repro.py`，且 `--num-workers` 取 0/2/4 结果完全相同。MPS 未在硬件上验证过，请先在 Mac 上跑一次 `check_repro.py`。

做到这一点靠的是：

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在 `msa/__init__.py` 里于 torch 建立 CUDA context 之前设好
- `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True` + `cudnn.benchmark=False`
- 全部 RNG（random / numpy / torch / cuda / mps）统一播种；`num_workers>0` 时 worker 用 `worker_init_fn` 派生播种
- 训练集打乱使用采样器**私有**的 generator，不用 DataLoader 自己的 generator——后者还要为 worker 抽种子，抽取次数随 `num_workers`、`persistent_workers` 变化，会连带改变打乱顺序

**跨设备类型的数值不保证一致**（CUDA / MPS / CPU 的归约顺序不同），这是硬件事实，不是 bug；各设备各自内部可复现。**CPU 还须钉住线程数**：同一份代码在 1 / 4 / 12 线程下会得到三份不同结果，跨机器比对时请显式传 `--num-threads N`（该参数覆盖 `OMP_NUM_THREADS`，已实测）。

若某个算子在当前后端没有确定性实现（MPS 的算子覆盖比 CUDA 薄），用 `--deterministic-warn-only` 降级为告警，或 `--no-deterministic` 关掉——两种情况都会记录进 `result.json` 的 `env` 字段。

## 如何审计一个数字

论文里的表格数字应当可以一路追溯到产生它的代码和预测。本仓库的链路：

```
outputs/<group>/summary.json          多 seed 的 mean/std/min/max + 每个 seed 的值
outputs/<group>/seedN/result.json     超参、环境指纹、git commit 与是否 dirty、逐 epoch 历史
outputs/<group>/seedN/test_predictions.npy  按测试集顺序排列的预测（第 i 行对应第 i 个样本）
outputs/<group>/seedN/best.pt         被选中的那个 epoch 的权重
```

`python scripts/verify_runs.py` 会遍历全部运行，**从预测重新算一遍指标**并与 `result.json` 比对、校验预测哈希、再用每个 seed 的结果重新推导 `summary.json`。任何对不上的地方要么是产物损坏，要么是有人改了指标或汇总口径却没重跑实验——两种都会让结论失效。它还会标记出自 dirty 工作树的运行（那种运行无法从 commit 还原出代码）。

几条口径约定，写在这里以免误读：

- **`std` 是样本标准差（ddof=1）**，即对"再跑几个 seed 会有多大波动"的无偏估计。numpy 默认的 ddof=0 在 n=5 时会低估 11%，而本项目的核心论点正是拿这个波动去比对已发表的差距。
- **模型选择只看验证集**，测试集在选完之后只碰一次。`--select-on mae` 等价于 MMSA 的 `KeyEval: Loss`（其回归损失即 L1）。
- **指标按整个 split 的拼接预测计算**，每个样本等权。MMSA 用于选模型的 `Loss` 是按 batch 平均的，最后一个不满批会被过度加权——细微但真实的协议差异。

## 代码结构

```
src/msa/config.py         数据集路径与特征维度（DatasetSpec）
src/msa/data.py           MMSADataset / build_dataloaders
src/msa/device.py         设备解析、线程钉住、pin_memory 判定
src/msa/repro.py          播种、确定性开关、环境指纹
src/msa/metrics.py        MAE, Corr, Acc-7, Acc-2, F1（has0 与 non0 两套口径）
src/msa/registry.py       模型注册表：名字 -> 类
src/msa/trainer.py        唯一的训练循环 + 多 seed 汇总
src/msa/models/base.py    MSAModel 契约：forward(batch)->{"M":...} / compute_loss / param_groups
src/msa/models/functional.py 序列池化（按长度取末状态 / 掩码均值）
src/msa/models/lf_lstm.py 后期融合 LSTM 基线（按真实长度取末状态）
src/msa/models/tfn.py     Tensor Fusion Network（移植自 MMSA，MIT）
scripts/check_data.py     数据体检
scripts/check_invariants.py 不变量回归检查
scripts/check_repro.py    复现性自检
scripts/verify_runs.py    审计已落盘结果（从预测重算指标）
scripts/train.py          训练 + 评测（任意注册模型，多 seed）
docs/                     roadmap / decisions / experiments
```

### 加一个新模型

```python
from msa.models.base import MSAModel
from msa.registry import register_model

@register_model("my_model")
class MyModel(MSAModel):
    def __init__(self, text_dim, audio_dim, vision_dim, **kw): ...
    def forward(self, batch):            # batch 已在正确设备上
        return {"M": ...}                # 键 "M" 必需，形状 (batch,)
    # 有辅助损失就覆写 compute_loss；文本编码器要单独学习率就覆写 param_groups
```

在 `src/msa/models/__init__.py` 里 import 一次即完成注册，训练循环无需改动。**这是与 MMSA 的关键差别**：它每加一个模型要复制一份 trainer（14 个文件 2325 行，`TFN.py` 与 `LMF.py` 名字归一后仅差 14 行），评测协议因此会静默漂移。本仓库全部代码 1932 行，对比其 10399 行。

## 当前结果

| 模型 | 数据 | seed | MAE ↓ | Acc-2(non0) | Acc-7 |
|------|------|------|-------|-------------|-------|
| LF-LSTM | MOSI aligned | 42-46 | 0.971 ± 0.039 | 76.4% ± 1.2% | 34.6% ± 2.6% |
| TFN | MOSI unaligned | 42-46 | 0.948 ± 0.029 | 77.5% ± 1.4% | 34.9% ± 2.3% |
| TFN | MOSI unaligned | 1111-1115 | 0.954 ± 0.037 | 78.5% ± 2.0% | 35.6% ± 2.8% |
| *MMSA 报告的 TFN* | MOSI unaligned | 1111-1115 | *0.947* | *79.1%* | *34.5%* |

TFN 复现命令：

```bash
python scripts/train.py --model tfn --unaligned --seeds 1111 1112 1113 1114 1115 \
    --lr 1e-3 --weight-decay 0 --batch-size 32 --patience 8
```

**单 seed 的 MAE 波动可达 ±0.039（样本标准差），比不少论文声称的模型改进还大——请始终多 seed 汇报。** 详见 `docs/experiments.md`。
