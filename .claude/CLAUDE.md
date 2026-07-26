# MSA — 给下一次会话的交接说明

多模态情感分析研究项目。**开工前先读 `docs/roadmap.md`**（研究方向与当前阶段）、`docs/decisions.md`（为什么这样做，含踩过的坑）、`docs/experiments.md`（全部实验数字）。

参考实现在 `/workspace/MMSA`（MIT，THUIAR）。我们移植它的模型定义，但训练循环、评测协议、复现性基础设施自己实现——原因见 roadmap「与 MMSA 的关系」。

## 环境

- venv 在 `.venv/`，所有命令用 `.venv/bin/python`
- GPU 是 RTX 5070 Ti（Blackwell sm_120），**PyTorch 必须是 cu128 及以上的轮子**，PyPI 默认版本没有 sm_120 kernel
- 数据集 `datasets/CMU-MOSI/`（不入库，879MB，sha256 记在 `DatasetSpec.file_sha256`）。MOSEI 尚未下载
- **换机器**：`bash scripts/setup.sh` 建环境 + 校验数据 + 跑闸门；完整步骤见 `docs/migration.md`

## 不可违背的约定

这些不是风格偏好，违反了会让结论失效：

1. **不接受单 seed 数字。** MOSI 上 LF-LSTM 的 seed 间 MAE 标准差是 0.039，宽于文献里大多数"改进"。汇报一律 `--seeds 42 43 44 45 46`，报 mean ± 样本标准差（ddof=1）。
2. **模型选择只看验证集**，测试集在选完之后只碰一次。
3. **进入文档的数字必须来自干净工作树**（`result.json` 里 `env.git.dirty == false`）。在后台批量跑实验时**不要改代码**——改了就得重跑。这个坑本项目已经踩过三次。
4. **改动 `msa/data.py`、`msa/metrics.py` 或任何模型后，先跑 `scripts/check_invariants.py`**（退出码非零即回归）。
5. **重构以预测哈希验收。** LF-LSTM 默认路径的哈希是 `172967c7dced83b2`，贯穿多次重构未变。若某次重构改变了它，要么是引入了 bug，要么必须解释清楚为什么该变。

## 常用命令

```bash
bash scripts/check_all.sh                      # 一条命令跑完所有闸门
.venv/bin/python scripts/train.py --list-models
.venv/bin/python scripts/train.py --model tfn --unaligned --seeds 42 43 44 45 46 \
    --lr 1e-3 --weight-decay 0                 # TFN 复现（MMSA 超参）
.venv/bin/python scripts/verify_runs.py        # 从落盘预测重算全部指标
```

四个闸门脚本：`check_data.py`（数据体检 + 泄漏检查）、`check_invariants.py`（指标/标签/padding/汇总口径）、`check_repro.py`（同设备连跑两次比哈希）、`verify_runs.py`（审计已落盘结果）。全部以退出码表示成败。

## 结果如何持久化

- `outputs/<group>/seed<N>/result.json` 与 `test_predictions.npy` **入库**（共 ~800KB），所以任何一次 clone 都能重新审计历史数字
- `best.pt` 不入库（584MB），需要时重跑即可——运行是逐比特可复现的
- 每次得到新结果，**同步更新 `docs/experiments.md`**，那是唯一的结论出处

## 加一个新模型

实现 `MSAModel` 契约（`forward(batch) -> {"M": ...}`，可选 `compute_loss` / `param_groups`），用 `@register_model("名字")` 注册，在 `src/msa/models/__init__.py` import 一次。**训练循环不需要改动**——这正是与 MMSA 的关键区别，它每加一个模型要复制一份 trainer。

## 当前进度

见 `docs/roadmap.md` 末尾的「当前进度与下一步」。
