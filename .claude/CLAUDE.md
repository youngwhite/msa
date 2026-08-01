# MSA — 给下一次会话的交接说明

多模态情感分析研究项目。**开工前先读 `docs/storyline.md`**——它是复盘与审计的主文档，按技术演进逐个模型记录"解决了什么问题 / 复现结果 / 判定 / 实现差异"。

其余：`docs/roadmap.md`（方向、验收标准、下一步）、`docs/investigations.md`（**排查记录，避免重复试错**）、`docs/decisions.md`（为什么这样做）、`docs/experiments.md`（全部数字）。

参考实现在 `/workspace/MMSA`（MIT，THUIAR）。我们移植它的模型定义，但训练循环、评测协议、复现性基础设施自己实现——原因见 roadmap「与 MMSA 的关系」。

## 环境

- venv 在 `.venv/`，所有命令用 `.venv/bin/python`
- GPU 是 RTX 5080（Blackwell sm_120，2026-08-01 起；此前是 RTX 5070 Ti，同属 sm_120），**PyTorch 必须是 cu128 及以上的轮子**，PyPI 默认版本没有 sm_120 kernel
- 数据集 `datasets/CMU-MOSI/`（不入库，879MB，sha256 记在 `DatasetSpec.file_sha256`）。MOSEI 尚未下载
- **换机器**：`bash scripts/setup.sh` 建环境 + 校验数据 + 跑闸门；完整步骤见 `docs/migration.md`

## 不可违背的约定

这些不是风格偏好，违反了会让结论失效：

1. **不接受单 seed 数字。** MOSI 上 LF-LSTM 的 seed 间 MAE 标准差是 0.039，宽于文献里大多数"改进"。汇报一律 `--seeds 42 43 44 45 46`，报 mean ± 样本标准差（ddof=1）。
2. **模型选择只看验证集**，测试集在选完之后只碰一次。
3. **进入文档的数字必须来自干净工作树**（`result.json` 里 `env.git.dirty == false`）。在后台批量跑实验时**不要改代码**——改了就得重跑。这个坑本项目已经踩过三次。
4. **改动 `msa/data.py`、`msa/metrics.py` 或任何模型后，先跑 `scripts/check_invariants.py`**（退出码非零即回归）。
5. **移植模型时必须同时对照 MMSA 与原论文的开源实现。** MMSA 是二次实现，已发现它与原作者配置不一致、以及自身的多处 bug。**只对照 MMSA 会忠实复现它偏离论文的地方**——MulT 的位置编码就是这样丢的（MMSA 默认关闭，原作者始终启用，我们照抄了）。完整的十一项核对清单与原始仓库获取方式见 `docs/investigations.md#protocol-faithful`。最常踩的：梯度裁剪、轮数上限、**梯度累积 `update_epochs`**、padding/池化、optimizer 的位置切片分组、**声明了但不生效的层（三种形态）**、参照实现是否关掉了论文里的结构件、`use_bert`、aligned/unaligned。**重写而非转抄的模型，必须做权重复制的数值等价测试**——ALMT 的错误版本指标全面优于参照却验收通过，是这条测试抓出来的（`#almt-better-than-reference`）。**比参照『更好』要与『更差』同等力度排查。**发现参照实现有 bug 时**必须回原作者确认是继承的还是二次实现引入的**——两者处理方式相反。
6. **重构以预测哈希验收。** LF-LSTM 默认路径的哈希是 `172967c7dced83b2`，贯穿多次重构未变。若某次重构改变了它，要么是引入了 bug，要么必须解释清楚为什么该变。**但这个哈希绑定产生它的那台机器**（RTX 5070 Ti + 当时的 CPU）——换机器后它必然对不上，且无法靠任何代码改动恢复。**换机器后哈希对不上时，不要直接归因于硬件**：先用 2×2 交叉排除代码（`git worktree` 签出旧 commit，在新机器上跑新旧两份代码 × CUDA/CPU 两个设备，同机器上逐比特相同才能说差异来自机器），然后在新机器上重立基线并标注它绑定哪台。方法与实测见 `docs/investigations.md#cross-machine-hash`。**绝不要为了让哈希对上去改代码。**

## 常用命令

```bash
bash scripts/check_all.sh                      # 一条命令跑完所有闸门
.venv/bin/python scripts/train.py --list-models
.venv/bin/python scripts/train.py --model tfn --unaligned --seeds 42 43 44 45 46 \
    --lr 1e-3 --weight-decay 0                 # TFN 复现（MMSA 超参）
.venv/bin/python scripts/verify_runs.py        # 从落盘预测重算全部指标
```

闸门脚本（全部以退出码表示成败）：

| 脚本 | 回答什么问题 |
|---|---|
| `check_data.py` | 数据是不是那份数据？split 有没有泄漏？（`--verify-files` 查 sha256） |
| `check_invariants.py` | 指标/标签/padding/汇总口径的实现对不对？ |
| `check_repro.py` | 同一台机器连跑两次，结果是否逐比特一致？ |
| `verify_runs.py` | 已落盘的指标能否由落盘预测重算出来？ |
| `check_reproduction.py` | **重新训练**得到的预测，与 git 里committed 的是否一致？ |

`bash scripts/reproduce_all.sh` 重跑 `docs/experiments.md` 里的全部实验（约 25 分钟）并自动做最后一项比对。**新实验进入文档时，必须同时在这个脚本里加一条**，否则下一个人无法重现它。

## 阶段性任务完成后：同步到远端

```bash
bash scripts/sync.sh          # 先跑全部闸门，全绿才推送；有一项失败就不推
```

远端是 `git@github.com:youngwhite/msa.git`。**用户已授权：每次阶段性任务审核通过后自动同步，不必逐次确认。** 闸门不过就不要推——远端不应该保存一个自己的检查都过不了的状态，因为下一个人（或下一台机器）是从远端开始的。

## 结果如何持久化

- `outputs/<group>/seed<N>/result.json` 与 `test_predictions.npy` **入库**（共 ~800KB），所以任何一次 clone 都能重新审计历史数字
- `best.pt` 默认训练完即删除（需要权重时加 `--keep-checkpoint`）；运行逐比特可复现，重跑即可拿回
- 每次得到新结果，**同步更新 `docs/experiments.md`**，那是唯一的结论出处

## 加一个新模型

实现 `MSAModel` 契约（`forward(batch) -> {"M": ...}`，可选 `compute_loss` / `param_groups`），用 `@register_model("名字")` 注册，在 `src/msa/models/__init__.py` import 一次。**训练循环不需要改动**——这正是与 MMSA 的关键区别，它每加一个模型要复制一份 trainer。

## 当前进度

见 `docs/roadmap.md` 末尾的「当前进度与下一步」。
