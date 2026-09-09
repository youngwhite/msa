"""MOSEI→MOSI 迁移：预训练能不能把 MOSI 上的提升做出来。

阶段二的结论是 MOSI 分辨不了 0.01 量级的效应（MDE 0.013，批间极差
0.0103），而对比学习损失的效应就在那个量级以下。用户要的是 **MOSI 上的**
提升，所以这里换的不是损失而是数据量——先在 MOSEI 的 16326 条上预训练
self_mm，再迁到 MOSI。三臂：

  A transfer_mosi_scratch  MOSI 从头训（对照）
  B transfer_mosi_full     用 MOSEI 权重初始化，全部参数继续训
  C transfer_mosi_frozen   同上，但冻结 encoder / audio_model / vision_model

判据事先定死：**MOSI 测试集** MAE 与 Corr，Welch 单侧，BH q=0.10 覆盖
4 个检验（2 臂 × 2 指标），seeds 42-61。这里读测试集而非验证集，与本项目
约定 2 不冲突——模型选择在每次运行内部只看验证集，这一步是在模型都选完
之后做一次组间比较。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _stats import benjamini_hochberg, minimum_detectable_effect, welch_one_sided

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(42, 62))
CONTROL = "transfer_mosi_scratch"
ARMS = ["transfer_mosi_full", "transfer_mosi_frozen"]
METRICS = ("mae", "corr")
Q = 0.10


def read_test(group: str) -> dict[str, list[float]] | None:
    directory = ROOT / "outputs" / group
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    seeds = []
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        if result["env"]["git"]["dirty"]:
            print(f"  warning: {group}/{path.parent.name} from a dirty tree")
        for metric in METRICS:
            per_metric[metric].append(result["test"][metric])
        seeds.append(result["seed"])
    if sorted(seeds) != SEEDS:
        print(f"  warning: {group} has {len(seeds)} seeds, expected {len(SEEDS)}"
              " — excluded")
        return None
    return per_metric


def main() -> int:
    groups = {g: read_test(g) for g in [CONTROL, *ARMS]}
    if any(v is None for v in groups.values()):
        return 1

    print(f"MOSEI→MOSI 迁移，MOSI 测试集，seeds {SEEDS[0]}-{SEEDS[-1]}\n")
    print(f"{'arm':<26}{'test mae':>18}{'test corr':>18}")
    for group, per_metric in groups.items():
        cells = "".join(
            f"{np.mean(per_metric[m]):>11.4f} ±{np.std(per_metric[m], ddof=1):<6.4f}"
            for m in METRICS
        )
        print(f"{group:<26}{cells}")

    tests = []
    for arm in ARMS:
        for metric in METRICS:
            result = welch_one_sided(groups[arm][metric], groups[CONTROL][metric],
                                     metric)
            result.update(arm=arm, metric=metric)
            tests.append(result)
    rejected = benjamini_hochberg([t["p_one_sided"] for t in tests], Q)

    print(f"\nWelch 单侧 + BH q={Q}，{len(tests)} 个检验")
    print(f"{'arm':<26}{'metric':<8}{'improvement':>13}{'d':>8}{'p':>9}{'MDE':>9}  判定")
    for test, keep in zip(tests, rejected):
        mde = minimum_detectable_effect(test["pooled_sd"], len(SEEDS))
        verdict = "检出" if keep else "未检出"
        print(f"{test['arm']:<26}{test['metric']:<8}{test['improvement']:>13.4f}"
              f"{test['cohens_d']:>8.2f}{test['p_one_sided']:>9.4f}{mde:>9.4f}"
              f"  {verdict}")

    print("\n双侧同时报，方向本身是结论的一部分（见 investigations 里那次"
          "『未检出』盖住了 |d|=1.39 的有害效应）：")
    from scipy import stats
    for arm in ARMS:
        for metric in METRICS:
            _, p_two = stats.ttest_ind(groups[arm][metric], groups[CONTROL][metric],
                                       equal_var=False)
            delta = np.mean(groups[arm][metric]) - np.mean(groups[CONTROL][metric])
            print(f"  {arm:<26}{metric:<6}Δ={delta:+.4f}  p_two={p_two:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
