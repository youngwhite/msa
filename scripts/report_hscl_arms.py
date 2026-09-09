"""汇总 HSCL 两臂的结果，并检验"发布版的选择规则值多少"。

HSCL 不落盘任何结果文件，指标只打在 stdout（见
`investigations.md#hscl-test-leak`），所以这里解析日志。它在每个"best"轮都打一遍
那个指标块、最后再打一遍，**取最后一次**才是它上报的数字。

两臂的训练轨迹逐轮相同（补丁没碰早停），唯一差别是哪一轮被上报，因此这里用配对
检验——与本项目别处用 Welch 非配对的理由正相反：那些地方是加了损失项、整条轨迹都
变了，这里是同一条轨迹上换了个读数点。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ("as_released", "clean")
PATTERNS = {
    "mae": re.compile(r"^MAE:\s+([0-9.]+)", re.M),
    "corr": re.compile(r"^Correlation Coefficient:\s+([0-9.]+)", re.M),
    "acc7": re.compile(r"^mult_acc_7:\s+([0-9.]+)", re.M),
    "acc5": re.compile(r"^mult_acc_5:\s+([0-9.]+)", re.M),
    "acc2_non0": re.compile(r"^Accuracy all/non0:\s+[0-9.]+/([0-9.]+)", re.M),
    "acc2_has0": re.compile(r"^Accuracy all/non0:\s+([0-9.]+)/", re.M),
}
LOWER_IS_BETTER = {"mae"}


def parse(log: Path) -> dict[str, float] | None:
    text = log.read_text(errors="replace")
    out = {}
    for name, pattern in PATTERNS.items():
        found = pattern.findall(text)
        if not found:
            print(f"    {log.name}: 日志里没有 {name}——运行可能中途死了", file=sys.stderr)
            return None
        out[name] = float(found[-1])       # 最后一次才是上报值
    return out


def main(directory: Path) -> int:
    arms: dict[str, dict[int, dict[str, float]]] = {}
    for arm in ARMS:
        folder = directory / arm
        if not folder.is_dir():
            print(f"没有 {folder}，先跑 scripts/run_hscl_arms.sh", file=sys.stderr)
            return 1
        per_seed = {}
        for log in sorted(folder.glob("seed*.log")):
            seed = int(log.stem.removeprefix("seed"))
            parsed = parse(log)
            if parsed is not None:
                per_seed[seed] = parsed
        arms[arm] = per_seed

    shared = sorted(set(arms[ARMS[0]]) & set(arms[ARMS[1]]))
    if not shared:
        print("两臂没有共同完成的 seed", file=sys.stderr)
        return 1
    for arm in ARMS:
        missing = sorted(set(shared) ^ set(arms[arm]))
        if missing:
            print(f"警告：{arm} 的 seed {missing} 不在配对集合里，已排除",
                  file=sys.stderr)

    print(f"HSCL / MOSI，seeds {shared}（n={len(shared)}）\n")
    metrics = list(PATTERNS)
    header = "".join(f"{m:>17}" for m in metrics)
    print(f"{'arm':<14}{header}")
    for arm in ARMS:
        cells = ""
        for metric in metrics:
            values = np.array([arms[arm][s][metric] for s in shared])
            cells += f"{values.mean():>10.4f} ±{values.std(ddof=1):<6.4f}"
        print(f"{arm:<14}{cells}")

    print("\n发布版的选择规则值多少（配对，同一条训练轨迹上换读数点）")
    print(f"{'metric':<12}{'as_released':>13}{'clean':>10}{'差':>10}"
          f"{'配对 p':>10}  方向")
    from scipy import stats
    for metric in metrics:
        released = np.array([arms["as_released"][s][metric] for s in shared])
        clean = np.array([arms["clean"][s][metric] for s in shared])
        difference = released - clean
        if np.allclose(difference, 0):
            print(f"{metric:<12}{released.mean():>13.4f}{clean.mean():>10.4f}"
                  f"{0.0:>10.4f}{'—':>10}  两臂逐 seed 完全相同")
            continue
        _, p_two = stats.ttest_rel(released, clean)
        better = ("发布版更好" if (difference.mean() < 0) == (metric in LOWER_IS_BETTER)
                  else "发布版更差")
        print(f"{metric:<12}{released.mean():>13.4f}{clean.mean():>10.4f}"
              f"{difference.mean():>+10.4f}{p_two:>10.4f}  {better}")

    print("\n预期方向是「发布版更好」——那个门槛只在测试集变好时才更新。若在某个指标上"
          "\n出现相反方向，说明该指标与被用作门槛的 test_loss 不同向，不构成反例。")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/hscl_arms")
    raise SystemExit(main(target))
