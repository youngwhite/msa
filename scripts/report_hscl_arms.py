"""汇总 HSCL 两臂的结果，并检验"发布版的选择规则值多少"。

HSCL 不落盘任何结果文件，指标只打在 stdout（见
`investigations.md#hscl-test-leak`），所以这里解析日志。它在每个"best"轮都打一遍
那个指标块、最后再打一遍，**取最后一次**才是它上报的数字。

检验用**非配对 Welch**，与本仓库别处一致。

这里原本用的是配对检验，前提是"补丁只改上报轮次、两臂轨迹逐轮相同"。**那个前提
是错的**：实测 seed 42 第 1 轮两臂就不同，而第 1 轮那行打印在任何 save 分支之前，
所以不是补丁造成的——HSCL 在固定 seed 下本身就不可复现，见
`investigations.md#hscl-nondeterministic`。因此两臂之差里混进了运行间噪声，配对的
前提不成立。
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

    print("\n发布版的选择规则值多少（非配对 Welch；差里含运行间噪声，见模块文档串）")
    print(f"{'metric':<12}{'as_released':>13}{'clean':>10}{'差':>10}"
          f"{'p 双侧':>10}  方向")
    from scipy import stats
    for metric in metrics:
        released = np.array([arms["as_released"][s][metric] for s in shared])
        clean = np.array([arms["clean"][s][metric] for s in shared])
        difference = released - clean
        if np.allclose(difference, 0):
            print(f"{metric:<12}{released.mean():>13.4f}{clean.mean():>10.4f}"
                  f"{0.0:>10.4f}{'—':>10}  两臂逐 seed 完全相同")
            continue
        _, p_two = stats.ttest_ind(released, clean, equal_var=False)
        better = ("发布版更好" if (difference.mean() < 0) == (metric in LOWER_IS_BETTER)
                  else "发布版更差")
        print(f"{metric:<12}{released.mean():>13.4f}{clean.mean():>10.4f}"
              f"{difference.mean():>+10.4f}{p_two:>10.4f}  {better}")

    print("\n方向不要凭机制推断——我推错过一次。发布版的接受集是 clean 的子集"
          "（要求验证改善**且**测试改善），\n所以它上报的轮次不晚于 clean，"
          "那个合取条件会让它提早冻结在训练不足的一轮上。门槛作用在测试 MSE 上、"
          "\n确实压低了所选轮次的测试 MSE，但论文报的是 MAE / Corr / Acc。"
          "两半各自的数字见 scripts/hscl_selection_trace.py。")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/hscl_arms")
    raise SystemExit(main(target))
