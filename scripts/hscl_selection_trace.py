"""HSCL 两臂各自上报了第几轮，以及那一轮的验证 / 测试损失。

这个脚本存在的原因是我把机制方向想反了。发布版的门槛是
`if val_loss < best_valid: if test_loss < best_mae:`，我据此推断"泄漏必然抬高
论文数字"——**错了**，clean 臂逐 seed 更好。重新推导：

  clean      上报的是**验证损失最低**的那一轮（最后一次验证改善）
  as-released 要求验证改善**且**测试改善，接受集是 clean 的子集，
              因此它上报的轮次**不晚于** clean 的

所以那个合取条件让它**提早冻结**：测试损失一停止改善它就不再更新，上报一个
训练不足的轮次。门槛作用在测试 **MSE** 上、确实让测试 MSE 更好，但论文报的是
MAE / Corr / Acc-7，早冻结的代价大于偷看的收益。

这个脚本把上面这段话的两半都落到数字上：上报轮次的先后，以及测试 MSE 的方向。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

EPOCH = re.compile(
    r"^Epoch\s+(\d+) \| Time [\d.]+ sec \| Valid Loss ([\d.]+) \| Test Loss ([\d.]+)",
    re.M)
SAVED = "Saved model at pre_trained_models/MM.pt!"
ARMS = ("as_released", "clean")


def trace(log: Path) -> tuple[int, float, float] | None:
    """(上报轮次, 该轮验证损失, 该轮测试损失)。"""
    text = log.read_text(errors="replace")
    last_save = text.rfind(SAVED)
    if last_save < 0:
        print(f"  {log.name}: 一次都没保存过", file=sys.stderr)
        return None
    epochs = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), m.start())
              for m in EPOCH.finditer(text)]
    before = [e for e in epochs if e[3] < last_save]
    if not before:
        return None
    epoch, valid, test, _ = before[-1]
    best_valid = min(e[1] for e in epochs)
    return epoch, valid, test, best_valid, epochs[-1][0]


def main(directory: Path) -> int:
    print(f"{'arm':<13}{'seed':>5}{'上报轮':>8}{'该轮 valid':>12}"
          f"{'该轮 test':>11}{'全程最低 valid':>15}{'总轮数':>8}")
    collected: dict[str, dict[int, tuple]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for log in sorted((directory / arm).glob("seed*.log")):
            seed = int(log.stem.removeprefix("seed"))
            result = trace(log)
            if result is None:
                continue
            collected[arm][seed] = result
            epoch, valid, test, best_valid, total = result
            flag = "" if abs(valid - best_valid) < 1e-9 else "  ← 不是验证最优轮"
            print(f"{arm:<13}{seed:>5}{epoch:>8}{valid:>12.4f}{test:>11.4f}"
                  f"{best_valid:>15.4f}{total:>8}{flag}")

    seeds = sorted(set(collected[ARMS[0]]) & set(collected[ARMS[1]]))
    if not seeds:
        return 1
    released_epoch = np.array([collected["as_released"][s][0] for s in seeds])
    clean_epoch = np.array([collected["clean"][s][0] for s in seeds])
    released_test = np.array([collected["as_released"][s][2] for s in seeds])
    clean_test = np.array([collected["clean"][s][2] for s in seeds])

    print(f"\n配对比较（n={len(seeds)}）")
    print(f"  上报轮次      as-released {released_epoch.mean():.1f}"
          f"   clean {clean_epoch.mean():.1f}"
          f"   发布版更早的 seed 数 {(released_epoch < clean_epoch).sum()}/{len(seeds)}"
          f"，更晚 {(released_epoch > clean_epoch).sum()}")
    print(f"  该轮测试损失  as-released {released_test.mean():.4f}"
          f"   clean {clean_test.mean():.4f}"
          f"   差 {released_test.mean() - clean_test.mean():+.4f}"
          f"（负号=发布版的门槛确实压低了它所选轮次的测试 MSE）")
    print("\n上报轮次不晚于 clean 是这个规则的结构性后果，不是巧合：它的接受集是"
          "\nclean 接受集的子集。指标方向见 scripts/report_hscl_arms.py。")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/hscl_arms")
    raise SystemExit(main(target))
