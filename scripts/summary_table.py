"""Build the cross-model results table from every stored run.

Reads each acceptance group's `summary.json`, pairs it with MMSA's reported
number, and prints a Markdown table with the shortfall in standard errors. This
is what goes into docs/storyline.md, generated rather than typed so the document
cannot drift from the runs.

Usage:
    python scripts/summary_table.py               # acceptance groups
    python scripts/summary_table.py --all         # every group, ablations included
"""

from __future__ import annotations

import argparse
import json
import math

from msa.config import OUTPUT_ROOT, PROJECT_ROOT

REFERENCE_PATH = PROJECT_ROOT / "docs" / "mmsa_reference_mosi.json"
LOWER_IS_BETTER = {"mae"}

#: The order the storyline tells it in, not alphabetical.
STORY_ORDER = [
    "ef_lstm", "lf_dnn", "tfn", "lmf", "mfn", "graph_mfn", "mctn", "mfm",
    "mult", "misa", "self_mm", "dlf", "dpdf_lq", "text_bert", "lf_lstm",
]
COLUMNS = ["mae", "corr", "acc2_non0", "acc7"]


def load_groups() -> list[dict]:
    groups = []
    for path in sorted(OUTPUT_ROOT.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        payload["group"] = path.parent.name
        groups.append(payload)
    return groups


def is_acceptance(payload: dict) -> bool:
    dataset = payload.get("dataset", "").lower().replace("cmu-", "")
    return payload["group"] == f"{payload.get('model', '')}_{dataset}"


def shortfall_in_se(stats: dict, reference: float, metric: str) -> float | None:
    n = stats.get("n", 1)
    if n < 2 or reference is None:
        return None
    se = stats["std"] / math.sqrt(n)
    gap = (stats["mean"] - reference) if metric in LOWER_IS_BETTER else (reference - stats["mean"])
    return gap / se if se > 0 else 0.0


def cell(stats: dict, reference: float | None, metric: str) -> str:
    text = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
    se = shortfall_in_se(stats, reference, metric) if reference is not None else None
    return f"{text} ({se:+.1f})" if se is not None else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include ablations and diagnostics")
    args = ap.parse_args()

    reference = json.loads(REFERENCE_PATH.read_text())["models"]
    groups = load_groups()
    if not args.all:
        groups = [g for g in groups if is_acceptance(g)]
    order = {name: i for i, name in enumerate(STORY_ORDER)}
    groups.sort(key=lambda g: (order.get(g.get("model", ""), 99), g["group"]))

    header = "| 模型 | 数据 | seed | " + " | ".join(
        {"mae": "MAE ↓", "corr": "Corr ↑", "acc2_non0": "Acc-2(non0)", "acc7": "Acc-7"}[c]
        for c in COLUMNS
    ) + " |"
    print(header)
    print("|" + "---|" * (len(COLUMNS) + 3))

    for payload in groups:
        model, stats = payload.get("model", "?"), payload["summary"]
        ref = reference.get(model, {})
        setting = "unaligned" if not payload.get("aligned", True) else "aligned"
        n = stats["mae"]["n"]
        cells = [cell(stats[c], ref.get(c), c) for c in COLUMNS]
        print(f"| {payload['group']} | {setting} | {n} | " + " | ".join(cells) + " |")
        if ref:
            mmsa = [f"*{ref[c]:.4f}*" if c in ref else "—" for c in COLUMNS]
            print("| ↳ *MMSA 报告* | | | " + " | ".join(mmsa) + " |")

    print("\n括号内为落后 MMSA 报告值的标准误倍数（负数表示更优）。"
          "≤1 通过，1-2 通过但标记，>2 未复现。")


if __name__ == "__main__":
    main()
