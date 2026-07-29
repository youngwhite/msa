"""Paired bootstrap test between two groups' stored predictions.

Phase 1's protocol asks for this and it was never implemented, so every
cross-model claim in the docs so far — including "text-only fine-tuned BERT
beats MulT", the whole point of the control group — has been a comparison of
means with no test behind it.

What is paired and what is not:

* **Test samples are paired.** Both models are scored on the same resampled
  indices in each replicate. This is the real pairing: the two models see the
  same hard and easy utterances, so the sampling noise that dominates a 686-item
  test set cancels out of the difference.
* **Seeds are resampled independently.** Seed 42 of one model has no meaningful
  correspondence to seed 42 of another — same integer, unrelated trajectories.
  Pairing them would fabricate a correlation that is not there.

The estimand is the difference of the *seed-mean* metrics, which is what the
tables report. Each replicate resamples seeds with replacement inside each model
and averages, so the interval covers seed variability as well as sample
variability. That is deliberately wider than a sample-only bootstrap: on this
dataset seed spread is the larger term (LF-LSTM's MAE moves +-0.039 across
seeds), and an interval that hid it would be the wrong answer to the question
the storyline actually asks.

The vectorised metrics below are checked against `msa.metrics.eval_sentiment` on
the unresampled data before any resampling happens, so this file cannot quietly
drift into being a second, disagreeing implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msa.config import OUTPUT_ROOT  # noqa: E402
from msa.data import build_dataloaders  # noqa: E402
from msa.metrics import LOWER_IS_BETTER, eval_sentiment  # noqa: E402

SUPPORTED = ("mae", "corr", "acc7", "acc5", "acc2_non0", "acc2_has0")


def metric_on(pred: np.ndarray, true: np.ndarray, name: str) -> float:
    """One metric, numpy only. Validated against eval_sentiment at startup."""
    if name == "mae":
        return float(np.mean(np.abs(pred - true)))
    if name == "corr":
        if pred.size < 2 or pred.std() == 0 or true.std() == 0:
            return 0.0
        return float(np.corrcoef(pred, true)[0, 1])
    if name in ("acc7", "acc5"):
        bound = 3.0 if name == "acc7" else 2.0
        return float(np.mean(np.round(np.clip(pred, -bound, bound))
                             == np.round(np.clip(true, -bound, bound))))
    if name == "acc2_has0":
        return float(np.mean((pred >= 0) == (true >= 0)))
    if name == "acc2_non0":
        keep = true != 0
        if not keep.any():
            return float("nan")
        return float(np.mean((pred[keep] > 0) == (true[keep] > 0)))
    raise ValueError(f"unsupported metric {name!r}")


def load_group(group: str) -> tuple[np.ndarray, list[int]]:
    """Per-seed test predictions, shape (n_seeds, n_samples)."""
    paths = sorted((OUTPUT_ROOT / group).glob("seed*/test_predictions.npy"),
                   key=lambda p: int(p.parent.name.removeprefix("seed")))
    if not paths:
        raise SystemExit(f"{group}: no test_predictions.npy found")
    seeds = [int(p.parent.name.removeprefix("seed")) for p in paths]
    return np.stack([np.load(p).reshape(-1) for p in paths]), seeds


def compare(a: str, b: str, metric: str, replicates: int, rng: np.random.Generator,
            labels: np.ndarray) -> dict:
    pred_a, seeds_a = load_group(a)
    pred_b, seeds_b = load_group(b)
    if pred_a.shape[1] != labels.size or pred_b.shape[1] != labels.size:
        raise SystemExit("prediction length does not match the test split")

    obs_a = float(np.mean([metric_on(p, labels, metric) for p in pred_a]))
    obs_b = float(np.mean([metric_on(p, labels, metric) for p in pred_b]))
    # "Better" is signed so a positive difference always means a beats b.
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    observed = sign * (obs_a - obs_b)

    n = labels.size
    deltas = np.empty(replicates)
    for i in range(replicates):
        idx = rng.integers(0, n, n)                     # paired across models
        y = labels[idx]
        pick_a = rng.integers(0, len(seeds_a), len(seeds_a))
        pick_b = rng.integers(0, len(seeds_b), len(seeds_b))
        ma = np.mean([metric_on(pred_a[s][idx], y, metric) for s in pick_a])
        mb = np.mean([metric_on(pred_b[s][idx], y, metric) for s in pick_b])
        deltas[i] = sign * (ma - mb)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided: how often the resampled difference lands on the other side of 0.
    p = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return {
        "a": a, "b": b, "metric": metric,
        "a_value": obs_a, "b_value": obs_b,
        "difference": observed, "ci95": [float(lo), float(hi)],
        "p_value": min(p, 1.0), "replicates": replicates,
        "seeds_a": len(seeds_a), "seeds_b": len(seeds_b),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="A:B",
                    help="group pairs to compare, e.g. text_bert_mosi:mult_mosi")
    ap.add_argument("--metrics", nargs="+", default=["mae", "corr"], choices=SUPPORTED)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args()

    loaders, _ = build_dataloaders(args.dataset, aligned=True, batch_size=64,
                                   num_workers=0, seed=0)
    labels = np.concatenate([b["label"].numpy() for b in loaders["test"]]).astype(np.float64)

    # Guard: the fast metrics must agree with the shared implementation.
    probe, _ = load_group(args.pairs[0].split(":")[0])
    reference = eval_sentiment(probe[0], labels)
    for name in SUPPORTED:
        mine, theirs = metric_on(probe[0], labels, name), reference[name]
        if not np.isclose(mine, theirs, atol=1e-12):
            raise SystemExit(f"metric {name} disagrees with eval_sentiment: "
                             f"{mine} vs {theirs}")
    print(f"metrics agree with msa.metrics on {labels.size} test samples\n")

    rng = np.random.default_rng(args.seed)
    out = []
    for pair in args.pairs:
        a, b = pair.split(":")
        print(f"=== {a}  vs  {b} ===")
        for metric in args.metrics:
            r = compare(a, b, metric, args.replicates, rng, labels)
            arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
            verdict = ("a better" if r["difference"] > 0 else "b better")
            sig = "significant" if r["p_value"] < 0.05 else "NOT significant"
            print(f"  {metric}{arrow}: {r['a_value']:.4f} vs {r['b_value']:.4f}   "
                  f"diff {r['difference']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, "
                  f"{r['ci95'][1]:+.4f}]  p={r['p_value']:.4f}  {verdict}, {sig}")
            out.append(r)
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
