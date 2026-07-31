"""Regression metrics for CMU-MOSI / CMU-MOSEI / CH-SIMS.

Deliberately written to match MMSA's `utils/metricsTop.py` operation for
operation, because our numbers are compared directly against its published
table. Verified on real predictions: every metric agrees to within its 4-decimal
rounding (`scripts/check_invariants.py` re-checks the formulas on random data).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _safe_corr(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pearson correlation, 0.0 when it is undefined.

    `np.corrcoef` divides by both standard deviations, so a constant prediction
    *or* a constant ground truth yields NaN. A NaN would then poison any mean
    over seeds, so it is reported as 0 — no correlation is measurable.
    """
    if y_pred.size < 2 or y_pred.std() == 0 or y_true.std() == 0:
        return 0.0
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _multiclass_acc(y_pred: np.ndarray, y_true: np.ndarray, bound: float) -> float:
    """Accuracy after clipping to +/-bound and rounding — MMSA's `__multiclass_acc`.

    Acc-7 uses bound 3 and Acc-5 uses bound 2. Clipping before rounding is MMSA's
    order; rounding first is equivalent (checked over 2e5 points including every
    .5 boundary), but we keep its order so the two implementations can be diffed
    line by line.
    """
    return float(
        np.mean(np.round(np.clip(y_pred, -bound, bound))
                == np.round(np.clip(y_true, -bound, bound)))
    )


def eval_sentiment(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """MAE / Corr / Acc-7 / Acc-5 / Acc-2 / F1 over a whole split.

    Acc-2 and F1 come in the two flavours reported in the literature:
      * `_non0`: zero-labelled (neutral) samples are dropped, positive vs
        negative (Zadeh et al.)
      * `_has0`: all samples, non-negative vs negative (Yu et al.)
    They differ by 1-2 points, so a comparison that mixes them is meaningless.

    Note the metrics are computed over the *concatenated* predictions of a split,
    i.e. every sample weighs the same. MMSA does this too for its table, but its
    model-selection signal ("Loss") is a mean over batches, which over-weights a
    final partial batch — a small protocol difference, documented in
    docs/decisions.md.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: predictions {y_pred.shape} vs labels {y_true.shape}")

    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = _safe_corr(y_pred, y_true)

    # non-negative vs negative, all samples
    has0_true, has0_pred = y_true >= 0, y_pred >= 0
    acc2_has0 = float(accuracy_score(has0_true, has0_pred))
    f1_has0 = float(f1_score(has0_true, has0_pred, average="weighted"))

    # positive vs negative, neutral samples excluded
    nonzero = y_true != 0
    if nonzero.any():
        non0_true, non0_pred = y_true[nonzero] > 0, y_pred[nonzero] > 0
        acc2_non0 = float(accuracy_score(non0_true, non0_pred))
        f1_non0 = float(f1_score(non0_true, non0_pred, average="weighted"))
    else:
        acc2_non0 = f1_non0 = float("nan")

    return {
        "mae": mae,
        # ALMT's reference trains on MSE and selects on it too, so it has to be
        # available as a selection metric. Nothing else here reads it.
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "corr": corr,
        "acc7": _multiclass_acc(y_pred, y_true, bound=3.0),
        "acc5": _multiclass_acc(y_pred, y_true, bound=2.0),
        "acc2_has0": acc2_has0,
        "f1_has0": f1_has0,
        "acc2_non0": acc2_non0,
        "f1_non0": f1_non0,
    }


#: Metrics where a smaller value is better. Everything else is "higher is better".
LOWER_IS_BETTER = frozenset({"mae", "mse"})

#: The keys `eval_sentiment` returns, for validating --select-on and friends.
METRIC_KEYS = (
    "mae", "mse", "corr", "acc7", "acc5", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0",
)


def format_metrics(m: dict[str, float]) -> str:
    return "  ".join(f"{k}={m[k]:.4f}" for k in METRIC_KEYS if k in m)
