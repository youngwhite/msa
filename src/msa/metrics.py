"""Standard CMU-MOSI/MOSEI regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def eval_sentiment(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """MAE / Corr / Acc-7 / Acc-2 / F1, following the MMSA protocol.

    Acc2 and F1 come in two flavours reported in the literature:
      * `_non0`: the zero-labelled samples are dropped (Zadeh et al.)
      * `_has0`: non-negative vs negative over all samples (Yu et al.)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if y_pred.std() > 0 else 0.0

    pred7 = np.clip(np.rint(y_pred), -3, 3)
    true7 = np.clip(np.rint(y_true), -3, 3)
    acc7 = float(accuracy_score(true7, pred7))

    # non-negative vs negative, all samples
    has0_pred = y_pred >= 0
    has0_true = y_true >= 0
    acc2_has0 = float(accuracy_score(has0_true, has0_pred))
    f1_has0 = float(f1_score(has0_true, has0_pred, average="weighted"))

    # positive vs negative, neutral samples excluded
    nz = y_true != 0
    if nz.any():
        acc2_non0 = float(accuracy_score(y_true[nz] > 0, y_pred[nz] > 0))
        f1_non0 = float(f1_score(y_true[nz] > 0, y_pred[nz] > 0, average="weighted"))
    else:
        acc2_non0 = f1_non0 = float("nan")

    return {
        "mae": mae,
        "corr": corr,
        "acc7": acc7,
        "acc2_has0": acc2_has0,
        "f1_has0": f1_has0,
        "acc2_non0": acc2_non0,
        "f1_non0": f1_non0,
    }


def format_metrics(m: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())
