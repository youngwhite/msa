"""Regression checks for the data conventions, metrics and model plumbing.

These are the assumptions the rest of the code is built on. Run after touching
`msa.data`, `msa.metrics` or a model; exits non-zero on the first broken one.

Usage:
    python scripts/check_invariants.py [--dataset mosi]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from msa.config import get_dataset_spec
from msa.data import SPLITS, MMSADataset, load_pickle
from msa.metrics import eval_sentiment
from msa.models.functional import masked_mean
from msa.models.lf_lstm import _ModalityEncoder

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def tail_absmax(t: torch.Tensor) -> float:
    """abs().max() of a possibly-empty slice — a sample whose length equals the
    padded width has no tail at all, and max() on an empty tensor raises."""
    return float(t.abs().max()) if t.numel() else 0.0


def check_label_conventions(spec) -> None:
    print("\n== label conventions ==")
    raw = load_pickle(spec.aligned_pkl)
    for split in SPLITS:
        ds = MMSADataset(spec, split)
        reg = ds.labels.numpy()
        stored_cls = np.asarray(raw[split]["classification_labels"])
        check(
            f"pickle classification_labels == sign(regression)+1 [{split}]",
            np.array_equal(stored_cls, np.sign(reg) + 1),
        )
        label_7 = np.array([int(ds[i]["label_7"]) for i in range(len(ds))])
        check(
            f"our label_7 in 0..6 and consistent with the score [{split}]",
            label_7.min() >= 0 and label_7.max() <= 6
            and np.array_equal(label_7, np.clip(np.rint(reg), -3, 3) + 3),
        )


def check_padding_conventions(spec) -> None:
    print("\n== padding conventions ==")
    for aligned in (True, False):
        tag = "aligned" if aligned else "unaligned"
        ds = MMSADataset(spec, "train", aligned=aligned)
        lens = ds.text_lengths
        check(f"[{tag}] lengths are >=1 and <= sequence length",
              bool((lens >= 1).all() and (lens <= ds.text.shape[1]).all()
                   and (ds.audio_lengths >= 1).all()
                   and (ds.audio_lengths <= ds.audio.shape[1]).all()))
        # BERT emits non-zero vectors for [PAD]; anything reading past the length
        # is reading junk, which is exactly why the encoders take lengths.
        nonzero_tail = sum(
            tail_absmax(ds.text[i, int(lens[i]):]) > 0
            for i in range(min(200, len(ds)))
        )
        check(f"[{tag}] text padding is non-zero (encoders must mask)",
              nonzero_tail > 0, f"{nonzero_tail}/200 samples have non-zero [PAD] rows")
        if aligned:
            bad = sum(
                tail_absmax(ds.audio[i, int(lens[i]):]) > 0
                or tail_absmax(ds.vision[i, int(lens[i]):]) > 0
                for i in range(len(ds))
            )
            check("[aligned] audio/vision are zero past the text length "
                  "(so one length serves all three modalities)", bad == 0,
                  f"{bad} violations")


def check_encoder_masking() -> None:
    print("\n== encoder length masking ==")
    torch.manual_seed(0)
    enc = _ModalityEncoder(4, 8, 0.0).eval()
    x = torch.randn(3, 10, 4)
    lengths = torch.tensor([10, 4, 1])
    with torch.no_grad():
        out = enc(x, lengths)
        truncated = torch.cat(
            [enc(x[i:i + 1, :int(n)], torch.tensor([int(n)])) for i, n in enumerate(lengths)]
        )
        junk = torch.cat([x, torch.randn(3, 20, 4)], dim=1)
        with_junk = enc(junk, lengths)
        unmasked = enc(junk, None)
    check("masked state == running the LSTM on the truncated sequence",
          torch.allclose(out, truncated, atol=1e-6),
          f"max diff {(out - truncated).abs().max():.2e}")
    check("appending junk padding does not change the masked state",
          torch.equal(with_junk, out))
    check("the unmasked h[-1] really is different (ablation is meaningful)",
          not torch.allclose(unmasked, out, atol=1e-3),
          f"max diff {(unmasked - out).abs().max():.2e}")


def check_pooling() -> None:
    print("\n== sequence pooling ==")
    x = torch.zeros(2, 6, 3)
    x[0, :2] = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    x[1, :6] = 2.0
    lengths = torch.tensor([2, 6])

    got = masked_mean(x, lengths)
    want = torch.tensor([[2.0, 3.0, 4.0], [2.0, 2.0, 2.0]])
    check("masked_mean averages over real frames only", torch.allclose(got, want),
          f"got {got.tolist()}")

    padded = masked_mean(x, None)
    ratio = padded[0] / got[0]
    check("unmasked mean scales a sample by valid_len/padded_width "
          "(MMSA's __normalize behaviour)",
          torch.allclose(ratio, torch.full((3,), 2 / 6), atol=1e-6),
          f"ratio {ratio[0]:.4f}, expected {2/6:.4f}")
    check("both pooling modes agree when nothing is padded",
          torch.allclose(masked_mean(x[1:], lengths[1:]), masked_mean(x[1:], None)))


def check_metrics() -> None:
    print("\n== metrics ==")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(20, 200))
        y_true = rng.choice([-3, -2.4, -1, -0.5, 0, 0.6, 1, 2, 3], size=n)
        y_pred = y_true + rng.normal(0, 1.2, size=n)
        m = eval_sentiment(y_pred, y_true)
        nz = y_true != 0
        ref = {
            "mae": np.abs(y_pred - y_true).mean(),
            "corr": np.corrcoef(y_pred, y_true)[0, 1],
            "acc7": (np.clip(np.rint(y_pred), -3, 3) == np.clip(np.rint(y_true), -3, 3)).mean(),
            "acc2_has0": ((y_pred >= 0) == (y_true >= 0)).mean(),
            "acc2_non0": ((y_pred[nz] > 0) == (y_true[nz] > 0)).mean(),
        }
        worst = max(worst, max(abs(m[k] - v) for k, v in ref.items()))
    check("200 random cases match an independent implementation", worst < 1e-12,
          f"max deviation {worst:.2e}")

    perfect = eval_sentiment(np.array([1.0, -1.0, 2.0]), np.array([1.0, -1.0, 2.0]))
    check("perfect predictions give mae 0, acc 1, f1 1",
          perfect["mae"] == 0 and perfect["acc2_non0"] == 1 and perfect["f1_non0"] == 1)
    inverted = eval_sentiment(np.array([-1.0, 1.0]), np.array([1.0, -1.0]))
    check("inverted predictions give acc2 0", inverted["acc2_non0"] == 0)
    constant = eval_sentiment(np.ones(10), np.arange(10, dtype=float))
    check("a constant prediction gives corr 0 rather than NaN", constant["corr"] == 0.0)


def check_loader_order(spec) -> None:
    print("\n== evaluation ordering ==")
    test = MMSADataset(spec, "test")
    seq = DataLoader(test, batch_size=32, shuffle=False)
    labels = torch.cat([b["label"] for b in seq]).numpy()
    check("eval loader yields samples in dataset order (predictions stay aligned "
          "with ids)", np.array_equal(labels, test.labels.numpy()))


def check_pickle_cache(spec) -> None:
    print("\n== feature cache ==")
    load_pickle.cache_clear()
    load_pickle(spec.aligned_pkl)
    load_pickle(str(spec.aligned_pkl))
    size = load_pickle.cache_info().currsize
    check("str and Path for the same file share one cache entry", size == 1,
          f"currsize={size}; each entry holds a ~400MB pickle")


def check_beats_trivial(spec) -> None:
    print("\n== sanity floor ==")
    train, test = MMSADataset(spec, "train"), MMSADataset(spec, "test")
    y_train, y_test = train.labels.numpy(), test.labels.numpy()
    floors = {
        "predict 0": np.zeros_like(y_test),
        "predict train mean": np.full_like(y_test, y_train.mean()),
        "predict train median": np.full_like(y_test, float(np.median(y_train))),
    }
    for name, pred in floors.items():
        m = eval_sentiment(pred, y_test)
        print(f"       {name:22s} mae={m['mae']:.4f} acc2_non0={m['acc2_non0']:.4f}")
    print("       (any trained model must clear these)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mosi")
    args = ap.parse_args()
    spec = get_dataset_spec(args.dataset)
    print(f"invariant checks on {spec.name}")

    check_label_conventions(spec)
    check_padding_conventions(spec)
    check_encoder_masking()
    check_pooling()
    check_metrics()
    check_loader_order(spec)
    check_pickle_cache(spec)
    check_beats_trivial(spec)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nall invariants hold")


if __name__ == "__main__":
    main()
