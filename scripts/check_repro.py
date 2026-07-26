"""Verify that a run is reproducible on the current device.

Trains a model twice in-process with the same seed, through the same Trainer the
real experiments use, and compares weights and predictions bit-for-bit. Run this
on any new machine (Apple Silicon included) before trusting its numbers.

Usage:
    python scripts/check_repro.py [--device auto|cuda|mps|cpu] [--epochs 2]
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from msa.data import build_dataloaders
from msa.device import (
    DEVICE_CHOICES,
    configure_threads,
    describe_device,
    resolve_device,
)
from msa.registry import build_model
from msa.repro import set_seed
from msa.trainer import TrainConfig, Trainer


def digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def one_run(args, device, ckpt: Path) -> tuple[str, str]:
    set_seed(args.seed, args.deterministic, warn_only=args.deterministic_warn_only)
    configure_threads(args.num_threads)
    loaders, spec = build_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    model = build_model(args.model, spec).to(device)
    cfg = TrainConfig(epochs=args.epochs, patience=args.epochs, seed=args.seed)
    _, preds = Trainer(model, loaders, device, cfg).fit(ckpt, verbose=False)
    weights = [p.detach().float().cpu().numpy() for p in model.parameters()]
    return digest(weights), digest([preds])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lf_lstm")
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--deterministic-warn-only", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    threads = configure_threads(args.num_threads)
    print(f"model {args.model}  device {describe_device(device)}  seed {args.seed}  "
          f"deterministic {args.deterministic}  workers {args.num_workers}  "
          f"cpu_threads {threads}")

    with tempfile.TemporaryDirectory() as tmp:
        w1, p1 = one_run(args, device, Path(tmp) / "a.pt")
        w2, p2 = one_run(args, device, Path(tmp) / "b.pt")
    print(f"run 1: weights {w1}  predictions {p1}")
    print(f"run 2: weights {w2}  predictions {p2}")

    if w1 == w2 and p1 == p2:
        print("\nREPRODUCIBLE: both runs are bit-identical on this device.")
        return
    raise SystemExit(
        "\nNOT REPRODUCIBLE on this device. Check that --no-deterministic is off "
        "and that CUBLAS_WORKSPACE_CONFIG is set before the first CUDA call."
    )


if __name__ == "__main__":
    main()
