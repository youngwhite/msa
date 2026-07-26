"""Sanity report for a preprocessed MMSA dataset.

Usage:
    python scripts/check_data.py --dataset mosi [--unaligned]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

from msa.config import get_dataset_spec, verify_files
from msa.data import SPLITS, MMSADataset, load_pickle
from msa.device import resolve_device
from msa.repro import collect_env


def report_environment() -> None:
    print("== environment ==")
    for key, value in collect_env(resolve_device("auto")).items():
        print(f"  {key:26s} {value}")


def report_raw_pickle(path) -> None:
    print(f"\n== raw pickle: {path.name} ==")
    data = load_pickle(path)
    for split in data:
        d = data[split]
        keys = ", ".join(
            f"{k}{tuple(v.shape)}" if isinstance(v, np.ndarray) else f"{k}[{len(v)}]"
            for k, v in d.items()
        )
        print(f"  {split:5s}: {keys}")


PROBLEMS: list[str] = []


def report_split(ds: MMSADataset) -> None:
    y = ds.labels.numpy()
    counts = Counter(ds.annotations)
    print(f"\n-- {ds.split} (n={len(ds)}) --")
    print(f"  dims         {ds.feature_dims}")
    print(f"  seq lens     text={ds.text.shape[1]} audio={ds.audio.shape[1]} "
          f"vision={ds.vision.shape[1]}")
    print(f"  text tokens  mean={ds.text_lengths.float().mean():.1f} "
          f"min={ds.text_lengths.min()} max={ds.text_lengths.max()} "
          f"(padding = {1 - ds.text_lengths.float().mean()/ds.text.shape[1]:.1%} of steps)")
    if not ds.aligned:
        print(f"  valid frames audio mean={ds.audio_lengths.float().mean():.1f} "
              f"max={ds.audio_lengths.max()}  "
              f"vision mean={ds.vision_lengths.float().mean():.1f} "
              f"max={ds.vision_lengths.max()}")
    print(f"  label        min={y.min():.2f} max={y.max():.2f} "
          f"mean={y.mean():.3f} std={y.std():.3f}")
    print(f"  annotations  {dict(counts)}")
    for name, t in (("text", ds.text), ("audio", ds.audio), ("vision", ds.vision)):
        finite = bool(torch.isfinite(t).all())
        print(f"  {name:6s}       finite={finite} mean={t.mean():.4f} std={t.std():.4f}")
        if not finite:
            PROBLEMS.append(f"{ds.split}/{name}: non-finite values survived _clean()")


def cross_check_labels(spec, datasets: dict[str, MMSADataset]) -> None:
    print("\n== cross-check against label.csv ==")
    df = pd.read_csv(spec.label_csv)
    print(f"  csv rows {len(df)}, modes {dict(df['mode'].value_counts())}")
    csv_key = {
        f"{r.video_id}$_${r.clip_id}": float(r.label) for r in df.itertuples()
    }
    for split, ds in datasets.items():
        missing = [i for i in ds.ids if i not in csv_key]
        diffs = [
            abs(csv_key[sample_id] - float(label))
            for sample_id, label in zip(ds.ids, ds.labels, strict=True)
            if sample_id in csv_key
        ]
        worst = max(diffs) if diffs else float("nan")
        print(f"  {split:5s}: ids missing from csv={len(missing)}  "
              f"max |label diff|={worst:.6f}")
        if missing:
            PROBLEMS.append(f"{split}: {len(missing)} sample ids are absent from label.csv")
        if diffs and worst > 1e-6:
            PROBLEMS.append(
                f"{split}: a feature-file label differs from label.csv by {worst:.6f}"
            )

    # The one failure that invalidates every number downstream.
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        overlap = set(datasets[a].ids) & set(datasets[b].ids)
        print(f"  {a}/{b} id overlap: {len(overlap)}")
        if overlap:
            PROBLEMS.append(
                f"{a} and {b} share {len(overlap)} sample id(s) — the split leaks"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--unaligned", action="store_true",
                    help="inspect unaligned_50.pkl instead of aligned_50.pkl")
    ap.add_argument("--verify-files", action="store_true",
                    help="also hash the dataset files (~900MB, a few seconds) and "
                         "compare against the sha256 recorded in the DatasetSpec")
    args = ap.parse_args()

    spec = get_dataset_spec(args.dataset)
    aligned = not args.unaligned
    print(f"dataset {spec.name} at {spec.root} "
          f"({'aligned' if aligned else 'unaligned'})")

    if args.verify_files:
        print("\n== dataset file hashes ==")
        file_problems = verify_files(spec)
        for problem in file_problems:
            print(f"  {problem}")
        if file_problems:
            PROBLEMS.append(f"{len(file_problems)} dataset file(s) do not match their "
                            f"recorded sha256. Source:\n  {spec.source}")
        else:
            print(f"  all {len(spec.file_sha256)} files match their recorded sha256")

    report_environment()
    report_raw_pickle(spec.aligned_pkl if aligned else spec.unaligned_pkl)

    datasets = {s: MMSADataset(spec, s, aligned=aligned) for s in SPLITS}
    print("\n== per split ==")
    for ds in datasets.values():
        report_split(ds)

    cross_check_labels(spec, datasets)

    print("\n== one training batch ==")
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(datasets["train"], batch_size=4, shuffle=False)))
    for k, v in batch.items():
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")
    print(f"\n  sample id   {datasets['train'].ids[0]}")
    print(f"  raw text    {datasets['train'].raw_text[0][:80]!r}")
    print(f"  label       {batch['label'][0].item():.2f} "
          f"(7-way class {batch['label_7'][0].item()})")

    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} PROBLEM(S):")
        for problem in PROBLEMS:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nOK: splits are disjoint and every label agrees with label.csv")


if __name__ == "__main__":
    main()
