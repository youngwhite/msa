"""Train any registered model, over one or more seeds.

Usage:
    python scripts/train.py --model lf_lstm --dataset mosi --seeds 42 43 44 45 46
    python scripts/train.py --list-models
    python scripts/train.py --model lf_lstm --model-arg modalities=t --model-arg dropout=0.3
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict

from msa.config import OUTPUT_ROOT
from msa.data import build_dataloaders
from msa.device import (
    DEVICE_CHOICES,
    configure_threads,
    describe_device,
    resolve_device,
)
from msa.metrics import format_metrics
from msa.registry import available_models, build_model
from msa.repro import set_seed
from msa.trainer import TrainConfig, Trainer, format_summary, save_run, summarize


def parse_model_args(pairs: list[str]) -> dict:
    """`--model-arg dropout=0.3` -> {"dropout": 0.3}, values parsed as Python literals."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--model-arg expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw  # plain string, e.g. modalities=tav
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--model", default="lf_lstm")
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--unaligned", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one run per seed; report mean±std over all of them")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select-on", default="mae",
                    help="valid metric used for model selection (mae, corr, acc2_non0, ...)")
    ap.add_argument("--model-arg", action="append", default=[],
                    help="model kwarg as key=value; repeatable")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--deterministic-warn-only", action="store_true")
    ap.add_argument("--run-group", default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress per-epoch lines")
    args = ap.parse_args()

    if args.list_models:
        print("\n".join(available_models()))
        return

    model_kwargs = parse_model_args(args.model_arg)
    results, all_preds = [], {}

    for seed in args.seeds:
        # Seed first: dataloader shuffling and weight init both draw from here.
        set_seed(seed, args.deterministic, warn_only=args.deterministic_warn_only)
        device = resolve_device(args.device)
        threads = configure_threads(args.num_threads)

        loaders, spec = build_dataloaders(
            args.dataset,
            aligned=not args.unaligned,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
            device=device,
        )
        model = build_model(args.model, spec, **model_kwargs).to(device)

        group = args.run_group or f"{args.model}_{args.dataset}_{device.type}"
        run_dir = OUTPUT_ROOT / group / f"seed{seed}"

        if seed == args.seeds[0]:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"model {args.model} ({n_params/1e6:.2f}M params) "
                  f"{model_kwargs if model_kwargs else ''}")
            print(f"device {describe_device(device)}  cpu_threads {threads}  "
                  f"deterministic {args.deterministic}")
            print(f"{spec.name}: " + " ".join(
                f"{name}={len(loader.dataset)}" for name, loader in loaders.items()))
        print(f"\n=== seed {seed} -> {run_dir.relative_to(OUTPUT_ROOT.parent)} ===")

        cfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            patience=args.patience,
            select_on=args.select_on,
            seed=seed,
        )
        result, preds = Trainer(model, loaders, device, cfg).fit(
            run_dir / "best.pt", verbose=not args.quiet
        )
        result.dataset = spec.name
        print(f"best epoch {result.best_epoch} "
              f"(valid {cfg.select_on} {result.best_valid_score:.4f}), "
              f"{result.elapsed_sec:.1f}s")
        print("test:", format_metrics(result.test))
        print(f"test prediction checksum: {result.test_pred_sha256_16}")

        save_run(result, preds, run_dir, extra={
            "cli": vars(args), "model_kwargs": model_kwargs,
            "aligned": not args.unaligned,
        })
        results.append(result)
        all_preds[seed] = preds

    group_dir = OUTPUT_ROOT / (args.run_group or
                               f"{args.model}_{args.dataset}_{results[0].env['device']}")
    summary = summarize(results)
    print("\n" + format_summary(summary, len(results)))
    if len(results) == 1:
        print("\nsingle seed: not reportable on its own — LF-LSTM alone varies by "
              "±0.035 MAE across seeds. Use --seeds 42 43 44 45 46.")
    (group_dir / "summary.json").write_text(json.dumps({
        "model": args.model,
        "dataset": results[0].dataset,
        "seeds": args.seeds,
        "cli": vars(args),
        "model_kwargs": model_kwargs,
        "summary": summary,
        "checksums": {str(r.seed): r.test_pred_sha256_16 for r in results},
        "config": asdict(results[0].config) if hasattr(results[0].config, "__dict__")
        else results[0].config,
    }, indent=2))
    print(f"\nsaved {group_dir/'summary.json'}")


if __name__ == "__main__":
    main()
