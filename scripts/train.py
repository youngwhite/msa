"""Train any registered model, over one or more seeds.

Usage:
    python scripts/train.py --model lf_lstm --dataset mosi --seeds 42 43 44 45 46
    python scripts/train.py --list-models
    python scripts/train.py --model lf_lstm --model-arg modalities=t --model-arg dropout=0.3

Layout of the results (everything needed to audit a number):
    outputs/<group>/seed<N>/result.json          metrics, config, env, git commit
    outputs/<group>/seed<N>/test_predictions.npy predictions in test-split order
    outputs/<group>/seed<N>/best.pt              only with --keep-checkpoint
    outputs/<group>/summary.json                 mean/std across the seeds
"""

from __future__ import annotations

import argparse
import ast
import json

from msa.config import OUTPUT_ROOT
from msa.data import build_dataloaders
from msa.device import (
    DEVICE_CHOICES,
    configure_threads,
    describe_device,
    resolve_device,
)
from msa.metrics import METRIC_KEYS, format_metrics
from msa.registry import available_models, build_model
from msa.repro import set_seed
from msa.trainer import TrainConfig, Trainer, format_summary, save_run, summarize


def parse_model_args(pairs: list[str]) -> dict:
    """`--model-arg dropout=0.3` -> {"dropout": 0.3}, values as Python literals.

    Anything that is not a literal stays a string, so `modalities=tav` works
    without quoting while `use_lengths=False` becomes a real bool.
    """
    parsed = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--model-arg expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            parsed[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed[key.strip()] = raw
    return parsed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--model", default="lf_lstm")
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--unaligned", action="store_true",
                    help="use unaligned_50.pkl (audio/vision keep their own timelines)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one run per seed; the report is mean+-std over all of them")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--clip-mode", default="norm", choices=("norm", "value"),
                    help="clip gradients by norm or by value; MulT and MISA clip by value")
    ap.add_argument("--optimizer", default="adam", choices=("adam", "adamw"),
                    help="AdamW decouples weight decay; ALMT's reference uses it")
    ap.add_argument("--lr-schedule", default="none",
                    choices=("none", "plateau", "warmup_cosine", "warmup_linear"),
                    help="ReduceLROnPlateau on the validation selection metric")
    ap.add_argument("--lr-schedule-factor", type=float, default=0.1)
    ap.add_argument("--lr-schedule-patience", type=int, default=5)
    ap.add_argument("--lr-warmup-epochs", type=int, default=1,
                    help="warmup_linear only: epochs spent climbing to the full rate")
    ap.add_argument("--accumulate-steps", type=int, default=1,
                    help="optimiser step once per N batches (MMSA's update_epochs)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select-on", default="mae", choices=METRIC_KEYS,
                    help="validation metric used to pick the reported epoch")
    ap.add_argument("--select-reduction", default="sample", choices=("sample", "mmsa"),
                    help="how that metric is reduced: 'sample' over the split "
                         "(default), or 'mmsa' as the mean of per-batch means "
                         "rounded to 1e-4. The second is a diagnostic for what "
                         "the reference's protocol is worth, not a mode to "
                         "reproduce in — see TrainConfig")
    ap.add_argument("--model-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="model constructor argument; repeatable")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="pin CPU intra-op threads; required for identical CPU "
                         "results across machines")
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="allow nondeterministic kernels; the run stops being reproducible")
    ap.add_argument("--deterministic-warn-only", action="store_true",
                    help="warn instead of raising when an op has no deterministic kernel")
    ap.add_argument("--keep-checkpoint", action="store_true",
                    help="keep best.pt after the run (default: discard it — runs are "
                         "reproducible and the predictions are saved anyway)")
    ap.add_argument("--run-group", default=None,
                    help="output subdirectory (default: <model>_<dataset>_<device>)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-epoch lines")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.list_models:
        print("\n".join(available_models()))
        return

    model_kwargs = parse_model_args(args.model_arg)
    results, group_dir = [], None

    for seed in args.seeds:
        # Seed before anything random exists: the sampler's shuffling order and
        # the model's weight init both draw from the generators seeded here.
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

        if group_dir is None:  # one name for the whole sweep, computed once
            group_dir = OUTPUT_ROOT / (
                args.run_group or f"{args.model}_{args.dataset}_{device.type}"
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"model {args.model} ({n_params/1e6:.2f}M params) {model_kwargs or ''}")
            print(f"device {describe_device(device)}  cpu_threads {threads}  "
                  f"deterministic {args.deterministic}")
            print(f"{spec.name} ({'unaligned' if args.unaligned else 'aligned'}): "
                  + " ".join(f"{name}={len(loader.dataset)}"
                             for name, loader in loaders.items()))
        run_dir = group_dir / f"seed{seed}"
        print(f"\n=== seed {seed} -> {run_dir} ===")

        cfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            clip_mode=args.clip_mode,
            optimizer=args.optimizer,
            lr_schedule=args.lr_schedule,
            lr_schedule_factor=args.lr_schedule_factor,
            lr_schedule_patience=args.lr_schedule_patience,
            lr_warmup_epochs=args.lr_warmup_epochs,
            accumulate_steps=args.accumulate_steps,
            patience=args.patience,
            select_on=args.select_on,
            select_reduction=args.select_reduction,
            seed=seed,
            keep_checkpoint=args.keep_checkpoint,
        )
        trainer = Trainer(model, loaders, device, cfg, dataset=spec.name)
        result, preds = trainer.fit(run_dir / "best.pt", verbose=not args.quiet)

        print(f"best epoch {result.best_epoch} "
              f"(valid {cfg.select_on} {result.best_valid_score:.4f}), "
              f"{result.elapsed_sec:.1f}s")
        print("test:", format_metrics(result.test))
        print(f"test prediction checksum: {result.test_pred_sha256_16}")

        save_run(result, preds, run_dir, extra={
            "cli": vars(args),
            "model_kwargs": model_kwargs,
            "aligned": not args.unaligned,
            "batch_size": args.batch_size,
        })
        results.append(result)

    summary = summarize(results)
    print("\n" + format_summary(summary, len(results)))
    if len(results) == 1:
        print("\nSingle seed: not reportable on its own. Seed spread on these "
              "datasets is wider than most published gaps — use e.g. "
              "--seeds 42 43 44 45 46.")

    (group_dir / "summary.json").write_text(json.dumps({
        "model": args.model,
        "dataset": results[0].dataset,
        "aligned": not args.unaligned,
        "seeds": args.seeds,
        "cli": vars(args),
        "model_kwargs": model_kwargs,
        "config": results[0].config,
        "env": results[0].env,
        "summary": summary,
        "checksums": {str(r.seed): r.test_pred_sha256_16 for r in results},
    }, indent=2))
    print(f"\nsaved {group_dir/'summary.json'}")


if __name__ == "__main__":
    main()
