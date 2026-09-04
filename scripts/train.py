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

import torch

from msa.config import OUTPUT_ROOT
from msa.data import build_dataloaders
from msa.device import (
    DEVICE_CHOICES,
    configure_threads,
    describe_device,
    resolve_device,
)
from msa.losses import available_losses, get_loss_class
from msa.losses.composite import SCHEMES, ContrastiveHead
from msa.metrics import METRIC_KEYS, format_metrics
from msa.models.contrastive import ContrastiveModel, feature_dims
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
                    choices=("none", "plateau", "warmup_cosine"),
                    help="ReduceLROnPlateau on the validation selection metric")
    ap.add_argument("--lr-schedule-factor", type=float, default=0.1)
    ap.add_argument("--lr-schedule-patience", type=int, default=5)
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

    contrastive = ap.add_argument_group(
        "contrastive auxiliary loss",
        "Adds `lambda * sum(w_i * loss_i)` on top of the model's own loss, on its "
        "pooled modality features. Without --contrastive nothing here has any "
        "effect and the run is byte-identical to one from before these options "
        "existed.",
    )
    contrastive.add_argument("--list-losses", action="store_true")
    contrastive.add_argument("--contrastive", nargs="+", default=[], metavar="NAME",
                             help="one or more registered contrastive objectives; "
                                  "phase 2 caps a combination at 4")
    contrastive.add_argument("--contrastive-lambda", type=float, default=0.1,
                             help="weight of the whole contrastive part against the "
                                  "task loss (default 0.1)")
    contrastive.add_argument("--contrastive-arg", action="append", default=[],
                             metavar="KEY=VALUE",
                             help="kwargs for the objectives, e.g. temperature=0.1")
    contrastive.add_argument("--projection-dim", type=int, default=64,
                             help="width of the shared space the terms are computed "
                                  "in; modality features differ in width so a "
                                  "projection is required, not optional")
    contrastive.add_argument("--weight-scheme", default="equal",
                             choices=sorted(SCHEMES),
                             help="equal: fixed 1/k, the control. grad_rate: shift "
                                  "weight every --weight-window epochs towards terms "
                                  "whose gradient is still moving. linear_ramp: move "
                                  "weight linearly towards --favour late in training")
    contrastive.add_argument("--weight-window", type=int, default=10,
                             help="grad_rate: epochs between weight updates")
    contrastive.add_argument("--weight-floor", type=float, default=0.05,
                             help="grad_rate: smallest weight a term can hold; zero "
                                  "would be an absorbing state")
    contrastive.add_argument("--favour", default=None,
                             help="linear_ramp: which term to ramp towards")
    contrastive.add_argument("--favour-target", type=float, default=0.7,
                             help="linear_ramp: weight it reaches by the end")
    contrastive.add_argument("--favour-end-fraction", type=float, default=1.0,
                             help="linear_ramp: fraction of --epochs by which the "
                                  "ramp completes. Below 1.0 because early stopping "
                                  "ends most runs well short of --epochs, and a ramp "
                                  "that never finishes is just the equal-weight arm")
    contrastive.add_argument("--raw-terms", dest="normalise_terms",
                             action="store_false",
                             help="weight the objectives' raw values instead of "
                                  "rescaling each by its own running magnitude. The "
                                  "terms differ by ~1e4 on real batches, so with this "
                                  "set a weight is not a share of influence and one "
                                  "lambda is not comparable across candidates")
    return ap


def build_contrastive(args, model, batch, spec) -> object:
    """Wrap `model` with a contrastive head, or return it untouched.

    The feature widths come from a forward pass rather than from the model's
    constructor arguments -- TFN's are 32, 32 and 128 and follow from three
    separate kwargs, so reading them off the model cannot drift out of date.
    """
    if not args.contrastive:
        return model
    if len(args.contrastive) > 4:
        raise SystemExit(
            f"phase 2 caps a combination at 4 terms, got {len(args.contrastive)}: "
            f"{args.contrastive}"
        )
    duplicates = {n for n in args.contrastive if args.contrastive.count(n) > 1}
    if duplicates:
        raise SystemExit(f"repeated objective(s): {sorted(duplicates)}")

    loss_kwargs = parse_model_args(args.contrastive_arg)
    dims = feature_dims(model, batch)
    losses = {}
    for name in args.contrastive:
        cls = get_loss_class(name)
        kwargs = dict(loss_kwargs)
        if getattr(cls, "parametric", False):
            kwargs.setdefault("dim", args.projection_dim)
        losses[name] = cls(**kwargs)

    scheme_cls = SCHEMES[args.weight_scheme]
    scheme_kwargs: dict = {}
    if args.weight_scheme == "grad_rate":
        scheme_kwargs = {"window": args.weight_window, "floor": args.weight_floor}
    elif args.weight_scheme == "linear_ramp":
        if args.favour is None:
            raise SystemExit("--weight-scheme linear_ramp needs --favour <term>")
        scheme_kwargs = {"favour": args.favour, "total_epochs": args.epochs,
                         "target": args.favour_target,
                         "end_fraction": args.favour_end_fraction}
    scheme = scheme_cls(sorted(losses), **scheme_kwargs)

    head = ContrastiveHead(
        feature_dims=dims, losses=losses, scheme=scheme,
        projection_dim=args.projection_dim, lambda_=args.contrastive_lambda,
        normalise_terms=args.normalise_terms,
    )
    return ContrastiveModel(model, head)


def main() -> None:
    args = build_parser().parse_args()

    if args.list_models:
        print("\n".join(available_models()))
        return
    if args.list_losses:
        for name in available_losses():
            cls = get_loss_class(name)
            tags = " ".join(t for t, on in (
                ("class-labels", cls.needs_class_labels),
                ("continuous-labels", cls.uses_continuous_labels),
                ("parametric", cls.parametric),
            ) if on)
            print(f"{name:14s} {tags}")
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
        if args.contrastive:
            probe = next(iter(loaders["train"]))
            probe = {k: v.to(device) for k, v in probe.items()
                     if isinstance(v, torch.Tensor)}
            model = build_contrastive(args, model, probe, spec).to(device)
            # Re-seed: probing consumed nothing random, but building the head
            # drew from the generators, and the training run must not depend on
            # how many parameters the head happened to have.
            set_seed(seed, args.deterministic, warn_only=args.deterministic_warn_only)

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
