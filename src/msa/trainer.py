"""One training loop for every model.

MMSA carries 14 near-identical trainers (2325 lines; TFN's and LMF's differ by 14
lines after renaming), which is how evaluation protocols quietly drift apart
between models. Here a model customises behaviour through `MSAModel`
(`compute_loss`, `param_groups`) and everything else — selection metric, early
stopping, checkpointing, provenance — is shared and therefore comparable.

Reading order: `TrainConfig` (what is tunable) -> `Trainer.fit` (the loop) ->
`summarize` (how several seeds become one reportable number).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .device import supports_pin_memory
from .metrics import LOWER_IS_BETTER, METRIC_KEYS, eval_sentiment
from .models.base import MSAModel
from .repro import collect_env


@dataclass
class TrainConfig:
    """Everything the loop needs. Stored verbatim in each run's result.json."""

    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    #: How grad_clip is applied. MMSA's TFN/LMF/MFN do not clip at all, MulT and
    #: MISA clip by *value*, MMIM clips by norm — so this has to be selectable.
    clip_mode: str = "norm"          # "norm" | "value"
    #: Optimiser. Adam everywhere except ALMT, whose reference and whose authors
    #: both use AdamW — decoupled weight decay, which at decay 1e-4 is not the
    #: same update as Adam's L2 term.
    optimizer: str = "adam"          # "adam" | "adamw"
    #: Optional ReduceLROnPlateau on the validation selection metric, as MulT and
    #: MMIM use, or ALMT's linear warmup into cosine annealing. "none" leaves the
    #: learning rate alone.
    lr_schedule: str = "none"        # "none" | "plateau" | "warmup_cosine"
    lr_schedule_factor: float = 0.1
    lr_schedule_patience: int = 5
    #: Optimiser steps once per this many batches. MMSA calls it `update_epochs`
    #: and uses it for MulT (8), Self-MM (4) and MISA (2); a partial window at
    #: the end of an epoch is discarded, as there.
    accumulate_steps: int = 1
    patience: int = 8
    #: Validation metric used to pick the reported epoch. `mae` reproduces MMSA's
    #: "KeyEval: Loss" for regression, since its criterion is L1.
    select_on: str = "mae"
    seed: int = 42
    #: Keep `best.pt` after the run. Off by default: a checkpoint is ~13MB for
    #: LF-LSTM and 38MB for TFN, the run is bit-for-bit reproducible from the
    #: recorded config, and the predictions it would produce are already saved.
    #: Turn it on when you actually need the weights (inspection, fine-tuning).
    keep_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.clip_mode not in ("norm", "value"):
            raise ValueError(f"clip_mode must be 'norm' or 'value', got {self.clip_mode!r}")
        if self.lr_schedule not in ("none", "plateau", "warmup_cosine"):
            raise ValueError("lr_schedule must be 'none', 'plateau' or 'warmup_cosine', "
                             f"got {self.lr_schedule!r}")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}")
        if self.select_on not in METRIC_KEYS:
            raise ValueError(
                f"select_on={self.select_on!r} is not a metric; "
                f"choose from {list(METRIC_KEYS)}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.accumulate_steps < 1:
            raise ValueError(f"accumulate_steps must be >= 1, got {self.accumulate_steps}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        if self.select_on in LOWER_IS_BETTER:
            return candidate < incumbent
        return candidate > incumbent

    @property
    def worst_score(self) -> float:
        return float("inf") if self.select_on in LOWER_IS_BETTER else float("-inf")


@dataclass
class RunResult:
    """One (model, dataset, seed) run — everything needed to audit its number."""

    model: str
    dataset: str
    seed: int
    best_epoch: int
    best_valid_score: float
    select_on: str
    test: dict[str, float]
    valid: dict[str, float]
    test_pred_sha256_16: str
    history: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    env: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def checksum(x: np.ndarray) -> str:
    """Fingerprint of predictions — two runs are identical iff these match.

    float32 is also the storage dtype of `test_predictions.npy`, so this hash
    describes exactly what lands on disk.
    """
    return hashlib.sha256(
        np.ascontiguousarray(x, dtype=np.float32).tobytes()
    ).hexdigest()[:16]


class Trainer:
    def __init__(
        self,
        model: MSAModel,
        loaders: dict[str, DataLoader],
        device: torch.device,
        cfg: TrainConfig,
        dataset: str = "",
    ) -> None:
        missing = {"train", "valid", "test"} - set(loaders)
        if missing:
            raise KeyError(f"loaders is missing the {sorted(missing)} split(s)")
        self.model = model
        self.loaders = loaders
        self.device = device
        self.cfg = cfg
        self.dataset = dataset
        # Only CUDA has pinned host memory, so only there is an async copy real.
        self.non_blocking = supports_pin_memory(device)
        build = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = build(model.param_groups(cfg.lr, cfg.weight_decay))
        # None for every model but MMIM; see MSAModel.auxiliary_optimizer.
        self.aux_optimizer = model.auxiliary_optimizer(cfg.lr, cfg.weight_decay)
        self.scheduler = None
        if cfg.lr_schedule == "plateau":
            direction = "min" if cfg.select_on in LOWER_IS_BETTER else "max"
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=direction,
                factor=cfg.lr_schedule_factor, patience=cfg.lr_schedule_patience,
            )
        elif cfg.lr_schedule == "warmup_cosine":
            # ALMT: the rate climbs linearly over the first tenth of the epoch
            # budget, then anneals by cosine over the remaining nine tenths. Both
            # the reference and the authors compute the two spans from the
            # configured epoch cap, not from the epochs actually run — early
            # stopping shortens the run without rescaling the schedule.
            warmup = max(1, int(0.1 * cfg.epochs))
            # The factor is `epoch / warmup` counting from zero, so the FIRST
            # epoch runs at a learning rate of exactly 0 and trains nothing. That
            # is what the authors' GradualWarmupScheduler does — verified against
            # their scheduler.py epoch by epoch — and MMSA copied the class
            # unchanged. Reproduced rather than corrected; see
            # docs/investigations.md#almt-warmup.
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        self.optimizer, lr_lambda=lambda e, w=warmup: e / w
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=0.9 * cfg.epochs
                    ),
                ],
                milestones=[warmup + 1],
            )

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=self.non_blocking) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def evaluate(self, split: str) -> tuple[dict[str, float], np.ndarray]:
        """Metrics over the whole split, plus the predictions in dataset order.

        Evaluation loaders are never shuffled (see `msa.data.build_dataloaders`),
        so row i of the returned array belongs to sample i of that split.
        """
        self.model.eval()
        preds, trues = [], []
        for batch in self.loaders[split]:
            batch = self._to_device(batch)
            preds.append(self.model(batch)["M"].float().cpu().numpy())
            trues.append(batch["label"].float().cpu().numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        return eval_sentiment(preds, trues), preds

    def _clip(self) -> None:
        if not self.cfg.grad_clip:
            return
        if self.cfg.clip_mode == "value":
            nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.grad_clip)
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

    def auxiliary_epoch(self) -> float:
        """A full pass fitting the auxiliary objective, before the main epoch.

        Grads are zeroed on the whole model rather than on the optimiser: both
        passes clip over `model.parameters()`, so gradients belonging to
        parameters the stepping optimiser does not own still enter the norm.
        Leaving them to accumulate would quietly change the clipping factor.
        """
        total, seen = 0.0, 0
        for batch in self.loaders["train"]:
            batch = self._to_device(batch)
            self.model.zero_grad(set_to_none=True)
            loss = self.model.auxiliary_loss(batch)
            loss.backward()
            self._clip()
            self.aux_optimizer.step()
            n = batch["label"].numel()
            total += loss.item() * n
            seen += n
        return total / seen

    def train_one_epoch(self, epoch: int = 1) -> float:
        """One pass over the training split; returns the sample-weighted mean loss."""
        self.model.train()
        self.model.on_train_epoch_start(epoch)
        if self.aux_optimizer is not None:
            self.auxiliary_epoch()
        accumulate = self.cfg.accumulate_steps
        total, seen = 0.0, 0
        for step, batch in enumerate(self.loaders["train"]):
            batch = self._to_device(batch)
            if step % accumulate == 0:
                # Same reason as in auxiliary_epoch: with a second optimiser in
                # play, clipping sees parameters this one does not own.
                if self.aux_optimizer is not None:
                    self.model.zero_grad(set_to_none=True)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch)
            loss = self.model.compute_loss(outputs, batch)
            # Gradients are summed, not averaged, over the window — what MMSA
            # does; averaging would change the effective step size.
            loss.backward()
            self._clip()
            if (step + 1) % accumulate == 0:
                self.optimizer.step()
            self.model.on_train_batch_end(outputs, batch, epoch)
            n = batch["label"].numel()
            # Weight by batch size: the last batch is usually partial, and a plain
            # mean over batches would over-weight it (MMSA reports that variant).
            total += loss.item() * n
            seen += n
        return total / seen

    def fit(self, ckpt_path: Path, verbose: bool = True) -> tuple[RunResult, np.ndarray]:
        """Train, keep the best epoch by validation score, report it on test.

        The test split is touched once, after selection, and never influences
        which epoch is chosen.
        """
        cfg = self.cfg
        best_score, best_epoch, history = cfg.worst_score, -1, []
        start = perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            valid_metrics, _ = self.evaluate("valid")
            score = valid_metrics[cfg.select_on]
            history.append(
                {"epoch": epoch, "train_loss": train_loss,
                 **{f"valid_{k}": v for k, v in valid_metrics.items()}}
            )
            improved = cfg.is_better(score, best_score)
            if improved:
                best_score, best_epoch = score, epoch
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), ckpt_path)
            if verbose:
                print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                      f"valid_mae={valid_metrics['mae']:.4f}  "
                      f"valid_corr={valid_metrics['corr']:.4f}  "
                      f"valid_acc2={valid_metrics['acc2_non0']:.4f}"
                      f"{' *' if improved else ''}")
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(score)
            elif self.scheduler is not None:
                # Epoch-driven schedules take no metric.
                self.scheduler.step()
            if epoch - best_epoch >= cfg.patience:
                if verbose:
                    print(f"early stop: no valid improvement for {cfg.patience} epochs")
                break

        if best_epoch < 0:
            # Only reachable if every epoch scored NaN — no comparison ever wins.
            raise RuntimeError(
                f"no epoch produced a usable valid {cfg.select_on}; the model "
                "diverged (check the learning rate) — there is nothing to report"
            )

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        valid_metrics, _ = self.evaluate("valid")
        test_metrics, test_preds = self.evaluate("test")
        if not cfg.keep_checkpoint:
            # The checkpoint has done its job: the selected epoch's weights are
            # loaded and its predictions are about to be written to disk.
            ckpt_path.unlink(missing_ok=True)
        result = RunResult(
            model=getattr(self.model, "name", type(self.model).__name__),
            dataset=self.dataset,
            seed=cfg.seed,
            best_epoch=best_epoch,
            best_valid_score=best_score,
            select_on=cfg.select_on,
            test=test_metrics,
            valid=valid_metrics,
            test_pred_sha256_16=checksum(test_preds),
            history=history,
            elapsed_sec=perf_counter() - start,
            env=collect_env(self.device),
            config=asdict(cfg),
        )
        return result, test_preds


def save_run(
    result: RunResult, preds: np.ndarray, run_dir: Path, extra: dict | None = None
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result) | (extra or {})
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2))
    np.save(run_dir / "test_predictions.npy", preds)


def summarize(
    results: list[RunResult], keys: tuple[str, ...] = METRIC_KEYS
) -> dict[str, dict]:
    """Mean/std/min/max/per-seed across runs.

    `std` is the *sample* standard deviation (ddof=1): the unbiased estimate of
    the spread further seeds would show. numpy's ddof=0 default understates it by
    sqrt((n-1)/n) — 11% at n=5 — which matters here, because the whole argument
    in docs/roadmap.md is a comparison between that spread and published gaps.
    A single run has no spread to estimate, so its std is reported as 0.

    A metric missing from *any* run is dropped rather than raising: a group can
    outlive the metric set it was produced under, and rebuilding a summary over
    a group where one seed was re-run after `mse` was added should not fail. It
    is dropped rather than averaged over the subset, because a mean taken over
    a different number of seeds than its neighbours is a trap.
    """
    if not results:
        raise ValueError("nothing to summarize")
    summary = {}
    for key in keys:
        if not all(key in r.test for r in results):
            continue
        values = np.array([r.test[key] for r in results], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "n": int(values.size),
            "per_seed": {str(r.seed): float(r.test[key]) for r in results},
        }
    return summary


def format_summary(summary: dict[str, dict], n_seeds: int) -> str:
    lines = [
        f"{'metric':14s}{'mean':>10s}{'std':>9s}{'min':>10s}{'max':>10s}",
        f"(n={n_seeds} seeds; std is the sample standard deviation, ddof=1)",
    ]
    lines += [
        f"{k:14s}{v['mean']:10.4f}{v['std']:9.4f}{v['min']:10.4f}{v['max']:10.4f}"
        for k, v in summary.items()
    ]
    return "\n".join(lines)
