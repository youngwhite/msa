"""One training loop for every model.

MMSA carries 14 near-identical trainers (2325 lines; TFN's and LMF's differ by 14
lines after renaming), which is how evaluation protocols quietly drift apart
between models. Here a model customises behaviour through `MSAModel`
(`compute_loss`, `param_groups`) and everything else — selection metric, early
stopping, checkpointing, provenance — is shared and therefore comparable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .device import supports_pin_memory
from .metrics import eval_sentiment
from .models.base import MSAModel
from .repro import collect_env

LOWER_IS_BETTER = {"mae"}


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8
    select_on: str = "mae"
    seed: int = 42

    def better(self, candidate: float, incumbent: float) -> bool:
        if self.select_on in LOWER_IS_BETTER:
            return candidate < incumbent
        return candidate > incumbent

    @property
    def worst_score(self) -> float:
        return float("inf") if self.select_on in LOWER_IS_BETTER else float("-inf")


@dataclass
class RunResult:
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
    """Fingerprint of predictions — two runs are identical iff these match."""
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
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.device = device
        self.cfg = cfg
        self.non_blocking = supports_pin_memory(device)
        self.optimizer = torch.optim.Adam(
            model.param_groups(cfg.lr, cfg.weight_decay)
        )

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=self.non_blocking)
            if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def evaluate(self, split: str) -> tuple[dict[str, float], np.ndarray]:
        self.model.eval()
        preds, trues = [], []
        for batch in self.loaders[split]:
            batch = self._to_device(batch)
            out = self.model(batch)["M"]
            preds.append(out.float().cpu().numpy())
            trues.append(batch["label"].float().cpu().numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        return eval_sentiment(preds, trues), preds

    def train_one_epoch(self) -> float:
        self.model.train()
        total, seen = 0.0, 0
        for batch in self.loaders["train"]:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(self.model(batch), batch)
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            n = batch["label"].numel()
            total += loss.item() * n
            seen += n
        return total / seen

    def fit(self, ckpt_path: Path, verbose: bool = True) -> tuple[RunResult, np.ndarray]:
        cfg = self.cfg
        best_score, best_epoch, history = cfg.worst_score, -1, []
        start = time.time()

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self.train_one_epoch()
            valid_metrics, _ = self.evaluate("valid")
            score = valid_metrics[cfg.select_on]
            history.append(
                {"epoch": epoch, "train_loss": train_loss,
                 **{f"valid_{k}": v for k, v in valid_metrics.items()}}
            )
            improved = cfg.better(score, best_score)
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
            if epoch - best_epoch >= cfg.patience:
                if verbose:
                    print(f"early stop: no valid improvement for {cfg.patience} epochs")
                break

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        valid_metrics, _ = self.evaluate("valid")
        test_metrics, test_preds = self.evaluate("test")
        return RunResult(
            model=getattr(self.model, "name", type(self.model).__name__),
            dataset="",
            seed=cfg.seed,
            best_epoch=best_epoch,
            best_valid_score=best_score,
            select_on=cfg.select_on,
            test=test_metrics,
            valid=valid_metrics,
            test_pred_sha256_16=checksum(test_preds),
            history=history,
            elapsed_sec=time.time() - start,
            env=collect_env(self.device),
            config=asdict(cfg),
        ), test_preds


def save_run(
    result: RunResult, preds: np.ndarray, run_dir: Path, extra: dict | None = None
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    if extra:
        payload.update(extra)
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2))
    np.save(run_dir / "test_predictions.npy", preds)


def summarize(results: list[RunResult], keys: tuple[str, ...] = (
    "mae", "corr", "acc2_non0", "f1_non0", "acc2_has0", "acc7",
)) -> dict[str, dict[str, float]]:
    """Mean/std/per-seed for a set of runs.

    Single-seed numbers are not reportable on these datasets: our LF-LSTM varies
    by ±0.035 MAE across seeds, which is wider than most published gaps.
    """
    summary = {}
    for key in keys:
        values = np.array([r.test[key] for r in results], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "per_seed": {str(r.seed): float(r.test[key]) for r in results},
        }
    return summary


def format_summary(summary: dict[str, dict[str, float]], n_seeds: int) -> str:
    head = f"{'metric':14s}{'mean':>10s}{'std':>9s}{'min':>10s}{'max':>10s}   (n={n_seeds})"
    rows = [
        f"{k:14s}{v['mean']:10.4f}{v['std']:9.4f}{v['min']:10.4f}{v['max']:10.4f}"
        for k, v in summary.items()
    ]
    return "\n".join([head, *rows])
