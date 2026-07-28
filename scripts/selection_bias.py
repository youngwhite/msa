"""How much does picking hyper-parameters on the test set flatter the result?

MMSA's tuning branch evaluates candidate configurations on the **test** split —
the line that would have used valid is commented out (docs/investigations.md
#mmsa-test-selection). Its published table therefore carries an unknown amount
of selection optimism. This measures that amount on our own implementation, so
the number is ours and not an accusation.

Design, fixed before running:

* Model: LMF. Cheapest to train (~12s/seed) and already reproduces MMSA, so the
  starting point is not itself in question.
* Space: MMSA's own `config_tune.json` entry for lmf, verbatim.
* N configurations sampled without replacement, K seeds each, all trained
  identically. No configuration is discarded for any reason.
* Two selections over the same runs:
    - valid-selection: argmin of mean **valid** MAE -> report its test MAE
    - test-selection:  argmin of mean **test** MAE  -> report that test MAE
  The difference is the optimism. It is an estimate of how much of a reported
  number can come from the choosing rather than the model.
* Reported at K seeds (primary, conservative) and at K=1 (secondary, closer to
  MMSA's protocol, where the noise available to exploit is larger).
* Acc-2(non0) alongside MAE, selected on its own metric.

Sampling uses a fixed seed so the configuration set is itself reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from msa.config import DATASETS
from msa.data import build_dataloaders
from msa.device import resolve_device
from msa.registry import build_model
from msa.repro import set_seed
from msa.trainer import TrainConfig, Trainer

# Verbatim from MMSA config_tune.json, "lmf" -> debugParams.
SPACE = {
    "hidden_dims": [[128, 16, 128], [64, 16, 64], [128, 32, 128], [256, 32, 256], [64, 32, 64]],
    "dropouts": [[0.3, 0.3, 0.3, 0.5], [0.3, 0.3, 0.3, 0.3], [0.2, 0.2, 0.2, 0.2],
                 [0.4, 0.4, 0.4, 0.4]],
    "rank": [3, 4, 5, 6],
    "batch_size": [32, 64, 128],
    "learning_rate": [0.0005, 0.001, 0.002, 0.005],
    "factor_lr": [0.0001, 0.0005, 0.001],
    "weight_decay": [0.0, 0.0001, 0.001, 0.005],
}
#: What MMSA settled on for mosi — every value is inside the space above.
MMSA_CHOICE = {
    "hidden_dims": [128, 16, 128], "dropouts": [0.3, 0.3, 0.3, 0.3], "rank": 3,
    "batch_size": 64, "learning_rate": 0.001, "factor_lr": 0.001, "weight_decay": 0.005,
}


def sample_configs(n: int, rng: random.Random) -> list[dict]:
    keys = list(SPACE)
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: rng.choice(SPACE[k]) for k in keys}
        token = json.dumps(cfg, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        out.append(cfg)
    return out


def run_one(cfg: dict, seed: int, device) -> dict:
    set_seed(seed, deterministic=True)
    loaders, spec = build_dataloaders(
        "mosi", aligned=False, batch_size=cfg["batch_size"], num_workers=0,
        seed=seed, device=device,
    )
    text_h, audio_h, vision_h = cfg["hidden_dims"]
    audio_d, vision_d, text_d, _ = cfg["dropouts"]
    model = build_model(
        "lmf", spec,
        text_hidden=text_h, audio_hidden=audio_h, vision_hidden=vision_h,
        rank=cfg["rank"], text_dropout=text_d, audio_dropout=audio_d,
        vision_dropout=vision_d, factor_lr=cfg["factor_lr"],
        use_lengths=False, mask_pooling=False,          # the faithful port
    ).to(device)
    tcfg = TrainConfig(
        epochs=200, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        grad_clip=0.0, patience=8, select_on="mae", seed=seed, keep_checkpoint=False,
    )
    result, _ = Trainer(model, loaders, device, tcfg, dataset=spec.name).fit(
        Path("/tmp/_sel_bias.pt"), verbose=False
    )
    return {"valid": result.valid, "test": result.test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="/tmp/selection_bias.json")
    args = ap.parse_args()

    device = resolve_device("cuda")
    configs = sample_configs(args.configs, random.Random(20260728))
    records = []
    for i, cfg in enumerate(configs, 1):
        runs = [run_one(cfg, s, device) for s in args.seeds]
        records.append({"config": cfg, "runs": runs})
        v = np.mean([r["valid"]["mae"] for r in runs])
        t = np.mean([r["test"]["mae"] for r in runs])
        print(f"[{i:3d}/{len(configs)}] valid_mae {v:.4f}  test_mae {t:.4f}  {cfg}",
              flush=True)
    Path(args.out).write_text(json.dumps(records, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
