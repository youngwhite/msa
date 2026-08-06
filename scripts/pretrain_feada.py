"""FeaDA stage one: pretrain the vision and audio encoders.

**Two encoders, not three.** The release ships `Vtrain.py` and `Atrain.py` and no
text equivalent: the fusion model loads vision and audio weights at construction
and freezes them with `set_froze()`, while BERT is fine-tuned during fusion from
its pretrained checkpoint.

    python scripts/pretrain_feada.py --seed 42
    python scripts/pretrain_feada.py --seed 42 --modality vision

Weights land in `outputs/feada_pretrain/seed<N>/<modality>_encoder.pt`,
git-ignored like `best.pt`: a reproducible intermediate, not a result.

Skipping this stage is not an option, and ConFEDE is why -- without its stage one
that model returned MAE 1.1549 against 0.7358, a wrong number that looked
entirely ordinary in a table. `FeaDA.on_run_start` therefore raises rather than
warns when the weights are absent.

Protocol from `train/{V,A}train.py`: `Adam(..., amsgrad=True)` at 1e-3 with decay
1e-3 -- amsgrad is not the default and is easy to drop -- 25 epochs, batch 32,
gradient accumulation 4, no schedule, selection on the validation regression
loss.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from msa.config import OUTPUT_ROOT, get_dataset_spec
from msa.models.feada import BaseClassifier, UnimodalEncoder, padding_mask
from msa.repro import set_seed

PRETRAIN_ROOT = OUTPUT_ROOT / "feada_pretrain"
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WIDTH, ACCUMULATE = 25, 32, 1e-3, 1e-3, 768, 4
LENGTHS = {"vision": 500, "audio": 375}


class _Split(Dataset):
    def __init__(self, data: dict, modality: str) -> None:
        features = torch.as_tensor(np.asarray(data[modality])).float()
        self.features = features
        self.mask = padding_mask(features)
        self.labels = torch.as_tensor(np.asarray(data["regression_labels"])).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.mask[index], self.labels[index]


def evaluate(encoder: nn.Module, head: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    encoder.eval()
    head.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for values, mask, label in loader:
            prediction = head(encoder(values.to(device), mask.to(device))).squeeze(-1)
            label = label.to(device)
            total += float(((prediction - label) ** 2).sum())
            seen += label.numel()
    return total / seen


def pretrain(modality: str, seed: int, device: torch.device, quick: int | None) -> None:
    spec = get_dataset_spec("mosi")
    with open(spec.unaligned_pkl, "rb") as handle:
        payload = pickle.load(handle)

    set_seed(seed)
    dims = {"vision": spec.vision_dim, "audio": spec.audio_dim}
    encoder = UnimodalEncoder(dims[modality], WIDTH, LENGTHS[modality],
                              heads=8, layers=2, dropout=0.5).to(device)
    head = BaseClassifier(WIDTH, [WIDTH // 2, WIDTH // 4, WIDTH // 8], 1).to(device)

    loaders = {split: DataLoader(_Split(payload[split], modality), batch_size=BATCH_SIZE,
                                 shuffle=(split == "train"), num_workers=0)
               for split in ("train", "valid")}
    # amsgrad=True is the release's, not the default, and changes the update rule.
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()],
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)
    criterion = nn.MSELoss()

    target = PRETRAIN_ROOT / f"seed{seed}" / f"{modality}_encoder.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    total_epochs = quick or EPOCHS
    for epoch in range(1, total_epochs + 1):
        encoder.train()
        head.train()
        for step, (values, mask, label) in enumerate(loaders["train"]):
            if step % ACCUMULATE == 0:
                optimizer.zero_grad()
            loss = criterion(head(encoder(values.to(device), mask.to(device))).squeeze(-1),
                             label.to(device))
            loss.backward()
            if (step + 1) % ACCUMULATE == 0:
                optimizer.step()
        validation = evaluate(encoder, head, loaders["valid"], device)
        if validation < best:
            best = validation
            torch.save(encoder.state_dict(), target)
        if epoch % 5 == 0 or epoch == total_epochs:
            print(f"  {modality} epoch {epoch:3d}  valid loss {validation:.4f}"
                  f"  best {best:.4f}", flush=True)

    (target.parent / f"{modality}.json").write_text(json.dumps({
        "modality": modality, "seed": seed, "epochs": total_epochs,
        "best_valid_loss": round(best, 6),
        "selected_on": "validation regression loss, as the release does",
    }, indent=2) + "\n")
    print(f"  saved {target.name}  (best valid loss {best:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modality", choices=sorted(LENGTHS), action="append")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", type=int, help="smoke tests only, never for a result")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for modality in args.modality or list(LENGTHS):
        print(f"=== pretraining {modality} (seed {args.seed})", flush=True)
        pretrain(modality, args.seed, device, args.quick)


if __name__ == "__main__":
    main()
