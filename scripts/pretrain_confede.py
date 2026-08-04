"""ConFEDE stage one: pretrain the three unimodal encoders.

The release trains a text, a vision and an audio encoder to regress sentiment on
their own, then the fusion stage loads all three (`TVA_fusion.load_model(
load_pretrain=True)`). Skipping it is not a small thing: without it the vision
and audio transformers start from random weights while BERT is frozen, and the
ten-seed run came out at MAE 1.1549 -- worse than this repository's own LSTM
baseline.

    python scripts/pretrain_confede.py --seed 42
    python scripts/pretrain_confede.py --seed 42 --modality text

Weights land in `outputs/confede_pretrain/seed<N>/<modality>_encoder.pt`, which
is git-ignored like `best.pt`: it is a reproducible intermediate, not a result.
Rerunning with the same seed reproduces it bit for bit.

This is a script rather than three more experiment groups, per the decision on
2026-08-04: pretraining is a means to the fusion run, not a number anyone
compares, and three extra rows in the results table would invite exactly that.

Protocol from the release's `train/constrastive/{T,V,A}train.py`:

* AdamW, lr 1e-4, weight decay 1e-3, over **all** parameters -- no separate BERT
  rate, and no no-decay group at this stage (that split exists only in fusion).
* Linear warmup over five epochs' worth of steps, then linear decay, counted in
  STEPS rather than epochs because that is how the release counts them.
* 200 epochs for text, 100 each for vision and audio; batch 128.
* Selection on **validation MAE**, which is both the release's `check={'MAE':
  ...}` and this repository's convention 2.
* **Text freezes BERT for the first third of training** (`train_all_epoch =
  total_epoch // 3`) and unfreezes it after. Vision and audio train everything
  throughout. That schedule is in the trainer, not the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from msa.config import OUTPUT_ROOT, get_dataset_spec
from msa.models.confede import BaseClassifier, SequenceEncoder, TextEncoder, padding_mask
from msa.repro import set_seed

PRETRAIN_ROOT = OUTPUT_ROOT / "confede_pretrain"

#: modality -> (epochs, warmup epochs). From config.py's {text,vision,audio}Pretrain.
SETTINGS = {"text": (200, 5), "vision": (100, 5), "audio": (100, 5)}
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WIDTH = 128, 1e-4, 1e-3, 768


class _Split(Dataset):
    """One split of the unaligned pickle, delivering exactly what each encoder needs."""

    def __init__(self, data: dict, modality: str) -> None:
        self.modality = modality
        self.labels = torch.as_tensor(np.asarray(data["regression_labels"])).float()
        if modality == "text":
            self.text = list(data["raw_text"])
        else:
            features = torch.as_tensor(np.asarray(data[modality])).float()
            self.features = features
            self.mask = padding_mask(features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if self.modality == "text":
            return self.text[index], self.labels[index]
        return self.features[index], self.mask[index], self.labels[index]


def build_model(modality: str, spec) -> tuple[nn.Module, nn.Module]:
    """The encoder the fusion stage will load, plus the head that trains it.

    The head is thrown away afterwards -- only the encoder is saved, which is
    what `TVA_fusion.load_model` reads.
    """
    if modality == "text":
        encoder: nn.Module = TextEncoder()
    else:
        dims = {"vision": spec.vision_dim, "audio": spec.audio_dim}
        # Unaligned MOSI: 500 vision frames, 375 audio. The positional table is
        # sized from these, so they must match what the fusion model builds.
        lengths = {"vision": 500, "audio": 375}
        encoder = SequenceEncoder(dims[modality], WIDTH, lengths[modality],
                                  heads=8, layers=2, dropout=0.5)
    head = BaseClassifier(WIDTH, [WIDTH // 2, WIDTH // 4, WIDTH // 8], 1)
    return encoder, head


def evaluate(encoder: nn.Module, head: nn.Module, loader: DataLoader,
             modality: str, device: torch.device) -> float:
    encoder.eval()
    head.eval()
    errors, seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if modality == "text":
                text, label = batch
                features = encoder(text)
            else:
                values, mask, label = batch
                features = encoder(values.to(device), mask.to(device))
            prediction = head(features).squeeze(-1)
            label = label.to(device)
            errors += float((prediction - label).abs().sum())
            seen += label.numel()
    return errors / seen


def pretrain(modality: str, seed: int, device: torch.device, quick: int | None) -> Path:
    import pickle

    spec = get_dataset_spec("mosi")
    with open(spec.unaligned_pkl, "rb") as handle:
        payload = pickle.load(handle)

    set_seed(seed)
    encoder, head = build_model(modality, spec)
    encoder.to(device)
    head.to(device)

    loaders = {
        split: DataLoader(_Split(payload[split], modality), batch_size=BATCH_SIZE,
                          shuffle=(split == "train"), num_workers=0)
        for split in ("train", "valid")
    }
    epochs, warmup_epochs = SETTINGS[modality]
    if quick:
        epochs = quick
    steps_per_epoch = len(loaders["train"])
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.MSELoss()

    # Text only: BERT is frozen for the first third, then unfrozen.
    freeze_until = epochs // 3 if modality == "text" else 0
    target = PRETRAIN_ROOT / f"seed{seed}" / f"{modality}_encoder.pt"
    target.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        if modality == "text":
            for parameter in encoder.extractor.parameters():
                parameter.requires_grad = epoch >= freeze_until
        encoder.train()
        head.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            if modality == "text":
                text, label = batch
                features = encoder(text)
            else:
                values, mask, label = batch
                features = encoder(values.to(device), mask.to(device))
            loss = criterion(head(features).squeeze(-1), label.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

        validation = evaluate(encoder, head, loaders["valid"], modality, device)
        if validation < best:
            best = validation
            torch.save(encoder.state_dict(), target)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  {modality} epoch {epoch:3d}/{epochs}  valid MAE {validation:.4f}"
                  f"  best {best:.4f}", flush=True)

    (target.parent / f"{modality}.json").write_text(json.dumps({
        "modality": modality, "seed": seed, "epochs": epochs,
        "best_valid_mae": round(best, 6),
        "selected_on": "validation MAE, as the release's check={'MAE': ...} does",
    }, indent=2) + "\n")
    print(f"  saved {target.relative_to(OUTPUT_ROOT.parent)}  (best valid MAE {best:.4f})")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modality", choices=sorted(SETTINGS), action="append",
                    help="repeatable; default is all three")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", type=int,
                    help="override the epoch count — smoke tests only, never for a result")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for modality in args.modality or list(SETTINGS):
        print(f"=== pretraining {modality} (seed {args.seed})", flush=True)
        pretrain(modality, args.seed, device, args.quick)


if __name__ == "__main__":
    main()
