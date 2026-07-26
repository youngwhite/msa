"""Loading of MMSA-style pre-extracted features (CMU-MOSI / CMU-MOSEI)."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .config import DatasetSpec, get_dataset_spec
from .device import supports_pin_memory
from .repro import seed_worker

SPLITS = ("train", "valid", "test")


@lru_cache(maxsize=4)
def _load_pickle_cached(resolved_path: str) -> dict:
    with open(resolved_path, "rb") as f:
        return pickle.load(f)


def load_pickle(path: str | Path) -> dict:
    """Read one MMSA feature pickle, cached by resolved path.

    The cache key is normalised: these files are ~400MB each, and caching the
    same file once as `Path` and once as `str` would silently double that.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"feature file not found: {path}\n"
            "Download the MMSA-preprocessed dataset into datasets/<NAME>/Processed/."
        )
    return _load_pickle_cached(str(path))


load_pickle.cache_clear = _load_pickle_cached.cache_clear
load_pickle.cache_info = _load_pickle_cached.cache_info


def _clean(x: np.ndarray) -> np.ndarray:
    """MMSA audio/vision features carry NaN/Inf in a few frames; zero them out."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class MMSADataset(Dataset):
    """One split of a dataset, with text/audio/vision features and labels.

    Each item is a dict of tensors:
        text   (seq_t, text_dim)    frozen BERT token embeddings
        audio  (seq_a, audio_dim)
        vision (seq_v, vision_dim)
        text_length / audio_length / vision_length  valid step counts
        text_bert (3, seq_t)        input_ids / attention_mask / token_type_ids
        label      scalar regression target in [-3, 3]
        label_7    scalar 7-way class index (label rounded, shifted to 0..6)

    Note the pickle's own `classification_labels` field is a 3-way sign label
    (`sign(regression) + 1`, so 0/1/2 = neg/exactly-zero/pos), *not* the 7-way
    index above; we derive `label_7` ourselves and leave that field unused.
    """

    def __init__(self, spec: DatasetSpec, split: str, aligned: bool = True) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.spec = spec
        self.split = split
        self.aligned = aligned

        path = spec.aligned_pkl if aligned else spec.unaligned_pkl
        data = load_pickle(path)
        if split not in data:
            raise KeyError(f"{path} has splits {list(data)}, no {split!r}")
        d = data[split]

        self.text = torch.from_numpy(_clean(d["text"]).astype(np.float32))
        self.audio = torch.from_numpy(_clean(d["audio"]).astype(np.float32))
        self.vision = torch.from_numpy(_clean(d["vision"]).astype(np.float32))
        self.text_bert = torch.from_numpy(np.asarray(d["text_bert"]).astype(np.int64))
        self.labels = torch.from_numpy(
            np.asarray(d["regression_labels"]).astype(np.float32)
        )
        self.ids = list(d["id"])
        self.raw_text = list(d["raw_text"])
        self.annotations = list(d["annotations"])

        # Number of real BERT tokens; the rest of the 50 steps are [PAD], whose
        # embeddings are *not* zero, so consumers must not read past this.
        # text_bert rows are (input_ids, attention_mask, token_type_ids).
        self.text_lengths = self.text_bert[:, 1, :].sum(dim=1).clamp(min=1).long()

        # 7-way class index, precomputed once: round to the nearest integer score
        # in [-3, 3] and shift to 0..6. Unused by the regression baselines, kept
        # for classification heads.
        self.labels_7 = (torch.clamp(torch.round(self.labels), -3, 3) + 3).long()

        if aligned:
            # Word-aligned: audio/vision share the text timeline. Verified on
            # CMU-MOSI that every frame past text_length is exactly zero.
            self.audio_lengths = self.text_lengths.clone()
            self.vision_lengths = self.text_lengths.clone()
        else:
            self.audio_lengths = torch.tensor(d["audio_lengths"], dtype=torch.long).clamp(
                min=1, max=self.audio.shape[1]
            )
            self.vision_lengths = torch.tensor(d["vision_lengths"], dtype=torch.long).clamp(
                min=1, max=self.vision.shape[1]
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "text_bert": self.text_bert[idx],
            "text_length": self.text_lengths[idx],
            "audio_length": self.audio_lengths[idx],
            "vision_length": self.vision_lengths[idx],
            "label": self.labels[idx],
            "label_7": self.labels_7[idx],
        }

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            "text": self.text.shape[-1],
            "audio": self.audio.shape[-1],
            "vision": self.vision.shape[-1],
        }


def build_dataloaders(
    dataset: str = "mosi",
    aligned: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[dict[str, DataLoader], DatasetSpec]:
    """One DataLoader per split; only `train` is shuffled.

    Shuffling uses a generator owned by the sampler, deliberately *not* the
    DataLoader's own generator: the loader also draws from that one to seed
    workers, and it draws a different number of times depending on num_workers
    and persistent_workers. Keeping the sampler's stream private makes the batch
    order a function of the seed alone — identical across worker counts and
    across devices.
    """
    spec = get_dataset_spec(dataset)
    pin = supports_pin_memory(device) if device is not None else False
    loaders, datasets = {}, {}
    for split in SPLITS:
        ds = MMSADataset(spec, split, aligned=aligned)
        datasets[split] = ds
        sampler = None
        if split == "train":
            sampler = RandomSampler(
                ds, generator=torch.Generator().manual_seed(seed)
            )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed + 1),
            worker_init_fn=seed_worker if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    check_matches_spec(datasets, spec)
    return loaders, spec


def check_matches_spec(datasets: dict[str, MMSADataset], spec: DatasetSpec) -> None:
    """Fail loudly if the feature file is not the dataset the spec describes.

    Models are constructed from `spec`'s dimensions, so a mismatch here would
    otherwise surface as an obscure shape error inside a model — or not at all,
    if the shapes happen to line up on different data.
    """
    problems = []
    for split, ds in datasets.items():
        expected = spec.split_sizes.get(split)
        if expected is not None and len(ds) != expected:
            problems.append(f"{split} has {len(ds)} samples, spec says {expected}")
    dims = next(iter(datasets.values())).feature_dims
    for modality, declared in (
        ("text", spec.text_dim), ("audio", spec.audio_dim), ("vision", spec.vision_dim)
    ):
        if dims[modality] != declared:
            problems.append(
                f"{modality} features are {dims[modality]}-dimensional, spec says {declared}"
            )
    if problems:
        raise ValueError(
            f"{spec.name} does not match its DatasetSpec:\n  "
            + "\n  ".join(problems)
            + f"\nEither the feature files were replaced or {spec.name}'s entry in "
              "msa/config.py is stale — fix one of them before trusting any result."
        )
