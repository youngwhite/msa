"""Paths and dataset-level constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives and what its feature tensors look like.

    The dimensions and split sizes are declared here and checked against the
    pickle at load time (`msa.data.build_dataloaders`). Models are built from
    these numbers, so a feature file that silently disagrees with them — a
    re-extracted set of features, a half-downloaded pickle — would otherwise
    produce either a confusing shape error deep in a model or, worse, a
    plausible-looking result on the wrong data.
    """

    name: str
    root: Path
    aligned_pkl: Path
    unaligned_pkl: Path
    label_csv: Path
    text_dim: int
    audio_dim: int
    vision_dim: int
    #: Expected number of samples per split, as published for the dataset.
    split_sizes: dict[str, int]


def mosi() -> DatasetSpec:
    root = DATASET_ROOT / "CMU-MOSI"
    return DatasetSpec(
        name="CMU-MOSI",
        root=root,
        aligned_pkl=root / "Processed" / "aligned_50.pkl",
        unaligned_pkl=root / "Processed" / "unaligned_50.pkl",
        label_csv=root / "label.csv",
        text_dim=768,   # frozen BERT-base token embeddings
        audio_dim=5,    # COVAREP
        vision_dim=20,  # Facet
        split_sizes={"train": 1284, "valid": 229, "test": 686},
    )


DATASETS = {"mosi": mosi}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("cmu-", "").replace("-", "")
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}, available: {sorted(DATASETS)}")
    return DATASETS[key]()
