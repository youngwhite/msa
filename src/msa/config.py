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
    #: sha256 of each file, relative to `root`. The features are not in git, so
    #: this is what tells a fresh machine it downloaded the same data — a
    #: truncated or re-extracted file would otherwise be silently trained on.
    file_sha256: dict[str, str]
    #: Where the files come from, printed when they are missing.
    source: str


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
        file_sha256={
            "Processed/aligned_50.pkl":
                "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9",
            "Processed/unaligned_50.pkl":
                "78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524",
            "label.csv":
                "dec8b0affc5c7c923688040f091dddb61966afc729f54630ef91d3f51624b821",
        },
        source=(
            "MMSA's preprocessed release (THUIAR) — the MOSI folder of\n"
            "  https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk\n"
            "  or https://pan.baidu.com/s/1a1bDX5htPsZjsRyHcvCKHw?pwd=qq0b (code qq0b)\n"
            "  Text features are BERT-based, not the GloVe ones from the original CMU SDK."
        ),
    )


DATASETS = {"mosi": mosi}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("cmu-", "").replace("-", "")
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}, available: {sorted(DATASETS)}")
    return DATASETS[key]()


def verify_files(spec: DatasetSpec) -> list[str]:
    """Check the dataset files against their recorded sha256.

    Returns a list of problems (empty means everything matches). Hashing ~900MB
    takes a few seconds, so this is called explicitly — by `scripts/setup.sh` on
    a new machine and by `check_data.py --verify-files` — not on every load.
    """
    import hashlib

    problems = []
    for relative, expected in spec.file_sha256.items():
        path = spec.root / relative
        if not path.exists():
            problems.append(f"missing: {path}")
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            problems.append(
                f"{path}\n    expected sha256 {expected}\n    actual   sha256 {actual}"
            )
    return problems
