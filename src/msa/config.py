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


def mosei() -> DatasetSpec:
    """CMU-MOSEI, the dataset the roadmap intends as primary.

    Ten times MOSI's size, and that is the point: phase 2 established that a
    20-seed experiment on MOSI resolves about 0.012 MAE while the effects it was
    chasing sit below that (docs/investigations.md#contrastive-below-floor).

    Every number here was read off the downloaded pickles rather than taken from
    the paper, because the dimensions differ from MOSI's in ways that would
    otherwise be easy to mistype: **audio is 74 and vision 35**, against MOSI's
    5 and 20. A model built from the wrong numbers fails with a shape error deep
    inside itself, which is the good case; the bad case is a plausible result on
    misread data.

    Verified before this spec was written, since `msa.data` assumes it: in the
    aligned pickle every audio and vision frame past `text_length` is exactly
    zero, across all 22,856 samples in the three splits. That assumption was
    previously checked only on MOSI.
    """
    root = DATASET_ROOT / "CMU-MOSEI"
    return DatasetSpec(
        name="CMU-MOSEI",
        root=root,
        aligned_pkl=root / "Processed" / "aligned_50.pkl",
        unaligned_pkl=root / "Processed" / "unaligned_50.pkl",
        label_csv=root / "label.csv",
        text_dim=768,   # frozen BERT-base token embeddings, as MOSI
        audio_dim=74,   # COVAREP — not MOSI's 5
        vision_dim=35,  # Facet — not MOSI's 20
        split_sizes={"train": 16326, "valid": 1871, "test": 4659},
        file_sha256={
            "Processed/aligned_50.pkl":
                "45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791",
            "Processed/unaligned_50.pkl":
                "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f",
            "label.csv":
                "52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521",
        },
        source=(
            "MMSA's preprocessed release (THUIAR) — the MOSEI folder of\n"
            "  https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk\n"
            "  or https://pan.baidu.com/s/1a1bDX5htPsZjsRyHcvCKHw?pwd=qq0b (code qq0b)\n"
            "  18GB in total: aligned 4.7GB, unaligned 13.7GB. Run\n"
            "  `bash scripts/fetch_dataset.sh mosei`, which handles the "
            "virus-scan\n  interstitial Drive serves for files this large."
        ),
    )


DATASETS = {"mosi": mosi, "mosei": mosei}


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
