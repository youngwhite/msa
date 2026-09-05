"""A float32 sidecar for the feature pickles, memory-mapped rather than loaded.

The pickles store audio and vision as float64 and every consumer wants float32,
so `MMSADataset` used to hold both at once: MOSEI's unaligned features measure
12.6GB unpickled against 7.9GB of tensors built from them, a peak of 22.9GB
(measured). On a machine whose free memory sits near 33GB that peak got four
long runs killed by a memory watchdog.

This writes the arrays the dataset actually wants -- cleaned, cast, contiguous --
once, and memory-maps them thereafter. Steady-state RSS drops to almost nothing:
the pages a batch touches are read on demand and the rest stay in the page cache,
which is reclaimable.

**Building it is also chunked**, and that is the point of `_copy_in_chunks`. The
naive build has the same 22.9GB peak it is meant to remove, because it holds the
source array and the destination at once. Filling the destination a few thousand
rows at a time keeps the float32 side near zero, so the build peaks at roughly
the pickle alone.

**Invalidation is by the pinned hash, not by re-hashing.** `DatasetSpec` already
records each file's sha256 and `check_data --verify-files` is what confirms the
file matches it; re-hashing 13.65GB on every load would cost more than the load.
So the cache records the *pinned* hash for its source file, plus that file's size
and mtime. A new data release changes the pinned hash and the cache rebuilds; a
file swapped underneath without the spec changing is caught by check_data, which
is where that check belongs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .config import DatasetSpec

#: Bumped when the set of stored arrays or their meaning changes, so an older
#: cache is rebuilt rather than misread.
FORMAT_VERSION = 1
#: Rows per copy. 4096 rows of MOSEI unaligned audio is ~90MB, small enough that
#: the build peak is the pickle and nothing else.
CHUNK_ROWS = 4096

#: float32 for anything a model consumes as a float, int64 for indices.
FLOAT_FIELDS = ("text", "audio", "vision")
INT_FIELDS = ("text_bert",)
#: Present only in the unaligned pickles.
OPTIONAL_INT_FIELDS = ("audio_lengths", "vision_lengths")
#: Python objects rather than arrays; small, so they go in one pickle.
OBJECT_FIELDS = ("id", "raw_text", "annotations")


def cache_root(spec: DatasetSpec) -> Path:
    return spec.root / ".f32cache"


def cache_dir(spec: DatasetSpec, split: str, aligned: bool) -> Path:
    setting = "aligned" if aligned else "unaligned"
    return cache_root(spec) / f"v{FORMAT_VERSION}_{setting}_{split}"


def _source(spec: DatasetSpec, aligned: bool) -> tuple[Path, str]:
    """The pickle for this setting, and its key in `spec.file_sha256`."""
    path = spec.aligned_pkl if aligned else spec.unaligned_pkl
    key = str(path.relative_to(spec.root))
    return path, key


def _expected_meta(spec: DatasetSpec, split: str, aligned: bool) -> dict:
    path, key = _source(spec, aligned)
    stat = path.stat()
    return {
        "format_version": FORMAT_VERSION,
        "dataset": spec.name,
        "split": split,
        "aligned": aligned,
        "source": key,
        "source_sha256": spec.file_sha256.get(key),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _is_valid(directory: Path, expected: dict) -> bool:
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        return False
    try:
        found = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    # mtime is excluded on purpose: copying a dataset between machines changes it
    # while the bytes are identical, and the pinned hash is the real identity.
    keys = [k for k in expected if k != "source_mtime_ns"]
    return all(found.get(k) == expected[k] for k in keys)


def _copy_in_chunks(source: np.ndarray, destination: np.ndarray, clean: bool) -> None:
    """Fill `destination` from `source` a chunk of rows at a time.

    Casting the whole array in one expression would materialise a second full
    copy, which is exactly the peak this module exists to avoid.
    """
    for start in range(0, source.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, source.shape[0])
        block = np.asarray(source[start:stop], dtype=destination.dtype)
        if clean:
            # MMSA's audio/vision carry NaN/Inf in a few frames.
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        destination[start:stop] = block


def build(spec: DatasetSpec, split: str, aligned: bool) -> Path:
    """Write the cache for one split. Returns its directory."""
    from .data import load_pickle  # deferred: data imports this module

    path, _ = _source(spec, aligned)
    directory = cache_dir(spec, split, aligned)
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        for leftover in staging.iterdir():
            leftover.unlink()
    staging.mkdir(parents=True, exist_ok=True)

    data = load_pickle(path)
    if split not in data:
        raise KeyError(f"{path} has splits {list(data)}, no {split!r}")
    d = data[split]

    fields = [(name, np.float32, True) for name in FLOAT_FIELDS]
    fields += [(name, np.int64, False) for name in INT_FIELDS]
    fields += [(name, np.int64, False) for name in OPTIONAL_INT_FIELDS if name in d]
    fields += [("regression_labels", np.float32, False)]

    for name, dtype, clean in fields:
        source = d[name]
        shape = np.shape(source)
        out = np.lib.format.open_memmap(
            staging / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        if len(shape) == 0:
            out[()] = np.asarray(source, dtype=dtype)
        else:
            _copy_in_chunks(np.asarray(source), out, clean)
        out.flush()
        del out

    (staging / "objects.pkl").write_bytes(
        pickle.dumps({name: list(d[name]) for name in OBJECT_FIELDS if name in d})
    )
    (staging / "meta.json").write_text(
        json.dumps(_expected_meta(spec, split, aligned), indent=2) + "\n"
    )
    # Rename last: a cache is either complete or absent, never half-written.
    if directory.exists():
        for leftover in directory.iterdir():
            leftover.unlink()
        directory.rmdir()
    staging.rename(directory)
    return directory


def load(spec: DatasetSpec, split: str, aligned: bool) -> dict:
    """The cached arrays, memory-mapped, building the cache first if needed."""
    directory = cache_dir(spec, split, aligned)
    expected = _expected_meta(spec, split, aligned)
    if not _is_valid(directory, expected):
        print(f"  building float32 cache: {directory.relative_to(spec.root.parent)}")
        directory = build(spec, split, aligned)

    out: dict[str, object] = {}
    for npy in sorted(directory.glob("*.npy")):
        out[npy.stem] = np.load(npy, mmap_mode="r")
    out.update(pickle.loads((directory / "objects.pkl").read_bytes()))
    return out
