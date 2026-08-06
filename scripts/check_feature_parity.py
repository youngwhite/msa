"""Do all the compared groups actually run on the same features?

Having author code is necessary and not sufficient. EBMC (CVPR 2026) has a
release and uses CMU-MOSI, and still cannot join the comparison: it runs on
DeBERTa-large text, wav2vec-large audio and MANet vision at utterance level,
while every group here runs on BERT-base with 5-dimensional COVAREP and
20-dimensional Facet sequences. Those are different inputs, not a different
preprocessing of the same ones, and a number obtained on stronger features
sitting in the same column reads as a better method when it may only be a better
encoder.

That check was being done by eye, per paper, which is exactly how the third
thing gets forgotten. This makes it mechanical:

    python scripts/check_feature_parity.py            # report
    python scripts/check_feature_parity.py --check    # exit non-zero on a mismatch

Three things are verified for every acceptance group:

* the dataset is the one this repository ships, and the tensors come from its
  pickles -- guaranteed structurally, since `msa.data` reads nothing else, and
  asserted here so that stops being an argument from memory;
* which pickle (aligned or unaligned) each group used, because those are also
  different inputs and the table must say which;
* the text encoder each model builds, because a model may re-encode raw text
  itself. ConFEDE tokenises raw strings rather than consuming `text_bert`, which
  is a different code path to the same BERT-base weights -- fine, and worth
  showing rather than assuming.

`check_data.py` already verifies the pickles themselves by sha256. This is the
layer above: not "is the file the right file" but "did every compared number
come from it".
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys

from msa.config import OUTPUT_ROOT, PROJECT_ROOT, get_dataset_spec

EXPECTED_DATASET = "CMU-MOSI"
EXPECTED_TEXT_ENCODER = "bert-base-uncased"
EXPECTED_DIMS = (768, 5, 20)


def acceptance_groups() -> list[dict]:
    groups = []
    for path in sorted(OUTPUT_ROOT.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        dataset = payload.get("dataset", "").lower().replace("cmu-", "")
        if path.parent.name != f"{payload.get('model', '')}_{dataset}":
            continue                      # ablations and diagnostics, not compared
        payload["group"] = path.parent.name
        groups.append(payload)
    return groups


def text_encoder_of(model_name: str) -> str:
    """What the model would build, read from its own signature.

    Read rather than declared in a table here: a table would be one more thing
    that can silently disagree with the code.
    """
    from msa.registry import get_model_class

    try:
        cls = get_model_class(model_name)
    except (KeyError, ValueError):
        return "unknown"
    for parameter in inspect.signature(cls.__init__).parameters.values():
        if parameter.name == "pretrained" and parameter.default is not inspect.Parameter.empty:
            return str(parameter.default)
    source = inspect.getsource(cls)
    if "BertTextEncoder" in source or "BertModel" in source:
        return EXPECTED_TEXT_ENCODER
    return "none (no text encoder)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit non-zero on a mismatch")
    args = ap.parse_args()

    spec = get_dataset_spec("mosi")
    problems: list[str] = []
    rows = []
    for payload in acceptance_groups():
        model = payload.get("model", "?")
        encoder = text_encoder_of(model)
        aligned = "aligned" if payload.get("aligned") else "unaligned"
        rows.append((payload["group"], aligned, encoder))
        if payload.get("dataset") != EXPECTED_DATASET:
            problems.append(f"{payload['group']}: dataset is {payload.get('dataset')!r}")
        if encoder not in (EXPECTED_TEXT_ENCODER, "none (no text encoder)"):
            problems.append(f"{payload['group']}: text encoder is {encoder!r}")

    if (spec.text_dim, spec.audio_dim, spec.vision_dim) != EXPECTED_DIMS:
        problems.append(f"dataset dims are {(spec.text_dim, spec.audio_dim, spec.vision_dim)}, "
                        f"expected {EXPECTED_DIMS}")

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"{len(rows)} acceptance group(s); dataset {EXPECTED_DATASET} "
          f"{EXPECTED_DIMS} from {spec.aligned_pkl.name} / {spec.unaligned_pkl.name}\n")
    by_alignment: dict[str, list[str]] = {}
    for group, aligned, encoder in rows:
        by_alignment.setdefault(aligned, []).append(group)
        print(f"  {group:{width}s}  {aligned:9s}  {encoder}")
    print("\n" + "  ".join(f"{k}: {len(v)}" for k, v in sorted(by_alignment.items())))

    if problems:
        print("\nMISMATCH:")
        for problem in problems:
            print(f"  ! {problem}")
        if args.check:
            sys.exit(1)
    else:
        print("\nall compared groups share one feature pipeline")
    print(f"\n(see {PROJECT_ROOT.name}/docs/survey.md for why EBMC is excluded on this basis)")


if __name__ == "__main__":
    main()
