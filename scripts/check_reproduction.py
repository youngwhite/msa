"""Did re-training actually reproduce the committed predictions?

`verify_runs.py` answers a different question — whether each stored metric follows
from its stored predictions. This one compares the predictions themselves against
the ones committed to git, which is what catches a code change that silently
alters the numbers.

Checksums are device-specific by construction (CUDA / MPS / CPU reduce in
different orders), so a run made on different hardware than the committed one is
reported as a difference to expect, not as a failure.

Usage:
    python scripts/check_reproduction.py                # compare against HEAD
    python scripts/check_reproduction.py --ref HEAD~3   # or any other revision
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from msa.config import OUTPUT_ROOT, PROJECT_ROOT


def committed(path: Path, ref: str) -> dict | None:
    """Read a JSON file as of `ref`, or None if it is not in that revision."""
    relative = path.relative_to(PROJECT_ROOT)
    done = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "show", f"{ref}:{relative}"],
        capture_output=True, text=True, check=False,
    )
    return json.loads(done.stdout) if done.returncode == 0 else None


def device_of(run_dir: Path) -> str:
    result = run_dir / "result.json"
    if not result.exists():
        return "?"
    return json.loads(result.read_text()).get("env", {}).get("device_description", "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git revision to compare against")
    args = ap.parse_args()

    if not OUTPUT_ROOT.exists():
        raise SystemExit(f"no results at {OUTPUT_ROOT}; run scripts/reproduce_all.sh first")

    identical, different_hardware, mismatched, missing = [], [], [], []
    for summary_path in sorted(OUTPUT_ROOT.glob("*/summary.json")):
        group = summary_path.parent.name
        reference = committed(summary_path, args.ref)
        if reference is None:
            missing.append(f"{group}: not present in {args.ref}")
            continue
        current = json.loads(summary_path.read_text())
        ref_device = reference.get("env", {}).get("device_description", "?")
        now_device = current.get("env", {}).get("device_description", "?")

        for seed, expected_hash in reference["checksums"].items():
            actual_hash = current["checksums"].get(seed)
            label = f"{group}/seed{seed}"
            if actual_hash is None:
                missing.append(f"{label}: not re-run")
            elif actual_hash == expected_hash:
                identical.append(label)
            elif ref_device != now_device:
                different_hardware.append(f"{label}: {ref_device} -> {now_device}")
            else:
                mismatched.append(
                    f"{label}: committed {expected_hash}, now {actual_hash} "
                    f"(same device: {now_device})"
                )

    print(f"identical to {args.ref}: {len(identical)} run(s)")
    if different_hardware:
        print(f"\ndifferent hardware — a difference to expect, not a regression "
              f"({len(different_hardware)}):")
        for line in different_hardware:
            print(f"  ~ {line}")
        print("  Update docs/experiments.md with this machine's numbers if it "
              "becomes the reference.")
    if missing:
        print(f"\nnot compared ({len(missing)}):")
        for line in missing:
            print(f"  ? {line}")
    if mismatched:
        print(f"\n{len(mismatched)} RUN(S) CHANGED ON THE SAME HARDWARE:")
        for line in mismatched:
            print(f"  - {line}")
        print("\nThe code no longer produces the committed numbers. Either a change "
              "altered them (say so and re-document) or something regressed.")
        sys.exit(1)
    print("\nevery re-run matches the committed predictions")


if __name__ == "__main__":
    main()
