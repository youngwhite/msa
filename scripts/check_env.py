"""Is this environment the one the recorded numbers were produced in?

Every other gate reads results that already exist. None of them constructs a
model, installs anything, or looks at the interpreter, so an environment that
cannot train at all still goes green. That gap has now cost this project two
machine moves:

* 2026-08-01: `transformers` was in nobody's dependency list, so `setup.sh`
  built an environment in which eight of the fourteen models could not be
  instantiated. Every gate passed.
* 2026-09-04: on a machine whose system python was 3.10, `pip install -r
  requirements-lock.txt` could not resolve at all (`networkx==3.6.1` needs
  >=3.11), so `setup.sh` failed before reaching the gates -- and the version
  requirement was written down nowhere.

Both are the same shape: the environment is a silent input to every number in
docs/, and nothing checked it. This does.

What it asserts:

1. The interpreter's minor version matches the one every committed run recorded.
   Not cosmetic -- it decides which wheels resolve, and a different numpy or
   scipy build is exactly the kind of thing that moves the last bits.
2. Every pin in requirements-lock.txt is installed at that exact version.
   Catches both a missing package and a drifted one.
3. torch imports and reports a device.

torch itself is deliberately not version-checked: its wheel is platform-specific
and deliberately absent from the lock file (see the header of scripts/setup.sh).
Its version is recorded per run instead.

Exit code 0 if the environment matches, non-zero otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "requirements-lock.txt"
OUTPUTS = REPO / "outputs"

#: The interpreter every committed result was produced under. Read from the
#: results themselves rather than hard-coded, so it cannot drift out of date --
#: but pinned to major.minor, because a patch release does not change resolution.
FALLBACK_PYTHON = (3, 12)


def recorded_python() -> tuple[tuple[int, int], Counter[str]]:
    """The (major, minor) every stored run agrees on, and the raw tally."""
    seen: Counter[str] = Counter()
    for result in OUTPUTS.glob("*/seed*/result.json"):
        try:
            recorded = json.loads(result.read_text()).get("env", {}).get("python")
        except (OSError, json.JSONDecodeError):
            continue
        if recorded:
            seen[recorded] += 1
    if not seen:
        return FALLBACK_PYTHON, seen
    minors = {tuple(int(p) for p in v.split(".")[:2]) for v in seen}
    if len(minors) > 1:
        # Not this gate's call to make: say so and check against the commonest.
        print(f"  note: stored runs span more than one python minor: {sorted(seen)}")
    common = max(seen.items(), key=lambda kv: kv[1])[0]
    return tuple(int(p) for p in common.split(".")[:2]), seen


def lock_pins() -> dict[str, str]:
    pins = {}
    for line in LOCK.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9._-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def main() -> int:
    problems: list[str] = []

    print("== interpreter ==")
    want, seen = recorded_python()
    have = sys.version_info[:2]
    running = ".".join(str(p) for p in sys.version_info[:3])
    total = sum(seen.values())
    where = f"{total} stored run(s)" if total else "the fallback (no stored runs)"
    print(f"  running {running}, {where} recorded {want[0]}.{want[1]}.x")
    if have != want:
        problems.append(
            f"python {have[0]}.{have[1]} but every recorded number was produced under "
            f"{want[0]}.{want[1]}. requirements-lock.txt does not even resolve below 3.11 "
            f"(networkx==3.6.1). Build .venv with {want[0]}.{want[1]}: "
            f"`uv python install {want[0]}.{want[1]}` then "
            f"`PYTHON=$(uv python find {want[0]}.{want[1]}) bash scripts/setup.sh`."
        )

    print("== pinned dependencies ==")
    pins = lock_pins()
    missing, drifted = [], []
    for name, want_version in sorted(pins.items()):
        try:
            have_version = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if have_version != want_version:
            drifted.append(f"{name} {have_version} != {want_version}")
    print(f"  {len(pins)} pin(s) in requirements-lock.txt, "
          f"{len(pins) - len(missing) - len(drifted)} satisfied")
    if missing:
        problems.append(f"not installed: {', '.join(missing)}")
    if drifted:
        problems.append(f"wrong version: {'; '.join(drifted)}")

    print("== torch ==")
    try:
        import torch

        sys.path.insert(0, str(REPO / "src"))
        from msa.device import describe_device, resolve_device

        print(f"  torch {torch.__version__}  device {describe_device(resolve_device('auto'))}")
    except Exception as exc:  # noqa: BLE001 -- any import failure is the finding
        problems.append(f"torch unusable: {exc!r}")

    print()
    if problems:
        print("environment does not match the one the recorded numbers came from:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("environment matches the recorded one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
