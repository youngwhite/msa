#!/usr/bin/env bash
# Bring this project up on a fresh machine.
#
#   bash scripts/setup.sh              # detect the platform and pick a torch wheel
#   TORCH_INDEX=cpu bash scripts/setup.sh   # force a CPU-only install
#
# What it does, in order: create .venv, install PyTorch for this platform, install
# the pinned dependencies, install the package, then verify the dataset files
# against their recorded sha256 and run every gate.
#
# PyTorch is not in requirements-lock.txt because its wheel is platform-specific.
# The development machine used torch 2.11.0+cu128 — a Blackwell (sm_120) GPU
# needs cu128 or newer; PyPI's default wheels have no sm_120 kernels and fail on
# the first CUDA op.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

if [ ! -d "$VENV" ]; then
    echo "==> creating $VENV with $PYTHON"
    "$PYTHON" -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

# --- PyTorch -----------------------------------------------------------------
if ! "$PY" -c "import torch" 2>/dev/null; then
    if [ -n "${TORCH_INDEX:-}" ]; then
        CHANNEL=$TORCH_INDEX
    elif [ "$(uname -s)" = "Darwin" ]; then
        CHANNEL=default          # macOS wheels ship MPS support
    elif command -v nvidia-smi >/dev/null 2>&1; then
        CHANNEL=cu128
    else
        CHANNEL=cpu
    fi
    echo "==> installing PyTorch (channel: $CHANNEL)"
    case "$CHANNEL" in
        default) "$PIP" install --quiet torch ;;
        *)       "$PIP" install --quiet torch \
                     --index-url "https://download.pytorch.org/whl/$CHANNEL" ;;
    esac
else
    echo "==> PyTorch already present: $("$PY" -c 'import torch; print(torch.__version__)')"
fi

echo "==> installing pinned dependencies"
"$PIP" install --quiet -r requirements-lock.txt
"$PIP" install --quiet -e .

echo "==> environment"
"$PY" - <<'PYCODE'
import torch
from msa.device import describe_device, resolve_device
print(f"    torch {torch.__version__}  device {describe_device(resolve_device('auto'))}")
PYCODE

# --- dataset -----------------------------------------------------------------
echo "==> dataset"
if ! "$PY" - <<'PYCODE'
import sys
from msa.config import get_dataset_spec, verify_files
spec = get_dataset_spec("mosi")
problems = verify_files(spec)
for problem in problems:
    print(f"    {problem}")
if problems:
    print(f"\n    Put the files under {spec.root} .")
    print(f"    Source: {spec.source}")
    sys.exit(1)
print(f"    {spec.name}: all {len(spec.file_sha256)} files match their recorded sha256")
PYCODE
then
    echo
    echo "Dataset is missing or does not match. Everything else is installed;"
    echo "re-run this script once the files are in place."
    exit 1
fi

echo
echo "==> running every gate"
bash scripts/check_all.sh
