#!/usr/bin/env bash
# Rebuild the MMSA reference checkout and the environment it needs.
#
#   bash scripts/setup_mmsa_reference.sh
#
# Neither /workspace/MMSA nor /workspace/mmsa_env is in this repository (the
# first is someone else's project, the second is 250MB of wheels), so both are
# lost on every machine move. Two things need them:
#
#   * scripts/check_almt_equivalence.py -- convention 5's numerical-equivalence
#     test. Without a checkout it prints SKIP and still exits 0, so a missing
#     MMSA does not turn a gate red; it silently removes one. That test is what
#     caught the ALMT port whose metrics beat the reference on all seven
#     measures because it had lost positional information.
#   * scripts/mmsa_reference.py -- runs MMSA's own code to produce the reference
#     the aggregate verdict is measured against.
#
# The environment is separate from .venv on purpose. MMSA's models cannot import
# under transformers 5, which this project targets, so its transformers 4.x tree
# is put on sys.path ahead of ours only for the processes that run MMSA code.
# torch and numpy are deliberately NOT in that tree: they come from .venv, so
# both implementations run on the same tensor library. That is the whole point
# of the comparison, and it is why the shim is a directory of symlinks to
# chosen packages rather than the whole site-packages.
set -euo pipefail

MMSA_DIR=${MMSA_DIR:-/workspace/MMSA}
ENV_DIR=${ENV_DIR:-/workspace/mmsa_env}
SHIM_DIR="$ENV_DIR/shim"
PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")/.."

# Pinned because the reference numbers in docs/mmsa_code_runs_mosi.json were
# produced against exactly these. transformers 4.44.2 is also the version
# docs/investigations.md#bert-mag checked our BERT-MAG port against.
PACKAGES=(
    "transformers==4.44.2"      # MMSA's BertTextEncoder; 5.x cannot import its models
    "einops==0.8.2"             # ALMT
    "easydict==1.13"            # MMSA's config objects
    "nvidia-ml-py3==7.352.0"    # MMSA/utils/functions.py imports pynvml at module level
)
# pytorch_transformers is CENET's, and is long dead: it declares torch, which
# must not be installed here. --no-deps, then its own imports by hand.
NO_DEPS=("pytorch-transformers==1.2.0")
NO_DEPS_SUPPORT=("boto3==1.43.62" "sentencepiece==0.2.2" "sacremoses==0.1.1")

echo "==> MMSA checkout at $MMSA_DIR"
if [ -d "$MMSA_DIR/.git" ]; then
    echo "    already present: $(git -C "$MMSA_DIR" log --oneline -1)"
else
    git clone --depth 1 https://github.com/thuiar/MMSA.git "$MMSA_DIR"
fi

echo "==> reference environment at $ENV_DIR"
[ -d "$ENV_DIR" ] || "$PYTHON" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet "${PACKAGES[@]}"
"$PIP" install --quiet --no-deps "${NO_DEPS[@]}"
"$PIP" install --quiet "${NO_DEPS_SUPPORT[@]}"

# The shim: symlinks to everything in that environment except numpy and pip.
# numpy is excluded so MMSA's models see the same array library as ours -- two
# numpy copies on one sys.path is the kind of thing that produces a difference
# nobody can attribute afterwards.
echo "==> shim at $SHIM_DIR"
SITE=$("$ENV_DIR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
rm -rf "$SHIM_DIR"
mkdir -p "$SHIM_DIR"
for entry in "$SITE"/*; do
    name=$(basename "$entry")
    case "$name" in
        numpy|numpy.libs|numpy-*|pip|pip-*|__pycache__|*.pth) continue ;;
    esac
    ln -s "$entry" "$SHIM_DIR/$name"
done
echo "    $(find "$SHIM_DIR" -maxdepth 1 -mindepth 1 | wc -l) entries linked"

# Proof, not assumption: this is the gate that was down, so run it. It exits
# non-zero on a real mismatch and prints SKIP if either half is still missing --
# grep for the verdict rather than trusting the exit code, which is 0 for SKIP.
echo
echo "==> verifying: scripts/check_almt_equivalence.py"
output=$(MMSA_SHIM="$SHIM_DIR" .venv/bin/python scripts/check_almt_equivalence.py 2>&1)
echo "$output" | grep -vE "FutureWarning|warnings.warn" || true
if ! grep -q "^EQUIVALENT$" <<<"$output"; then
    echo
    echo "The equivalence check did not reach a verdict -- the reference is not"
    echo "usable yet. Fix that before relying on the gate; it reports PASS when"
    echo "it skips."
    exit 1
fi

echo
echo "Reference ready. Run MMSA's own code with:"
echo "  MMSA_SHIM=$SHIM_DIR .venv/bin/python scripts/mmsa_reference.py run <model> <seeds...>"
echo "Note that 'collect' overwrites docs/mmsa_code_runs_mosi.json, whose numbers"
echo "were produced on the RTX 5070 Ti machine and do not reproduce elsewhere."
