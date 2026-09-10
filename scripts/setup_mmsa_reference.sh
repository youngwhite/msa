#!/usr/bin/env bash
# Rebuild the MMSA reference checkout and the environment it needs.
#
#   bash scripts/setup_mmsa_reference.sh
#
# Neither the MMSA checkout nor its environment is in this repository (the first
# is someone else's project, the second is 250MB of wheels), so both are lost on
# every machine move. They default to <repo>/.mmsa-reference/ , which is
# gitignored; override with MMSA_DIR / ENV_DIR. Two things need them:
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

cd "$(dirname "$0")/.."

# The interpreter that builds this environment must be the one that will import
# the shim -- .venv's, not the system's. Half these packages ship C extensions
# (tokenizers, safetensors, sentencepiece) whose wheels are built per CPython
# ABI, so a shim built by python3.10 and put on a python3.12 sys.path fails at
# import. It never showed on the first two machines because the system python
# happened to be the same version as .venv's.
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=$PWD/.venv/bin/python
    else
        echo "No .venv -- run scripts/setup.sh first." >&2
        exit 1
    fi
fi

# Same three candidates, same order, as scripts/_reference_paths.py: an explicit
# override, then the gitignored directory inside the repo, then /workspace where
# the first two machines kept it. The repo-local default is what stops this from
# silently disappearing again -- nothing has to be exported for the gate to find
# it, and it moves with the checkout.
if [ -z "${MMSA_DIR:-}" ]; then
    if [ -d /workspace/MMSA/.git ] && [ ! -d .mmsa-reference/MMSA ]; then
        MMSA_DIR=/workspace/MMSA
    else
        MMSA_DIR=$PWD/.mmsa-reference/MMSA
    fi
fi
if [ -z "${ENV_DIR:-}" ]; then
    if [ -d /workspace/mmsa_env ] && [ ! -d .mmsa-reference/env ]; then
        ENV_DIR=/workspace/mmsa_env
    else
        ENV_DIR=$PWD/.mmsa-reference/env
    fi
fi
SHIM_DIR="$ENV_DIR/shim"
mkdir -p "$(dirname "$MMSA_DIR")" "$(dirname "$ENV_DIR")"

# Pinned because the reference numbers in docs/mmsa_code_runs_mosi.json were
# produced against exactly these. transformers 4.44.2 is also the version
# docs/investigations.md#bert-mag checked our BERT-MAG port against.
PACKAGES=(
    "transformers==4.44.2"      # MMSA's BertTextEncoder; 5.x cannot import its models
    "einops==0.8.2"             # ALMT
    "easydict==1.13"            # MMSA's config objects
    "nvidia-ml-py3==7.352.0"    # MMSA/utils/functions.py imports pynvml at module level
    "matplotlib==3.11.1"        # UniMSE's modules/adapters.py imports pyplot at module level
)
# These declare torch, which must not be installed here (see the shim comment
# below). --no-deps, then their own imports by hand.
#   pytorch_transformers  CENET's, long dead
#   pytorch_metric_learning  ConFEDE's sample-level contrastive losses
NO_DEPS=("pytorch-transformers==1.2.0" "pytorch-metric-learning==0.9.99")
NO_DEPS_SUPPORT=("boto3==1.43.62" "sentencepiece==0.2.2" "sacremoses==0.1.1"
                 "scikit-learn==1.9.0" "tqdm==4.67.1")

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

# The shim: symlinks to everything in that environment except numpy, torch and
# pip. numpy and torch are excluded so the reference implementations see the same
# array and tensor libraries as ours -- two copies of either on one sys.path is
# the kind of thing that produces a difference nobody can attribute afterwards,
# and a reference run is only worth having when both sides share one torch.
#
# The torch exclusion is defensive, not decorative. Every torch-declaring package
# above is installed --no-deps for this reason, but that is easy to forget when
# adding one: installing pytorch-metric-learning for ConFEDE without --no-deps
# pulled torch into this environment, and a later rebuild of the shim linked it,
# silently shadowing ours. Keep both halves -- --no-deps on the install, and the
# exclusion here -- so forgetting one is not enough to break it.
echo "==> shim at $SHIM_DIR"
SITE=$("$ENV_DIR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
rm -rf "$SHIM_DIR"
mkdir -p "$SHIM_DIR"
for entry in "$SITE"/*; do
    name=$(basename "$entry")
    case "$name" in
        numpy|numpy.libs|numpy-*|pip|pip-*|__pycache__|*.pth) continue ;;
        torch|torch-*|torchvision|torchvision-*|torchvision.libs|torchgen) continue ;;
        functorch|triton|triton-*|nvidia|nvidia_c*|nvidia_n*|cuda|cuda_*) continue ;;
    esac
    ln -s "$entry" "$SHIM_DIR/$name"
done
echo "    $(find "$SHIM_DIR" -maxdepth 1 -mindepth 1 | wc -l) entries linked"

# Prove the exclusion held rather than trusting the case list above.
torch_from=$(PYTHONPATH="$SHIM_DIR" .venv/bin/python -c "import torch; print(torch.__file__)")
case "$torch_from" in
    "$PWD/.venv/"*) echo "    torch resolves to our venv: ok" ;;
    *) echo "    torch resolves to $torch_from -- NOT our venv" >&2; exit 1 ;;
esac

# Proof, not assumption: this is the gate that was down, so run it. It exits
# non-zero on a real mismatch and prints SKIP if either half is still missing --
# grep for the verdict rather than trusting the exit code, which is 0 for SKIP.
echo
echo "==> verifying: scripts/check_almt_equivalence.py"
output=$(MMSA_DIR="$MMSA_DIR" MMSA_SHIM="$SHIM_DIR" .venv/bin/python scripts/check_almt_equivalence.py 2>&1)
echo "$output" | grep -vE "FutureWarning|warnings.warn" || true
if ! grep -q "^EQUIVALENT$" <<<"$output"; then
    echo
    echo "The equivalence check did not reach a verdict -- the reference is not"
    echo "usable yet. Fix that before relying on the gate; it reports PASS when"
    echo "it skips."
    exit 1
fi

echo
echo "Reference ready at $MMSA_DIR . Run MMSA's own code with:"
echo "  .venv/bin/python scripts/mmsa_reference.py run <model> <seeds...>"
echo "Note that 'collect' overwrites docs/mmsa_code_runs_mosi.json, whose numbers"
echo "were produced on the RTX 5070 Ti machine and do not reproduce elsewhere."
