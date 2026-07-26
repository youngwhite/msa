#!/usr/bin/env bash
# Regenerate every number in docs/experiments.md, then check the predictions
# against the ones committed to git.
#
#   bash scripts/reproduce_all.sh                  # everything, ~25 min on an RTX 5070 Ti
#   bash scripts/reproduce_all.sh tfn_mosi_masked  # one group
#   bash scripts/reproduce_all.sh --list           # what the groups are
#
# This file is the link between the tables in docs/experiments.md and the code:
# every documented number comes from exactly one entry below. Add one when an
# experiment enters the docs, or the next person cannot regenerate it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

RUN_GROUPS=(  # note: not GROUPS — that is a read-only bash builtin (the user's gids)
    lf_lstm_mosi_cuda
    lf_lstm_mosi_cpu
    abl_lf_padded
    abl_lf_unaligned_masked
    abl_lf_unaligned_padded
    mod_t
    mod_a
    mod_v
    mod_av
    mod_tav
    tfn_mosi_masked
    tfn_mosi_mmsaseeds
    tfn_mosi_faithful
)

# TFN reproduces MMSA's reported MOSI result, so those groups use MMSA's
# hyper-parameters (lr 1e-3, no weight decay) rather than this repo's defaults.
args_for() {
    case "$1" in
    lf_lstm_mosi_cuda)
        echo "--model lf_lstm --seeds 42 43 44 45 46 --device cuda" ;;
    lf_lstm_mosi_cpu)
        echo "--model lf_lstm --seeds 42 --device cpu --num-threads 8" ;;
    abl_lf_padded)
        echo "--model lf_lstm --seeds 42 43 44 45 46 --device cuda --model-arg use_lengths=False" ;;
    abl_lf_unaligned_masked)
        echo "--model lf_lstm --unaligned --epochs 20 --seeds 42 43 --device cuda" ;;
    abl_lf_unaligned_padded)
        echo "--model lf_lstm --unaligned --epochs 20 --seeds 42 43 --device cuda --model-arg use_lengths=False" ;;
    mod_t)   echo "--model lf_lstm --seeds 42 43 44 --device cuda --model-arg modalities=t" ;;
    mod_a)   echo "--model lf_lstm --seeds 42 43 44 --device cuda --model-arg modalities=a" ;;
    mod_v)   echo "--model lf_lstm --seeds 42 43 44 --device cuda --model-arg modalities=v" ;;
    mod_av)  echo "--model lf_lstm --seeds 42 43 44 --device cuda --model-arg modalities=av" ;;
    mod_tav) echo "--model lf_lstm --seeds 42 43 44 --device cuda --model-arg modalities=tav" ;;
    tfn_mosi_masked)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 --device cuda --lr 1e-3 --weight-decay 0" ;;
    tfn_mosi_mmsaseeds)
        echo "--model tfn --unaligned --seeds 1111 1112 1113 1114 1115 --device cuda --lr 1e-3 --weight-decay 0" ;;
    tfn_mosi_faithful)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 --device cuda --lr 1e-3 --weight-decay 0 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
    *)  return 1 ;;
    esac
}

WANTED=${1:-}

if [ "$WANTED" = "--list" ]; then
    printf '%s\n' "${RUN_GROUPS[@]}"
    exit 0
fi

if [ -n "$WANTED" ] && ! args_for "$WANTED" >/dev/null; then
    echo "unknown group: $WANTED"
    echo "known groups:"
    printf '  %s\n' "${RUN_GROUPS[@]}"
    exit 1
fi

# A dirty tree would stamp every result with dirty=true, and documented numbers
# must be traceable to a commit.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Working tree has uncommitted changes. Commit them first: every run"
    echo "records the commit it came from, and results from a dirty tree cannot"
    echo "be traced back to code."
    exit 1
fi

FAILED=()
for group in "${RUN_GROUPS[@]}"; do
    if [ -n "$WANTED" ] && [ "$WANTED" != "$group" ]; then
        continue
    fi
    printf '\n\033[1m=== %s ===\033[0m\n' "$group"
    # shellcheck disable=SC2046,SC2086
    if ! $PY scripts/train.py $(args_for "$group") --quiet --run-group "$group"; then
        FAILED+=("$group")
    fi
done

if [ ${#FAILED[@]} -ne 0 ]; then
    printf '\n\033[31m%d group(s) failed to run:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"
    exit 1
fi

printf '\n\033[1m=== comparing against the committed predictions ===\033[0m\n'
$PY scripts/check_reproduction.py
