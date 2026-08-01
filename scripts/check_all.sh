#!/usr/bin/env bash
# Every gate this repo has, in one command. Run it after any change, and at the
# start of a session to confirm the checkout is in the state its docs describe.
#
#   bash scripts/check_all.sh [--fast]
#
# --fast skips the two GPU-training checks (check_repro), which take ~1 minute.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FAST=${1:-}
FAILED=()

run() {
    local name=$1; shift
    printf '\n\033[1m=== %s ===\033[0m\n' "$name"
    if "$@"; then
        printf '\033[32mPASS\033[0m %s\n' "$name"
    else
        printf '\033[31mFAIL\033[0m %s\n' "$name"
        FAILED+=("$name")
    fi
}

run "lint"                 $PY -m ruff check src scripts --select E,F,W,B,SIM,I,UP --line-length 100
run "data integrity"       $PY scripts/check_data.py
run "invariants"           $PY scripts/check_invariants.py
run "stored results"       $PY scripts/verify_runs.py --quiet
run "per-model acceptance" $PY scripts/check_acceptance.py --all
# Against MMSA's *code*, not its published table. The table is the claim under
# test, but it is unreachable by MMSA's own implementation too (by more than it
# is by ours), so gating on it would keep this red for a reason no change to this
# repository can fix. See docs/investigations.md#aggregate-verdict.
run "aggregate acceptance" $PY scripts/aggregate_acceptance.py --reference code
# Skips itself when MMSA is not checked out, so a fresh clone still goes green.
run "almt equivalence"     $PY scripts/check_almt_equivalence.py
if [ "$FAST" != "--fast" ]; then
    run "reproducibility (lf_lstm)" $PY scripts/check_repro.py --epochs 1
    run "reproducibility (tfn)"     $PY scripts/check_repro.py --model tfn --epochs 1
fi

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
    printf '\033[32mall gates passed\033[0m\n'
    exit 0
fi
printf '\033[31m%d gate(s) failed:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"
exit 1
