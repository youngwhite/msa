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

# Disk headroom, and how much work is only on this machine. A full disk on this
# kind of instance does not merely stop a training run: it can take SSH down
# with it, and /workspace is not a persistent volume, so anything unpushed goes
# with the container. That has already cost this project a session's work once.
#
# Deliberately a WARNING and never a gate. A failing gate stops sync.sh from
# pushing, and a low disk is exactly when pushing matters most -- a check that
# blocks the escape route because the room is on fire is worse than no check.
DISK_WARN_GB=${DISK_WARN_GB:-3}
disk_note() {
    local free_gb unpushed=0 dirty=0
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if git rev-parse '@{upstream}' >/dev/null 2>&1; then
        unpushed=$(git rev-list --count '@{upstream}..HEAD' 2>/dev/null || echo 0)
    fi
    [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && dirty=1

    if [ "$free_gb" -lt "$DISK_WARN_GB" ]; then
        printf '\033[31m!! %sG disk free (warn below %sG)\033[0m\n' "$free_gb" "$DISK_WARN_GB"
        printf '   Free space before training anything: pip cache purge, rm -rf mmsa_runs,\n'
        printf '   outputs/**/best.pt. A full disk here can lock you out of the instance.\n'
    else
        printf 'disk: %sG free\n' "$free_gb"
    fi
    local what=""
    [ "$unpushed" -gt 0 ] && what="$unpushed commit(s) unpushed"
    if [ "$dirty" -eq 1 ]; then
        [ -n "$what" ] && what="$what, plus uncommitted changes" || what="uncommitted changes"
    fi
    [ -n "$what" ] && printf '\033[33m   %s — only on this machine. bash scripts/sync.sh\033[0m\n' "$what"
}
disk_note

# Regenerated every run so the checklist cannot go stale, and reported rather
# than gated. Its drift count is a to-do list (papers filed but never verified),
# and no code change can clear it -- only triage can. A gate that is red by
# design teaches people to ignore gates, which is the same reason the disk
# warning is a warning.
printf '\n'
$PY scripts/survey_checklist.py | sed 's/^/survey: /'

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
# Same rule: skips itself when the reference is not checked out.
run "dpdf-lq equivalence"  $PY scripts/check_dpdf_equivalence.py
run "dlf equivalence"      $PY scripts/check_dlf_equivalence.py
run "dmd equivalence"      $PY scripts/check_dmd_equivalence.py
run "confede equivalence"  $PY scripts/check_confede_equivalence.py
if [ "$FAST" != "--fast" ]; then
    run "reproducibility (lf_lstm)" $PY scripts/check_repro.py --epochs 1
    run "reproducibility (tfn)"     $PY scripts/check_repro.py --model tfn --epochs 1
fi

printf '\n'
# Repeated at the end because the tail is what anyone actually reads.
disk_note
if [ ${#FAILED[@]} -eq 0 ]; then
    printf '\033[32mall gates passed\033[0m\n'
    exit 0
fi
printf '\033[31m%d gate(s) failed:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"
exit 1
