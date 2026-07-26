#!/usr/bin/env bash
# Publish a milestone: verify everything, then push.
#
#   bash scripts/sync.sh           # full check (~1 min), then push
#   bash scripts/sync.sh --fast    # skip the two GPU reproducibility checks
#
# Nothing is pushed unless every gate passes. That is the whole point: the remote
# should never hold a state whose own checks fail, because the next person (or
# the next machine) starts from whatever is on the remote.
set -uo pipefail
cd "$(dirname "$0")/.."

BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE=${REMOTE:-origin}

# 1. Everything committed? An unpushed working-tree change is invisible to the
#    remote, so a "synced" repo that still has local edits is a lie.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Uncommitted changes — commit them first, then sync:"
    git status --short --untracked-files=no
    exit 1
fi

UNTRACKED=$(git status --porcelain --untracked-files=normal | grep -c '^??' || true)
if [ "$UNTRACKED" -gt 0 ]; then
    echo "note: $UNTRACKED untracked file(s) will not be pushed:"
    git status --short --untracked-files=normal | grep '^??' | sed 's/^/  /'
    echo
fi

# 2. Anything to push?
if git rev-parse --abbrev-ref "$BRANCH@{upstream}" >/dev/null 2>&1; then
    AHEAD=$(git rev-list --count "@{upstream}..HEAD")
    if [ "$AHEAD" -eq 0 ]; then
        echo "already in sync with $REMOTE/$BRANCH — nothing to push"
        exit 0
    fi
    echo "==> $AHEAD commit(s) to push to $REMOTE/$BRANCH"
    SET_UPSTREAM=""
else
    echo "==> $(git rev-list --count HEAD) commit(s) to push; setting upstream to $REMOTE/$BRANCH"
    SET_UPSTREAM="-u"
fi

# 3. The review: every gate, or no push.
echo "==> running gates"
if ! bash scripts/check_all.sh "${1:-}"; then
    echo
    echo "Gates failed — nothing pushed. Fix the failures, commit, and sync again."
    exit 1
fi

# 4. Publish.
echo
echo "==> pushing"
# shellcheck disable=SC2086
if ! git push $SET_UPSTREAM "$REMOTE" "$BRANCH"; then
    echo "Push failed. If it is an auth problem: ssh -T git@github.com should"
    echo "greet you by name; otherwise the key is not on the account that owns"
    echo "$(git remote get-url "$REMOTE")."
    exit 1
fi

echo
echo "local  HEAD: $(git log --oneline -1 HEAD)"
echo "remote HEAD: $(git log --oneline -1 "$REMOTE/$BRANCH")"
echo "synced."
