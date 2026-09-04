#!/usr/bin/env bash
# Download a dataset into datasets/ and verify it against the recorded sha256.
#
#   bash scripts/fetch_dataset.sh            # mosi (879MB)
#   bash scripts/fetch_dataset.sh mosei      # 18GB
#   bash scripts/fetch_dataset.sh all
#
# Two things about Google Drive that this script exists to encode, because both
# were found the hard way:
#
# 1. **Never `gdown --folder` the release root.** It recurses the entire
#    publication -- CH-SIMS and MOSI raw video, several thousand .mp4 -- and
#    Drive answers with a 500 partway through enumeration, so it exits having
#    downloaded nothing at all. The folder is therefore resolved one level at a
#    time and only the Processed/ features and label.csv are fetched; the Raw/
#    subtree is useless here since the features are already extracted.
#
# 2. **Files over a few GB serve a virus-scan interstitial, and gdown fails on
#    it.** MOSEI's unaligned_50.pkl (13.7GB) died with "Cannot retrieve the
#    public link... but Gdown can't", which reads like a permissions or quota
#    problem and is neither. The page carries a `uuid` confirmation token, so
#    those files go through curl against drive.usercontent.google.com with the
#    token attached -- and with `-C -`, because resuming a 13.7GB download beats
#    restarting it.
#
# The file IDs are recorded rather than re-derived each run: if the release is
# re-uploaded they 404, which is a loud failure rather than a quietly wrong file.
#
# The hashes are the point. A truncated download or a re-extracted feature set
# would otherwise be trained on silently, and every number in docs/ rests on
# this exact data.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "No .venv -- run scripts/setup.sh first." >&2; exit 1; }
UA='Mozilla/5.0'

# --- what each dataset needs ------------------------------------------------
# Children of the MOSI / MOSEI folders in MMSA's preprocessed release (THUIAR).
MOSI_DIR=datasets/CMU-MOSI
MOSI_PROCESSED_FOLDER=1qHa_gMOG-ZFt9hfDocRxB0uIh4hymnHo   # both .pkl, ~880MB
MOSI_LABEL=1dqyQI8iOHGjwofgq1DvAD-inW-zrTa2h

MOSEI_DIR=datasets/CMU-MOSEI
MOSEI_ALIGNED=16LfoCDw8LncJMVHaCOhN3H3_jjiKBmPu          # 4.7GB
MOSEI_UNALIGNED=13yKFqrS6v95QzH29h-zSg7ZuJdn6G5yA        # 13.7GB, needs the token path
MOSEI_LABEL=17_IhQLXHqON78XX5i0h9c0ywXXpt60PX

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

need_space() {
    local want=$1 have
    have=$(free_gb)
    echo "==> ${have}G free, this needs about ${want}G"
    if [ "$have" -lt "$want" ]; then
        echo "Not enough disk. Free some first: pip cache purge, rm -rf mmsa_runs," >&2
        echo "outputs/**/best.pt. A full disk on this kind of box can lock you out." >&2
        exit 1
    fi
}

# One large file, through the virus-scan confirmation. See note 2 above.
fetch_large() {
    local id=$1 out=$2 uuid
    [ -f "$out" ] && { echo "    $out already present"; return; }
    uuid=$(curl -sL "https://drive.google.com/uc?export=download&id=$id" -H "User-Agent: $UA" \
           | grep -oE 'name="uuid" value="[^"]+"' | sed 's/.*value="//;s/"//')
    [ -n "$uuid" ] || { echo "Could not get a confirmation token for $id" >&2; exit 1; }
    echo "    $out (confirm token $uuid)"
    curl -L --fail -C - --retry 5 --retry-delay 10 --retry-all-errors \
         -H "User-Agent: $UA" -o "$out" \
         "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t&uuid=$uuid"
}

fetch_mosi() {
    if "$PY" scripts/check_data.py --dataset mosi --verify-files >/dev/null 2>&1; then
        echo "==> mosi already present and matching its sha256"
        return
    fi
    need_space 3
    mkdir -p "$MOSI_DIR"
    [ -f "$MOSI_DIR/Processed/aligned_50.pkl" ] && [ -f "$MOSI_DIR/Processed/unaligned_50.pkl" ] || \
        "$PY" -m gdown --folder -O "$MOSI_DIR/Processed" "$MOSI_PROCESSED_FOLDER"
    [ -f "$MOSI_DIR/label.csv" ] || "$PY" -m gdown -O "$MOSI_DIR/label.csv" "$MOSI_LABEL"
    echo; echo "==> verifying mosi"
    "$PY" scripts/check_data.py --dataset mosi --verify-files
}

fetch_mosei() {
    if "$PY" scripts/check_data.py --dataset mosei --verify-files >/dev/null 2>&1; then
        echo "==> mosei already present and matching its sha256"
        return
    fi
    need_space 22
    mkdir -p "$MOSEI_DIR/Processed"
    [ -f "$MOSEI_DIR/label.csv" ] || "$PY" -m gdown -O "$MOSEI_DIR/label.csv" "$MOSEI_LABEL"
    # 4.7GB: gdown copes with this one.
    [ -f "$MOSEI_DIR/Processed/aligned_50.pkl" ] || \
        "$PY" -m gdown -O "$MOSEI_DIR/Processed/aligned_50.pkl" "$MOSEI_ALIGNED"
    # 13.7GB: gdown does not.
    fetch_large "$MOSEI_UNALIGNED" "$MOSEI_DIR/Processed/unaligned_50.pkl"
    echo; echo "==> verifying mosei"
    "$PY" scripts/check_data.py --dataset mosei --verify-files
}

case "${1:-mosi}" in
    mosi)  fetch_mosi ;;
    mosei) fetch_mosei ;;
    all)   fetch_mosi; echo; fetch_mosei ;;
    *)     echo "usage: $0 [mosi|mosei|all]" >&2; exit 1 ;;
esac
