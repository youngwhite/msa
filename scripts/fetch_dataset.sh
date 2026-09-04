#!/usr/bin/env bash
# Download CMU-MOSI into datasets/ and verify it against the recorded sha256.
#
#   bash scripts/fetch_dataset.sh
#
# Until 2026-09-04 this step was two links in docs/migration.md and a manual
# copy. The obvious automation does not work and it is worth writing down why:
#
#   gdown --folder <the MMSA release folder>
#
# recurses the *whole* release -- CH-SIMS and MOSI raw video, several thousand
# .mp4 files -- and Google Drive answers with a 500 partway through enumeration,
# so it exits having downloaded nothing. The three files this project needs are
# 879MB out of a release that is far larger, and the Raw/ subtree is of no use
# here: the features are already extracted.
#
# So the folder is resolved one level at a time and only MOSI's Processed/ and
# label.csv are fetched. The folder IDs below were read out of the Drive listing
# and are recorded rather than re-derived on every run; if the release is ever
# re-uploaded they will 404, which is a loud failure, not a silent wrong file.
#
# The hashes are the real check. A truncated download or a re-extracted feature
# set would otherwise be trained on silently, and every recorded number in
# docs/ is built on this exact data.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "No .venv -- run scripts/setup.sh first." >&2; exit 1; }

DEST=datasets/CMU-MOSI
# Children of the MOSI folder in MMSA's preprocessed release (THUIAR).
PROCESSED_FOLDER_ID=1qHa_gMOG-ZFt9hfDocRxB0uIh4hymnHo   # Processed/ : the two .pkl
LABEL_FILE_ID=1dqyQI8iOHGjwofgq1DvAD-inW-zrTa2h        # label.csv

if "$PY" scripts/check_data.py --verify-files >/dev/null 2>&1; then
    echo "Dataset already present and matching its recorded sha256; nothing to do."
    exit 0
fi

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 3 ]; then
    echo "Only ${FREE_GB}G free -- MOSI is 879MB and this machine is too tight." >&2
    echo "Free space first: pip cache purge, rm -rf mmsa_runs, outputs/**/best.pt." >&2
    exit 1
fi
echo "==> ${FREE_GB}G free, fetching 879MB into $DEST"

mkdir -p "$DEST"
[ -f "$DEST/Processed/aligned_50.pkl" ] && [ -f "$DEST/Processed/unaligned_50.pkl" ] || \
    "$PY" -m gdown --folder -O "$DEST/Processed" "$PROCESSED_FOLDER_ID"
[ -f "$DEST/label.csv" ] || "$PY" -m gdown -O "$DEST/label.csv" "$LABEL_FILE_ID"

# Not optional. This is the whole reason the script exists rather than a link.
echo
echo "==> verifying sha256"
"$PY" scripts/check_data.py --verify-files
