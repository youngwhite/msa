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
    ef_lstm_mosi
    lf_dnn_mosi
    tfn_mosi
    lmf_mosi
    mfn_mosi
    graph_mfn_mosi
    mult_mosi
    text_bert_mosi
    misa_mosi
    self_mm_mosi
    tfn_mosi_ablation_masked
    lmf_mosi_ablation_masked
    mfn_mosi_ablation_realseq
    graph_mfn_mosi_ablation_frozen
    tfn_mosi_mmsaseeds
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
    # Acceptance run: 10 registered seeds, judged by scripts/check_acceptance.py
    tfn_mosi)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
    # LMF: MMSA hyper-parameters for MOSI (bs 64, lr 1e-3, weight decay 5e-3,
    # rank 3, hidden [128,16,128]); no clipping and no epoch cap, as MMSA trains.
    lmf_mosi)
        echo "--model lmf --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.005 --batch-size 64 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
    # MFN: MMSA hyper-parameters for MOSI (bs 128, lr 2e-3, memsize 400,
    # hidden [256,32,256]) on ALIGNED data, and with audio/vision collapsed to
    # their utterance mean the way MMSA's config does it.
    mfn_mosi)
        echo "--model mfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0 --batch-size 128 --grad-clip 0 --epochs 200" ;;
    mfn_mosi_ablation_realseq)
        echo "--model mfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0 --batch-size 128 --grad-clip 0 --epochs 200 --model-arg collapse_av_to_mean=False" ;;
    # MulT: MMSA hyper-parameters for MOSI. First model here that clips (by
    # value, 0.6) and decays its learning rate on plateau (factor 0.1,
    # patience 5); early stopping still uses patience 8.
    mult_mosi)
        echo "--model mult --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0.005 --batch-size 16 --grad-clip 0.6 --clip-mode value --lr-schedule plateau --lr-schedule-patience 5 --patience 8 --epochs 200" ;;
    # Text-only fine-tuned BERT: our control group, not a reproduction target —
    # MMSA has no text-only entry. Standard BERT fine-tuning settings (lr 2e-5,
    # head at 10x, bs 16); it converges in a few epochs.
    text_bert_mosi)
        echo "--model text_bert --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-5 --weight-decay 0.01 --batch-size 16 --grad-clip 1.0 --patience 3 --epochs 12" ;;
    # MISA: MMSA hyper-parameters (lr 1e-4, bs 16, hidden 128, clip 0.8 by value,
    # weights diff 0.1 / sim 0.3 / recon 1.0). Fine-tunes BERT.
    misa_mosi)
        echo "--model misa --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-4 --weight-decay 0 --batch-size 16 --grad-clip 0.8 --clip-mode value --epochs 200 --patience 8" ;;
    # Self-MM: MMSA's four learning rates expressed relative to --lr (its
    # "other" rate, 1e-3): BERT at 0.05x, audio/vision at 5x.
    self_mm_mosi)
        echo "--model self_mm --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.001 --batch-size 16 --grad-clip 0 --epochs 200 --patience 8" ;;
    # EF-LSTM / LF-DNN: the two pre-TFN baselines. EF-LSTM needs aligned data
    # (a per-step concatenation requires a shared clock); LF-DNN is unaligned and
    # pools each modality first.
    ef_lstm_mosi)
        echo "--model ef_lstm --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.005 --batch-size 32 --grad-clip 0 --epochs 200" ;;
    lf_dnn_mosi)
        echo "--model lf_dnn --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.01 --batch-size 128 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
    # Graph-MFN: aligned, and unlike MFN its config leaves need_normalized False,
    # so it receives the real per-step audio and vision sequences.
    graph_mfn_mosi)
        echo "--model graph_mfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0.005 --batch-size 32 --grad-clip 0 --epochs 200" ;;
    graph_mfn_mosi_ablation_frozen)
        echo "--model graph_mfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0.005 --batch-size 32 --grad-clip 0 --epochs 200 --model-arg freeze_graph_networks=True" ;;
    # Ablations: our masked/length-aware defaults instead of MMSA's padding
    # behaviour. Kept apart from the acceptance runs so reproduction fidelity and
    # our own changes are never confounded.
    tfn_mosi_ablation_masked)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0 --grad-clip 0 --epochs 200" ;;
    lmf_mosi_ablation_masked)
        echo "--model lmf --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.005 --batch-size 64 --grad-clip 0 --epochs 200" ;;
    # Same faithful config as the acceptance run, but on MMSA's own default
    # seeds — a check on how much of any residual gap is just the seed draw.
    tfn_mosi_mmsaseeds)
        echo "--model tfn --unaligned --seeds 1111 1112 1113 1114 1115 --device cuda --lr 1e-3 --weight-decay 0 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
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
