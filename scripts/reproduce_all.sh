#!/usr/bin/env bash
# Regenerate every number in docs/experiments.md, then check the predictions
# against the ones committed to git.
#
#   bash scripts/reproduce_all.sh                  # everything, serially
#   bash scripts/reproduce_all.sh tfn_mosi_masked  # one group
#   bash scripts/reproduce_all.sh --list           # what the groups are
#   JOBS=auto bash scripts/reproduce_all.sh        # seeds in parallel, per-group limit
#   JOBS=3 bash scripts/reproduce_all.sh lmf_mosi  # explicit degree
#
# JOBS>1 runs a group's seeds as concurrent processes and rebuilds summary.json
# afterwards. The subprocesses are the same `train.py` invocations, so results
# are unchanged — verified bit-identical on lmf_mosi. Roughly 2.9x on three
# seeds; the card sits near a third utilisation with one.
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
    lf_lstm_mosei_cuda
    abl_lf_padded
    abl_lf_unaligned_masked
    abl_lf_unaligned_padded
    mod_t
    mod_a
    mod_v
    mod_av
    mod_tav
    ef_lstm_mosi
    ef_lstm_mosi_collapse_rate
    lf_dnn_mosi
    tfn_mosi
    lmf_mosi
    mfn_mosi
    graph_mfn_mosi
    mult_mosi
    mult_mosi_posenc
    text_bert_mosi
    misa_mosi
    self_mm_mosi
    cenet_mosi
    tetfn_mosi
    bert_mag_mosi
    mmim_mosi
    almt_mosi
    tfn_mosi_ablation_masked
    lmf_mosi_ablation_masked
    mfn_mosi_ablation_realseq
    graph_mfn_mosi_ablation_frozen
    tfn_mosi_ablation_mmsaselect
    mfn_mosi_ablation_mmsaselect
    misa_mosi_ablation_mmsaselect
    self_mm_mosi_ablation_mmsaselect
    tfn_mosi_2021config
    ef_lstm_mosi_2021config
    tfn_mosi_mmsaseeds
)

# The phase-2 contrastive screen has its own driver (scripts/screen_contrastive.py,
# which is what docs/experiments.md tells you to run), but its groups belong here
# as well: their predictions are committed like any others, so check_reproduction.py
# has to cover them or they are documented numbers nothing verifies.
#
# Enumerated from what is on disk rather than listed by hand. The groups are
# mechanical -- one per (candidate, lambda) -- and a hand-written list of seventy
# would go stale the first time the sweep grew an axis. They are committed, so
# this is deterministic from a clean clone.
#
# They roughly triple this script's runtime. `reproduce_all.sh <group>` still does
# one, and SKIP_SCREEN=1 leaves them all out.
if [ "${SKIP_SCREEN:-0}" != "1" ]; then
    while IFS= read -r _group; do
        [ -n "$_group" ] && RUN_GROUPS+=("$_group")
    done < <(find outputs -maxdepth 1 -name 'screen_*' -type d -printf '%f\n' 2>/dev/null | sort)
fi

# The weighting arms and their confirmation. Few enough to list, and unlike the
# screen each one differs in more than a single number, so args_for spells them
# out rather than deriving them from the name.
RUN_GROUPS+=(
    weight_control
    weight_equal
    weight_grad10
    weight_grad5
    weight_ramp
    confirm_l1
    confirm_combo
    mmim_contrast_off
    mmim_contrast_on
    mmim_diag_mosi_unaligned_off
    mmim_diag_mosi_unaligned_on
)

# TFN reproduces MMSA's reported MOSI result, so those groups use MMSA's
# hyper-parameters (lr 1e-3, no weight decay) rather than this repo's defaults.
#: TFN with MMSA's hyper-parameters, which is what the screen is built on.
SCREEN_BACKBONE="--model tfn --unaligned --lr 1e-3 --weight-decay 0 --seeds 42 43 44 45 46"
#: The same backbone, but seeds 100-119 for the weighting arms and 120-139 for
#: their confirmation. Three disjoint seed sets on purpose: 42-46 chose the four
#: terms, so reusing them downstream would confirm this work's own selection.
TFN_BACKBONE="--model tfn --unaligned --lr 1e-3 --weight-decay 0"
WEIGHT_SEEDS=$(seq -s' ' 100 119)
CONFIRM_SEEDS=$(seq -s' ' 120 139)
COMBO="--contrastive hcl infonce simsiam supcon --contrastive-lambda 0.1"

args_for() {
    case "$1" in
    screen_control)
        echo "$SCREEN_BACKBONE" ;;
    weight_control)
        echo "$TFN_BACKBONE --seeds $WEIGHT_SEEDS" ;;
    weight_equal)
        echo "$TFN_BACKBONE --seeds $WEIGHT_SEEDS $COMBO --weight-scheme equal" ;;
    weight_grad10)
        echo "$TFN_BACKBONE --seeds $WEIGHT_SEEDS $COMBO --weight-scheme grad_rate --weight-window 10" ;;
    weight_grad5)
        echo "$TFN_BACKBONE --seeds $WEIGHT_SEEDS $COMBO --weight-scheme grad_rate --weight-window 5" ;;
    weight_ramp)
        echo "$TFN_BACKBONE --seeds $WEIGHT_SEEDS $COMBO --weight-scheme linear_ramp --favour infonce --favour-target 0.7 --favour-end-fraction 0.5" ;;
    confirm_l1)
        echo "$TFN_BACKBONE --seeds $CONFIRM_SEEDS" ;;
    confirm_combo)
        echo "$TFN_BACKBONE --seeds $CONFIRM_SEEDS $COMBO --weight-scheme equal" ;;
    mmim_contrast_on)
        echo "--model mmim --seeds $WEIGHT_SEEDS" ;;
    mmim_contrast_off)
        echo "--model mmim --seeds $WEIGHT_SEEDS --model-arg contrast=False" ;;
    # The correct diagnostic: MMIM takes unaligned data (MMSA's own config, and
    # how mmim_mosi was accepted). The unsuffixed mmim_contrast_* groups above
    # are the mis-specified first attempt, kept as the record.
    mmim_diag_mosi_unaligned_on)
        echo "--model mmim --dataset mosi --unaligned --seeds $WEIGHT_SEEDS" ;;
    mmim_diag_mosi_unaligned_off)
        echo "--model mmim --dataset mosi --unaligned --seeds $WEIGHT_SEEDS --model-arg contrast=False" ;;
    screen_*)
        # screen_<candidate>, or screen_<candidate>_l<lambda> with the decimal
        # point written as `p` so it survives being a directory name. No
        # candidate name contains `_l` followed by a digit, so the split is
        # unambiguous.
        _rest=${1#screen_}
        case "$_rest" in
        *_l[0-9]*)
            _candidate=${_rest%_l*}
            _lambda=${_rest##*_l}
            _lambda=${_lambda//p/.} ;;
        *)
            _candidate=$_rest
            _lambda=0.1 ;;
        esac
        echo "$SCREEN_BACKBONE --contrastive $_candidate --contrastive-lambda $_lambda" ;;
    lf_lstm_mosi_cuda)
        echo "--model lf_lstm --seeds 42 43 44 45 46 --device cuda" ;;
    lf_lstm_mosi_cpu)
        echo "--model lf_lstm --seeds 42 --device cpu --num-threads 8" ;;
    lf_lstm_mosei_cuda)
        # The group NOISE_FLOOR_BY_DATASET["mosei"] is measured from.
        echo "--model lf_lstm --dataset mosei --seeds 42 43 44 45 46 --device cuda" ;;
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
    # weight-decay 0, not the 0.005 in MMSA's config: its MulT trainer builds
    # Adam without the weight_decay argument, so the configured value never
    # reaches the optimiser. We match the behaviour, not the config file.
    mult_mosi)
        echo "--model mult --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0 --batch-size 16 --grad-clip 0.6 --clip-mode value --lr-schedule plateau --lr-schedule-patience 5 --patience 8 --epochs 200 --accumulate-steps 8" ;;
    # Ablation, NOT an acceptance group. Identical to mult_mosi except for the
    # positional encoding the paper always builds and MMSA never switches on.
    # One variable, so the difference is attributable. It must not be compared
    # with MMSA's table (different model) nor with the paper's (different
    # features on all three modalities) — only with mult_mosi.
    mult_mosi_posenc)
        echo "--model mult --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0 --batch-size 16 --grad-clip 0.6 --clip-mode value --lr-schedule plateau --lr-schedule-patience 5 --patience 8 --epochs 200 --accumulate-steps 8 --model-arg position_embedding=True" ;;
    # Text-only fine-tuned BERT: our control group, not a reproduction target —
    # MMSA has no text-only entry. Standard BERT fine-tuning settings (lr 2e-5,
    # head at 10x, bs 16); it converges in a few epochs.
    text_bert_mosi)
        echo "--model text_bert --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-5 --weight-decay 0.01 --batch-size 16 --grad-clip 1.0 --patience 3 --epochs 12" ;;
    # MISA: MMSA hyper-parameters (lr 1e-4, bs 16, hidden 128, clip 0.8 by value,
    # weights diff 0.1 / sim 0.3 / recon 1.0). Fine-tunes BERT.
    misa_mosi)
        echo "--model misa --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-4 --weight-decay 0 --batch-size 16 --grad-clip 0.8 --clip-mode value --epochs 200 --patience 8 --accumulate-steps 2" ;;
    # Self-MM: MMSA's four learning rates expressed relative to --lr (its
    # "other" rate, 1e-3): BERT at 0.05x, audio/vision at 5x.
    self_mm_mosi)
        echo "--model self_mm --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.001 --batch-size 16 --grad-clip 0 --epochs 200 --patience 8 --accumulate-steps 4" ;;
    # CENET: MMSA hyper-parameters (lr 1e-5, weight decay 1e-4, bs 64, clip by
    # norm at 2, Adam eps 3e-8 supplied through the model's param_groups).
    # Reproduces MMSA's CENET, which differs from the paper's in backbone and in
    # what the CE module consumes — see docs/investigations.md#cenet-vs-paper.
    cenet_mosi)
        echo "--model cenet --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-5 --weight-decay 1e-4 --batch-size 64 --grad-clip 2.0 --epochs 200 --patience 8" ;;
    # ALMT: MMSA hyper-parameters, which match the authors' mosi.yaml. UNALIGNED.
    # AdamW rather than Adam, MSE rather than L1 (the model supplies the loss),
    # selection on validation MSE because that is what the reference's KeyEval
    # measures here, and NO clipping: max_grad_norm sits in the config and
    # neither implementation ever applies it.
    almt_mosi)
        echo "--model almt --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --optimizer adamw --lr 1e-4 --weight-decay 1e-4 --batch-size 64 --grad-clip 0 --lr-schedule warmup_cosine --epochs 200 --patience 32 --select-on mse" ;;
    # MMIM: MMSA hyper-parameters. UNALIGNED — each stream keeps its own clock,
    # and the LSTM encoders read every stream at its true final step. lr is the
    # main rate; BERT runs at a twentieth of it (see MMIM.param_groups).
    mmim_mosi)
        echo "--model mmim --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 1e-4 --batch-size 32 --grad-clip 1.0 --epochs 200 --patience 8" ;;
    # BERT-MAG: MMSA hyper-parameters (lr 2e-5, bs 32, no weight decay, no
    # clipping). ALIGNED — the gate displaces token embeddings position by
    # position, so audio and vision must share the text's clock.
    bert_mag_mosi)
        echo "--model bert_mag --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-5 --weight-decay 0 --batch-size 32 --grad-clip 0 --epochs 200 --patience 8" ;;
    # TETFN: MMSA hyper-parameters. ALIGNED data. Its four learning rates and
    # four weight decays are expressed relative to the CLI pair by the model's
    # param_groups, so --lr 3e-4 --weight-decay 0.01 reproduces all of them.
    tetfn_mosi)
        echo "--model tetfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 3e-4 --weight-decay 0.01 --batch-size 64 --grad-clip 0 --epochs 200 --patience 8 --accumulate-steps 4" ;;
    # EF-LSTM / LF-DNN: the two pre-TFN baselines. EF-LSTM needs aligned data
    # (a per-step concatenation requires a shared clock); LF-DNN is unaligned and
    # pools each modality first.
    ef_lstm_mosi)
        echo "--model ef_lstm --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.005 --batch-size 32 --grad-clip 0 --epochs 200" ;;
    # 20 extra seeds (52-71) purely to measure how often this configuration fails
    # to train at all. NOT an acceptance group: the registered set stays 42-51.
    ef_lstm_mosi_collapse_rate)
        echo "--model ef_lstm --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 --device cuda --lr 1e-3 --weight-decay 0.005 --batch-size 32 --grad-clip 0 --epochs 200" ;;
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
    # Same faithful config as the acceptance runs, but picking the epoch on
    # MMSA's own reduction of the validation loss (mean of per-batch means,
    # rounded to 1e-4) instead of ours (over samples, unrounded). The one
    # protocol difference that applies to every model, never quantified until
    # now — see docs/investigations.md#mmim and #select-reduction.
    tfn_mosi_ablation_mmsaselect)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False --select-reduction mmsa" ;;
    mfn_mosi_ablation_mmsaselect)
        echo "--model mfn --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 2e-3 --weight-decay 0 --batch-size 128 --grad-clip 0 --epochs 200 --select-reduction mmsa" ;;
    # Two more at batch 16, testing the prediction that the effect tracks the
    # final batch's overweight (3.1x here, against TFN's 5.7x and MFN's 1.13x).
    # MISA and Self-MM because their protocol matches TFN's and MFN's — 200
    # epochs, patience 8, no scheduler. MulT is batch 16 too but decays on
    # plateau off the same quantity, and text_bert stops at 12 epochs; in either
    # the reduction would change more than one thing.
    misa_mosi_ablation_mmsaselect)
        echo "--model misa --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-4 --weight-decay 0 --batch-size 16 --grad-clip 0.8 --clip-mode value --epochs 200 --patience 8 --accumulate-steps 2 --select-reduction mmsa" ;;
    self_mm_mosi_ablation_mmsaselect)
        echo "--model self_mm --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 0.001 --batch-size 16 --grad-clip 0 --epochs 200 --patience 8 --accumulate-steps 4 --select-reduction mmsa" ;;
    # TFN under the hyper-parameters MMSA's config held on 2021-05-06, the day
    # its results table was written — not the ones in its config today. Eight of
    # the eleven models were retuned afterwards without the table being
    # regenerated. See docs/investigations.md#table-predates-config.
    tfn_mosi_2021config)
        echo "--model tfn --unaligned --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 5e-4 --weight-decay 0 --batch-size 32 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False --model-arg text_hidden=128 --model-arg audio_hidden=16 --model-arg vision_hidden=128 --model-arg text_out=128 --model-arg post_fusion_dim=32 --model-arg text_dropout=0.2 --model-arg audio_dropout=0.2 --model-arg vision_dropout=0.2 --model-arg post_fusion_dropout=0.2" ;;
    # EF-LSTM under the 2021 config too, for a different question than TFN's:
    # today's config stacks 4 LSTM layers where 2021's stacked 2, and the 23%
    # collapse rate in #ef-lstm-collapse is a property of the 4-layer stack. Does
    # the configuration the table was written under collapse at all?
    ef_lstm_mosi_2021config)
        echo "--model ef_lstm --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda --lr 1e-3 --weight-decay 1e-4 --batch-size 128 --grad-clip 0 --epochs 200 --model-arg hidden_size=256 --model-arg num_layers=2 --model-arg dropout=0.3" ;;
    # Same faithful config as the acceptance run, but on MMSA's own default
    # seeds — a check on how much of any residual gap is just the seed draw.
    tfn_mosi_mmsaseeds)
        echo "--model tfn --unaligned --seeds 1111 1112 1113 1114 1115 --device cuda --lr 1e-3 --weight-decay 0 --grad-clip 0 --epochs 200 --model-arg use_lengths=False --model-arg mask_pooling=False" ;;
    *)  return 1 ;;
    esac
}

# How many seeds of a group may share the GPU. Bounded by memory, not by cores:
# a single seed leaves the card at roughly a third utilisation, but exceeding
# memory does not degrade, it raises out-of-memory partway through. Measured on
# a 16.3 GB card; lower these if yours is smaller.
jobs_for() {
    case "$1" in
    mult_mosi|mult_mosi_posenc) echo 2 ;;      # ~5.6 GB each
    misa_mosi|self_mm_mosi|text_bert_mosi|cenet_mosi|tetfn_mosi|bert_mag_mosi|mmim_mosi|almt_mosi) echo 2 ;;   # fine-tuned BERT
    *) echo 4 ;;                                # the frozen-feature models are small
    esac
}

WANTED=${1:-}
JOBS=${JOBS:-1}   # JOBS=n runs n seeds concurrently; JOBS=auto uses jobs_for()

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
    if [ "$JOBS" = "auto" ]; then
        group_jobs=$(jobs_for "$group")
    else
        group_jobs=$JOBS
    fi
    if [ "$group_jobs" -gt 1 ]; then
        # Same subprocesses, run concurrently; summary.json is rebuilt after.
        # Verified bit-identical to the serial path on lmf_mosi (2026-07-28).
        # shellcheck disable=SC2046,SC2086
        if ! $PY scripts/train_parallel.py --jobs "$group_jobs" --run-group "$group" \
                 $(args_for "$group"); then
            FAILED+=("$group")
        fi
    else
        # shellcheck disable=SC2046,SC2086
        if ! $PY scripts/train.py $(args_for "$group") --quiet --run-group "$group"; then
            FAILED+=("$group")
        fi
    fi
done

if [ ${#FAILED[@]} -ne 0 ]; then
    printf '\n\033[31m%d group(s) failed to run:\033[0m %s\n' "${#FAILED[@]}" "${FAILED[*]}"
    exit 1
fi

printf '\n\033[1m=== comparing against the committed predictions ===\033[0m\n'
$PY scripts/check_reproduction.py
