"""Is our DLF numerically identical to the authors' implementation?

DLF was rebuilt from the release rather than transcribed, so "the maths is the
same" is a claim until something checks it. This is that check, in the shape
convention 5 requires: construct both models, copy every weight across, feed one
input, compare the outputs.

The comparison covers all five prediction heads. It began as the final
prediction alone, reasoning that anything feeding it is checked implicitly --
true, but the release has four *auxiliary* heads whose outputs reach the loss
and never reach the prediction, so they sat outside the argument entirely. That
blind spot is how a missing loss term survived a passing test: the port dropped
four of the five task terms, and nothing on the prediction path could tell.

The lesson is narrow and worth keeping: "everything upstream is implied" only
covers what is upstream *of the tensor being compared*. Heads that exist solely
to be supervised have to be named.

**Seven modules the release builds and never calls** are excluded rather than
reproduced, per convention 5. Three are `proj_cosine_l/v/a`, commented "for
calculate cosine sim between s_x" while the orthogonality loss actually reshapes
instead. The other four are `trans_a_with_l`, `trans_a_with_v`, `trans_v_with_l`
and `trans_v_with_a` -- the paths by which audio and vision would attend to each
other or act as queries. Their absence from `forward` is how "language-focused"
is implemented: the paths are constructed and then simply not wired. Together
they account for exactly the 110 tensors by which the reference exceeds this
port.

BERT is stubbed on both sides -- identical by construction, both wrapping the
same pretrained weights -- so what is under test is everything that was
rewritten.

Exit code 0 if the outputs match, non-zero otherwise. SKIP also exits 0, so read
the verdict rather than the code.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ORIGIN = Path(os.environ.get("DLF_ORIGIN", "/workspace/DLF"))
#: The release imports pynvml, which lives in the reference environment rather
#: than ours. Without this the test SKIPs -- and a SKIP exits 0, so check_all.sh
#: would record a silent PASS. That is precisely how check_almt_equivalence.py
#: went unnoticed-dead for months, so the path is resolved here rather than left
#: to whoever remembers to export it.
SHIM = Path(os.environ.get("MMSA_SHIM", "/workspace/mmsa_env/shim"))
BATCH, LENGTH, BERT_DIM, AUDIO_DIM, VISION_DIM = 4, 50, 768, 5, 20
TOLERANCE = 1e-5

#: config/config.json, DLF section, MOSI, verbatim.
MODEL_ARGS = dict(
    need_data_aligned=True, need_model_aligned=True, early_stop=10,
    use_bert=True, use_finetune=True, attn_mask=True, update_epochs=10,
    attn_dropout_a=0.2, attn_dropout_v=0.0, relu_dropout=0.0, embed_dropout=0.2,
    res_dropout=0.0, dst_feature_dim_nheads=[50, 10], batch_size=16,
    learning_rate=0.0001, nlevels=2, conv1d_kernel_size_l=5,
    conv1d_kernel_size_a=5, conv1d_kernel_size_v=5, text_dropout=0.5,
    attn_dropout=0.3, output_dropout=0.5, grad_clip=0.6, patience=5,
    weight_decay=0.005, transformers="bert", pretrained="bert-base-uncased",
    feature_dims=[768, 5, 20], dataset_name="mosi", train_mode="regression",
)


#: Their head name -> ours. All five are supervised by the task loss, with the
#: language head weighted 3 and the rest 1 -- so all five have to agree, not just
#: the one the model finally predicts with.
HEADS = {
    "output_logit": "M",
    "logits_c": "shared_logit",
    "logits_l_hetero": "high_text",
    "logits_v_hetero": "high_vision",
    "logits_a_hetero": "high_audio",
}


class _StubBert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "trains").exists():
        return _skip(f"no DLF checkout at {ORIGIN}")
    for path in (SHIM, ORIGIN):
        if path.exists():
            sys.path.insert(0, str(path))
    try:
        from trains.singleTask.model.DLF import DLF as TheirDLF
    except ImportError as exc:
        return _skip(f"cannot import the reference ({exc})")
    from msa.models.dlf import DisentangledLanguageFocused

    torch.manual_seed(0)
    theirs = TheirDLF(types.SimpleNamespace(**MODEL_ARGS))
    ours = DisentangledLanguageFocused(
        text_dim=BERT_DIM, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM,
        text_length=LENGTH, audio_length=LENGTH, vision_length=LENGTH,
    )
    theirs.text_model = _StubBert()
    ours.encoder = _StubBert()

    unused = ("proj_cosine_l.", "proj_cosine_v.", "proj_cosine_a.",
              "trans_a_with_l.", "trans_a_with_v.", "trans_v_with_l.", "trans_v_with_a.")
    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("text_model.")
                    and not any(n.startswith(u) for u in unused)]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("encoder.")]
    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} parameter tensors in the reference, "
              f"{len(our_params)} in ours")
        their_shapes = [tuple(p.shape) for _, p in their_params]
        our_shapes = [tuple(p.shape) for _, p in our_params]
        print(f"  shapes only in theirs: {[s for s in their_shapes if s not in our_shapes][:6]}")
        print(f"  shapes only in ours:   {[s for s in our_shapes if s not in their_shapes][:6]}")
        return 1

    for (their_name, x), (our_name, y) in zip(their_params, our_params, strict=True):
        if x.shape != y.shape:
            print(f"FAIL: {their_name} is {tuple(x.shape)}, "
                  f"ours has {our_name} at {tuple(y.shape)}")
            return 1
    with torch.no_grad():
        for (_, x), (_, y) in zip(their_params, our_params, strict=True):
            y.copy_(x)

    text = torch.randn(BATCH, LENGTH, BERT_DIM)
    audio = torch.randn(BATCH, LENGTH, AUDIO_DIM)
    vision = torch.randn(BATCH, LENGTH, VISION_DIM)
    theirs.eval()
    ours.eval()
    with torch.no_grad():
        their_out = theirs(text, audio, vision)
        our_out = ours({"text_bert": text, "audio": audio, "vision": vision})

    print(f"{len(their_params)} parameter tensors matched and copied")
    worst = 0.0
    for their_key, our_key in HEADS.items():
        a = their_out[their_key].view(-1)
        b = our_out[our_key].view(-1)
        difference = float((a - b).abs().max())
        worst = max(worst, difference)
        print(f"  {their_key:18s} {difference:.3e}"
              f"   {[round(float(v), 6) for v in a[:2]]} vs {[round(float(v), 6) for v in b[:2]]}")
    print(f"max abs difference: {worst:.3e}")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
