"""Is our DLF numerically identical to the authors' implementation?

DLF was rebuilt from the release rather than transcribed, so "the maths is the
same" is a claim until something checks it. This is that check, in the shape
convention 5 requires: construct both models, copy every weight across, feed one
input, compare the outputs.

The comparison is on the prediction only. The release returns nineteen tensors
from `forward`, most of them intermediates the loss consumes; if the final
prediction agrees to 1e-5 after a full weight copy, every stage that feeds it
agrees too, and the intermediates are checked implicitly.

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


class _StubBert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "trains").exists():
        return _skip(f"no DLF checkout at {ORIGIN}")
    sys.path.insert(0, str(ORIGIN))
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
        their_out = theirs(text, audio, vision)["output_logit"].view(-1)
        our_out = ours({"text_bert": text, "audio": audio, "vision": vision})["M"]

    difference = float((their_out - our_out).abs().max())
    print(f"{len(their_params)} parameter tensors matched and copied")
    print(f"reference: {[round(float(v), 6) for v in their_out]}")
    print(f"ours:      {[round(float(v), 6) for v in our_out]}")
    print(f"max abs difference: {difference:.3e}")
    if difference > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
