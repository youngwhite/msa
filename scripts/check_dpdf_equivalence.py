"""Is our DPDF-LQ numerically identical to the authors' implementation?

We rebuilt DPDF-LQ from the authors' code rather than transcribing it, so "the
maths is the same" is a claim, not a fact. This tests it the way convention 5
requires: construct both models, copy every weight across, feed one input,
compare the outputs.

Two known and deliberate differences, both excluded from the comparison rather
than papered over:

* The reference's **global path constructs `conv_att_v` and never calls it** --
  vision reaches `proj_v` directly. It is the "declared but never reached" form
  from convention 5: real parameters, no effect on the output. We do not build
  it, so its tensors are skipped here. If a future reference version starts
  calling it, the shapes stop lining up and this test says so.
* The reference builds **three modules it never calls**: `conv_att_v` and
  `regression_layer` in the global path, and `classifier` in the local one. All
  three hold real parameters and none reach the output -- regression is done by
  `Gate_fusion` instead. Convention 5 says an inherited defect is recorded, not
  reproduced, so we do not build them and their tensors are skipped here.
* The reference hard-codes `source_num_frames=58` in the local cross-transformer
  (8 tokens + 50 frames), which would break on any other sequence length. Ours
  derives it, and the test passes 58 explicitly so the two positional embeddings
  are the same size.

BERT is stubbed on both sides -- identical by construction, both wrapping the
same pretrained weights, and loading four copies would make the test slow for
nothing. What is under test is everything that was rewritten.

Exit code 0 if the outputs match to 1e-5, non-zero otherwise. Note that SKIP
also exits 0, so read the verdict, not the code.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ORIGIN = Path(os.environ.get("DPDF_LQ_ORIGIN", "/workspace/DPDF-LQ"))
#: einops lives here, not in this project's venv -- the reference needs it and
#: we deliberately do not depend on it. Same directory mmsa_reference.py uses.
SHIM = Path(os.environ.get("MMSA_SHIM", "/workspace/mmsa_env/shim"))

BATCH, TEXT_LEN, AUDIO_LEN, VISION_LEN = 4, 50, 50, 50
#: The reference sizes every positional embedding for 500 frames regardless of
#: the data's actual 50, so both sides are built that way and fed 50 steps; the
#: unused tail is never read by either.
PROJ_LEN = 500
AUDIO_DIM, VISION_DIM, BERT_DIM = 5, 20, 768
TOLERANCE = 1e-5

#: configs/mosi.yaml, model section, verbatim.
MODEL_ARGS = dict(
    bert_pretrained="bert-base-uncased",
    l_proj_dim=768, a_proj_dim=5, v_proj_dim=20, proj_dst_dim=128,
    token_len=8, token_dim=128,
    l_proj_length=500, a_proj_length=500, v_proj_length=500,
    proj_input_dim=128, proj_depth=1, proj_heads=8, proj_mlp_dim=128,
    token_length=8, l_enc_heads=8, l_enc_mlp_dim=128,
    DGLQA_depth=3, DGLQA_heads=8, DGLQA_dim_head=16, DGLQA_droup=0.0,
    fusion_heads=8, fusion_mlp_dim=128, fusion_layer_depth=2,
)


class _StubBert(nn.Module):
    """Stands in for the text encoder; the test feeds its output directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "models").exists():
        return _skip(f"no DPDF-LQ checkout at {ORIGIN}")
    for path in (SHIM, ORIGIN):
        sys.path.insert(0, str(path))
    try:
        from models.Gate_fusion import Gate_fusion
    except ImportError as exc:
        return _skip(f"cannot import the reference ({exc})")
    from msa.models.dpdf_lq import DualPathDynamicFusion

    args = types.SimpleNamespace(model=types.SimpleNamespace(**MODEL_ARGS))
    torch.manual_seed(0)
    theirs = Gate_fusion(args)
    ours = DualPathDynamicFusion(
        text_dim=BERT_DIM, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM,
        text_length=PROJ_LEN, audio_length=PROJ_LEN, vision_length=PROJ_LEN,
        local_source_len=58, local_target_len=58,
    )
    for path in (theirs.Global_path, theirs.Local_path):
        path.bertmodel = _StubBert()
    for streams in (ours.global_streams, ours.local_streams):
        streams.encoder = _StubBert()

    skip_theirs = ("bertmodel.", "Global_path.conv_att_v.",
                   "Global_path.regression_layer.", "Local_path.classifier.")
    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not any(n.startswith(s) or f".{s}" in n for s in skip_theirs)]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if ".encoder.bert." not in n]

    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} parameter tensors in the reference "
              f"(after skipping the unused conv_att_v), {len(our_params)} in ours")
        their_shapes = [tuple(p.shape) for _, p in their_params]
        our_shapes = [tuple(p.shape) for _, p in our_params]
        extra = [s for s in their_shapes if s not in our_shapes][:6]
        missing = [s for s in our_shapes if s not in their_shapes][:6]
        print(f"  shapes only in theirs: {extra}")
        print(f"  shapes only in ours:   {missing}")
        return 1

    for (their_name, x), (our_name, y) in zip(their_params, our_params, strict=True):
        if x.shape != y.shape:
            print(f"FAIL: {their_name} is {tuple(x.shape)}, "
                  f"ours has {our_name} at {tuple(y.shape)}")
            return 1
    with torch.no_grad():
        for (_, x), (_, y) in zip(their_params, our_params, strict=True):
            y.copy_(x)

    text = torch.randn(BATCH, TEXT_LEN, BERT_DIM)
    audio = torch.randn(BATCH, AUDIO_LEN, AUDIO_DIM)
    vision = torch.randn(BATCH, VISION_LEN, VISION_DIM)
    theirs.eval()
    ours.eval()
    with torch.no_grad():
        their_out = theirs(vision, audio, text).view(-1)
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
