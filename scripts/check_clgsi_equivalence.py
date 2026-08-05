"""Is our CLGSI numerically identical to the authors' implementation?

Convention 5, in the shape the previous four checks settled on: build both
models, copy every weight across, feed one input, compare -- and compare **every
tensor the loss consumes**, not just the prediction.

That means all four returned features as well as `M`. Three of them
(`Feature_t/a/v`) feed the contrastive term and reach the prediction only
indirectly, and `Feature_f` reaches the prediction but, notably, is returned and
never used by the contrastive loss at all. A prediction-only comparison would be
blind to a wrong unimodal head, which is exactly how DLF shipped with four of
five loss terms missing (`docs/investigations.md#dlf-task-heads`).

The contrastive loss is checked separately, against the release's own
`contrastive_loss` module on the same features and labels. The weighting by
sentiment distance is the whole point of this paper, and a weight-copy test of
the forward pass says nothing about it.

Six modules the release declares and never calls are excluded, per convention 5:
`post_fusion_layer_1`, `post_{text,audio,video}_layer_2`,
`skip_connection_BatchNorm` and `post_fusion_dropout`.

BERT is stubbed on both sides. Exit code 0 on match; SKIP also exits 0, so read
the verdict rather than the code.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ORIGIN = Path(os.environ.get("CLGSI_ORIGIN", "/workspace/CLGSI"))
SHIM = Path(os.environ.get("MMSA_SHIM", "/workspace/mmsa_env/shim"))
#: UNALIGNED lengths, because that is what the release reads and what our
#: model now pools down. Testing at 50/50/50 would test a path the trained
#: model never takes -- the DPDF-LQ lesson, where a passing test covered
#: shapes training never used.
BATCH, LENGTH, TEXT_DIM, AUDIO_DIM, VISION_DIM = 8, 50, 768, 5, 20
AUDIO_LEN, VISION_LEN = 375, 500
TOLERANCE = 1e-5

UNUSED = ("post_fusion_layer_1.", "post_text_layer_2.", "post_audio_layer_2.",
          "post_video_layer_2.", "skip_connection_BatchNorm.",
          # Not merely unused -- duplicated. The release does
          # `self.encoder_layer = nn.TransformerEncoderLayer(...)` and then
          # `nn.TransformerEncoder(self.encoder_layer, n)`, which DEEP-COPIES the
          # layer. The original stays registered as a submodule, gets gradients
          # from nothing, and is 12 tensors per encoder -- 24 across audio and
          # vision, which is exactly the count this test first disagreed by.
          "audio_model.encoder_layer.", "video_model.encoder_layer.")

ARGS = dict(
    need_data_aligned=True, language="en", use_finetune=True,
    feature_dims=(768, 5, 20), seq_lens=(50, 50, 50),
    need_model_aligned=True, modelName="clgsi",
    a_encoder_heads=1, v_encoder_heads=4, a_encoder_layers=2, v_encoder_layers=2,
    text_out=768, audio_out=5, video_out=20, t_bert_dropout=0.1,
    post_fusion_dim=128, post_text_dim=64, post_audio_dim=64, post_video_dim=64,
    post_fusion_dropout=0.2, post_text_dropout=0.05, post_audio_dropout=0.05,
    post_video_dropout=0.05, skip_net_reduction=2, fusion_filter_nums=16,
    datasetName="mosi", device=torch.device("cpu"), H=3.0,
)

COMPARED = {"M": "M", "Feature_t": "Feature_t", "Feature_a": "Feature_a",
            "Feature_v": "Feature_v", "Feature_f": "Feature_f"}


class _StubBert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "models").exists():
        return _skip(f"no CLGSI checkout at {ORIGIN}")
    for path in (SHIM, ORIGIN):
        if path.exists():
            sys.path.insert(0, str(path))
    try:
        from models.contrastive_loss import contrastive_loss as TheirLoss
        from models.multiTask.CLGSI import CLGSI as TheirCLGSI
    except Exception as exc:                                    # noqa: BLE001
        return _skip(f"cannot import the reference ({type(exc).__name__}: {exc})")
    from msa.models import CLGSI as OurCLGSI
    from msa.models.clgsi import GAMMA

    torch.manual_seed(0)
    theirs = TheirCLGSI(types.SimpleNamespace(**ARGS))
    ours = OurCLGSI()
    text = torch.randn(BATCH, LENGTH, TEXT_DIM)
    theirs.text_model = _StubBert()
    ours.text_model = _StubBert()

    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("text_model.")
                    and not any(n.startswith(u) for u in UNUSED)]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("text_model.")]
    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} tensors in the reference, {len(our_params)} in ours")
        theirs_shapes = [tuple(p.shape) for _, p in their_params]
        ours_shapes = [tuple(p.shape) for _, p in our_params]
        for side, params, other in (("theirs", their_params, ours_shapes),
                                    ("ours", our_params, theirs_shapes)):
            extra = [(n, tuple(p.shape)) for n, p in params if tuple(p.shape) not in other]
            print(f"  only in {side}: {extra[:6]}")
        return 1
    for (tn, x), (on, y) in zip(their_params, our_params, strict=True):
        if x.shape != y.shape:
            print(f"FAIL: {tn} is {tuple(x.shape)}, ours has {on} at {tuple(y.shape)}")
            return 1
    with torch.no_grad():
        for (_, x), (_, y) in zip(their_params, our_params, strict=True):
            y.copy_(x)

    audio = torch.randn(BATCH, AUDIO_LEN, AUDIO_DIM)
    vision = torch.randn(BATCH, VISION_LEN, VISION_DIM)
    label = torch.randn(BATCH) * 3
    theirs.eval()
    ours.eval()
    with torch.no_grad():
        from models.subNets.AlignNets import AlignSubNet
        aligner = AlignSubNet(types.SimpleNamespace(**ARGS), 'avg_pool')
        _, their_audio, their_vision = aligner(text, audio, vision)
        their_out = theirs(text, (their_audio, None), (their_vision, None))
        our_out = ours({"text_bert": text, "audio": audio, "vision": vision, "label": label})

    worst, worst_name = 0.0, ""
    for their_key, our_key in COMPARED.items():
        a = their_out[their_key].reshape(-1)
        b = our_out[our_key].reshape(-1)
        if a.shape != b.shape:
            print(f"FAIL: {their_key} is {tuple(a.shape)}, ours {tuple(b.shape)}")
            return 1
        difference = float((a - b).abs().max())
        worst, worst_name = max((worst, worst_name), (difference, their_key))
        print(f"  {their_key:12s} {difference:.3e}")

    # The contrastive loss, against the release's own module on the same inputs.
    with torch.no_grad():
        their_loss = float(TheirLoss("mosi", torch.device("cpu"), 0.4)(their_out, label))
        our_loss = float(ours.contrastive(our_out, label))
    difference = abs(their_loss - our_loss)
    worst, worst_name = max((worst, worst_name), (difference, "contrastive loss"))
    print(f"  {'contrastive':12s} {difference:.3e}   theirs {their_loss:.6f} "
          f"ours {our_loss:.6f}   (weight {GAMMA})")

    print(f"{len(their_params)} parameter tensors matched and copied")
    print(f"max abs difference: {worst:.3e} ({worst_name})")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
