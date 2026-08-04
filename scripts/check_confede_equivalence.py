"""Is our ConFEDE numerically identical to the authors' implementation?

Convention 5, same shape as the DLF and DMD checks: build both models, copy every
weight across, feed one input, compare.

**Both encoders and all six projections are compared, not just the prediction.**
The six similar/dissimilar views are what the contrastive loss consumes and the
mono decoder scores; only their concatenation reaches the prediction, so a
prediction-only comparison would pass while any individual view was wrong. This
is the DLF lesson (`docs/investigations.md#dlf-task-heads`) applied rather than
relearned.

The text encoder is stubbed on both sides. The release tokenises raw strings
inside the encoder and takes `pooler_output`; that path is identical by
construction on both sides because it is the same pretrained BERT called the same
way, and stubbing it keeps the test independent of tokeniser downloads.

Dead weights in the release are excluded, as convention 5 directs: the vision and
audio encoders each declare an `fc`, a `dense`, an `activation` and an empty
`cls_embedding` that `forward` never touches -- the projection into the
transformer's width happens inside the positional encoder instead.

Exit code 0 if the outputs match, non-zero otherwise. SKIP also exits 0, so read
the verdict rather than the code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ORIGIN = Path(os.environ.get("CONFEDE_ORIGIN", "/workspace/ConFEDE")) / "MOSI"
BATCH, VISION_LEN, AUDIO_LEN, VISION_DIM, AUDIO_DIM, WIDTH = 4, 500, 375, 20, 5, 768
TOLERANCE = 1e-5

#: Declared by the release's vision and audio encoders and never used by their
#: forward. Anchored to those two modules on purpose: an unanchored "fc." also
#: matches the six projectors, whose only layer is called `fc`, and silently
#: excluded 24 real tensors -- the count mismatch was the only sign.
UNUSED = tuple(f"{module}_encoder.{attr}"
               for module in ("vision", "audio")
               for attr in ("fc.", "dense.", "cls_embedding", "activation."))


class _StubText(nn.Module):
    """Stands in for BERT-over-raw-strings on both sides."""

    def __init__(self, fixed: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed", fixed)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.fixed


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "model").exists():
        return _skip(f"no ConFEDE checkout at {ORIGIN}")
    sys.path.insert(0, str(ORIGIN))
    try:
        import config as their_config
        their_config.DEVICE = torch.device("cpu")
        from model.net.constrastive.TVA_fusion import TVA_fusion
    except Exception as exc:                                   # noqa: BLE001
        return _skip(f"cannot import the reference ({type(exc).__name__}: {exc})")
    from msa.models import ConFEDE

    torch.manual_seed(0)
    theirs = TVA_fusion(config=their_config)
    ours = ConFEDE(vision_dim=VISION_DIM, audio_dim=AUDIO_DIM,
                   vision_length=VISION_LEN, audio_length=AUDIO_LEN)

    text_feature = torch.randn(BATCH, WIDTH)
    theirs.text_encoder = _StubText(text_feature)
    ours.text_encoder = _StubText(text_feature)

    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("text_encoder.")
                    and not any(u in n for u in UNUSED)]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("text_encoder.")]
    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} parameter tensors in the reference, "
              f"{len(our_params)} in ours")
        their_shapes = [tuple(p.shape) for _, p in their_params]
        our_shapes = [tuple(p.shape) for _, p in our_params]
        for side, params, other in (("theirs", their_params, our_shapes),
                                    ("ours", our_params, their_shapes)):
            extra = [(n, tuple(p.shape)) for n, p in params if tuple(p.shape) not in other]
            print(f"  only in {side}: {extra[:6]}")
        return 1
    for (their_name, x), (our_name, y) in zip(their_params, our_params, strict=True):
        if x.shape != y.shape:
            print(f"FAIL: {their_name} is {tuple(x.shape)}, ours has {our_name} "
                  f"at {tuple(y.shape)}")
            return 1
    with torch.no_grad():
        for (_, x), (_, y) in zip(their_params, our_params, strict=True):
            y.copy_(x)

    vision = torch.randn(BATCH, VISION_LEN, VISION_DIM)
    audio = torch.randn(BATCH, AUDIO_LEN, AUDIO_DIM)
    # Zero a tail of frames on one sample so the padding mask is exercised rather
    # than being all-False: masks are where an encoder silently diverges.
    vision[0, 400:] = 0
    audio[1, 300:] = 0
    from msa.models.confede import padding_mask

    theirs.eval()
    ours.eval()
    sample = {
        "raw_text": ["unused"] * BATCH,
        "vision": vision, "audio": audio,
        "regression_labels": torch.randn(BATCH),
        "vision_padding_mask": padding_mask(vision),
        "audio_padding_mask": padding_mask(audio),
    }
    with torch.no_grad():
        their_pred, their_embed, *_ = theirs(sample, None, return_loss=True,
                                             device=torch.device("cpu"))
        our_out = ours({"raw_text": sample["raw_text"], "vision": vision,
                        "audio": audio, "index": torch.arange(BATCH),
                        "label": sample["regression_labels"]})

    worst, worst_name = 0.0, ""
    checks = [("vision embedding", their_embed[1], ours.vision_encoder(
        vision, padding_mask(vision))),
        ("audio embedding", their_embed[2], ours.audio_encoder(
            audio, padding_mask(audio))),
        ("prediction", their_pred.reshape(-1), our_out["prediction"].reshape(-1))]
    with torch.no_grad():
        for name, a, b in checks:
            a, b = a.reshape(-1), b.reshape(-1)
            if a.shape != b.shape:
                print(f"FAIL: {name} is {tuple(a.shape)} vs {tuple(b.shape)}")
                return 1
            difference = float((a - b).abs().max())
            if difference > worst:
                worst, worst_name = difference, name
            print(f"  {name:18s} {difference:.3e}")
        # All six views, in the release's order.
        views = our_out["views"]
        for i, projector in enumerate(("T_simi", "V_simi", "A_simi",
                                       "T_dissimi", "V_dissimi", "A_dissimi")):
            source = {"T": their_embed[0], "V": their_embed[1], "A": their_embed[2]}[
                projector[0]]
            a = getattr(theirs, f"{projector}_proj")(source).reshape(-1)
            b = views[i].reshape(-1)
            difference = float((a - b).abs().max())
            if difference > worst:
                worst, worst_name = difference, projector
            print(f"  {projector:18s} {difference:.3e}")

    print(f"{len(their_params)} parameter tensors matched and copied")
    print(f"max abs difference: {worst:.3e} ({worst_name})")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
