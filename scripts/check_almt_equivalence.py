"""Is our ALMT numerically identical to MMSA's?

We rewrote ALMT against plain tensor ops instead of transcribing MMSA's einops
version, so "the maths is the same" is a claim, not a fact. This tests it: build
both models, copy every weight across in registration order, feed the same input,
compare the outputs.

It has already earned its place. The first version of our port scored *better*
than the reference on all seven metrics — which is the wrong kind of good news,
and it was. This test found three things reading had missed:

* `l_encoder` is built from the same `Transformer` class with `token_len=None`,
  which still creates a positional embedding. The text stream had lost its only
  notion of order before the AHL layers query it.
* the fusion layer adds positional embeddings to *both* of its inputs.
* the fusion layer prepends a shared CLS token to both streams, and the final
  `[:, 0]` reads **that token**. Without it the regression head was reading the
  first hyper-modality token instead — a different quantity entirely.

BERT is stubbed on both sides: it is identical by construction (both wrap the
same pretrained weights) and loading it twice makes the test slow for nothing.
What is under test is everything that was rewritten.

Exit code 0 if the outputs match to 1e-6, non-zero otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reference_paths import mmsa_shim, mmsa_src  # noqa: E402

MMSA_SRC = mmsa_src()
#: transformers 4.x, einops and easydict live here, not in this project's venv —
#: MMSA needs all three and we deliberately do not depend on einops or easydict.
#: Resolved by _reference_paths, which both this and mmsa_reference.py share. It
#: was twice a hard-coded absolute path, and both times this check went quietly
#: to SKIP rather than red — see that module for why the constant had to go.
SHIM = mmsa_shim()

BATCH, DIM, TOKENS = 4, 128, 8
LEN_TEXT, LEN_AUDIO, LEN_VISION = 50, 375, 500
TOLERANCE = 1e-6


class _Args(dict):
    __getattr__ = dict.get


ARGS = _Args(
    use_bert=True, use_finetune=True, transformers="bert",
    pretrained="bert-base-uncased", feature_dims=[768, 5, 20],
    feature_length=[LEN_TEXT, LEN_AUDIO, LEN_VISION], dst_feature_dims=DIM,
    dst_feature_hidden_dims=DIM, dst_embedding_length=TOKENS,
    embedding_depth=[1, 1, 1], embedding_heads=[8, 8, 8], l_encoder_heads=8,
    AHL_depth=3, h_hyper_layer_heads=8, fusion_hidden_d=DIM, fusion_heads=8,
    fusion_layer_depth=2, train_mode="regression", num_classes=1,
)


class _StubBert(nn.Module):
    """Stands in for the text encoder; the test feeds its output directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def main() -> int:
    if not MMSA_SRC.exists():
        print(f"SKIP: no MMSA checkout at {MMSA_SRC}")
        return 0
    for path in (SHIM, MMSA_SRC):
        sys.path.insert(0, str(path))
    try:
        from MMSA.models.singleTask.ALMT import ALMT as TheirALMT
    except ImportError as exc:
        print(f"SKIP: cannot import MMSA's ALMT ({exc})")
        return 0
    from msa.models.almt import ALMT as OurALMT

    torch.manual_seed(0)
    theirs = TheirALMT(ARGS)
    theirs.bertmodel = _StubBert()
    ours = OurALMT(text_dim=768, audio_dim=5, vision_dim=20)
    ours.encoder = _StubBert()

    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("bertmodel.")]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("encoder.")]
    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} parameter tensors in MMSA's ALMT, "
              f"{len(our_params)} in ours")
        return 1
    for (their_name, x), (our_name, y) in zip(their_params, our_params, strict=True):
        if x.shape != y.shape:
            print(f"FAIL: {their_name} is {tuple(x.shape)}, "
                  f"ours has {our_name} at {tuple(y.shape)}")
            return 1
    with torch.no_grad():
        for (_, x), (_, y) in zip(their_params, our_params, strict=True):
            y.copy_(x)

    text = torch.randn(BATCH, LEN_TEXT, 768)
    audio = torch.randn(BATCH, LEN_AUDIO, 5)
    vision = torch.randn(BATCH, LEN_VISION, 20)
    theirs.eval()
    ours.eval()
    with torch.no_grad():
        their_out = theirs(text, audio, vision).view(-1)
        our_out = ours({"text_bert": text, "audio": audio, "vision": vision})["M"]

    difference = float((their_out - our_out).abs().max())
    print(f"{len(their_params)} parameter tensors matched and copied")
    print(f"MMSA: {[round(float(v), 6) for v in their_out]}")
    print(f"ours: {[round(float(v), 6) for v in our_out]}")
    print(f"max abs difference: {difference:.3e}")
    if difference > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
