"""Is our FeaDA numerically identical to the authors' implementation?

Convention 5, in the shape the previous five settled on: build both, copy every
weight across, feed one input, compare -- and compare every tensor a loss
consumes, not just the prediction.

Here that means the prediction, the six similar/dissimilar views, the two gates
`p2a` produces, and the two KL terms. The views feed the contrastive loss and
reach the prediction only through the gate; the KL terms are FeaDA's increment
over ConFEDE. A prediction-only check would be blind to all of it, which is how
DLF shipped missing four of five loss terms.

Inputs are **375 audio frames and 500 vision frames** -- the unaligned lengths the
data pipeline actually delivers. CLGSI's test used the config's post-alignment
numbers and so exercised shapes training never took, while passing.

BERT is stubbed on both sides, returning a fixed `last_hidden_state` and a fixed
`pooler_output`. Both are needed: the release projects the sequence for its
cross-modal streams and feeds the pooler to the six projections, and they are
different vectors.

`load_froze()` is not called from the release's `__init__`, so the model can be
built without its pretrained checkpoints.

Exit code 0 on match; SKIP also exits 0, so read the verdict.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

#: The release imports itself package-style (`import MOSI.config as cfg`), so
#: the PARENT goes on sys.path, not the MOSI directory.
ORIGIN = Path(os.environ.get("FEADA_ORIGIN", "/workspace/FeaDA-main"))
SHIM = Path(os.environ.get("MMSA_SHIM", "/workspace/mmsa_env/shim"))
BATCH, TEXT_LEN, TEXT_DIM, AUDIO_DIM, VISION_DIM = 4, 50, 768, 5, 20
AUDIO_LEN, VISION_LEN = 375, 500
TOLERANCE = 1e-5

#: Built by the release and never reached by its forward: the unimodal encoders'
#: second branch (proj_a into trans_encoder_a), whose output the fusion stage
#: binds and never reads.
UNUSED = ("vision_encoder.proj_a.", "vision_encoder.trans_encoder_a.",
          "audio_encoder.proj_a.", "audio_encoder.trans_encoder_a.")


class _PlaceholderText(nn.Module):
    """Stands in for the release's TextEncoder at construction time.

    Replaced with `_StubBert` once the model exists; this only has to build
    without reaching the authors' local filesystem.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("placeholder: replace with _StubBert before forward")


class _StubBert(nn.Module):
    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("hidden", hidden)
        self.register_buffer("pooled", pooled)

    def forward(self, *args, **kwargs):
        return self.hidden, self.pooled


class _StubBertModel(nn.Module):
    """Matches what `output.last_hidden_state` / `.pooler_output` access expects."""

    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("hidden", hidden)
        self.register_buffer("pooled", pooled)

    def forward(self, **kwargs):
        return types.SimpleNamespace(last_hidden_state=self.hidden, pooler_output=self.pooled)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main() -> int:
    if not (ORIGIN / "MOSI" / "models").exists():
        return _skip(f"no FeaDA checkout at {ORIGIN}")
    # Two entries: the parent for its package-style `import MOSI.config`, and
    # MOSI/models for its flat `from Text_encoder import TextEncoder`. The
    # release mixes both styles.
    for path in (SHIM, ORIGIN, ORIGIN / "MOSI" / "models"):
        if path.exists():
            sys.path.insert(0, str(path))
    try:
        import MOSI.config as their_config
        their_config.DEVICE = torch.device("cpu")

        # Its TextEncoder hardcodes a BERT path on the authors' own machine, and
        # BERT is stubbed here anyway, so the class is replaced before the model
        # is built rather than the path being repointed.
        import Text_encoder

        Text_encoder.TextEncoder = _PlaceholderText
        from MOSI.models.model import TVA_fusion as TheirFeaDA
    except Exception as exc:                                    # noqa: BLE001
        return _skip(f"cannot import the reference ({type(exc).__name__}: {exc})")
    from msa.models import FeaDA

    torch.manual_seed(0)
    theirs = TheirFeaDA(config=their_config)
    ours = FeaDA()

    hidden = torch.randn(BATCH, TEXT_LEN, TEXT_DIM)
    pooled = torch.randn(BATCH, TEXT_DIM)
    theirs.text_encoder = _StubBert(hidden, pooled)
    ours.text_encoder.bert = _StubBertModel(hidden, pooled)

    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("text_encoder.")
                    and not any(n.startswith(u) for u in UNUSED)]
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("text_encoder.")]
    if len(their_params) != len(our_params):
        print(f"FAIL: {len(their_params)} tensors in the reference, {len(our_params)} in ours")
        theirs_shapes = [tuple(p.shape) for _, p in their_params]
        ours_shapes = [tuple(p.shape) for _, p in our_params]
        for side, params, other in (("theirs", their_params, ours_shapes),
                                    ("ours", our_params, theirs_shapes)):
            extra = [(n, tuple(p.shape)) for n, p in params if tuple(p.shape) not in other]
            print(f"  only in {side}: {extra[:8]}")
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
    label = torch.randn(BATCH)
    from msa.models.feada import padding_mask

    theirs.eval()
    ours.eval()
    sample = {
        "raw_text": ["unused"] * BATCH, "vision": vision, "audio": audio,
        "labels": {"M": label},
        "vision_padding_mask": padding_mask(vision),
        "audio_padding_mask": padding_mask(audio),
    }
    with torch.no_grad():
        their_pred, *_ = theirs(sample, None, mode="eval", device=torch.device("cpu"))
        our_out = ours({"text_bert": torch.zeros(BATCH, 3, TEXT_LEN, dtype=torch.long),
                        "vision": vision, "audio": audio, "label": label})

        # The intermediates, recomputed through their own submodules.
        xv = theirs.vision_encoder(vision, key_padding_mask=padding_mask(vision))[1].squeeze()
        xa = theirs.audio_encoder(audio, key_padding_mask=padding_mask(audio))[1].squeeze()
        their_views = {
            "t_simi": theirs.T_simi_proj(pooled), "v_simi": theirs.V_simi_proj(xv),
            "a_simi": theirs.A_simi_proj(xa), "t_dissimi": theirs.T_dissimi_proj(pooled),
            "v_dissimi": theirs.V_dissimi_proj(xv), "a_dissimi": theirs.A_dissimi_proj(xa),
        }

    worst, worst_name = 0.0, ""

    def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        nonlocal worst, worst_name
        a, b = a.reshape(-1), b.reshape(-1)
        if a.shape != b.shape:
            print(f"FAIL: {name} is {tuple(a.shape)}, ours {tuple(b.shape)}")
            return False
        difference = float((a - b).abs().max())
        if difference > worst:
            worst, worst_name = difference, name
        print(f"  {name:12s} {difference:.3e}")
        return True

    if not compare("prediction", their_pred, our_out["M"]):
        return 1
    scale = float(their_pred.abs().max())
    print(f"  (prediction magnitude {scale:.2f}; relative "
          f"{worst / max(scale, 1e-12):.2e})")
    for key, value in their_views.items():
        if not compare(key, value, our_out[f"view_{key}"]):
            return 1
    with torch.no_grad():
        for key, source in (("gate_v", "v_simi"), ("gate_a", "a_simi")):
            if not compare(key, theirs.p2a(their_views[source]), our_out[key]):
                return 1

    print(f"{len(their_params)} parameter tensors matched and copied")
    print(f"max abs difference: {worst:.3e} ({worst_name})")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
