"""Is our MMIM numerically identical to MMSA's?

Convention 5 requires this for any model rewritten rather than transcribed, and
MMIM never had it -- ALMT did, which is how the ALMT port that beat its
reference on all seven metrics was caught being wrong. This is the same test for
MMIM, and it exists because the port scores about one standard deviation better
than MMSA's code on MOSEI (test MAE 0.5774 against 0.5912, p=0.030) with every
cheap explanation ruled out: update_epochs, epoch selection, NaN/Inf cleaning,
preprocessing, all three learning rates, the validation selection quantity, and
the memory bank's scope. See docs/investigations.md#mmim-mosei-vs-mmsa.

It splits the question in one forward pass. Identical outputs mean the model
definitions agree and whatever is left lives in the training protocol; different
outputs mean the difference is in the model, and MMIM's phase-1 acceptance needs
revisiting along with everything MOSEI concluded from it.

Two ordering differences make a positional weight copy wrong here, unlike ALMT:
MMSA registers `visual_enc` before `acoustic_enc` where we register audio first,
and it puts `fusion_prj` after the three CPC heads where we put it before. So
the copy goes module by module through an explicit map. A blind zip would have
been caught by the shape check -- audio is 74-dimensional and vision 35 -- but
only by accident, and not at all if the dimensions had matched.

BERT is stubbed on both sides: identical by construction, and loading it twice
makes the test slow for nothing. What is under test is everything else.

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
SHIM = mmsa_shim()

BATCH = 4
TEXT_DIM, AUDIO_DIM, VISION_DIM = 768, 74, 35     # MOSEI's dimensions
LEN_TEXT, LEN_AUDIO, LEN_VISION = 50, 500, 500
TOLERANCE = 1e-6

#: MMSA's module name -> ours. Registration order differs, so the copy is by
#: name; see the module docstring.
MODULES = (
    ("acoustic_enc", "audio_enc"),
    ("visual_enc", "vision_enc"),
    ("mi_tv", "mi_tv"),
    ("mi_ta", "mi_ta"),
    ("fusion_prj", "fusion"),
    ("cpc_zt", "cpc_zt"),
    ("cpc_zv", "cpc_zv"),
    ("cpc_za", "cpc_za"),
)


class _Args(dict):
    __getattr__ = dict.get


#: MMSA's own MOSEI values for MMIM, which are field-for-field its MOSI values.
ARGS = _Args(
    use_bert=True, use_finetune=True, transformers="bert",
    pretrained="bert-base-uncased",
    feature_dims=[TEXT_DIM, AUDIO_DIM, VISION_DIM],
    need_data_aligned=False, add_va=False,
    d_ah=16, d_vh=16, d_aout=16, d_vout=16, d_prjh=128,
    n_layer=1, cpc_layers=1, bidirectional=True,
    dropout_a=0.1, dropout_v=0.1, dropout_prj=0.1,
    mmilb_mid_activation="ReLU", mmilb_last_activation="Tanh",
    cpc_activation="Tanh",
    train_mode="regression", num_classes=1,
)


class _StubBert(nn.Module):
    """Stands in for the text encoder; the test feeds its output directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def copy_weights(theirs: nn.Module, ours: nn.Module) -> tuple[int, str | None]:
    """Copy module by module. Returns (tensors copied, first problem or None)."""
    copied = 0
    for their_name, our_name in MODULES:
        their_module = getattr(theirs, their_name)
        our_module = getattr(ours, our_name)
        their_params = list(their_module.named_parameters())
        our_params = list(our_module.named_parameters())
        if len(their_params) != len(our_params):
            return copied, (f"{their_name} has {len(their_params)} tensors, our "
                            f"{our_name} has {len(our_params)}")
        for (tn, x), (on, y) in zip(their_params, our_params, strict=True):
            if x.shape != y.shape:
                return copied, (f"{their_name}.{tn} is {tuple(x.shape)}, our "
                                f"{our_name}.{on} is {tuple(y.shape)}")
            with torch.no_grad():
                y.copy_(x)
            copied += 1
    return copied, None


def main() -> int:
    if not MMSA_SRC.exists():
        print(f"SKIP: no MMSA checkout at {MMSA_SRC}")
        return 0
    for path in (SHIM, MMSA_SRC):
        sys.path.insert(0, str(path))
    try:
        from MMSA.models.singleTask.MMIM import MMIM as TheirMMIM
    except ImportError as exc:
        print(f"SKIP: cannot import MMSA's MMIM ({exc})")
        return 0
    from msa.models.mmim import MMIM as OurMMIM

    torch.manual_seed(0)
    theirs = TheirMMIM(ARGS)
    theirs.bertmodel = _StubBert()
    ours = OurMMIM(text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM)
    ours.encoder = _StubBert()

    # Every parameter outside BERT must be accounted for, or the copy is partial
    # and a match would mean nothing.
    their_total = sum(1 for n, _ in theirs.named_parameters()
                      if not n.startswith("bertmodel."))
    our_total = sum(1 for n, _ in ours.named_parameters()
                    if not n.startswith("encoder."))
    copied, problem = copy_weights(theirs, ours)
    if problem:
        print(f"FAIL: {problem}")
        return 1
    if copied != their_total or copied != our_total:
        print(f"FAIL: copied {copied} tensors but MMSA has {their_total} outside "
              f"BERT and we have {our_total} — the map in MODULES is incomplete")
        return 1

    text = torch.randn(BATCH, LEN_TEXT, TEXT_DIM)
    audio = torch.randn(BATCH, LEN_AUDIO, AUDIO_DIM)
    vision = torch.randn(BATCH, LEN_VISION, VISION_DIM)
    audio_len = torch.tensor([LEN_AUDIO, 300, 120, 7])
    vision_len = torch.tensor([LEN_VISION, 410, 90, 3])

    theirs.eval()
    ours.eval()
    with torch.no_grad():
        their_out = theirs(text, (audio, audio_len), (vision, vision_len))
        our_out = ours({"text_bert": text, "audio": audio, "vision": vision,
                        "audio_length": audio_len, "vision_length": vision_len})

    print(f"{copied} parameter tensors matched and copied")
    worst, worst_key = 0.0, None
    # M is the prediction; nce and lld are the contrastive terms, which is what
    # the MOSEI question is actually about, so a match on M alone is not enough.
    for their_key, our_key in (("M", "M"), ("nce", "nce"), ("lld", "lld")):
        a = their_out[their_key].reshape(-1)
        b = our_out[our_key].reshape(-1)
        difference = float((a - b).abs().max())
        print(f"  {our_key:4s} MMSA {[round(float(v), 6) for v in a[:4]]}")
        print(f"       ours {[round(float(v), 6) for v in b[:4]]}   "
              f"max abs diff {difference:.3e}")
        if difference > worst:
            worst, worst_key = difference, our_key

    print(f"\nworst: {worst_key} at {worst:.3e}")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e} — the difference is in the "
              f"model, not the training protocol")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
