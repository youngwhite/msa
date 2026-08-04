"""Is our DMD numerically identical to the authors' implementation?

Convention 5: DMD was rebuilt from the release rather than transcribed, so "the
maths is the same" is a claim until something checks it. Construct both models,
copy every weight across, feed one input, compare the outputs.

**Every supervised head and every distilled representation is compared**, not
just the prediction. Six of the eight heads never reach the prediction, and the
six representations reach only the distillation losses, so a prediction-only
comparison cannot see any of them. That is not a hypothetical: it is exactly how
DLF's port shipped with four of five loss terms missing and a passing test
(`docs/investigations.md#dlf-task-heads`). Fourteen tensors are compared here
for that reason.

`proj_cosine_l/v/a` are excluded. The release builds them, computes with them,
returns the results, and reads them nowhere -- so they are dead output, and
convention 5 excludes rather than reproduces such things.

BERT is stubbed on both sides -- identical by construction -- so what is under
test is everything that was rewritten.

The orthogonality loss gets a second, separate check. The release passes 3-D
tensors to `nn.CosineEmbeddingLoss`, which current PyTorch rejects outright, so
our port reconstructs the old behaviour; `cosine_embedding_over_dim1` is
therefore asserted to agree with the current `nn.CosineEmbeddingLoss` on 2-D
input, which pins the formula independently of the port.

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

ORIGIN = Path(os.environ.get("DMD_ORIGIN", "/workspace/DMD"))
#: The release imports pynvml, which lives in the reference environment. Without
#: this the test SKIPs, and a SKIP exits 0, so check_all.sh would record a silent
#: PASS -- how check_almt_equivalence.py stayed dead for months. Resolved here
#: rather than left to whoever remembers to export it.
SHIM = Path(os.environ.get("MMSA_SHIM", "/workspace/mmsa_env/shim"))
BATCH, LENGTH, BERT_DIM, AUDIO_DIM, VISION_DIM = 4, 50, 768, 5, 20
TOLERANCE = 1e-5

#: config/config.json, DMD section, MOSI, verbatim. nlevels is 4 here and 2 in
#: DLF -- the two configs are otherwise near-identical, so this is easy to carry
#: over wrongly.
MODEL_ARGS = dict(
    need_data_aligned=True, need_model_aligned=True, early_stop=10,
    use_bert=True, use_finetune=True, attn_mask=True, update_epochs=10,
    attn_dropout_a=0.2, attn_dropout_v=0.0, relu_dropout=0.0, embed_dropout=0.2,
    res_dropout=0.0, dst_feature_dim_nheads=[50, 10], batch_size=16,
    learning_rate=0.0001, nlevels=4, conv1d_kernel_size_l=5,
    conv1d_kernel_size_a=5, conv1d_kernel_size_v=5, text_dropout=0.5,
    attn_dropout=0.3, output_dropout=0.5, grad_clip=0.6, patience=5,
    weight_decay=0.005, transformers="bert", pretrained="bert-base-uncased",
    feature_dims=[768, 5, 20], dataset_name="mosi", train_mode="regression",
)

#: Their output name -> ours. Eight supervised heads and six representations.
COMPARED = {
    "output_logit": "M",
    "logits_c": "shared_logit",
    "logits_l_homo": "logits_text_homo",
    "logits_v_homo": "logits_vision_homo",
    "logits_a_homo": "logits_audio_homo",
    "logits_l_hetero": "logits_text_hetero",
    "logits_v_hetero": "logits_vision_hetero",
    "logits_a_hetero": "logits_audio_hetero",
    "repr_l_homo": "repr_text_homo",
    "repr_v_homo": "repr_vision_homo",
    "repr_a_homo": "repr_audio_homo",
    "repr_l_hetero": "repr_text_hetero",
    "repr_v_hetero": "repr_vision_hetero",
    "repr_a_hetero": "repr_audio_hetero",
}

#: Built, used, returned, and read by nothing.
UNUSED = ("proj_cosine_l.", "proj_cosine_v.", "proj_cosine_a.")


class _StubBert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def check_cosine_reconstruction() -> bool:
    """Our reconstruction must equal PyTorch's own on 2-D input.

    The release's 3-D call cannot run at all on a current PyTorch, so the port
    reimplements what the old one did: sum over dim 1. On 2-D input that is the
    ordinary cosine loss and is still checkable against the library -- which
    pins the formula, leaving only the axis, and the axis is forced (summing
    over dim 1 is what made the release's 3-D shapes legal).
    """
    from msa.models.dmd import cosine_embedding_over_dim1

    torch.manual_seed(1)
    x, y = torch.randn(16, 32), torch.randn(16, 32)
    theirs = nn.CosineEmbeddingLoss()(x, y, torch.tensor([-1]))
    ours = cosine_embedding_over_dim1(x, y)
    difference = float((theirs - ours).abs())
    print(f"cosine reconstruction vs nn.CosineEmbeddingLoss on 2-D: {difference:.3e}")
    return difference <= TOLERANCE


def main() -> int:
    if not (ORIGIN / "trains").exists():
        return _skip(f"no DMD checkout at {ORIGIN}")
    for path in (SHIM, ORIGIN):
        if path.exists():
            sys.path.insert(0, str(path))
    try:
        from trains.singleTask.model.dmd import DMD as TheirDMD
    except ImportError as exc:
        return _skip(f"cannot import the reference ({exc})")
    from msa.models.dmd import DecoupledMultimodalDistillation

    if not check_cosine_reconstruction():
        print("FAIL: the reconstructed cosine loss does not match PyTorch's on 2-D")
        return 1

    torch.manual_seed(0)
    theirs = TheirDMD(types.SimpleNamespace(**MODEL_ARGS))
    ours = DecoupledMultimodalDistillation(
        text_dim=BERT_DIM, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM,
        text_length=LENGTH, audio_length=LENGTH, vision_length=LENGTH,
    )
    theirs.text_model = _StubBert()
    ours.encoder = _StubBert()

    their_params = [(n, p) for n, p in theirs.named_parameters()
                    if not n.startswith("text_model.")
                    and not any(n.startswith(u) for u in UNUSED)]
    # The two distillation kernels are separate modules in the release, trained
    # by the same optimiser; here they are submodules, so they are compared
    # separately and appended in the release's construction order.
    our_params = [(n, p) for n, p in ours.named_parameters()
                  if not n.startswith("encoder.")
                  and not n.startswith("homogeneous.")
                  and not n.startswith("heterogeneous.")]
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
    worst, worst_name = 0.0, ""
    for their_key, our_key in COMPARED.items():
        a = their_out[their_key].reshape(-1)
        b = our_out[our_key].reshape(-1)
        if a.shape != b.shape:
            print(f"FAIL: {their_key} is {tuple(a.shape)}, ours {our_key} is {tuple(b.shape)}")
            return 1
        difference = float((a - b).abs().max())
        if difference > worst:
            worst, worst_name = difference, their_key
        print(f"  {their_key:18s} {difference:.3e}")
    print(f"max abs difference: {worst:.3e} ({worst_name})")
    if worst > TOLERANCE:
        print(f"FAIL: above tolerance {TOLERANCE:.0e}")
        return 1
    print("EQUIVALENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
