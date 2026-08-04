"""DMD — Decoupled Multimodal Distillation (Li et al., CVPR 2023 highlight).

Each modality is split into a **modality-specific** stream and a
**modality-shared** one from a single encoder all three pass through, then the
two halves are distilled against each other over a learned graph: a
*homogeneous* graph over the three shared streams, and a *heterogeneous* one
over the three cross-attended streams. The edge weights are themselves
parameters, so the model decides which modality teaches which.

Structure and protocol from the authors' release (`mdswyz/DMD`, MIT);
hyper-parameters from its `config/config.json`, which is MMSA's -- same feature
path, dims, sample count and KeyEval -- so it reads the same pickles as every
other aligned group here. Full reading in `docs/spec_dmd.md`.

**DLF is this model with four attention paths unwired.** DMD calls all six
cross-modal transformers; DLF builds the same six and wires only the two where
language is the query, and weights the language head 3 against everyone else's
1. Neither paper says so. That is why `dlf.py` is not imported here and nothing
is shared with it beyond the generic encoder: DLF is verified and committed, and
a shared edit would put its numbers at risk. Duplication is the cheaper mistake.

**Its transformer always adds sinusoidal positions**, like DLF's and unlike
MMSA's, whose defaulted-off switch is how our MulT lost positional encoding
(`docs/investigations.md#mult-position`). `position_embedding=True` is therefore
explicit below.

Two things that read like ordinary code and are not:

* The homogeneous representation handed to distillation is the *pre-residual*
  projection, and the heterogeneous one is *post-residual*. The release gets
  this from `F.relu(..., inplace=True)` mutating the tensor it saved, so the
  saved representation is the rectified one. Written out here rather than
  reproduced by side effect.
* `proj_cosine_l/v/a` are built, called, and their outputs returned and never
  read by anything. Not implemented; excluded from the equivalence test with
  everything else the release leaves dangling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder
from .transformers import TransformerEncoder

#: Modalities in the release's own registration order. Two orders appear in that
#: file -- `l, a, v` for the input convolutions and `l, v, a` everywhere after --
#: and a positional weight copy only lines up if both are kept.
PROJECTION_ORDER = ("text", "audio", "vision")
ORDER = ("text", "vision", "audio")


#: Old ATen's cosine_embedding_loss constant, added to each squared magnitude.
EPS = 1e-12


def _softmax_prior(values: list[float], temperature: float) -> torch.Tensor:
    """The release's `softmax(x, T)` helper: a temperature-scaled softmax."""
    tensor = torch.tensor(values, dtype=torch.float32) / temperature
    return torch.softmax(tensor, dim=0)


def cosine_embedding_over_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`CosineEmbeddingLoss(target=-1)` as PyTorch used to compute it.

    The release passes 3-D tensors, shaped (time, batch, channel), straight into
    `nn.CosineEmbeddingLoss`. **Current PyTorch refuses that outright** -- "1D
    target tensor expects 2D input tensors" -- so the release cannot run as
    written on any recent version, and the loss has to be reconstructed to run
    it at all.

    Old ATen summed over dim 1 unconditionally instead of requiring 2-D, which
    is what made 3-D input work. For a 2-D input dim 1 is the feature axis and
    the result is the ordinary cosine loss; **for the release's 3-D input dim 1
    is the batch axis**, so the vectors being compared are batch-length and the
    value depends on which samples happen to share a batch. That is the same
    defect class as `investigations.md#mctn-batch-axis`, and it is why the same
    authors' later DLF reshapes to 2-D at exactly this line.

    Reconstructed rather than assumed: `check_dmd_equivalence.py` asserts this
    agrees with the current `nn.CosineEmbeddingLoss` on 2-D input, which pins
    the formula. The axis is not a choice -- summing over dim 1 is what made the
    release's shapes legal in the first place.
    """
    dot = (x * y).sum(1)
    # EPS goes INSIDE the square root, as old ATen had it, and this is not
    # cosmetic: MOSI's padded audio and vision frames make ~750 of these slices
    # exactly zero, and sqrt has an infinite derivative at zero. Clamping the
    # norm afterwards instead gives a finite loss with NaN gradients, which
    # takes the whole model out on the first optimiser step.
    denominator = ((x.pow(2).sum(1) + EPS) * (y.pow(2).sum(1) + EPS)).sqrt()
    cosine = dot / denominator
    # target = -1, margin = 0: loss is max(0, cos), averaged over every element.
    return F.relu(cosine).mean()


def _residual_head(width: int, hidden: int) -> nn.ModuleDict:
    """`proj1 -> relu -> dropout -> proj2`, added back, then a linear head."""
    return nn.ModuleDict({
        "proj1": nn.Linear(width, hidden),
        "proj2": nn.Linear(hidden, width),
        "out": nn.Linear(width, 1),
    })


def _apply_residual_head(head: nn.ModuleDict, x: torch.Tensor,
                         dropout: float, training: bool) -> torch.Tensor:
    projected = head["proj2"](
        F.dropout(F.relu(head["proj1"](x)), p=dropout, training=training))
    return head["out"](projected + x)


def hinge_similarity_loss(ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """The release's `HingeLoss`: pull same-label pairs together, push others apart.

    Byte for byte the same objective DLF uses -- DLF inherited the file. The
    margin scales with how far apart two labels are, so disagreeing a little is
    cheaper than disagreeing a lot.
    """
    batch = feats.size(0)
    left = feats.repeat(1, batch).view(-1, feats.size(1))
    right = feats.repeat(batch, 1)
    # The release spells the cosine out rather than calling F.cosine_similarity,
    # with its own floor of 1e-8 on each magnitude. Kept literal: the floors sit
    # in different places in the two formulations, and zero-magnitude rows do
    # occur here.
    left_norm = (left.pow(2).sum(1) + 1e-8).sqrt().clamp_min(1e-8)
    right_norm = (right.pow(2).sum(1) + 1e-8).sqrt().clamp_min(1e-8)
    cosine = ((left * right).sum(1) / (left_norm * right_norm)).view(batch, batch)

    left_ids = ids.view(batch, 1).repeat(1, batch)
    right_ids = ids.view(1, batch).repeat(batch, 1)
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=feats.device)
    left_ids = left_ids[off_diagonal].view(batch, batch - 1)
    right_ids = right_ids[off_diagonal].view(batch, batch - 1)
    cosine = cosine[off_diagonal].view(batch, batch - 1)

    same = left_ids == right_ids
    margin = 0.15 * (left_ids - right_ids).abs()

    total, counted = feats.new_zeros(()), 0
    for i in range(batch):
        positives = int(same[i].sum())
        negatives = batch - 1 - positives
        if not positives or not negatives:
            continue
        positive = cosine[i, same[i]].reshape(-1, 1).repeat(1, negatives)
        negative = cosine[i, ~same[i]].reshape(-1, 1).repeat(1, positives).t()
        pair_margin = margin[i, ~same[i]].reshape(-1, 1).repeat(1, positives).t()
        total = total + torch.clamp(pair_margin - positive + negative, min=0).mean()
        counted += 1
    return total / max(counted, 1)


class DistillationKernel(nn.Module):
    """Learned edge weights e_{j->k}, plus the losses they weight.

    Both graphs use this. The release ships two near-identical copies whose only
    substantive difference is which distance the representation loss uses, and
    the homogeneous copy's representation loss is discarded by the trainer -- so
    the difference lands entirely on the one term that never counts. One class
    with a flag is the honest rendering of that.
    """

    def __init__(self, n_classes: int, hidden: int, gd_size: int,
                 prior: torch.Tensor, gd_reg: float, w_losses: tuple[float, float],
                 alpha: float = 1 / 8, *, min_cosine_repr: bool) -> None:
        super().__init__()
        self.logit = nn.Linear(n_classes, gd_size)
        self.repr = nn.Linear(hidden, gd_size)
        self.edge = nn.Linear(gd_size * 4, 1)
        self.gd_size = gd_size
        self.alpha = alpha
        self.gd_reg = gd_reg
        self.w_losses = w_losses
        self.min_cosine_repr = min_cosine_repr
        self.register_buffer("prior", prior)

    #: Ordered (to, from) pairs over three modalities, skipping self-pairs. The
    #: release derives these from `to_idx`/`from_idx`, both [0, 1, 2]. Its two
    #: loops skip on different quantities -- one on the modality index, one on
    #: the enumeration position -- which coincide only because those lists are
    #: [0, 1, 2]. Fixed here so the two can never disagree; see docs/spec_dmd.md.
    PAIRS = tuple((j, i) for j in range(3) for i in range(3) if i != j)

    def forward(self, logits: torch.Tensor, reprs: torch.Tensor) -> torch.Tensor:
        modalities, batch = logits.shape[:2]
        z_logits = self.logit(logits.reshape(modalities * batch, -1))
        z_reprs = self.repr(reprs.reshape(modalities * batch, -1))
        z = torch.cat((z_logits, z_reprs), dim=1).view(modalities, batch, self.gd_size * 2)
        edges = torch.cat(
            [self.edge(torch.cat((z[j], z[i]), dim=1)) for j, i in self.PAIRS], dim=1)
        return F.softmax(edges * self.alpha, dim=1).t()

    def distillation_loss(self, logits: torch.Tensor, reprs: torch.Tensor,
                          edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regularisation = (edges.mean(1) - self.prior).pow(2).sum() * self.gd_reg
        logit_loss = reprs.new_zeros(())
        repr_loss = reprs.new_zeros(())
        for x, (j, i) in enumerate(self.PAIRS):
            weight = edges[x] + self.prior[x]
            logit_loss = logit_loss + self.w_losses[0] * _l1_distance(
                logits[j], logits[i], weight)
            if self.min_cosine_repr:
                repr_loss = repr_loss + self.w_losses[1] * _min_cosine(
                    reprs[j], reprs[i], weight)
            else:
                repr_loss = repr_loss + self.w_losses[1] * _l1_distance(
                    reprs[j], reprs[i], weight)
        return regularisation, logit_loss, repr_loss


def _l1_distance(student: torch.Tensor, teacher: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """`distance_metric(..., 'l1', w)`: the teacher is detached -- it teaches."""
    distances = (student - teacher.detach()).abs().sum(1)
    return (distances * weights).mean()


def _min_cosine(student: torch.Tensor, teacher: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    """`min_cosine(..., w)`, which is not the per-sample distance it looks like.

    The release calls `nn.CosineEmbeddingLoss()` -- already reduced to a scalar
    over the batch -- and only then multiplies by the per-sample weights and
    takes a mean. A scalar times a vector, averaged, is the scalar times the
    weights' mean, so **the learned edge weights only scale this term as a
    whole**; they cannot weight one pair of samples above another, which is what
    the surrounding code is built to do. Reproduced as written.

    For target -1 and margin 0 that loss is `max(0, cos)`, not `1 - cos`. The
    sibling `distance_metric(..., 'cosine')` is the `1 - cos` one; they are
    different functions and only this one is used here.
    """
    cosine = F.cosine_similarity(student, teacher.detach(), dim=1)
    return (F.relu(cosine).mean() * weights).mean()


@register_model("dmd")
class DecoupledMultimodalDistillation(MSAModel):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 5,
        vision_dim: int = 20,
        text_length: int = 50,
        audio_length: int = 50,
        vision_length: int = 50,
        width: int = 50,
        heads: int = 10,
        levels: int = 4,
        kernel_size: int = 5,
        attn_dropout: float = 0.3,
        attn_dropout_audio: float = 0.2,
        attn_dropout_vision: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.2,
        output_dropout: float = 0.5,
        text_dropout: float = 0.5,
        attn_mask: bool = True,
        pretrained: str = "bert-base-uncased",
        finetune_bert: bool = True,
    ) -> None:
        super().__init__()
        self.output_dropout = output_dropout
        self.text_dropout = text_dropout
        self.encoder = BertTextEncoder(pretrained, finetune_bert)

        dims = {"text": text_dim, "audio": audio_dim, "vision": vision_dim}
        lengths = {"text": text_length, "audio": audio_length, "vision": vision_length}
        drops = {"text": attn_dropout, "audio": attn_dropout_audio,
                 "vision": attn_dropout_vision}
        #: Kernel 5 with no padding: 50 frames become 46, and every flattened
        #: linear below is sized from that.
        self.frames = {m: lengths[m] - kernel_size + 1 for m in lengths}

        def encoder_for(dropout: float, layers: int, dim: int = width) -> TransformerEncoder:
            return TransformerEncoder(
                embed_dim=dim, num_heads=heads, layers=layers, attn_dropout=dropout,
                relu_dropout=relu_dropout, res_dropout=res_dropout,
                embed_dropout=embed_dropout, attn_mask=attn_mask,
                position_embedding=True)

        # Declared in the release's registration order throughout: a positional
        # weight copy is what the equivalence test does, and grouping these by
        # modality instead would misalign it.
        self.project = nn.ModuleDict()
        for m in PROJECTION_ORDER:
            self.project[m] = nn.Conv1d(dims[m], width, kernel_size, bias=False)
        self.specific = nn.ModuleDict()
        for m in ORDER:
            self.specific[m] = nn.Conv1d(width, width, 1, bias=False)
        #: One instance for all three -- that sharing is what makes it shared.
        self.shared = nn.Conv1d(width, width, 1, bias=False)
        self.decode = nn.ModuleDict()
        for m in ORDER:
            self.decode[m] = nn.Conv1d(width * 2, width, 1, bias=False)
        # proj_cosine_l/v/a would be declared here. They are computed by the
        # release, returned, and read by nothing; not implemented.
        self.align = nn.ModuleDict()
        for m in ORDER:
            self.align[m] = nn.Linear(width * self.frames[m], width)
        self.shared_attention = nn.ModuleDict()
        for m in ORDER:
            self.shared_attention[m] = encoder_for(drops[m], levels)
        self.shared_head = _residual_head(width * 3, width * 3)

        # Six cross-modal transformers, all of them wired -- the difference from
        # DLF. Names read (query, key/value).
        self.cross = nn.ModuleDict()
        for query, other in (("text", "audio"), ("text", "vision"),
                             ("audio", "text"), ("audio", "vision"),
                             ("vision", "text"), ("vision", "audio")):
            self.cross[f"{query}_{other}"] = encoder_for(drops[query], levels)
        self.memory = nn.ModuleDict()
        for m in ("text", "audio", "vision"):
            self.memory[m] = encoder_for(attn_dropout, max(levels, 3), dim=width * 2)

        self.low_head = nn.ModuleDict()
        for m in ORDER:
            self.low_head[m] = _residual_head(width * self.frames[m], width)
        self.high_head = nn.ModuleDict()
        for m in ORDER:
            self.high_head[m] = _residual_head(width * 2, width * 2)

        self.gate = nn.ModuleDict()
        for m in ORDER:
            self.gate[m] = nn.Linear(width * 2, width * 2)
        self.gate_shared = nn.Linear(width * 3, width * 3)
        fused = 2 * width * 3 + width * 3
        self.fusion_head = _residual_head(fused, fused)

        # Both graphs. gd_size, priors and the representation distance differ;
        # everything else is shared. Values from run.py:123-157.
        self.homogeneous = DistillationKernel(
            n_classes=1, hidden=width, gd_size=64,
            prior=_softmax_prior([0, 0, 1, 0, 1, 0], 0.25), gd_reg=10.0,
            w_losses=(1.0, 10.0), min_cosine_repr=False)
        self.heterogeneous = DistillationKernel(
            n_classes=1, hidden=width * 2, gd_size=32,
            prior=_softmax_prior([0, 0, 1, 0, 1, 1], 0.25), gd_reg=10.0,
            w_losses=(1.0, 10.0), min_cosine_repr=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.encoder(batch["text_bert"])
        raw = {
            "text": F.dropout(text.transpose(1, 2), p=self.text_dropout,
                              training=self.training),
            "audio": batch["audio"].transpose(1, 2),
            "vision": batch["vision"].transpose(1, 2),
        }
        batch_size = raw["text"].size(0)

        projected = {m: self.project[m](raw[m]) for m in PROJECTION_ORDER}
        specific = {m: self.specific[m](projected[m]) for m in ORDER}
        shared = {m: self.shared(projected[m]) for m in ORDER}

        aligned = {m: self.align[m](shared[m].contiguous().view(batch_size, -1))
                   for m in ORDER}
        reconstructed = {m: self.decode[m](torch.cat([specific[m], shared[m]], dim=1))
                         for m in ORDER}
        respecified = {m: self.specific[m](reconstructed[m]) for m in ORDER}

        # (channel, time) -> (time, batch, channel) for the transformers.
        specific = {m: specific[m].permute(2, 0, 1) for m in ORDER}
        shared = {m: shared[m].permute(2, 0, 1) for m in ORDER}

        # Homogeneous branch: the shared streams, flattened. Its representation
        # is the PRE-residual projection, rectified.
        low_repr, low_logits = {}, {}
        for m in ORDER:
            flat = shared[m].transpose(0, 1).contiguous().view(batch_size, -1)
            head = self.low_head[m]
            hidden = F.relu(head["proj1"](flat))
            projected_back = head["proj2"](
                F.dropout(hidden, p=self.output_dropout, training=self.training))
            low_repr[m] = hidden
            low_logits[m] = head["out"](projected_back + flat)

        attended = torch.cat(
            [self.shared_attention[m](shared[m])[-1] for m in ORDER], dim=1)
        shared_logit = _apply_residual_head(
            self.shared_head, attended, self.output_dropout, self.training)

        # Heterogeneous branch: every modality queries the other two.
        last = {}
        for query, others in (("text", ("audio", "vision")),
                              ("audio", ("text", "vision")),
                              ("vision", ("text", "audio"))):
            crossed = torch.cat(
                [self.cross[f"{query}_{other}"](specific[query], specific[other],
                                                specific[other])
                 for other in others], dim=2)
            last[query] = self.memory[query](crossed)[-1]

        # Its representation is the POST-residual projection -- the asymmetry
        # with the homogeneous branch above is the release's, not ours.
        high_repr, high_logits = {}, {}
        for m in ORDER:
            head = self.high_head[m]
            projected_back = head["proj2"](F.dropout(
                F.relu(head["proj1"](last[m])), p=self.output_dropout,
                training=self.training))
            high_repr[m] = projected_back + last[m]
            high_logits[m] = head["out"](high_repr[m])

        # Language, vision, audio, then the shared stream -- the release's order.
        gated = [torch.sigmoid(self.gate[m](last[m])) for m in ORDER]
        gated.append(torch.sigmoid(self.gate_shared(attended)))
        prediction = _apply_residual_head(
            self.fusion_head, torch.cat(gated, dim=1),
            self.output_dropout, self.training)

        out: dict[str, torch.Tensor] = {"M": prediction.squeeze(-1)}
        for m in ORDER:
            out[f"origin_{m}"] = projected[m]
            out[f"specific_{m}"] = specific[m]
            out[f"shared_{m}"] = shared[m]
            out[f"recon_{m}"] = reconstructed[m]
            out[f"respecified_{m}"] = respecified[m]
            out[f"aligned_{m}"] = aligned[m]
            out[f"logits_{m}_homo"] = low_logits[m].squeeze(-1)
            out[f"repr_{m}_homo"] = low_repr[m]
            out[f"logits_{m}_hetero"] = high_logits[m].squeeze(-1)
            out[f"repr_{m}_hetero"] = high_repr[m]
        out["shared_logit"] = shared_logit.squeeze(-1)
        return out

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        label = batch["label"]

        # Eight supervised heads, all at weight 1. Six of them never reach the
        # prediction, so a forward-only comparison cannot see them missing --
        # which is exactly how DLF's five-head loss shipped as one head
        # (docs/investigations.md#dlf-task-heads).
        task = F.l1_loss(outputs["M"], label) + F.l1_loss(outputs["shared_logit"], label)
        for m in ORDER:
            task = task + F.l1_loss(outputs[f"logits_{m}_homo"], label)
            task = task + F.l1_loss(outputs[f"logits_{m}_hetero"], label)

        reconstruction = sum(
            F.mse_loss(outputs[f"recon_{m}"], outputs[f"origin_{m}"]) for m in ORDER)
        round_trip = sum(
            F.mse_loss(outputs[f"specific_{m}"].permute(1, 2, 0),
                       outputs[f"respecified_{m}"]) for m in ORDER)
        orthogonality = sum(
            cosine_embedding_over_dim1(outputs[f"specific_{m}"], outputs[f"shared_{m}"])
            for m in ORDER)

        feats = torch.cat([outputs[f"aligned_{m}"] for m in ORDER], dim=0)
        similarity = hinge_similarity_loss(label.repeat(3), feats)

        distillation = label.new_zeros(())
        for kernel, suffix in ((self.homogeneous, "homo"), (self.heterogeneous, "hetero")):
            logits = torch.stack([outputs[f"logits_{m}_{suffix}"].unsqueeze(-1)
                                  for m in ORDER])
            reprs = torch.stack([outputs[f"repr_{m}_{suffix}"] for m in ORDER])
            edges = kernel(logits, reprs)
            regularisation, logit_loss, repr_loss = kernel.distillation_loss(
                logits, reprs, edges)
            # The homogeneous branch's representation loss is computed by the
            # release and then dropped by its trainer; only the heterogeneous
            # one counts it. Undocumented, and reproduced deliberately.
            terms = logit_loss + regularisation
            if suffix == "hetero":
                terms = terms + repr_loss
            distillation = distillation + 0.05 * terms

        return (task + distillation
                + 0.1 * (round_trip + reconstruction + 0.1 * (similarity + orthogonality)))
