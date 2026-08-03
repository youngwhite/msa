"""DLF — Disentangled-Language-Focused MSA (Wang et al., AAAI 2025).

MulT's backbone with two additions. Each modality is split into a
**modality-specific** stream and a **modality-shared** one produced by a single
encoder all three pass through; the pair is required to reconstruct the original
signal, and pushed apart so the split means something. Then the language-focused
part: audio and vision never attend to each other, only into language
(`trans_l_with_a(s_l, s_a, s_a)`), so both are read on the text's clock.

Structure and protocol from the authors' release (`pwang322/DLF`, MIT);
hyper-parameters from its `config/config.json`. Its config is MMSA's -- same
feature path, dims, sample count and KeyEval -- so it is a fork of that
framework and reads the same pickles as every other aligned group here.

**Its transformer always adds sinusoidal positions.** MMSA turned that into a
switch defaulted off, which is how our MulT lost its positional encoding
(`docs/investigations.md#mult-position`); the DLF release does not have the
switch. So `position_embedding=True` is passed explicitly here -- reusing the
shared `TransformerEncoder` with its default would silently reintroduce the
exact bug that cost MulT.

Two structural details that are easy to read past:

* `encoder_c` is **one instance shared by all three modalities**, not one each.
  That sharing is what makes the "shared" stream shared.
* The temporal convolutions use kernel 5 with **no padding**, so 50 frames
  become 46, and every flattened linear downstream is sized from that. Changing
  the kernel changes eleven layer widths.

Selection is this repository's validation MAE, matching the release's
`KeyEval: Loss` for an L1 criterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder
from .transformers import TransformerEncoder


def _residual_head(width: int, hidden: int, dropout: float) -> nn.ModuleDict:
    """`proj1 -> relu -> dropout -> proj2`, added back to the input, then a head.

    The release writes this out nine times; it is one shape with three widths.
    """
    return nn.ModuleDict({
        "proj1": nn.Linear(width, hidden),
        "proj2": nn.Linear(hidden, width),
        "out": nn.Linear(width, 1),
    })


def _apply_residual_head(head: nn.ModuleDict, x: torch.Tensor,
                         dropout: float, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    projected = head["proj2"](F.dropout(F.relu(head["proj1"](x)), p=dropout, training=training))
    projected = projected + x
    return projected, head["out"](projected)


def hinge_similarity_loss(ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Triplet margin over the shared representations, keyed by label.

    Every sample contributes its three shared vectors under one id, so pairs
    with the same id should be more alike than pairs with different ones. The
    margin is not fixed: it is `0.15 * |label_i - label_j|`, so a pair of
    samples far apart in sentiment must be separated further than a near pair.
    Transcribed from the release's HingeLoss.
    """
    batch, width = feats.shape
    source = feats.repeat(1, batch).view(-1, width)
    target = feats.repeat(batch, 1)
    cosine = F.cosine_similarity(source, target, dim=1, eps=1e-8).view(batch, batch)

    source_ids = ids.view(batch, 1).repeat(1, batch)
    target_ids = ids.view(1, batch).repeat(batch, 1)
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=feats.device)
    source_ids = source_ids[off_diagonal].view(batch, batch - 1)
    target_ids = target_ids[off_diagonal].view(batch, batch - 1)
    cosine = cosine[off_diagonal].view(batch, batch - 1)

    same = source_ids == target_ids
    margin = 0.15 * (source_ids - target_ids).abs()

    total = feats.new_zeros(())
    counted = 0
    for row in range(batch):
        positives = int(same[row].sum())
        negatives = batch - 1 - positives
        if not positives or not negatives:
            continue
        positive_cos = cosine[row, same[row]].reshape(-1, 1).repeat(1, negatives)
        negative_cos = cosine[row, ~same[row]].reshape(-1, 1).repeat(1, positives).T
        row_margin = margin[row, ~same[row]].reshape(-1, 1).repeat(1, positives).T
        total = total + torch.clamp(row_margin - positive_cos + negative_cos, min=0).mean()
        counted += 1
    return total / max(counted, 1)


@register_model("dlf")
class DisentangledLanguageFocused(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_length: int = 50,
        audio_length: int = 50,
        vision_length: int = 50,
        width: int = 50,
        heads: int = 10,
        levels: int = 2,
        memory_levels: int = 3,
        kernel_size: int = 5,
        attn_dropout: float = 0.3,
        attn_dropout_a: float = 0.2,
        attn_dropout_v: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.2,
        output_dropout: float = 0.5,
        text_dropout: float = 0.5,
        attn_mask: bool = True,
        orthogonality_group: int = 50,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.text_dropout, self.output_dropout = text_dropout, output_dropout
        self.orthogonality_group = orthogonality_group
        self.encoder = BertTextEncoder(pretrained, finetune)

        # Kernel 5 with no padding: 50 frames become 46, and every flattened
        # width below is derived from that rather than written out.
        lengths = {"text": text_length, "audio": audio_length, "vision": vision_length}
        self.spans = {m: n - kernel_size + 1 for m, n in lengths.items()}
        dims = {"text": 768, "audio": audio_dim, "vision": vision_dim}
        drops = {"text": attn_dropout, "audio": attn_dropout_a, "vision": attn_dropout_v}

        def encoder_for(dropout: float, layers: int) -> TransformerEncoder:
            # position_embedding=True: the release's transformer always adds
            # sinusoidal positions. See the module docstring.
            return TransformerEncoder(
                width, heads, layers, attn_dropout=dropout, relu_dropout=relu_dropout,
                res_dropout=res_dropout, embed_dropout=embed_dropout,
                attn_mask=attn_mask, position_embedding=True)

        # Modules are declared in the reference's registration order -- proj_*,
        # encoder_s_*, encoder_c, decoder_*, align_c_*, ... -- so the weight-copy
        # equivalence test can walk both parameter lists positionally. Grouping
        # them by modality reads better and made the copy misalign.
        ORDER = ("text", "vision", "audio")
        self.project = nn.ModuleDict()
        for m in ("text", "audio", "vision"):        # proj_l, proj_a, proj_v
            self.project[m] = nn.Conv1d(dims[m], width, kernel_size, bias=False)
        self.specific = nn.ModuleDict()
        for m in ORDER:                              # encoder_s_l, _v, _a
            self.specific[m] = encoder_for(drops[m], levels)
        #: One instance for all three modalities -- that sharing is the point.
        self.shared = encoder_for(attn_dropout, levels)
        self.decode = nn.ModuleDict()
        for m in ("text", "vision", "audio"):        # decoder_l, _v, _a
            self.decode[m] = nn.Conv1d(width * 2, width, 1, bias=False)
        self.align = nn.ModuleDict()
        for m in ORDER:                              # align_c_l, _v, _a
            self.align[m] = nn.Linear(width * self.spans[m], width)
        self.shared_attention = nn.ModuleDict()
        for m in ORDER:                              # self_attentions_c_l, _v, _a
            self.shared_attention[m] = encoder_for(drops[m], levels)
        self.shared_head = _residual_head(width * 3, width * 3, output_dropout)

        # Language-focused cross attention: audio and vision into language only.
        self.into_language = nn.ModuleDict()
        self.into_language["audio"] = encoder_for(attn_dropout_a, levels)   # trans_l_with_a
        self.into_language["vision"] = encoder_for(attn_dropout_v, levels)  # trans_l_with_v
        self.memory = nn.ModuleDict()
        self.memory["text"] = encoder_for(attn_dropout, levels)             # trans_l_mem
        self.memory["audio"] = encoder_for(attn_dropout, memory_levels)     # trans_a_mem
        self.memory["vision"] = encoder_for(attn_dropout, memory_levels)    # trans_v_mem

        self.low_head = nn.ModuleDict()
        for m in ORDER:
            self.low_head[m] = _residual_head(width * self.spans[m], width, output_dropout)
        self.high_head = nn.ModuleDict()
        for m in ORDER:
            self.high_head[m] = _residual_head(width, width, output_dropout)

        self.gate = nn.ModuleDict()
        for m in ORDER:                              # projector_l, _v, _a
            self.gate[m] = nn.Linear(width, width)
        self.gate_shared = nn.Linear(width * 3, width * 3)
        combined = width * 3 + width * 3
        self.fusion_head = _residual_head(combined, combined, output_dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.encoder(batch["text_bert"])
        raw = {
            "text": F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training),
            "audio": batch["audio"].transpose(1, 2),
            "vision": batch["vision"].transpose(1, 2),
        }
        # (batch, dim, time) -> (time, batch, dim), which is what the shared
        # TransformerEncoder expects.
        projected = {m: self.project[m](x).permute(2, 0, 1) for m, x in raw.items()}

        specific = {m: self.specific[m](x) for m, x in projected.items()}
        shared = {m: self.shared(x) for m, x in projected.items()}

        batch_size = text.shape[0]
        aligned = {m: self.align[m](shared[m].permute(1, 2, 0).reshape(batch_size, -1))
                   for m in shared}

        # Reconstruct each modality from its two streams, then re-encode: the
        # specific stream should survive the round trip.
        reconstructed, respecified = {}, {}
        for m in projected:
            pair = torch.cat([specific[m].permute(1, 2, 0), shared[m].permute(1, 2, 0)], dim=1)
            recon = self.decode[m](pair).permute(2, 0, 1)
            reconstructed[m] = recon
            respecified[m] = self.specific[m](recon)

        low_logits = {}
        for m in shared:
            flat = shared[m].transpose(0, 1).reshape(batch_size, -1)
            _, low_logits[m] = _apply_residual_head(
                self.low_head[m], flat, self.output_dropout, self.training)

        attended = torch.cat(
            [self.shared_attention[m](shared[m])[-1] for m in ("text", "vision", "audio")], dim=1)
        _, shared_logit = _apply_residual_head(
            self.shared_head, attended, self.output_dropout, self.training)

        # Language-focused attention: each non-verbal stream queries language.
        last = {"text": self.memory["text"](specific["text"])[-1]}
        for m in ("audio", "vision"):
            crossed = self.into_language[m](specific["text"], specific[m], specific[m])
            last[m] = self.memory[m](crossed)[-1]

        high, high_logits = {}, {}
        for m in ("text", "vision", "audio"):
            high[m], high_logits[m] = _apply_residual_head(
                self.high_head[m], last[m], self.output_dropout, self.training)

        gated = [torch.sigmoid(self.gate[m](high[m])) for m in ("text", "vision", "audio")]
        gated.append(torch.sigmoid(self.gate_shared(attended)))
        _, prediction = _apply_residual_head(
            self.fusion_head, torch.cat(gated, dim=1), self.output_dropout, self.training)

        out: dict[str, torch.Tensor] = {"M": prediction.squeeze(-1)}
        for m in projected:
            out[f"origin_{m}"] = projected[m]
            out[f"specific_{m}"] = specific[m]
            out[f"shared_{m}"] = shared[m]
            out[f"recon_{m}"] = reconstructed[m]
            out[f"respecified_{m}"] = respecified[m]
            out[f"aligned_{m}"] = aligned[m]
            out[f"low_{m}"] = low_logits[m].squeeze(-1)
            out[f"high_{m}"] = high_logits[m].squeeze(-1)
        out["shared_logit"] = shared_logit.squeeze(-1)
        return out

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        label = batch["label"]
        modalities = ("text", "audio", "vision")

        # Five supervised heads, not one. The fused prediction is scored, and so
        # are the shared branch and each of the three language-focused branches
        # -- the last of which carries weight 3 while everything else carries 1.
        # That asymmetry is the second half of "language-focused": the first is
        # architectural (only language acts as a query), this one is in the
        # objective, and the paper states neither. From the release's
        # trains/singleTask/DLF.py:89.
        #
        # This port originally scored the fused prediction alone. The weight-copy
        # test passed anyway -- it compared the prediction, and three of the four
        # missing heads never reach it -- and ten seeds came out 2-4 SE behind
        # the authors' own code on every metric at once. Uniform, same-signed
        # deficits are a protocol difference, not noise, which is what sent us
        # back to the trainer. See docs/investigations.md#dlf-task-heads.
        task = (
            F.l1_loss(outputs["M"], label)
            + F.l1_loss(outputs["shared_logit"], label)
            + 3 * F.l1_loss(outputs["high_text"], label)
            + F.l1_loss(outputs["high_vision"], label)
            + F.l1_loss(outputs["high_audio"], label)
        )

        reconstruction = sum(
            F.mse_loss(outputs[f"recon_{m}"], outputs[f"origin_{m}"]) for m in modalities)
        round_trip = sum(
            F.mse_loss(outputs[f"specific_{m}"].permute(1, 2, 0),
                       outputs[f"respecified_{m}"].permute(1, 2, 0)) for m in modalities)

        # Orthogonality: specific and shared should point apart. The release
        # reshapes both to (-1, 50) before taking cosines, so the similarity is
        # computed over groups of 50 elements rather than over whole vectors.
        # Reproduced as written -- the grouping decides what the loss measures.
        group = self.orthogonality_group
        target = label.new_full((1,), -1.0)
        orthogonality = sum(
            F.cosine_embedding_loss(outputs[f"specific_{m}"].reshape(-1, group),
                                    outputs[f"shared_{m}"].reshape(-1, group), target)
            for m in modalities)

        # Every sample's three shared vectors carry that sample's label as their
        # id, so the triplet objective pulls a sample's modalities together.
        feats = torch.cat([outputs[f"aligned_{m}"] for m in ("text", "vision", "audio")], dim=0)
        ids = label.repeat(3)
        similarity = hinge_similarity_loss(ids, feats)

        return task + 0.1 * (round_trip + reconstruction + 0.1 * (similarity + orthogonality))
