"""BERT-MAG — Multimodal Adaptation Gate (Rahman et al., ACL 2020).

The first model in this storyline to fuse *inside* the language model rather
than after it. MAG turns audio and vision into a displacement of BERT's token
embeddings, applied once between the embedding layer and the encoder stack, and
then lets BERT run untouched. CENET (2022) later moves the same idea between two
encoder layers; this is where it starts.

The gate is the interesting part. A raw multimodal displacement can be large
enough to push the embedding somewhere BERT has never seen, so its norm is
capped relative to the embedding it is modifying:

    alpha = min( ||text|| / (||h_m|| + eps) * beta_shift ,  1 )

With `beta_shift = 1` the shift can never exceed the magnitude of what it
shifts. That single line is what makes injecting into a pretrained encoder
survivable.

Written against `transformers` directly rather than transcribed from MMSA's
copy, which subclasses a `BertPreTrainedModel` contract that no longer exists —
its version cannot be imported at all under transformers 5. The computation
below follows MMSA's `BERT_MAG.py` line for line (readable even where it is not
runnable); only the plumbing is modern. Validated against MMSA's own run under
transformers 4.44, see docs/investigations.md#bert-mag.

Uses **aligned** data: the displacement is per token, so audio and vision must
already share the text's clock.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder


class _MultimodalAdaptationGate(nn.Module):
    """One displacement of the token embeddings, norm-capped against them."""

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int,
                 beta_shift: float, dropout: float) -> None:
        super().__init__()
        # Each modality proposes a direction, weighted by how much it agrees
        # with the text at that position.
        self.gate_v = nn.Linear(vision_dim + text_dim, text_dim)
        self.gate_a = nn.Linear(audio_dim + text_dim, text_dim)
        self.project_v = nn.Linear(vision_dim, text_dim)
        self.project_a = nn.Linear(audio_dim, text_dim)
        self.beta_shift = beta_shift
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor) -> torch.Tensor:
        weight_v = F.relu(self.gate_v(torch.cat([vision, text], dim=-1)))
        weight_a = F.relu(self.gate_a(torch.cat([audio, text], dim=-1)))
        shift = weight_v * self.project_v(vision) + weight_a * self.project_a(audio)

        text_norm = text.norm(2, dim=-1)
        shift_norm = shift.norm(2, dim=-1)
        # A position with no multimodal signal at all would divide by zero; the
        # reference substitutes 1, which makes alpha fall out at beta_shift.
        shift_norm = torch.where(shift_norm == 0, torch.ones_like(shift_norm), shift_norm)

        scale = (text_norm / (shift_norm + 1e-6)) * self.beta_shift
        alpha = torch.min(scale, torch.ones_like(scale)).unsqueeze(-1)
        return self.dropout(self.norm(alpha * shift + text))


@register_model("bert_mag")
class BertMAG(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
        beta_shift: float = 1.0,
        gate_dropout: float = 0.3,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(pretrained, finetune)
        hidden = self.encoder.hidden_size
        self.gate = _MultimodalAdaptationGate(
            hidden, audio_dim, vision_dim, beta_shift, gate_dropout
        )
        # BertForSequenceClassification's head: pooler output -> dropout -> linear.
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_bert = batch["text_bert"]
        input_ids = text_bert[:, 0].long()
        attention_mask = text_bert[:, 1].long()
        token_type_ids = text_bert[:, 2].long()

        bert = self.encoder.bert
        embedded = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        # The whole model, in one line: shift the embeddings, then run BERT.
        shifted = self.gate(embedded, batch["audio"], batch["vision"])

        mask = bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        hidden = shifted
        for layer in bert.encoder.layer:
            out = layer(hidden, attention_mask=mask)
            hidden = out if torch.is_tensor(out) else out[0]
        pooled = bert.pooler(hidden) if bert.pooler is not None else hidden[:, 0]
        return {"M": self.head(pooled).view(-1)}
