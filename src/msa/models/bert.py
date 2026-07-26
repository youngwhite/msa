"""BERT text encoder, and the text-only control that gives the storyline meaning.

From MISA onward the reference models stop using frozen BERT features and
fine-tune BERT itself. That is a change of encoder, not of fusion architecture,
and it lands in the middle of the storyline — MulT (frozen features) scores 0.810
Acc-2 in MMSA's table, MISA (fine-tuned) 0.835, Self-MM 0.855.

`TextOnlyBert` exists to tell those two explanations apart. It is the same
fine-tuned encoder with no audio and no vision at all. Whatever it scores is the
part of the "progress" that owes nothing to multimodal fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_model
from .base import MSAModel


class BertTextEncoder(nn.Module):
    """Wraps HuggingFace BERT over the tokenised text the dataset already carries.

    `batch["text_bert"]` is (batch, 3, seq): input_ids, attention_mask,
    token_type_ids — the same tensor MMSA feeds its BERT-based models.
    """

    def __init__(self, pretrained: str = "bert-base-uncased", finetune: bool = True) -> None:
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(pretrained)
        self.finetune = finetune
        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, text_bert: torch.Tensor) -> torch.Tensor:
        input_ids = text_bert[:, 0].long()
        attention_mask = text_bert[:, 1].long()
        token_type_ids = text_bert[:, 2].long()
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return output.last_hidden_state          # (batch, seq, hidden)


@register_model("text_bert")
class TextOnlyBert(MSAModel):
    """Fine-tuned BERT on the transcript alone — the storyline's control group."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
        dropout: float = 0.1,
        head_dim: int = 128,
        head_lr_multiplier: float = 10.0,
    ) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(pretrained, finetune)
        self.head_lr_multiplier = head_lr_multiplier
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """The pretrained encoder wants a small step; the fresh head does not.

        One learning rate for both either wrecks the pretrained weights or leaves
        the head barely trained.
        """
        return [
            {"params": list(self.encoder.parameters()), "lr": lr,
             "weight_decay": weight_decay},
            {"params": list(self.head.parameters()),
             "lr": lr * self.head_lr_multiplier, "weight_decay": weight_decay},
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encoder(batch["text_bert"])
        return {"M": self.head(hidden[:, 0]).view(-1)}   # [CLS]
