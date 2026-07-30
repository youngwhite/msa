"""Self-MM — Self-Supervised Multi-Task Learning (Yu et al., AAAI 2021).

MOSI labels the utterance, not the modalities. So a model with a per-modality
branch has nothing to train those branches on, and they end up supervised only
through the fusion head. Self-MM manufactures the missing supervision: it keeps a
running pseudo-label per modality per sample, and adjusts it from how far that
modality's representation sits from the positive and negative class centres
relative to how far the fused representation sits.

The pseudo-labels move while training runs, which is why this model needs the
`on_train_batch_end` hook. It is the only model in the storyline that carries
state between batches; everything else in the shared training loop is untouched.

Ported from MMSA (`models/multiTask/SELF_MM.py` and its trainer, MIT, THUIAR),
and checked line by line against the authors' own release
(github.com/thuiar/Self-MM). Note the two differ in hyper-parameters for MOSI —
the authors use batch 32, audio lr 1e-3, video lr 1e-4, LSTM widths 32/64, while
MMSA uses batch 16, both 5e-3, widths 16/32. We follow MMSA's, since MMSA's table
is what the acceptance criterion compares against.
The label update follows the trainer's `update_labels`, including its details:
the momentum term `(n-1)/(n+1)` with `n` the epoch number, clamping to +/-H, and
updates starting only from the second epoch (the first supplies the centres).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec
from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder
from .pseudo_labels import MODES, PseudoLabelMixin
from .functional import last_valid_state



class _AudioVisualSubNet(nn.Module):
    """LSTM over a modality, read at its last real step, then projected."""

    def __init__(self, in_size: int, hidden: int, out_size: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, out_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.linear(self.dropout(last_valid_state(outputs, lengths)))


class _Branch(nn.Module):
    """dropout -> linear -> relu (the representation) -> linear -> relu -> linear.

    The intermediate after the first ReLU is what the pseudo-label machinery
    measures distances in — checked against the authors' release, where
    `Feature_t`/`Feature_a`/`Feature_v`/`Feature_f` are these post-layer states,
    not the raw encoder outputs. Using the raw outputs puts the class centres in
    a different (and much wider) space and changes the label dynamics.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_1 = nn.Linear(in_dim, hidden)
        self.layer_2 = nn.Linear(hidden, hidden)
        self.layer_3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = F.relu(self.layer_1(self.dropout(x)))
        prediction = self.layer_3(F.relu(self.layer_2(representation)))
        return prediction.view(-1), representation


@register_model("self_mm")
class SelfMM(PseudoLabelMixin, MSAModel):
    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        # Needs the training-set size up front: the pseudo-labels are per sample.
        kwargs.setdefault("train_size", spec.split_sizes["train"])
        return cls(
            text_dim=spec.text_dim, audio_dim=spec.audio_dim,
            vision_dim=spec.vision_dim, **kwargs,
        )

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        train_size: int,
        audio_hidden: int = 16,
        vision_hidden: int = 32,
        audio_out: int = 16,
        vision_out: int = 32,
        audio_dropout: float = 0.0,
        vision_dropout: float = 0.0,
        fusion_dim: int = 128,
        post_text_dim: int = 32,
        post_audio_dim: int = 16,
        post_vision_dim: int = 32,
        fusion_dropout: float = 0.0,
        post_text_dropout: float = 0.1,
        post_audio_dropout: float = 0.1,
        post_vision_dropout: float = 0.0,
        label_bound: float = 3.0,
        exclude_zero: bool = True,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.label_bound = label_bound
        self.exclude_zero = exclude_zero

        self.encoder = BertTextEncoder(pretrained, finetune)
        text_out = self.encoder.hidden_size
        self.audio_model = _AudioVisualSubNet(audio_dim, audio_hidden, audio_out, audio_dropout)
        self.vision_model = _AudioVisualSubNet(
            vision_dim, vision_hidden, vision_out, vision_dropout
        )

        self.fusion_branch = _Branch(
            text_out + audio_out + vision_out, fusion_dim, fusion_dropout
        )
        self.text_branch = _Branch(text_out, post_text_dim, post_text_dropout)
        self.audio_branch = _Branch(audio_out, post_audio_dim, post_audio_dropout)
        self.vision_branch = _Branch(vision_out, post_vision_dim, post_vision_dropout)

        # The centres live in the branches' representation spaces, not the
        # encoders' output spaces.
        widths = {
            "fusion": fusion_dim, "text": post_text_dim,
            "audio": post_audio_dim, "vision": post_vision_dim,
        }
        self._register_pseudo_label_buffers(train_size, widths)

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Four rates, as MMSA uses: BERT 5e-5, audio/vision 5e-3, the rest 1e-3.

        Expressed relative to `lr` (the "other" rate) so one CLI flag scales all
        of them together.
        """
        # The standard BERT recipe MMSA follows: biases and LayerNorm weights are
        # exempt from weight decay.
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        decayed, plain = [], []
        for name, param in self.encoder.named_parameters():
            (plain if any(k in name for k in no_decay) else decayed).append(param)
        encoder = list(self.encoder.parameters())
        av = list(self.audio_model.parameters()) + list(self.vision_model.parameters())
        seen = {id(p) for p in encoder + av}
        rest = [p for p in self.parameters() if id(p) not in seen]
        return [
            {"params": decayed, "lr": lr * 0.05, "weight_decay": weight_decay},
            {"params": plain, "lr": lr * 0.05, "weight_decay": 0.0},
            {"params": av, "lr": lr * 5.0, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.encoder(batch["text_bert"])[:, 0]        # [CLS]
        audio = self.audio_model(batch["audio"], batch["audio_length"])
        vision = self.vision_model(batch["vision"], batch["vision_length"])

        m, f_fusion = self.fusion_branch(torch.cat([text, audio, vision], dim=1))
        t, f_text = self.text_branch(text)
        a, f_audio = self.audio_branch(audio)
        v, f_vision = self.vision_branch(vision)
        return {
            "M": m, "T": t, "A": a, "V": v,
            "feature_fusion": f_fusion,
            "feature_text": f_text,
            "feature_audio": f_audio,
            "feature_vision": f_vision,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.pseudo_label_loss(outputs, batch)

    def on_train_batch_end(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        self.pseudo_label_step(outputs, batch, epoch)
