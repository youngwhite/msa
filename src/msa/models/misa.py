"""MISA — Modality-Invariant and -Specific Representations (Hazarika et al., MM 2020).

Fusing raw modality representations mixes two different things: what the
modalities agree on, and what only one of them carries. MISA splits each
modality's utterance vector into a *shared* code and a *private* code, and adds
three auxiliary objectives to keep the split honest:

* similarity — the three shared codes should follow the same distribution, so
  their central moments are matched (CMD, up to the 5th moment)
* difference — shared and private codes of a modality, and the private codes of
  different modalities, should be orthogonal
* reconstruction — shared + private must still rebuild the original vector, so
  the split cannot discard information

This is the first model here whose loss is not just L1 on the prediction, which
makes it the first real test of the `MSAModel.compute_loss` contract: the trainer
needs no changes.

Ported from MMSA (`models/singleTask/MISA.py`, MIT, THUIAR). Two notes:

* MISA fine-tunes BERT (`use_bert`, `use_finetune` both true in MMSA's config)
  and takes the attention-masked mean of the last hidden states as the utterance
  vector — not [CLS].
* MMSA's regression criterion here is **MSE**, not the L1 every other model uses.
  Kept, since it is part of what produced the reference number.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder


def central_moment_discrepancy(x: torch.Tensor, y: torch.Tensor, moments: int = 5) -> torch.Tensor:
    """Distance between two distributions by their first `moments` central moments.

    Cheaper and more stable than an adversarial discriminator, which is why MISA
    prefers it (`use_cmd_sim=True` in MMSA's config).
    """
    def match(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((a - b) ** 2) + 1e-12)

    mean_x, mean_y = x.mean(0), y.mean(0)
    centred_x, centred_y = x - mean_x, y - mean_y
    total = match(mean_x, mean_y)
    for k in range(2, moments + 1):
        total = total + match((centred_x ** k).mean(0), (centred_y ** k).mean(0))
    return total


def orthogonality(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared cosine similarity after centring — zero when the codes are orthogonal."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / (a.norm(p=2, dim=1, keepdim=True).detach() + 1e-6)
    b = b / (b.norm(p=2, dim=1, keepdim=True).detach() + 1e-6)
    return (a.t() @ b).pow(2).mean()


def _projector(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim))


@register_model("misa")
class MISA(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
        diff_weight: float = 0.1,
        sim_weight: float = 0.3,
        recon_weight: float = 1.0,
        cmd_moments: int = 5,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
        bert_lr_divisor: float = 1.0,
    ) -> None:
        super().__init__()
        self.diff_weight = diff_weight
        self.sim_weight = sim_weight
        self.recon_weight = recon_weight
        self.cmd_moments = cmd_moments
        self.bert_lr_divisor = bert_lr_divisor

        self.encoder = BertTextEncoder(pretrained, finetune)

        # Audio and vision go through two stacked bidirectional LSTMs; the two
        # final states are concatenated, hence 4x the hidden width.
        self.arnn1 = nn.LSTM(audio_dim, audio_dim, bidirectional=True, batch_first=True)
        self.arnn2 = nn.LSTM(2 * audio_dim, audio_dim, bidirectional=True, batch_first=True)
        self.vrnn1 = nn.LSTM(vision_dim, vision_dim, bidirectional=True, batch_first=True)
        self.vrnn2 = nn.LSTM(2 * vision_dim, vision_dim, bidirectional=True, batch_first=True)
        self.anorm = nn.LayerNorm(2 * audio_dim)
        self.vnorm = nn.LayerNorm(2 * vision_dim)

        self.project_t = _projector(self.encoder.hidden_size, hidden_size)
        self.project_a = _projector(4 * audio_dim, hidden_size)
        self.project_v = _projector(4 * vision_dim, hidden_size)

        def gate() -> nn.Sequential:
            return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

        self.private_t, self.private_a, self.private_v = gate(), gate(), gate()
        self.shared = gate()
        self.recon_t = nn.Linear(hidden_size, hidden_size)
        self.recon_a = nn.Linear(hidden_size, hidden_size)
        self.recon_v = nn.Linear(hidden_size, hidden_size)

        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2)
        self.fusion_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 3), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden_size * 3, 1),
        )

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        encoder = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoder, "lr": lr / self.bert_lr_divisor, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def _sequence_features(
        self, x: torch.Tensor, lengths: torch.Tensor, rnn1, rnn2, norm
    ) -> torch.Tensor:
        """Two stacked BiLSTMs; returns both final states concatenated."""
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out1, (h1, _) = rnn1(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        normed = nn.utils.rnn.pack_padded_sequence(
            norm(padded), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h2, _) = rnn2(normed)
        batch = x.shape[0]
        return torch.cat(
            [h1.permute(1, 0, 2).reshape(batch, -1), h2.permute(1, 0, 2).reshape(batch, -1)],
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_bert = batch["text_bert"]
        mask = text_bert[:, 1].float()
        hidden = self.encoder(text_bert)
        # Attention-masked mean over real tokens, MISA's utterance vector.
        utterance_t = (hidden * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)

        lengths = batch["text_length"].clamp(min=1)
        utterance_a = self._sequence_features(
            batch["audio"], lengths, self.arnn1, self.arnn2, self.anorm
        )
        utterance_v = self._sequence_features(
            batch["vision"], lengths, self.vrnn1, self.vrnn2, self.vnorm
        )

        projected = {
            "t": self.project_t(utterance_t),
            "a": self.project_a(utterance_a),
            "v": self.project_v(utterance_v),
        }
        private = {
            "t": self.private_t(projected["t"]),
            "a": self.private_a(projected["a"]),
            "v": self.private_v(projected["v"]),
        }
        shared = {key: self.shared(value) for key, value in projected.items()}
        recon = {
            "t": self.recon_t(private["t"] + shared["t"]),
            "a": self.recon_a(private["a"] + shared["a"]),
            "v": self.recon_v(private["v"] + shared["v"]),
        }

        stacked = torch.stack(
            [private["t"], private["v"], private["a"], shared["t"], shared["v"], shared["a"]],
            dim=0,
        )
        fused = self.fusion_encoder(stacked)
        prediction = self.fusion(torch.cat(list(fused), dim=1)).view(-1)
        return {
            "M": prediction,
            "private": private,
            "shared": shared,
            "recon": recon,
            "projected": projected,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # MMSA uses MSE for MISA's regression term, unlike the L1 used elsewhere.
        task = F.mse_loss(outputs["M"], batch["label"])

        shared, private = outputs["shared"], outputs["private"]
        similarity = (
            central_moment_discrepancy(shared["t"], shared["v"], self.cmd_moments)
            + central_moment_discrepancy(shared["t"], shared["a"], self.cmd_moments)
            + central_moment_discrepancy(shared["a"], shared["v"], self.cmd_moments)
        ) / 3.0

        difference = sum(
            orthogonality(private[key], shared[key]) for key in ("t", "v", "a")
        ) + sum(
            orthogonality(private[a], private[b])
            for a, b in (("a", "t"), ("a", "v"), ("t", "v"))
        )

        reconstruction = sum(
            F.mse_loss(outputs["recon"][key], outputs["projected"][key])
            for key in ("t", "a", "v")
        ) / 3.0

        return (task
                + self.diff_weight * difference
                + self.sim_weight * similarity
                + self.recon_weight * reconstruction)
