"""Low-rank Multimodal Fusion (Liu et al., ACL 2018).

TFN's fusion tensor grows as the product of the modality dimensions — 140k
entries feeding one linear layer, which is where 9 of its 9.5M parameters live.
LMF never materialises that tensor: it factorises the fusion weights into `rank`
modality-specific factors, so the same trilinear interaction costs
O(rank * sum(dims)) instead of O(prod(dims)).

Ported from MMSA (`models/singleTask/LMF.py`, MIT, THUIAR), which in turn follows
the authors' release. Same deviations as our TFN port (`use_lengths`,
`mask_pooling`), plus one deliberate correction:

MMSA builds the optimiser as

    Adam([{"params": list(model.parameters())[:3], "lr": factor_lr},
          {"params": list(model.parameters())[5:], "lr": learning_rate}], ...)

`parameters()` yields a module's own `nn.Parameter`s before it recurses into
submodules, so indices 0-2 really are the three fusion factors and the slicing
gets that part right. What it drops is indices 3 and 4 — `fusion_weights` and
`fusion_bias` — which land in no group and keep their initial values for the
whole run. That is the more damaging pair: `fusion_weights` is the (1, rank)
vector that combines the rank components, and `fusion_bias` starts at zero while
MOSI's labels do not have zero mean, so the head cannot learn an offset.

The authors' release slices `[:3]` / `[3:]` — contiguous, every parameter
covered. MMSA introduced the gap when it changed `[3:]` to `[5:]`. We train every
parameter and give the factors `factor_lr` through `param_groups`, which is both
what MMSA was evidently trying to express and what the original does.
`freeze_mmsa_quirk=True` restores MMSA's behaviour for comparison (MAE 0.9806 vs
0.9628 over 10 seeds).

A second discrepancy: the config assigns 0.3 to a `post_fusion_dropout` slot, but
forward never applies that layer — declared and forgotten. This one is *not*
MMSA's doing; the authors' `model.py` has the identical omission and MMSA
inherited it. Applying the dropout costs ~0.04 MAE, so our default is 0.0, i.e.
the behaviour both implementations actually have rather than the one their
configs state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from ..registry import register_model
from .base import MSAModel
from .blocks import SubNet, TextSubNet
from .functional import masked_mean


@register_model("lmf")
class LowRankFusion(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_hidden: int = 128,
        audio_hidden: int = 16,
        vision_hidden: int = 128,
        rank: int = 3,
        text_dropout: float = 0.3,
        audio_dropout: float = 0.3,
        vision_dropout: float = 0.3,
        post_fusion_dropout: float = 0.0,
        factor_lr: float | None = None,
        use_lengths: bool = True,
        mask_pooling: bool = True,
        freeze_mmsa_quirk: bool = False,
    ) -> None:
        super().__init__()
        self.use_lengths = use_lengths
        self.mask_pooling = mask_pooling
        self.rank = rank
        self.factor_lr = factor_lr
        self.freeze_mmsa_quirk = freeze_mmsa_quirk
        text_out = text_hidden // 2

        self.audio_subnet = SubNet(audio_dim, audio_hidden, audio_dropout)
        self.vision_subnet = SubNet(vision_dim, vision_hidden, vision_dropout)
        self.text_subnet = TextSubNet(text_dim, text_hidden, text_out, text_dropout)

        self.post_fusion_dropout = nn.Dropout(p=post_fusion_dropout)
        self.audio_factor = nn.Parameter(torch.empty(rank, audio_hidden + 1, 1))
        self.vision_factor = nn.Parameter(torch.empty(rank, vision_hidden + 1, 1))
        self.text_factor = nn.Parameter(torch.empty(rank, text_out + 1, 1))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, 1))

        for factor in (self.audio_factor, self.vision_factor, self.text_factor,
                       self.fusion_weights):
            xavier_normal_(factor)

    @property
    def factors(self) -> list[nn.Parameter]:
        return [self.audio_factor, self.vision_factor, self.text_factor,
                self.fusion_weights, self.fusion_bias]

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """The factors may train at their own rate; everything else uses `lr`."""
        if self.freeze_mmsa_quirk:
            # Reproduce MMSA's positional slicing verbatim, frozen tensors and all.
            params = list(self.parameters())
            return [
                {"params": params[:3], "lr": self.factor_lr or lr,
                 "weight_decay": weight_decay},
                {"params": params[5:], "lr": lr, "weight_decay": weight_decay},
            ]
        factor_ids = {id(p) for p in self.factors}
        rest = [p for p in self.parameters() if id(p) not in factor_ids]
        return [
            {"params": self.factors, "lr": self.factor_lr or lr,
             "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return masked_mean(x, lengths if self.mask_pooling else None)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        audio_h = self.audio_subnet(self._pool(batch["audio"], batch["audio_length"]))
        vision_h = self.vision_subnet(self._pool(batch["vision"], batch["vision_length"]))
        text_h = self.text_subnet(
            batch["text"], batch["text_length"] if self.use_lengths else None
        )

        ones = torch.ones(audio_h.shape[0], 1, dtype=audio_h.dtype, device=audio_h.device)
        a = torch.cat((ones, audio_h), dim=1)      # (batch, audio_hidden + 1)
        v = torch.cat((ones, vision_h), dim=1)
        t = torch.cat((ones, text_h), dim=1)

        # Each (batch, d+1) x (rank, d+1, 1) -> (rank, batch, 1). Multiplying the
        # three elementwise is the low-rank stand-in for the full outer product:
        # summing over rank afterwards reconstructs the trilinear term without
        # ever building it.
        fused = (torch.matmul(a, self.audio_factor)
                 * torch.matmul(v, self.vision_factor)
                 * torch.matmul(t, self.text_factor))
        # No-op at the default 0.0; see the module docstring for why.
        fused = self.post_fusion_dropout(fused)
        # (1, rank) x (batch, rank, 1) -> (batch, 1, 1)
        out = torch.matmul(self.fusion_weights, fused.permute(1, 0, 2)).squeeze(1)
        out = out + self.fusion_bias
        return {
            "M": out.view(-1),
            "feature_t": text_h,
            "feature_a": audio_h,
            "feature_v": vision_h,
        }
