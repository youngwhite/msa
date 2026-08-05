"""CLGSI — Contrastive Learning Guided by Sentiment Intensity (Yang et al., Findings of NAACL 2024).

Positive and negative pairs are not chosen by label equality but **weighted by
how far apart two labels are**: pairs closer than a dividing line count as
positive with weight `-tanh(d - 2·line)·gain`, pairs beyond it count as negative
with weight `tanh(d)·gain`. Both a cross-modal and a within-modal term use the
same supervision mask.

That mechanism is why this model is here. **ConFEDE implements the same idea and
never switches it on** -- its `cont_NTXentLoss` weights negatives by label
distance only after `update_label` is called, and nothing calls it
(`storyline.md` §18). Read together the two ask whether the weighting does
anything, which is a question rather than another table row.

Structure and protocol from the authors' release (`AZYoung233/CLGSI`, MIT);
hyper-parameters from its `config/config_regression.py`. It is MMSA-derived, so
it reads the same aligned pickles as most groups here. Full reading in
`docs/spec_clgsi.md`.

**BERT gets its own learning rate here (5e-5)**, unlike DPDF-LQ, DLF and DMD
which put it in one group at the shared rate, and unlike ConFEDE which freezes
it. Three different answers in five recent methods: there is no house convention
to apply, only the paper in front of you.

Two things the paper does not mention, reproduced deliberately:

* **Negative pairs are multiplied by a hardcoded 0.8** before the loss. That is
  a temperature change on the negative term under another name.
* **Six declared modules are never called** -- `post_fusion_layer_1`,
  `post_{text,audio,video}_layer_2`, `skip_connection_BatchNorm` and
  `post_fusion_dropout`. Not implemented here; excluded from the equivalence
  test, as convention 5 directs.

The multi-task scaffolding it inherited from Self-MM is inert: all four entries
of `label_map` are assigned the same true label, so there are no pseudo-labels
despite the machinery for them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder

#: config/config_regression.py, mosi. The dividing line is in sentiment units
#: AFTER labels are mapped to [-1, 1].
DIVIDING_LINE, GAIN, TEMPERATURE, GAMMA = 0.4, 1.5, 0.03, 0.95
#: Applied to every negative term by the release; see the module docstring.
NEGATIVE_SCALE = 0.8


def intensity_mask(labels: torch.Tensor, positive: bool,
                   weighted: bool = True) -> torch.Tensor:
    """Pair weights from the distance between two samples' sentiment.

    Labels are first mapped to [-1, 1] the way the release does, `(y + 3)/3 - 1`,
    which for MOSI's [-3, 3] is `y/3`.

    `weighted=False` is the release's own alternative, sitting commented out one
    line below each weighted assignment: the same pairs selected, every weight 1.
    The authors tried it and did not report it. See
    `investigations.md#intensity-weighting`.
    """
    mapped = (labels + 3) / 3 - 1
    distance = (mapped.unsqueeze(0) - mapped.unsqueeze(1)).abs()
    mask = torch.zeros_like(distance)
    if positive:
        near = distance <= DIVIDING_LINE
        mask[near] = (-torch.tanh(distance[near] - DIVIDING_LINE * 2) * GAIN
                      if weighted else 1.0)
    else:
        far = distance > DIVIDING_LINE
        mask[far] = torch.tanh(distance[far]) * GAIN if weighted else 1.0
    return mask


def align_avg_pool(x: torch.Tensor, dst_len: int) -> torch.Tensor:
    """The release's `AlignSubNet('avg_pool')`, applied before the model proper.

    CLGSI reads the **unaligned** pickle -- audio 375 frames, vision 500 -- and
    pools both down to the text length inside `AMIO.forward`. It does NOT read
    the word-aligned pickle, which is a different set of features entirely; a
    first run here used the aligned one and had to be discarded.

    The pooling is **strided, not blocked**. After padding with the last frame
    repeated, it views as (batch, pool, dst, dim) and averages over `pool`, so
    output position j is the mean of input positions j, j+dst, j+2·dst, ... --
    not of a contiguous window. Reproduced as written; it reads like block
    pooling at a glance.
    """
    raw_len = x.size(1)
    if raw_len == dst_len:
        return x
    if raw_len % dst_len == 0:
        pad_len, pool = 0, raw_len // dst_len
    else:
        pad_len, pool = dst_len - raw_len % dst_len, raw_len // dst_len + 1
    if pad_len:
        pad = x[:, -1, :].unsqueeze(1).expand(x.size(0), pad_len, x.size(-1))
        x = torch.cat([x, pad], dim=1)
    return x.view(x.size(0), pool, dst_len, -1).mean(dim=1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positions, sized to the release's own max length."""

    def __init__(self, width: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, width)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2) * -(math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)
        pe[:, 1::2] = cos if width % 2 == 0 else cos[:, :-1]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class SequenceEncoder(nn.Module):
    """Positions then a transformer, **at the raw feature width**.

    Audio is 5-dimensional and vision 20, and the release runs the transformer at
    those widths rather than projecting up first -- so audio attention has one
    head over five channels. Faithful, and worth noticing before assuming a
    typo.
    """

    def __init__(self, width: int, heads: int, layers: int, max_len: int) -> None:
        super().__init__()
        self.position_embbeding = PositionalEncoding(width, 0.1, max_len)
        layer = nn.TransformerEncoderLayer(width, heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(self.position_embbeding(x))


class GLFK(nn.Module):
    """Global-Local-Fine-Knowledge: a small 2-D convolution stack over the three
    stacked modality vectors, collapsing the modality axis."""

    def __init__(self, enlarge: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1),
            nn.Conv2d(1, enlarge // 2, kernel_size=(1, 1), stride=1),
            nn.PReLU(),
            nn.Conv2d(enlarge // 2, enlarge, kernel_size=(1, 1), stride=1),
            nn.Conv2d(enlarge, 1, kernel_size=(1, 1), stride=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.unsqueeze(1)).squeeze(1).squeeze(-1)


class UnimodalSkipNet(nn.Module):
    """Squeeze-and-excitation over the sequence: average the time axis, then a
    bottleneck MLP with a sigmoid."""

    def __init__(self, channels: int, enlarge: int, reduction: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, enlarge, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


@register_model("clgsi")
class CLGSI(MSAModel):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 5,
        vision_dim: int = 20,
        audio_length: int = 50,
        vision_length: int = 50,
        audio_heads: int = 1,
        vision_heads: int = 4,
        audio_layers: int = 2,
        vision_layers: int = 2,
        post_dim: int = 64,
        fusion_dim: int = 128,
        fusion_filters: int = 16,
        skip_reduction: int = 2,
        text_dropout: float = 0.05,
        audio_dropout: float = 0.05,
        vision_dropout: float = 0.05,
        fusion_dropout: float = 0.2,
        pretrained: str = "bert-base-uncased",
        finetune_bert: bool = True,
        intensity_weighting: bool = True,
    ) -> None:
        super().__init__()
        #: False reproduces the release's own commented-out ablation.
        self.intensity_weighting = intensity_weighting
        self.relu = nn.ReLU()
        self.text_model = BertTextEncoder(pretrained, finetune_bert)
        self.audio_model = SequenceEncoder(audio_dim, audio_heads, audio_layers, audio_length)
        self.video_model = SequenceEncoder(vision_dim, vision_heads, vision_layers, vision_length)

        self.GLFK = GLFK(fusion_filters)
        skip_length = post_dim * 4
        self.skip_connection_dropout = nn.Dropout(p=fusion_dropout)
        self.post_fusion_layer_skip_connection = nn.Linear(skip_length, fusion_dim * 3)
        self.post_fusion_layer_2 = nn.Linear(fusion_dim * 3, fusion_dim * 2)
        self.post_fusion_layer_3 = nn.Linear(fusion_dim * 2, 1)
        self.fusionBatchNorm = nn.BatchNorm1d(fusion_dim * 2)

        self.post_text_dropout = nn.Dropout(p=text_dropout)
        self.text_skip_net = UnimodalSkipNet(text_dim, post_dim, skip_reduction)
        self.post_text_layer_1 = nn.Linear(text_dim, post_dim)
        self.post_audio_dropout = nn.Dropout(p=audio_dropout)
        self.audio_skip_net = UnimodalSkipNet(audio_dim, post_dim, skip_reduction)
        self.post_audio_layer_1 = nn.Linear(audio_dim, post_dim)
        self.post_video_dropout = nn.Dropout(p=vision_dropout)
        self.video_skip_net = UnimodalSkipNet(vision_dim, post_dim, skip_reduction)
        self.post_video_layer_1 = nn.Linear(vision_dim, post_dim)

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Five groups at four rates, as the release's AdamW has.

        `lr` and `weight_decay` from the command line are ignored on purpose:
        this model's rates are per-module and come from its own config, and
        silently applying one rate to all of it is the mistake that cost 2.7 SE
        on DPDF-LQ.
        """
        no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
        bert = list(self.text_model.named_parameters())
        groups = [
            {"params": [p for n, p in bert if not any(k in n for k in no_decay)],
             "lr": 5e-5, "weight_decay": 0.01},
            {"params": [p for n, p in bert if any(k in n for k in no_decay)],
             "lr": 5e-5, "weight_decay": 0.0},
            {"params": list(self.audio_model.parameters()),
             "lr": 5e-3, "weight_decay": 0.01},
            {"params": list(self.video_model.parameters()),
             "lr": 1e-3, "weight_decay": 0.001},
        ]
        named = dict(self.named_parameters())
        covered = {id(p) for group in groups for p in group["params"]}
        groups.append({"params": [p for p in named.values() if id(p) not in covered],
                       "lr": 1e-2, "weight_decay": 0.001})
        return groups

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.text_model(batch["text_bert"])
        # Align first, exactly where AMIO does it -- before the model proper.
        audio = self.audio_model(align_avg_pool(batch["audio"].float(), text.size(1)))
        video = self.video_model(align_avg_pool(batch["vision"].float(), text.size(1)))

        # Text takes BERT's [CLS]; audio and vision take the LAST position of
        # their transformer, not the first. The asymmetry is the release's.
        text_h = self.relu(self.post_text_layer_1(self.post_text_dropout(text[:, 0, :])))
        audio_h = self.relu(self.post_audio_layer_1(self.post_audio_dropout(audio[:, -1, :])))
        video_h = self.relu(self.post_video_layer_1(self.post_video_dropout(video[:, -1, :])))

        fusion = self.GLFK(torch.cat(
            [text_h.unsqueeze(-1), audio_h.unsqueeze(-1), video_h.unsqueeze(-1)], dim=-1))
        fusion = torch.cat([fusion, self.text_skip_net(text),
                            self.audio_skip_net(audio), self.video_skip_net(video)], dim=-1)
        fusion = self.post_fusion_layer_skip_connection(
            self.skip_connection_dropout(fusion))

        hidden = self.fusionBatchNorm(self.relu(self.post_fusion_layer_2(fusion)))
        return {
            "M": self.post_fusion_layer_3(hidden).squeeze(-1),
            "Feature_t": text_h, "Feature_a": audio_h,
            "Feature_v": video_h, "Feature_f": fusion,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        label = batch["label"]
        task = F.l1_loss(outputs["M"], label)
        return task + GAMMA * self.contrastive(outputs, label)

    def contrastive(self, outputs: dict[str, torch.Tensor],
                    label: torch.Tensor) -> torch.Tensor:
        """The release's own composition, read rather than inferred.

        Three pairs only -- (v,a), (v,t), (t,a) -- and their losses are **summed**,
        not averaged. `Feature_f`, the fused representation, never enters the
        contrastive loss at all despite being returned alongside the other three.
        A first pass here assumed all four views and six pairs, which is the kind
        of plausible guess this repository keeps paying for.
        """
        self._masks = (intensity_mask(label, True, self.intensity_weighting),
                       intensity_mask(label, False, self.intensity_weighting))
        return (self._single(outputs["Feature_v"], outputs["Feature_a"])
                + self._single(outputs["Feature_v"], outputs["Feature_t"])
                + self._single(outputs["Feature_t"], outputs["Feature_a"]))

    def _single(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """One pair, scored from both sides and averaged.

        Both views are L2-normalised first, so the inner product IS the cosine.
        """
        positive, negative = self._masks
        first = F.normalize(first, dim=1)
        second = F.normalize(second, dim=1)

        def weighted(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            similarity = a @ b.T
            return (torch.exp(similarity * positive / TEMPERATURE),
                    torch.exp(similarity * negative * NEGATIVE_SCALE / TEMPERATURE))

        inter_pos_1, inter_neg_1 = weighted(first, second)
        inter_pos_2, inter_neg_2 = weighted(second, first)
        intra_pos_1, intra_neg_1 = weighted(first, first)
        intra_pos_2, intra_neg_2 = weighted(second, second)

        def one(inter_pos, inter_neg, intra_pos, intra_neg) -> torch.Tensor:
            numerator = (inter_pos + intra_pos).sum(1)
            denominator = numerator + (inter_neg + intra_neg).sum(1)
            return -torch.log(numerator / denominator.clamp_min(1e-12)).mean()

        return (one(inter_pos_1, inter_neg_1, intra_pos_1, intra_neg_1)
                + one(inter_pos_2, inter_neg_2, intra_pos_2, intra_neg_2)) / 2
