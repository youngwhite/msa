"""CENET — Cross-modal Enhancement Network (Wang et al., IEEE TMM vol.25, 2023).

Where every earlier model in the storyline fuses *after* the encoders have run,
CENET reaches inside the text encoder: audio and vision are turned into a shift
applied to BERT's hidden states between two of its own layers. The text tower is
no longer a component that produces a vector to be fused — it is the fusion.

Ported from MMSA (`models/singleTask/CENET.py`, MIT, THUIAR). **MMSA's CENET is
not the paper's CENET**, and the acceptance group reproduces MMSA's, because
that is what its table reports. The three divergences, all verified against the
authors' release (`Say2L/CENet`):

1. **Backbone.** The authors build on SentiLARE, a RoBERTa variant pretrained
   with part-of-speech and sentiment-word knowledge. MMSA rebuilds the whole
   thing on `bert-base-uncased`.
2. **What the CE module consumes.** The authors quantise each modality into 16
   discrete labels and look those up in an `nn.Embedding` — their CE takes
   `visual_ids`/`acoustic_ids` and *ignores* the continuous `visual`/`acoustic`
   arguments it accepts. MMSA replaces the lookup with an MLP over the raw
   continuous features and ignores the ids instead. This is the mechanism of the
   paper, swapped.
3. **Features.** The authors use 74-dim COVAREP audio and 27-dim Facet vision on
   MOSI; MMSA's pipeline gives 5 and 20.

The attention itself is byte-identical between the two, including
`softmax(scores * 8)` where scaled dot-product attention would divide by
sqrt(d_k) ~= 27.7. That is the authors' own scaling, inherited rather than
introduced, so it is reproduced here. See docs/investigations.md#cenet-vs-paper.

Implementation note: MMSA re-implements BertLayer, BertEncoder, BertOutput and
BertIntermediate — 487 lines — because it needs a hook between layers, and its
copy still imports the long-deprecated `pytorch_transformers`. Iterating
HuggingFace's own `bert.encoder.layer` gets the same computation from the
maintained implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder


class _SelfAttention(nn.Module):
    """Text queries a modality. Reproduced from the authors' `CEmodule.py`.

    Despite the name it is cross-attention: Q comes from the text hidden states,
    K and V from the modality, so the output is always text-length however long
    the audio or vision stream is. That is what lets CENET take unaligned data
    without any resampling.
    """

    def __init__(self, hidden_size: int, head_num: int = 1) -> None:
        super().__init__()
        self.head_num = head_num
        self.all_head_size = (hidden_size // head_num) * head_num
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), x.size(1), self.head_num, -1).permute(0, 2, 1, 3)

    def forward(self, text: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.query(text))
        k = self._split_heads(self.key(modality))
        v = self._split_heads(self.value(modality))
        # *8, not /sqrt(d_k): the authors' scaling, kept deliberately.
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * 8, dim=-1)
        context = torch.matmul(weights, v).permute(0, 2, 1, 3).contiguous()
        return context.view(*context.size()[:-2], self.all_head_size)


class _CrossModalEnhancement(nn.Module):
    """The shift added to BERT's hidden states partway up the stack."""

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int) -> None:
        super().__init__()

        def project(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, text_dim)
            )

        self.vision_transform = project(vision_dim)
        self.audio_transform = project(audio_dim)
        self.hv = _SelfAttention(text_dim)
        self.ha = _SelfAttention(text_dim)
        self.cat_connect = nn.Linear(2 * text_dim, text_dim)

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor
    ) -> torch.Tensor:
        seen_v = self.hv(text, self.vision_transform(vision))
        seen_a = self.ha(text, self.audio_transform(audio))
        shift = self.cat_connect(torch.cat([seen_v, seen_a], dim=-1))
        return shift + text


@register_model("cenet")
class CENet(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
        injection_index: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(pretrained, finetune)
        hidden = self.encoder.hidden_size
        # The shift is applied *before* the layer at this index runs, so index 1
        # means "after layer 0". Matches the authors' ROBERTA_INJECTION_INDEX.
        self.injection_index = injection_index
        self.ce = _CrossModalEnhancement(hidden, audio_dim, vision_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """MMSA's three-way split, plus its Adam epsilon.

        Biases and LayerNorm weights are exempt from weight decay — except
        inside the CE module, which MMSA groups wholesale and therefore decays.
        That asymmetry is almost certainly accidental, but it is what produced
        the reference number.

        `eps` rides along per group: MMSA passes 3e-8 where torch defaults to
        1e-8, and a param group is the contract's own way to say that without
        the trainer needing to know.
        """
        eps = 3e-8
        no_decay = ("bias", "LayerNorm.weight")
        ce = list(self.ce.parameters())
        ce_ids = {id(p) for p in ce}
        decayed, plain = [], []
        for name, param in self.named_parameters():
            if id(param) in ce_ids:
                continue
            (plain if any(k in name for k in no_decay) else decayed).append(param)
        return [
            {"params": decayed, "lr": lr, "weight_decay": weight_decay, "eps": eps},
            {"params": ce, "lr": lr, "weight_decay": weight_decay, "eps": eps},
            {"params": plain, "lr": lr, "weight_decay": 0.0, "eps": eps},
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_bert = batch["text_bert"]
        input_ids = text_bert[:, 0].long()
        attention_mask = text_bert[:, 1].long()
        token_type_ids = text_bert[:, 2].long()

        bert = self.encoder.bert
        hidden = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        mask = bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        for index, layer in enumerate(bert.encoder.layer):
            if index == self.injection_index:
                hidden = self.ce(hidden, batch["audio"], batch["vision"])
            # transformers >= 5 returns the tensor directly; older versions (and
            # MMSA's vendored copy) returned a tuple. Indexing [0] on the new
            # API silently takes the first *sample* instead, which shows up as a
            # batch-size mismatch several frames away.
            out = layer(hidden, attention_mask=mask)
            hidden = out if torch.is_tensor(out) else out[0]
        return {"M": self.head(hidden[:, 0]).view(-1)}
