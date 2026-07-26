"""Multimodal Transformer (Tsai et al., ACL 2019).

Everything before this in the storyline needs the modalities to share a clock:
audio and vision are resampled onto word boundaries so that step i means the same
moment everywhere. That alignment is itself an estimate, and it throws away the
fact that a gesture may lag the word it modifies.

MulT drops the requirement. Each modality keeps its own timeline, and a
cross-modal attention block lets every step of one modality query the whole
sequence of another — six such blocks, one per ordered pair. The results for each
target modality are concatenated, passed through a self-attention stack, and the
final step is read out.

Ported from MMSA (`models/singleTask/MULT.py`, MIT, THUIAR). Notes for a reader
comparing the two:

* The temporal Conv1d uses `padding=0`, so it shortens each sequence by
  `kernel_size - 1` (50 -> 46 for text, 375 -> 371 audio, 500 -> 496 vision).
  That is the reference's behaviour, kept.
* MMSA constructs its encoders without positional embeddings, so MulT here has
  none either. See `msa/models/transformers.py`.
* MulT is the first model in the storyline that clips gradients (by *value*, at
  0.6) and decays its learning rate on plateau — both supplied by TrainConfig
  rather than by a model-specific trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .transformers import TransformerEncoder


@register_model("mult")
class MultimodalTransformer(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        model_dim: int = 50,
        num_heads: int = 10,
        layers: int = 4,
        self_attention_layers: int = 3,
        kernel_size: int = 5,
        attn_dropout: float = 0.3,
        attn_dropout_audio: float = 0.2,
        attn_dropout_vision: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.2,
        text_dropout: float = 0.5,
        output_dropout: float = 0.5,
        attn_mask: bool = True,
    ) -> None:
        super().__init__()
        self.text_dropout = text_dropout
        self.output_dropout = output_dropout
        d = model_dim

        # Temporal convolution: project every modality to a common width while
        # giving each position a local receptive field.
        self.proj_t = nn.Conv1d(text_dim, d, kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, d, kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(vision_dim, d, kernel_size, padding=0, bias=False)

        def cross(dropout: float) -> TransformerEncoder:
            return TransformerEncoder(d, num_heads, layers, attn_dropout=dropout,
                                      relu_dropout=relu_dropout, res_dropout=res_dropout,
                                      embed_dropout=embed_dropout, attn_mask=attn_mask)

        # Six directed pairs: the target modality attends to the source.
        self.t_with_a, self.t_with_v = cross(attn_dropout_audio), cross(attn_dropout_vision)
        self.a_with_t, self.a_with_v = cross(attn_dropout), cross(attn_dropout_vision)
        self.v_with_t, self.v_with_a = cross(attn_dropout), cross(attn_dropout_audio)

        def memory() -> TransformerEncoder:
            return TransformerEncoder(2 * d, num_heads, max(layers, self_attention_layers),
                                      attn_dropout=attn_dropout, relu_dropout=relu_dropout,
                                      res_dropout=res_dropout, embed_dropout=embed_dropout,
                                      attn_mask=attn_mask)

        self.mem_t, self.mem_a, self.mem_v = memory(), memory(), memory()

        combined = 6 * d
        self.proj1 = nn.Linear(combined, combined)
        self.proj2 = nn.Linear(combined, combined)
        self.out_layer = nn.Linear(combined, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # (batch, seq, dim) -> (dim, seq) for the convolution -> (seq, batch, dim)
        text = F.dropout(batch["text"].transpose(1, 2), p=self.text_dropout,
                         training=self.training)
        x_t = self.proj_t(text).permute(2, 0, 1)
        x_a = self.proj_a(batch["audio"].transpose(1, 2)).permute(2, 0, 1)
        x_v = self.proj_v(batch["vision"].transpose(1, 2)).permute(2, 0, 1)

        def fuse(target: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                 blocks, memory) -> torch.Tensor:
            a = blocks[0](target, first, first)
            b = blocks[1](target, second, second)
            return memory(torch.cat([a, b], dim=2))[-1]

        last_t = fuse(x_t, x_a, x_v, (self.t_with_a, self.t_with_v), self.mem_t)
        last_a = fuse(x_a, x_t, x_v, (self.a_with_t, self.a_with_v), self.mem_a)
        last_v = fuse(x_v, x_t, x_a, (self.v_with_t, self.v_with_a), self.mem_v)

        fused = torch.cat([last_t, last_a, last_v], dim=1)
        projected = self.proj2(
            F.dropout(F.relu(self.proj1(fused)), p=self.output_dropout,
                      training=self.training)
        )
        return {"M": self.out_layer(projected + fused).view(-1)}
