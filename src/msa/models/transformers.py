"""Cross-modal transformer encoder, the machinery MulT is built from.

Ported from MMSA's `models/subNets/transformers_encoder/` (MIT, THUIAR), which
is itself a trimmed fairseq encoder. Two substitutions, both documented because
they are the kind of thing a reader will want to check:

* `nn.MultiheadAttention` replaces the hand-rolled attention. The maths is the
  same scaled dot-product attention with separate q/k/v projections; PyTorch's
  version is the well-tested one. Parameter initialisation differs in detail.
* Positional embeddings are absent — not an omission on our part. MMSA builds
  every encoder without `position_embedding=True`, so its MulT runs with none,
  and a faithful port has to do the same. Order still reaches the model through
  the causal mask and the temporal convolution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    """MMSA's `buffered_future_mask`: -inf above the diagonal.

    With different query and key lengths the diagonal is offset by their
    difference, which keeps the mask rectangular for cross-modal attention.
    """
    mask = torch.full((query_len, key_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1 + abs(key_len - query_len))


class _EncoderLayer(nn.Module):
    """Pre-LayerNorm block: norm -> attention -> residual, norm -> FFN -> residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
        attn_mask: bool,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])
        for linear in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0.0)

    def forward(
        self, x: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        x = self.norms[0](x)
        if k is None:
            k = v = x
        else:
            k, v = self.norms[0](k), self.norms[0](v)
        mask = causal_mask(x.shape[0], k.shape[0], x.device) if self.attn_mask else None
        attended, _ = self.attention(x, k, v, attn_mask=mask, need_weights=False)
        x = residual + F.dropout(attended, p=self.res_dropout, training=self.training)

        residual = x
        y = self.norms[1](x)
        y = F.dropout(F.relu(self.fc1(y)), p=self.relu_dropout, training=self.training)
        y = F.dropout(self.fc2(y), p=self.res_dropout, training=self.training)
        return residual + y


class TransformerEncoder(nn.Module):
    """Stack of pre-LN layers. Called with one argument it is self-attention;
    called with (query, key, value) it is cross-modal attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layers: int,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        attn_mask: bool = False,
    ) -> None:
        super().__init__()
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_dropout = embed_dropout
        self.layers = nn.ModuleList(
            _EncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout,
                          res_dropout, attn_mask)
            for _ in range(layers)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """All tensors are (seq, batch, embed_dim), fairseq's layout."""
        def prepare(t: torch.Tensor) -> torch.Tensor:
            return F.dropout(self.embed_scale * t, p=self.embed_dropout,
                             training=self.training)

        x = prepare(x)
        if k is not None and v is not None:
            k, v = prepare(k), prepare(v)
            for layer in self.layers:
                x = layer(x, k, v)
        else:
            for layer in self.layers:
                x = layer(x)
        return self.layer_norm(x)
