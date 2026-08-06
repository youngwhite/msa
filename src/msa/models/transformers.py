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


class SinusoidalPositionalEmbedding(nn.Module):
    """The reference's sinusoidal table, ported from fairseq via MMSA.

    MMSA ships this file but never switches it on for MulT, so the faithful-to-
    MMSA port runs without it. It exists here for the faithful-to-paper variant,
    where the original always builds one.

    Two quirks of the reference are kept deliberately:

    * A step counts as padding when its **first feature channel** is exactly
      zero. That is a heuristic, not a real mask, but it is what the reference
      does and the numbers have to come from the same rule.
    * Non-padding steps take their *absolute* index + 1, not a compacted
      numbering, so right-padding leaves the earlier positions untouched.
    """

    def __init__(self, embedding_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer("_table", self._build(128 + padding_idx + 1), persistent=False)

    def _build(self, num_embeddings: int) -> torch.Tensor:
        half = self.embedding_dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, dtype=torch.float) * -scale)
        angles = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * freqs.unsqueeze(0)
        table = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1).view(num_embeddings, -1)
        if self.embedding_dim % 2 == 1:
            table = torch.cat([table, torch.zeros(num_embeddings, 1)], dim=1)
        table[self.padding_idx, :] = 0
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is (seq, batch, dim); the result is the same shape."""
        seq, batch, _ = x.shape
        max_pos = self.padding_idx + 1 + seq
        if self._table.size(0) < max_pos:
            self._table = self._build(max_pos).to(self._table.device)
        table = self._table.to(dtype=x.dtype, device=x.device)

        probe = x[:, :, 0].t()                                    # (batch, seq)
        index = torch.arange(self.padding_idx + 1, max_pos, device=x.device)
        index = index.unsqueeze(0).expand(batch, seq)
        positions = torch.where(probe.ne(self.padding_idx), index, torch.zeros_like(index))
        embedded = table.index_select(0, positions.reshape(-1)).view(batch, seq, -1)
        return embedded.transpose(0, 1).detach()


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
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        # PyTorch xavier-initialises in_proj_weight (matching the reference) but
        # leaves out_proj.weight at nn.Linear's kaiming_uniform(a=sqrt(5)) — a
        # bound 1.73x narrower than the reference's xavier_uniform. The forward
        # pass is identical either way, so the weight-copy equivalence test in
        # scripts/check_mult_encoder.py cannot see this; only the starting point
        # differs, and on a stack this deep that is not a detail.
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        # 4x is the MulT default every earlier model here uses. FeaDA's release
        # sets its feed-forward to embed_dim, so the width is a parameter --
        # defaulted to 4x, which leaves every existing model byte-identical.
        hidden = ffn_dim if ffn_dim is not None else 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
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
        position_embedding: bool = False,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_dropout = embed_dropout
        self.embed_positions = (
            SinusoidalPositionalEmbedding(embed_dim) if position_embedding else None
        )
        self.layers = nn.ModuleList(
            _EncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout,
                          res_dropout, attn_mask, ffn_dim)
            for _ in range(layers)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """All tensors are (seq, batch, embed_dim), fairseq's layout."""
        def prepare(t: torch.Tensor) -> torch.Tensor:
            # Scale, then add positions, then drop — the reference's order.
            scaled = self.embed_scale * t
            if self.embed_positions is not None:
                scaled = scaled + self.embed_positions(t)
            return F.dropout(scaled, p=self.embed_dropout, training=self.training)

        x = prepare(x)
        if k is not None and v is not None:
            k, v = prepare(k), prepare(v)
            for layer in self.layers:
                x = layer(x, k, v)
        else:
            for layer in self.layers:
                x = layer(x)
        return self.layer_norm(x)
