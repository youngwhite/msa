"""Graph-MFN — MFN with a Dynamic Fusion Graph (Zadeh et al., ACL 2018).

MFN writes into its shared memory through a single attention block, which has to
represent every kind of cross-modal interaction at once. Graph-MFN replaces that
block with a small graph: one vertex per non-empty subset of modalities (7 for
three modalities), each vertex fed by the vertices below it, and a learned
*efficacy* per edge that decides how much that particular interaction matters for
this sample. A trimodal cue and a text-only cue no longer share one pathway.

Ported from MMSA (`models/singleTask/Graph_MFN.py`, MIT, THUIAR), with one
correction.

**MMSA's vertex networks never train.** Its `DynamicFusionGraph` stores them in a
plain dict:

    self.networks = {}
    ...
    self.networks[key] = final_model

A plain dict is not tracked by `nn.Module`, so those sub-networks are absent from
`model.parameters()` — the optimiser never sees them and they stay at their
initial values for the whole run. Only the efficacy model and the top-level
network, which are ordinary attributes, actually learn. We use `nn.ModuleDict`;
`freeze_graph_networks=True` restores the original behaviour so the difference
can be measured rather than argued about.

Note also that unlike MFN, Graph-MFN's MMSA config leaves `need_normalized`
False, so it receives the real per-step audio and vision sequences. The two are
therefore not directly comparable in MMSA's table.
"""

from __future__ import annotations

from itertools import chain, combinations

import torch
import torch.nn as nn

from ..registry import register_model
from .base import MSAModel


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


class DynamicFusionGraph(nn.Module):
    """One vertex per non-empty subset of modalities, edges weighted by efficacy."""

    def __init__(
        self,
        in_dims: list[int],
        node_dim: int,
        hidden: int,
        dropout: float,
        freeze_networks: bool = False,
    ) -> None:
        super().__init__()
        self.num_modalities = len(in_dims)
        self.in_dims = in_dims
        self.node_dim = node_dim
        # [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
        self.powerset = list(
            chain.from_iterable(
                combinations(range(self.num_modalities), r)
                for r in range(self.num_modalities + 1)
            )
        )[1:]

        shapes = {(i,): dim for i, dim in enumerate(in_dims)}
        networks, total_efficacies = {}, 0
        for key in self.powerset[self.num_modalities:]:
            unimodal = sum(in_dims[m] for m in key)
            multimodal = ((2 ** len(key) - 2) - len(key)) * node_dim
            total_efficacies += 2 ** len(key) - 2
            shapes[key] = unimodal + multimodal
            networks["_".join(map(str, key))] = _mlp(shapes[key], hidden, node_dim, dropout)
        # nn.ModuleDict, not a plain dict — see the module docstring.
        self.networks = nn.ModuleDict(networks)
        if freeze_networks:
            for param in self.networks.parameters():
                param.requires_grad = False

        total_efficacies += 2 ** self.num_modalities - 1
        top_in = sum(in_dims) + (2 ** self.num_modalities - self.num_modalities - 1) * node_dim
        self.top_network = _mlp(top_in, hidden, node_dim, dropout)
        self.efficacy_model = _mlp(sum(in_dims), hidden, total_efficacies, dropout)

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        outputs = {(i,): tensor for i, tensor in enumerate(modalities)}
        efficacies = self.efficacy_model(torch.cat(modalities, dim=1))

        index = 0
        subsets: list[tuple[int, ...]] = []
        for key in self.powerset[self.num_modalities:]:
            # every proper non-empty subset of `key` feeds this vertex
            subsets = list(
                chain.from_iterable(combinations(key, r) for r in range(len(key) + 1))
            )[1:-1]
            gated = torch.cat(
                [outputs[sub] * efficacies[:, index + i].view(-1, 1)
                 for i, sub in enumerate(subsets)],
                dim=1,
            )
            outputs[key] = self.networks["_".join(map(str, key))](gated)
            index += len(subsets)

        top_inputs = [*subsets, tuple(range(self.num_modalities))]
        gated = torch.cat(
            [outputs[sub] * efficacies[:, index + i].view(-1, 1)
             for i, sub in enumerate(top_inputs)],
            dim=1,
        )
        return self.top_network(gated)


@register_model("graph_mfn")
class GraphMemoryFusionNetwork(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_hidden: int = 256,
        audio_hidden: int = 32,
        vision_hidden: int = 256,
        memsize: int = 300,
        node_dim: int = 64,
        graph_hidden: int = 32,
        graph_dropout: float = 0.0,
        gamma1_hidden: int = 128,
        gamma1_dropout: float = 0.5,
        gamma2_hidden: int = 64,
        gamma2_dropout: float = 0.5,
        out_hidden: int = 256,
        out_dropout: float = 0.0,
        freeze_graph_networks: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_sizes = (text_hidden, audio_hidden, vision_hidden)
        self.memsize = memsize

        self.lstm_t = nn.LSTMCell(text_dim, text_hidden)
        self.lstm_a = nn.LSTMCell(audio_dim, audio_hidden)
        self.lstm_v = nn.LSTMCell(vision_dim, vision_hidden)

        # Each modality contributes the concatenation of its previous and current
        # memory cell, projected to a common width.
        self.transform_t = nn.Linear(2 * text_hidden, text_hidden)
        self.transform_a = nn.Linear(2 * audio_hidden, audio_hidden)
        self.transform_v = nn.Linear(2 * vision_hidden, vision_hidden)

        self.graph = DynamicFusionGraph(
            [text_hidden, audio_hidden, vision_hidden], node_dim,
            graph_hidden, graph_dropout, freeze_graph_networks,
        )
        gamma_in = node_dim + memsize
        self.gamma1 = _mlp(gamma_in, gamma1_hidden, memsize, gamma1_dropout)
        self.gamma2 = _mlp(gamma_in, gamma2_hidden, memsize, gamma2_dropout)
        total_hidden = text_hidden + audio_hidden + vision_hidden
        self.head = _mlp(total_hidden + memsize, out_hidden, 1, out_dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        batch_size, steps = text.shape[0], text.shape[1]
        device, dtype = text.device, text.dtype
        dh_t, dh_a, dh_v = self.hidden_sizes

        def zeros(width: int) -> torch.Tensor:
            return torch.zeros(batch_size, width, device=device, dtype=dtype)

        h_t, c_t = zeros(dh_t), zeros(dh_t)
        h_a, c_a = zeros(dh_a), zeros(dh_a)
        h_v, c_v = zeros(dh_v), zeros(dh_v)
        memory = zeros(self.memsize)

        for step in range(steps):
            prev_t, prev_a, prev_v = c_t, c_a, c_v
            h_t, c_t = self.lstm_t(text[:, step], (h_t, c_t))
            h_a, c_a = self.lstm_a(audio[:, step], (h_a, c_a))
            h_v, c_v = self.lstm_v(vision[:, step], (h_v, c_v))

            singletons = [
                self.transform_t(torch.cat([prev_t, c_t], dim=1)),
                self.transform_a(torch.cat([prev_a, c_a], dim=1)),
                self.transform_v(torch.cat([prev_v, c_v], dim=1)),
            ]
            proposal = torch.tanh(self.graph(singletons))
            gated = torch.cat([proposal, memory], dim=1)
            memory = (torch.sigmoid(self.gamma1(gated)) * memory
                      + torch.sigmoid(self.gamma2(gated)) * proposal)

        return {"M": self.head(torch.cat([h_t, h_a, h_v, memory], dim=1)).view(-1)}
