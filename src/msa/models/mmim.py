"""MMIM — MultiModal InfoMax (Han et al., EMNLP 2021).

Where every earlier model in this storyline asks "how should the three modalities
be combined", MMIM asks "how much does the fused representation actually keep".
It adds two information-theoretic objectives on top of an ordinary
concatenate-and-predict head:

* **MMILB**, a lower bound on the mutual information between text and each other
  modality, fitted in its own pass each epoch. It models one modality as a
  Gaussian conditioned on the other and maximises the log-likelihood.
* **CPC**, contrastive predictive coding between the fused vector and each of the
  three unimodal vectors, so the fusion is pushed to stay predictive of its
  parts rather than collapsing onto text.

The two-stage epoch is the part that does not fit an ordinary training loop: the
MMILB parameters are fitted over the whole training split *before* the main pass
runs, with their own optimiser. Folding that into the main batch loop would be a
different algorithm, so `MSAModel` grew `auxiliary_optimizer` / `auxiliary_loss`
instead — a hook, not an MMIM special case.

Ported from MMSA (`models/singleTask/MMIM.py`, MIT, THUIAR) and checked against
the authors' own repository (`declare-lab/Multimodal-Infomax`). Those two agree:
MMILB and CPC are line-for-line identical, and the main model differs only in
naming and plumbing.

Three things are reproduced as they behave rather than as they read — see
docs/investigations.md#mmim:

* The entropy term `H` is *computed* from batch 1 but only *subtracted* from
  batch 2 onward: the memory is read at `i_batch >= mem_size` while the loss uses
  it at `i_batch > mem_size`. An off-by-one, inherited from the authors.
* `update_epochs: 2` sits in MMSA's config and its MMIM trainer never reads it.
  There is no gradient accumulation here, whatever the config says.
* The learning-rate scheduler (`when: 20`) can never fire, because early stopping
  is at 8 epochs on the same quantity. Verified against ten reference runs: not
  one logged a reduction.

What is **not** reproduced is the reference's checkpoint selection. The authors
gate saving on the *test* loss (`solver.py`, `elif test_loss < best_mae`), so the
published number comes from a protocol that reads the test set. MMSA already
selects on validation and so do we; see #originals-select-on-test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec
from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder


class _RNNEncoder(nn.Module):
    """LSTM over the padded stream, read at its true final step."""

    def __init__(self, in_size: int, hidden: int, out_size: int, num_layers: int,
                 dropout: float, bidirectional: bool) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(in_size, hidden, num_layers=num_layers, batch_first=True,
                           dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((2 if bidirectional else 1) * hidden, out_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.rnn(packed)
        h = torch.cat((h[0], h[1]), dim=-1) if self.bidirectional else h.squeeze()
        return self.linear(self.dropout(h))


class _MMILB(nn.Module):
    """A lower bound on I(x; y), plus a Gaussian entropy term over a memory bank.

    `y` is modelled as Normal(mu(x), exp(logvar(x))); `lld` is its mean
    log-likelihood, which is the bound being maximised. The entropy term looks at
    the covariance of positive- and negative-label samples separately, pooled
    with a small history of earlier batches, and rewards keeping the two clouds
    distinguishable.
    """

    def __init__(self, x_size: int, y_size: int, mid_activation: str = "ReLU",
                 last_activation: str = "Tanh") -> None:
        # `last_activation` is accepted and never applied — see the comment on
        # mlp_logvar. Kept in the signature so the config maps across unchanged.
        super().__init__()
        mid = getattr(nn, mid_activation)
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size), mid(), nn.Linear(y_size, y_size)
        )
        # `last_activation` is configured (Tanh) and unused, exactly as in both
        # the reference and the authors' code: logvar comes out of a bare Linear.
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size), mid(), nn.Linear(y_size, y_size)
        )
        self.entropy_prj = nn.Sequential(nn.Linear(y_size, y_size // 4), nn.Tanh())

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                labels: torch.Tensor | None = None,
                memory: dict[str, list[torch.Tensor]] | None = None):
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)
        lld = torch.mean(torch.sum(-((mu - y) ** 2) / 2.0 / torch.exp(logvar), dim=-1))

        entropy = x.new_zeros(())
        samples: dict[str, torch.Tensor | None] = {"pos": None, "neg": None}
        if labels is None:
            return lld, samples, entropy

        projected = self.entropy_prj(y)
        flat = labels.view(-1)
        samples = {"pos": projected[flat > 0], "neg": projected[flat < 0]}
        if memory is not None and memory.get("pos") is not None:
            halves = []
            for sign in ("pos", "neg"):
                pooled = torch.cat([*memory[sign], samples[sign]], dim=0)
                centred = pooled - pooled.mean(dim=0)
                covariance = torch.mean(
                    torch.bmm(centred.unsqueeze(-1), centred.unsqueeze(1)), dim=0
                )
                halves.append(torch.logdet(covariance))
            entropy = 0.25 * (halves[0] + halves[1])
        return lld, samples, entropy


class _CPC(nn.Module):
    """Contrastive predictive coding: can `y` pick its own `x` out of the batch?

    Both sides are L2-normalised, so the score is a cosine similarity; the
    negatives are the other samples in the same batch.
    """

    def __init__(self, x_size: int, y_size: int, n_layers: int = 1,
                 activation: str = "Tanh") -> None:
        super().__init__()
        act = getattr(nn, activation)
        if n_layers == 1:
            self.net: nn.Module = nn.Linear(y_size, x_size)
        else:
            layers: list[nn.Module] = [nn.Linear(y_size, x_size), act()]
            layers += [nn.Linear(x_size, x_size) for _ in range(n_layers - 1)]
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        predicted = F.normalize(self.net(y), dim=1)
        x = F.normalize(x, dim=1)
        positive = torch.sum(x * predicted, dim=-1)
        negative = torch.logsumexp(x @ predicted.t(), dim=-1)
        return -(positive - negative).mean()


class _Fusion(nn.Module):
    """Concatenate, two tanh layers, predict. Returns the representation too."""

    def __init__(self, in_size: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.linear_1(self.drop(x)))
        representation = torch.tanh(self.linear_2(hidden))
        return representation, self.linear_3(representation).view(-1)


@register_model("mmim")
class MMIM(MSAModel):
    """Uses **unaligned** data: each stream keeps its own clock and true length."""

    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        return cls(text_dim=spec.text_dim, audio_dim=spec.audio_dim,
                   vision_dim=spec.vision_dim, **kwargs)

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        d_ah: int = 16,
        d_vh: int = 16,
        d_aout: int = 16,
        d_vout: int = 16,
        d_prjh: int = 128,
        n_layer: int = 1,
        cpc_layers: int = 1,
        bidirectional: bool = True,
        dropout_a: float = 0.1,
        dropout_v: float = 0.1,
        dropout_prj: float = 0.1,
        mmilb_mid_activation: str = "ReLU",
        mmilb_last_activation: str = "Tanh",
        cpc_activation: str = "Tanh",
        alpha: float = 0.1,
        beta: float = 0.1,
        mem_size: int = 1,
        add_va: bool = False,
        contrast: bool = True,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.mem_size, self.add_va, self.contrast = mem_size, add_va, contrast

        self.encoder = BertTextEncoder(pretrained, finetune)
        d_tout = self.encoder.hidden_size
        # dropout applies between LSTM layers only, so it is zero at one layer.
        rnn_dropout = (dropout_a if n_layer > 1 else 0.0, dropout_v if n_layer > 1 else 0.0)
        self.audio_enc = _RNNEncoder(audio_dim, d_ah, d_aout, n_layer,
                                     rnn_dropout[0], bidirectional)
        self.vision_enc = _RNNEncoder(vision_dim, d_vh, d_vout, n_layer,
                                      rnn_dropout[1], bidirectional)

        def bound(x: int, y: int) -> _MMILB:
            return _MMILB(x, y, mmilb_mid_activation, mmilb_last_activation)

        # Named with "mi" because the optimiser split keys off that substring —
        # see param_groups.
        self.mi_tv = bound(d_tout, d_vout)
        self.mi_ta = bound(d_tout, d_aout)
        self.mi_va = bound(d_vout, d_aout) if add_va else None

        self.fusion = _Fusion(d_aout + d_vout + d_tout, d_prjh, dropout_prj)
        self.cpc_zt = _CPC(d_tout, d_prjh, cpc_layers, cpc_activation)
        self.cpc_zv = _CPC(d_vout, d_prjh, cpc_layers, cpc_activation)
        self.cpc_za = _CPC(d_aout, d_prjh, cpc_layers, cpc_activation)

        self._init_non_bert()
        self._reset_memory()

    def _init_non_bert(self) -> None:
        """Xavier-normal on everything except BERT, matching the reference.

        Both MMSA and the authors nest this loop inside the one that classifies
        parameters, so each tensor is re-drawn once per remaining parameter —
        O(n^2) draws instead of n. Every draw overwrites the last, so the final
        weights are distributed exactly as one clean pass would leave them; only
        the RNG stream position differs. One pass is written here, which means
        this port is distribution-equivalent to the reference but not
        bit-equivalent. See docs/investigations.md#mmim.
        """
        for name, p in self.named_parameters():
            if "encoder.bert" not in name and p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _reset_memory(self) -> None:
        self._batch = 0
        self._memory: dict[str, dict[str, list[torch.Tensor]]] = {
            pair: {"pos": [], "neg": []} for pair in ("tv", "ta", "va")
        }

    def on_train_epoch_start(self, epoch: int) -> None:
        """The memory bank does not survive an epoch boundary in the reference."""
        self._reset_memory()

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """BERT at a twentieth of the main rate; the MMILB parameters step apart.

        MMSA's MOSI values are main 1e-3, BERT 5e-5, MMILB 1e-3, all with decay
        1e-4 — so `--lr 1e-3 --weight-decay 1e-4` lands every group where the
        reference has it. The MMILB parameters are excluded here because they are
        driven by `auxiliary_optimizer`, and the reference splits them by the
        substring "mi" in the parameter name, which is why they are named so.
        """
        bert, mmilb, main = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (bert if "bert" in name else mmilb if "mi" in name else main).append(p)
        self._mmilb_params = mmilb
        return [
            {"params": bert, "lr": lr * 0.05, "weight_decay": weight_decay},
            {"params": main, "lr": lr, "weight_decay": weight_decay},
        ]

    def auxiliary_optimizer(self, lr: float, weight_decay: float):
        """The MMILB stage: same rate as the main group, its own pass."""
        if not self.contrast:
            return None
        if not hasattr(self, "_mmilb_params"):
            self.param_groups(lr, weight_decay)
        return torch.optim.Adam(self._mmilb_params, lr=lr, weight_decay=weight_decay)

    def _encode(self, batch: dict[str, torch.Tensor]):
        text = self.encoder(batch["text_bert"])[:, 0, :]
        # pack_padded_sequence wants the lengths on the CPU.
        audio = self.audio_enc(batch["audio"], batch["audio_length"].cpu())
        vision = self.vision_enc(batch["vision"], batch["vision_length"].cpu())
        return text, audio, vision

    def auxiliary_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage one: fit the bound only, with no labels and no memory."""
        text, audio, vision = self._encode(batch)
        lld_tv, _, _ = self.mi_tv(text, vision)
        lld_ta, _, _ = self.mi_ta(text, audio)
        lld = lld_tv + lld_ta
        if self.mi_va is not None:
            lld = lld + self.mi_va(vision, audio)[0]
        return -lld

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text, audio, vision = self._encode(batch)
        labels = batch["label"] if self.training else None
        # The memory is read from batch `mem_size` onward; the entropy it feeds
        # is only *used* from `mem_size + 1`. See compute_loss.
        use_memory = self.training and self._batch >= self.mem_size

        pairs, lld, entropy, samples = (
            [("tv", self.mi_tv, text, vision), ("ta", self.mi_ta, text, audio)],
            0.0, 0.0, {},
        )
        if self.mi_va is not None:
            pairs.append(("va", self.mi_va, vision, audio))
        for key, module, x, y in pairs:
            memory = self._memory[key] if use_memory and self._memory[key]["pos"] else None
            pair_lld, pair_samples, pair_entropy = module(x, y, labels, memory)
            lld, entropy, samples[key] = lld + pair_lld, entropy + pair_entropy, pair_samples

        representation, prediction = self.fusion(torch.cat([text, audio, vision], dim=1))
        nce = (self.cpc_zt(text, representation)
               + self.cpc_zv(vision, representation)
               + self.cpc_za(audio, representation))
        return {"M": prediction, "lld": lld, "nce": nce, "entropy": entropy,
                "samples": samples}

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = F.l1_loss(outputs["M"], batch["label"])
        if not self.contrast:
            return loss
        loss = loss + self.alpha * outputs["nce"] - self.beta * outputs["lld"]
        # Strictly greater, not >=: the reference reads the memory one batch
        # before it starts subtracting the entropy computed from it.
        if self._batch > self.mem_size:
            loss = loss - self.beta * outputs["entropy"]
        return loss

    def on_train_batch_end(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        """Push this batch's positive/negative samples into the ring buffer."""
        for key, sample in outputs["samples"].items():
            if sample["pos"] is None:
                continue
            slot = self._memory[key]
            for sign in ("pos", "neg"):
                if len(slot[sign]) < self.mem_size:
                    slot[sign].append(sample[sign].detach())
                else:
                    slot[sign][self._batch % self.mem_size] = sample[sign].detach()
        self._batch += 1
