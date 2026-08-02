"""MFM — Multimodal Factorization Model (Tsai et al., ICLR 2019).

MFM splits the representation in two: a **multimodal discriminative factor**
`F_y`, shared across modalities and carrying what the label needs, and
**modality-specific generative factors** `F_a{1..M}`, each carrying what its own
modality needs to be regenerated. The discriminative head reads only `F_y`; the
decoders read `F_y` together with their own `F_a`.

Structure and objective come from the paper (arXiv:1806.06176); hyper-parameters
come from MMSA's MOSI configuration, with the authors' implementation
(`pliang279/factorized`) as the tie-breaker. `MFN` is reused as the multimodal
encoder for `Z_y`, which is what MMSA does and one of the encoder choices the
paper explores (§2.4: "can be parametrized by any model").

**Claim 2 of the paper is implemented here for the first time in this lineage.**
MMSA hard-codes `missing_loss = 0.0` and never computes it, so the paper's
headline robustness property — reconstructing a missing modality from the
observed ones without losing discriminative performance — has no realisation in
the reference implementation everyone ports from. §2.3 defines it as *surrogate
inference*: a network Φ infers the latent codes from the observed modalities,
and the existing decoders and predictor then run unchanged. "In the presence of
missing modalities, we only need to infer the latent codes rather than the
entire modality." The paper states it uses deterministic mappings for Φ, which
is what `_Surrogate` is.

Two departures from MMSA are parameterised rather than argued about:

* **Learning rate.** MMSA's config sets `learning_rate: 0.002` and its trainer
  builds `optim.Adam(model.parameters(), weight_decay=...)` without passing it,
  so every published MMSA MFM number was trained at Adam's default 1e-3. We take
  the rate from the CLI, and the acceptance group passes 1e-3 to match behaviour
  rather than the config file — the same choice already made for MulT's
  never-applied weight decay.
* **MMD sampling.** MMSA draws `torch.randn` from the global RNG on CPU and then
  moves it to the device. We draw on the device through the seeded generator so
  the run stays bit-reproducible.

Selection here is this repository's default (validation MAE) while MMSA's
`KeyEval` is the composite loss — a known protocol difference, decided
2026-08-02.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .mfn import MemoryFusionNetwork


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dim = x.shape[1]
    tiled_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], dim)
    tiled_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], dim)
    return torch.exp(-(tiled_x - tiled_y).pow(2).mean(2) / float(dim))


def mmd_to_gaussian(z: torch.Tensor) -> torch.Tensor:
    """MMD between the batch of codes and a standard normal sample.

    The paper regularises the latent codes towards a prior; the implementation
    uses MMD rather than a KL term. The reference draws its normal sample from
    the global RNG on CPU; drawing on the device keeps runs reproducible.
    """
    prior = torch.randn(z.shape, device=z.device, dtype=z.dtype)
    return (
        _gaussian_kernel(prior, prior).mean()
        + _gaussian_kernel(z, z).mean()
        - 2.0 * _gaussian_kernel(prior, z).mean()
    )


class _EncoderLSTM(nn.Module):
    """Q(Z_a | X): a sequence to one code (§2.4 uses encoder LSTMs)."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(input_dim, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden)
        c = x.new_zeros(batch, self.hidden)
        for step in range(x.shape[1]):
            h, c = self.cell(x[:, step], (h, c))
        return self.fc(h)


class _DecoderLSTM(nn.Module):
    """F_m: one code back to a sequence of that modality's frames."""

    def __init__(self, code_dim: int, output_dim: int) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(code_dim, code_dim)
        self.fc = nn.Linear(code_dim, output_dim)
        self.code_dim = code_dim

    def forward(self, code: torch.Tensor, steps: int) -> torch.Tensor:
        batch = code.shape[0]
        h = code.new_zeros(batch, self.code_dim)
        c = code.new_zeros(batch, self.code_dim)
        frames = []
        for step in range(steps):
            h, c = self.cell(code if step == 0 else h, (h, c))
            frames.append(h)
        return self.fc(torch.stack(frames, dim=1))


def _factor_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim), nn.ReLU(),
    )


class _Surrogate(nn.Module):
    """Φ of §2.3: latent codes inferred from a *subset* of the modalities.

    Eq. 5 with deterministic mappings. One head per modality that may go
    missing, each taking the codes of the two that remain and producing both the
    missing modality's own code and a replacement for the shared discriminative
    code. Nothing downstream changes: the existing decoders and predictor run on
    whatever codes they are handed.
    """

    def __init__(self, code_dims: dict[str, int], shared_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for missing, dim in code_dims.items():
            observed = sum(d for name, d in code_dims.items() if name != missing)
            self.heads[missing] = nn.Sequential(
                nn.Linear(observed, shared_dim + dim), nn.ReLU(),
                nn.Linear(shared_dim + dim, shared_dim + dim),
            )
        self.shared_dim = shared_dim

    def forward(
        self, missing: str, observed: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.heads[missing](torch.cat(observed, dim=1))
        return out[:, : self.shared_dim], out[:, self.shared_dim :]


@register_model("mfm")
class MultimodalFactorizationModel(MSAModel):
    MODALITIES = ("text", "audio", "vision")

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        zy_size: int = 80,
        zl_size: int = 64,
        za_size: int = 80,
        zv_size: int = 80,
        fy_size: int = 64,
        fl_size: int = 32,
        fa_size: int = 16,
        fv_size: int = 64,
        zy_dropout: float = 0.5,
        zl_dropout: float = 0.5,
        za_dropout: float = 0.7,
        zv_dropout: float = 0.5,
        fy_to_y_dropout: float = 0.0,
        lambda_mmd: float = 100.0,
        lambda_xl: float = 0.5,
        lambda_xa: float = 0.01,
        lambda_xv: float = 0.5,
        lambda_missing: float = 1.0,
        surrogate_inference: bool = False,
        **mfn_kwargs,
    ) -> None:
        super().__init__()
        self.lambdas = {"text": lambda_xl, "audio": lambda_xa, "vision": lambda_xv}
        self.lambda_mmd = lambda_mmd
        self.lambda_missing = lambda_missing
        self.surrogate_inference = surrogate_inference

        dims = {"text": text_dim, "audio": audio_dim, "vision": vision_dim}
        z_sizes = {"text": zl_size, "audio": za_size, "vision": zv_size}
        f_sizes = {"text": fl_size, "audio": fa_size, "vision": fv_size}
        z_drops = {"text": zl_dropout, "audio": za_dropout, "vision": zv_dropout}

        self.encoders = nn.ModuleDict(
            {m: _EncoderLSTM(dims[m], z_sizes[m]) for m in self.MODALITIES}
        )
        self.to_factor = nn.ModuleDict(
            {m: _factor_mlp(z_sizes[m], f_sizes[m], z_drops[m]) for m in self.MODALITIES}
        )
        self.decoders = nn.ModuleDict(
            {m: _DecoderLSTM(fy_size + f_sizes[m], dims[m]) for m in self.MODALITIES}
        )

        self.mfn = MemoryFusionNetwork(text_dim, audio_dim, vision_dim, **mfn_kwargs)
        self.to_zy = nn.Linear(self.mfn.joint_dim, zy_size)
        self.zy_to_fy = _factor_mlp(zy_size, fy_size, zy_dropout)
        self.head = nn.Sequential(
            nn.Linear(fy_size, fy_size), nn.ReLU(),
            nn.Dropout(fy_to_y_dropout), nn.Linear(fy_size, 1),
        )
        self.surrogate = (
            _Surrogate(z_sizes, zy_size) if surrogate_inference else None
        )

    def _codes(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.encoders[m](batch[m]) for m in self.MODALITIES}

    def _decode(
        self, fy: torch.Tensor, factors: dict[str, torch.Tensor], steps: dict[str, int]
    ) -> dict[str, torch.Tensor]:
        return {
            m: self.decoders[m](torch.cat([fy, factors[m]], dim=1), steps[m])
            for m in self.MODALITIES
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        codes = self._codes(batch)
        zy = self.to_zy(self.mfn(batch)["joint"])
        fy = self.zy_to_fy(zy)
        factors = {m: self.to_factor[m](codes[m]) for m in self.MODALITIES}
        steps = {m: batch[m].shape[1] for m in self.MODALITIES}

        out: dict[str, torch.Tensor] = {
            "M": self.head(fy).squeeze(-1),
            "zy": zy,
            **{f"z_{m}": codes[m] for m in self.MODALITIES},
            **{f"recon_{m}": r for m, r in self._decode(fy, factors, steps).items()},
        }

        if self.surrogate is not None and self.training:
            # §2.3: infer the codes of a held-out modality from the other two,
            # then reuse the unchanged decoders and head. Rotating which
            # modality is dropped exercises all three heads every batch.
            for missing in self.MODALITIES:
                observed = [codes[m] for m in self.MODALITIES if m != missing]
                zy_hat, z_hat = self.surrogate(missing, observed)
                fy_hat = self.zy_to_fy(zy_hat)
                sub = dict(factors)
                sub[missing] = self.to_factor[missing](z_hat)
                out[f"missing_{missing}_M"] = self.head(fy_hat).squeeze(-1)
                out[f"missing_{missing}_recon"] = self.decoders[missing](
                    torch.cat([fy_hat, sub[missing]], dim=1), steps[missing]
                )
        return out

    def predict_with_missing(
        self, batch: dict[str, torch.Tensor], missing: str
    ) -> dict[str, torch.Tensor]:
        """Prediction and reconstruction when `missing` is unavailable at test time.

        The point of the surrogate: the missing modality is never read. Its code
        is inferred from the other two, and the ordinary decoder and head run on
        the inferred codes. Used by the missing-modality evaluation, not by
        training.
        """
        if self.surrogate is None:
            raise ValueError("model was built without surrogate_inference=True")
        codes = {m: self.encoders[m](batch[m]) for m in self.MODALITIES if m != missing}
        zy_hat, z_hat = self.surrogate(missing, [codes[m] for m in self.MODALITIES
                                                 if m != missing])
        fy_hat = self.zy_to_fy(zy_hat)
        reconstruction = self.decoders[missing](
            torch.cat([fy_hat, self.to_factor[missing](z_hat)], dim=1),
            batch[missing].shape[1],
        )
        return {"M": self.head(fy_hat).squeeze(-1), f"recon_{missing}": reconstruction}

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        discriminative = F.l1_loss(outputs["M"], batch["label"])
        generative = sum(
            self.lambdas[m] * F.mse_loss(outputs[f"recon_{m}"], batch[m])
            for m in self.MODALITIES
        )
        mmd = sum(
            mmd_to_gaussian(outputs[key])
            for key in ("zy", "z_text", "z_audio", "z_vision")
        )
        loss = discriminative + generative + self.lambda_mmd * mmd

        # Claim 2. Zero in MMSA; here it is the surrogate's own objective --
        # predict the label and rebuild the dropped modality from the two that
        # remain.
        missing = sum(
            F.l1_loss(outputs[f"missing_{m}_M"], batch["label"])
            + self.lambdas[m] * F.mse_loss(outputs[f"missing_{m}_recon"], batch[m])
            for m in self.MODALITIES
            if f"missing_{m}_M" in outputs
        )
        if isinstance(missing, torch.Tensor):
            loss = loss + self.lambda_missing * missing
        return loss
