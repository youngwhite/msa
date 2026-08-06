"""FeaDA (Yin et al., IJCNLP-AACL 2025).

Text queries vision and audio through two cross-modal transformers, each of
which reads its modality with an additive prompt. In parallel, frozen pretrained
unimodal encoders produce vision and audio summaries; those are projected into
similar/dissimilar views for a contrastive term, and the similar views also
**gate** the cross-modal embeddings by elementwise product. A KL term then pulls
each gated embedding towards the projection that gated it.

Structure and protocol from the authors' release
(`PowerLittleYin/FeaDA-main`). **That repository ships no LICENSE**, so rights
are reserved by default; reading it, reimplementing from it and running it
locally as a reference is ordinary academic practice, and it is not MIT. Full
reading in `docs/spec_feada.md`.

**It is ConFEDE's successor** -- same config names, same companion-sampling
dataloader, same 42-row contrastive block -- as DLF is DMD's. Nothing is shared
with `confede.py` all the same: that model is verified and committed, and a
shared edit would put its numbers at risk.

Three things reproduced deliberately, none of them in the paper:

* **Seven modules are used in forward and never trained.** The release's
  optimiser has four groups -- text, vision, audio, and anything whose name
  contains `_decoder` -- and the six similar/dissimilar projections plus `p2a`
  fall outside all four. They take gradients and no optimiser ever steps them,
  so they stay at their random initialisation. This is not the deliberate
  `set_froze()` applied to the pretrained encoders; it is an omission from the
  grouping, and it means **the contrastive loss operates through random
  projections**. `param_groups` below reproduces it.
* **The unimodal encoders' second output is discarded.** `VisionEncoder` returns
  `(last_h, x)` and the fusion stage binds only `x`, so the whole
  `trans_encoder_a` branch runs each step and reaches nothing.
* Vision and audio are pretrained and frozen; **text is not pretrained at all**
  (the release has `Vtrain.py` and `Atrain.py` and no text equivalent).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder
from .transformers import TransformerEncoder

#: config.py, MOSI. Loss weights are from the release's own line, not the paper.
CONST_WEIGHT, MONO_WEIGHT, KL_WEIGHT, TEMPERATURE = 0.02, 0.03, 0.09, 0.5
COMPANIONS = 6


class _Extractor(nn.Module):
    """LayerNorm, one linear, tanh, dropout.

    The release ships this twice under two names -- `common_feature_extractor`
    (dropout 0.3) for the similar views and `private_feature_extractor`
    (dropout 0.5) for the dissimilar ones -- with identical bodies. The only
    difference is the dropout rate, so it is one class with a parameter here and
    the two rates are passed at the call sites.

    Also worth stating: this is NOT the plain Linear a first pass here assumed.
    The LayerNorm is two more tensors per projection, twelve across the six, and
    the equivalence test's count was the only thing that said so.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BaseClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: list[int], output_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, width in enumerate(hidden_size):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size[i - 1], width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.MLP = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)


class _PositionEncodingTraining(nn.Module):
    """Projects to the transformer width, prepends CLS, adds learned positions."""

    def __init__(self, fea_size: int, hidden: int, patches: int, dropout: float) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, hidden))
        self.proj = nn.Linear(fea_size, hidden)
        self.position_embeddings = nn.Parameter(torch.zeros(1, patches + 1, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return self.dropout(torch.cat((cls, x), dim=1) + self.position_embeddings)


class UnimodalEncoder(nn.Module):
    """The pretrained, frozen vision/audio encoder.

    Returns only the mean-pooled summary. The release also builds a second
    branch (`proj_a` into `trans_encoder_a`) whose output the fusion stage binds
    and never reads, so it is not built here -- and is excluded from the
    equivalence test with everything else the release leaves dangling.
    """

    def __init__(self, fea_size: int, hidden: int, patches: int, heads: int,
                 layers: int, dropout: float) -> None:
        super().__init__()
        # The release declares layernorm before the encoder it wraps (its proj_a
        # and trans_encoder_a sit between, and are the dead branch), so a
        # positional weight copy needs the same order here.
        self.layernorm = nn.LayerNorm(hidden)
        self.pos_encoder = _PositionEncodingTraining(fea_size, hidden, patches, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden,
            dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x).transpose(0, 1)
        x = self.transformer_encoder(x, mask=None, src_key_padding_mask=key_padding_mask)
        return torch.mean(self.layernorm(x.transpose(0, 1)), dim=-2)


def padding_mask(features: torch.Tensor) -> torch.Tensor:
    """True where a frame is padding, with a slot prepended for the CLS token."""
    mask = features.sum(dim=-1) == 0
    mask[:, 0] = False
    return torch.cat((mask[:, 0:1], mask), dim=-1)


@register_model("feada")
class FeaDA(MSAModel):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 5,
        vision_dim: int = 20,
        audio_length: int = 375,
        vision_length: int = 500,
        width: int = 768,
        cross_heads: int = 8,
        vision_cross_layers: int = 2,
        audio_cross_layers: int = 5,
        unimodal_heads: int = 8,
        unimodal_layers: int = 2,
        cross_dropout: float = 0.1,
        unimodal_dropout: float = 0.5,
        text_dropout: float = 0.5,
        pretrained: str = "bert-base-uncased",
        finetune_bert: bool = True,
    ) -> None:
        super().__init__()
        self.text_dropout = text_dropout
        half = width // 2

        # Declared in the release's own order: the six projections come first,
        # before the prompts and encoders. A positional weight copy is what the
        # equivalence test does, and grouping these more readably misaligned it.
        # Similar views use the release's common_feature_extractor at dropout
        # 0.3; dissimilar use private_feature_extractor at 0.5. Same body.
        self.T_simi_proj = _Extractor(width, half, 0.3)
        self.V_simi_proj = _Extractor(width, half, 0.3)
        self.A_simi_proj = _Extractor(width, half, 0.3)
        self.T_dissimi_proj = _Extractor(width, half, 0.5)
        self.V_dissimi_proj = _Extractor(width, half, 0.5)
        self.A_dissimi_proj = _Extractor(width, half, 0.5)

        self.prompta_m = nn.Parameter(torch.rand(audio_length, width))
        self.promptv_m = nn.Parameter(torch.rand(vision_length, width))
        self.text_encoder = BertTextEncoder(pretrained, finetune_bert)
        self.proj_t = nn.Linear(text_dim, width)
        self.proj_v = nn.Linear(vision_dim, width)
        self.vision_with_text = TransformerEncoder(
            embed_dim=width, num_heads=cross_heads, layers=vision_cross_layers,
            attn_dropout=cross_dropout, relu_dropout=cross_dropout,
            res_dropout=cross_dropout, embed_dropout=cross_dropout, attn_mask=True,
            # Its feed-forward is embed_dim wide, not the 4x every earlier model
            # here uses. Reusing the default would have been a silently larger
            # model that still trained and still looked right.
            ffn_dim=width,
            # Its transformer builds SinusoidalPositionalEmbedding unconditionally
            # -- no switch. Our shared encoder defaults the switch OFF, which is
            # exactly how MulT lost its positional encoding here
            # (investigations.md#mult-position). Made that mistake again writing
            # this; the equivalence test caught it at 3.05e-03 relative.
            position_embedding=True)
        self.proj_a = nn.Linear(audio_dim, width)
        # Five layers for audio against two for vision -- the release's asymmetry.
        self.audio_with_text = TransformerEncoder(
            embed_dim=width, num_heads=cross_heads, layers=audio_cross_layers,
            attn_dropout=cross_dropout, relu_dropout=cross_dropout,
            res_dropout=cross_dropout, embed_dropout=cross_dropout, attn_mask=True,
            ffn_dim=width, position_embedding=True)

        self.vision_encoder = UnimodalEncoder(
            vision_dim, width, vision_length, unimodal_heads, unimodal_layers, unimodal_dropout)
        self.audio_encoder = UnimodalEncoder(
            audio_dim, width, audio_length, unimodal_heads, unimodal_layers, unimodal_dropout)

        self.p2a = nn.Linear(half, width)

        self.TVA_decoder = BaseClassifier(width * 3, [width, width // 2, width // 8], 1)
        self.mono_decoder = BaseClassifier(half, [width // 4, width // 8], 1)

    def on_run_start(self, seed: int) -> None:
        """Produce this seed's stage one if absent, load it, freeze it, drop the rest.

        Raises rather than warns when it cannot be produced: ConFEDE without its
        pretraining returned MAE 1.1549 against 0.7358, a number that would have
        sat in the comparison table looking ordinary.
        """
        import subprocess
        import sys
        from pathlib import Path as _Path

        from msa.config import OUTPUT_ROOT

        root = OUTPUT_ROOT / "feada_pretrain"
        current = root / f"seed{seed}"
        needed = [current / f"{m}_encoder.pt" for m in ("vision", "audio")]
        if not all(path.exists() for path in needed):
            script = _Path(__file__).resolve().parents[3] / "scripts" / "pretrain_feada.py"
            print(f"  stage one for seed {seed} is missing; running {script.name}", flush=True)
            subprocess.run([sys.executable, str(script), "--seed", str(seed)], check=True)
        for modality, module in (("vision", self.vision_encoder),
                                 ("audio", self.audio_encoder)):
            path = current / f"{modality}_encoder.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"no pretrained {modality} encoder at {path}. Run\n"
                    f"    python scripts/pretrain_feada.py --seed {seed}\n"
                    f"first -- FeaDA without stage one is not FeaDA.")
            module.load_state_dict(torch.load(path, map_location="cpu"))
        # set_froze(), as the release calls it right after loading.
        for module in (self.vision_encoder, self.audio_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for other in sorted(root.glob("seed*")):
            if other != current:
                for stale in other.glob("*.pt"):
                    stale.unlink()

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Four groups at three rates -- and seven modules in none of them.

        The omission is the release's, not a simplification here: its
        `model_params_other` collects only names containing `_decoder`, so the
        six projections and `p2a` never reach the optimiser. Reproduced, because
        correcting it would make this a different paper; see
        `docs/spec_feada.md`. The command line's `lr` and `weight_decay` are
        ignored, as the rates are per-module in its config.
        """
        return [
            {"params": [*self.text_encoder.parameters(), *self.proj_t.parameters()],
             "lr": 5e-5, "weight_decay": 1e-3},
            {"params": [*self.proj_a.parameters(), *self.audio_with_text.parameters(),
                        self.prompta_m], "lr": 1e-3, "weight_decay": 1e-3},
            {"params": [*self.proj_v.parameters(), *self.vision_with_text.parameters(),
                        self.promptv_m], "lr": 1e-3, "weight_decay": 1e-3},
            {"params": [*self.TVA_decoder.parameters(), *self.mono_decoder.parameters()],
             "lr": 1e-3, "weight_decay": 1e-3},
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Both outputs, and they are not interchangeable: the release projects
        # `last_hidden_state` for the cross-modal streams and feeds
        # `pooler_output` -- CLS through BERT's own dense+tanh -- to the six
        # projections. Taking hidden[:, 0] for the second would silently drop
        # that pooling layer, which is a different vector, not a shortcut to the
        # same one. `BertTextEncoder` returns only the sequence, so the pooler is
        # read from the wrapped model.
        tokens = batch["text_bert"]
        output = self.text_encoder.bert(
            input_ids=tokens[:, 0].long(), attention_mask=tokens[:, 1].long(),
            token_type_ids=tokens[:, 2].long())
        hidden, pooled = output.last_hidden_state, output.pooler_output
        text = F.dropout(self.proj_t(hidden.permute(1, 0, 2)),
                         p=self.text_dropout, training=self.training)

        vision, audio = batch["vision"].float(), batch["audio"].float()
        projected_vision = self.proj_v(vision).permute(1, 0, 2) + self.promptv_m.unsqueeze(1)
        projected_audio = self.proj_a(audio).permute(1, 0, 2) + self.prompta_m.unsqueeze(1)
        vision_embed = self.vision_with_text(text, projected_vision, projected_vision)[0]
        audio_embed = self.audio_with_text(text, projected_audio, projected_audio)[0]

        with torch.no_grad() if not self.training else torch.enable_grad():
            xv = self.vision_encoder(vision, padding_mask(vision))
            xa = self.audio_encoder(audio, padding_mask(audio))

        views = {
            "t_simi": self.T_simi_proj(pooled), "v_simi": self.V_simi_proj(xv),
            "a_simi": self.A_simi_proj(xa), "t_dissimi": self.T_dissimi_proj(pooled),
            "v_dissimi": self.V_dissimi_proj(xv), "a_dissimi": self.A_dissimi_proj(xa),
        }
        # The gate: the similar views are widened by p2a and multiplied in. The
        # PRE-p2a copies are what the contrastive term uses.
        gate_v, gate_a = self.p2a(views["v_simi"]), self.p2a(views["a_simi"])
        vision_gated, audio_gated = vision_embed * gate_v, audio_embed * gate_a

        fused = torch.cat([text[0], audio_gated, vision_gated], dim=-1)
        out = {"M": self.TVA_decoder(fused).view(-1)}
        out.update({f"view_{k}": v for k, v in views.items()})
        out["gate_v"], out["gate_a"] = gate_v, gate_a
        out["vision_gated"], out["audio_gated"] = vision_gated, audio_gated
        return out

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        label = batch["label"]
        prediction = F.mse_loss(outputs["M"], label)

        stacked = torch.cat([outputs[f"view_{k}"] for k in
                             ("t_simi", "v_simi", "a_simi",
                              "t_dissimi", "v_dissimi", "a_dissimi")], dim=0)
        mono = F.mse_loss(self.mono_decoder(stacked).squeeze(-1), label.repeat(6))

        distillation = (_kl(outputs["vision_gated"], outputs["gate_v"])
                        + _kl(outputs["audio_gated"], outputs["gate_a"]))
        return prediction + MONO_WEIGHT * mono + KL_WEIGHT * distillation


def _kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """The release's `get_KL_loss`: log-softmax the gated embedding, softmax the
    projection that gated it, then `KLDivLoss(reduction='batchmean')`.

    Worth stating plainly: the teacher here is `p2a` applied to a projection, and
    neither is ever trained, so this term pulls a learned embedding towards a
    fixed random transform of a fixed random projection."""
    return F.kl_div(F.log_softmax(student, dim=1), F.softmax(teacher, dim=1),
                    reduction="batchmean")
