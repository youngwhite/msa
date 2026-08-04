"""ConFEDE — Contrastive Feature Decomposition (Yang et al., ACL 2023).

Every modality is projected twice, into a *similar* and a *dissimilar* view, and
the six views are contrasted against six companion samples drawn per anchor from
similarity pools: two same-label-and-similar, two different-label-and-dissimilar,
two different-label-but-similar. The pools come from a cosine matrix over the
training set that is **rebuilt from the model's own features every other epoch**,
so what counts as a hard example moves as the model learns.

Structure, protocol and hyper-parameters from the authors' release
(`Haoyu-ha/ConFEDE`); full reading and the infrastructure argument in
`docs/spec_confede.md`.

**This is the first model here that needs more than one batch per step**, and the
training loop still does not change. The companion draw happens inside `forward`
using `batch["index"]`, which this repository already provides and which is also
how the release looks companions up; the rebuild hangs off
`on_train_epoch_start`. The loop is shared by twenty groups and changing it would
put every existing number's comparability up for re-argument.

Three things about the release that the paper does not say, all reproduced here
deliberately:

* **BERT is frozen for the whole fusion stage.** The default `train_module` is
  `[False, False, True, True]` and the unfreeze is gated on
  `epoch == finetune_epoch`, which is 200 against a 25-epoch run. The branch
  never fires. DPDF-LQ, DLF and DMD all fine-tune BERT; this one must not.
* **The label-distance weighting of negative pairs is inert.** `cont_NTXentLoss`
  weights negatives by `|label_i - label_j|` only when `update_label` has been
  called, and nothing in the release ever calls it. The loss it actually trains
  with is plain NT-Xent.
* **A fourth candidate pool (`ds`) is built and never sampled.**

Unlike every other model here, ConFEDE reads **unaligned** features and tokenises
**raw text itself** (`max_length=256`, `pooler_output`), rather than consuming
the pre-tokenised `text_bert`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec
from ..registry import register_model
from .base import MSAModel

#: Two from each pool per anchor, in the release's order.
POOLS = ("ss", "dd", "sd")
PER_POOL = 2
COMPANIONS = len(POOLS) * PER_POOL


def padding_mask(features: torch.Tensor) -> torch.Tensor:
    """True where a frame is padding, with a slot prepended for the CLS token.

    The release computes this in its dataloader; computing it here instead keeps
    `msa.data` -- shared by every other group -- untouched. A frame counts as
    padding when it sums to exactly zero, position 0 is always kept, and column 0
    is duplicated to cover the CLS token the positional encoder prepends.
    """
    mask = features.sum(dim=-1) == 0
    mask[:, 0] = False
    return torch.cat((mask[:, 0:1], mask), dim=-1)


class Projector(nn.Module):
    """LayerNorm, one linear, tanh, dropout.

    The release defines this locally inside `TVA_fusion.py` and *also* ships a
    much larger `FeatureProjector` in `model/projector.py`. Only the local one is
    used by the fusion stage; the larger one belongs to the unimodal pretraining
    encoders. Reading the wrong one costs 54 parameter tensors, which is how this
    was caught.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.5) -> None:
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
    """Projects into the transformer's width, prepends a CLS token, adds positions.

    The projection lives here rather than in the encoder that owns it: the
    encoder declares an `fc` of the right shape and never calls it.
    """

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


class SequenceEncoder(nn.Module):
    """Vision and audio: positions, a transformer, a layer norm, then a mean."""

    def __init__(self, fea_size: int, hidden: int, patches: int, heads: int,
                 layers: int, dropout: float) -> None:
        super().__init__()
        self.pos_encoder = _PositionEncodingTraining(fea_size, hidden, patches, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden,
            dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(layer, layers)
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x).transpose(0, 1)
        x = self.transformer_encoder(x, mask=None, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x.transpose(0, 1))
        return torch.mean(x, dim=-2)


class TextEncoder(nn.Module):
    """BERT over raw strings, taking `pooler_output`.

    Not the tokenised `text_bert` every other model here uses: the release
    tokenises inside the encoder at `max_length=256` with per-batch padding.
    """

    def __init__(self, pretrained: str = "bert-base-uncased") -> None:
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.extractor = BertModel.from_pretrained(pretrained)

    def forward(self, text: list[str]) -> torch.Tensor:
        device = next(self.extractor.parameters()).device
        tokens = self.tokenizer(list(text), padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(device)
        return self.extractor(**tokens)["pooler_output"]


class SimilarityPools:
    """The companion pools, rebuilt from features on demand.

    `ds` is built by the release and never sampled; it is not built here.
    """

    def __init__(self, labels: np.ndarray, depth: int = 10) -> None:
        self.labels = np.round(labels)
        self.depth = depth
        self.pools: dict[str, list[list[int]]] = {}

    def rebuild(self, features: torch.Tensor) -> None:
        size = features.size(0)
        normalised = F.normalize(features.float(), dim=-1, eps=1e-6)
        order = torch.argsort(normalised @ normalised.t(), dim=1, descending=True)
        pools: dict[str, list[list[int]]] = {name: [] for name in POOLS}
        for i in range(size):
            row = order[i].tolist()
            same, different = [], []
            for j in row:
                if i == j:
                    continue
                (same if self.labels[i] == self.labels[j] else different).append(j)
            pools["ss"].append(same[:self.depth])
            pools["sd"].append(different[:self.depth])
            # dd is the same partition read from the far end: least similar first.
            pools["dd"].append([j for j in reversed(row)
                                if j != i and self.labels[i] != self.labels[j]][:self.depth])
        self.pools = pools

    def draw(self, indices: list[int], rng: random.Random) -> list[int]:
        drawn: list[int] = []
        for i in indices:
            for name in POOLS:
                pool = self.pools[name][i]
                drawn += rng.sample(pool, PER_POOL) if len(pool) >= PER_POOL \
                    else [pool[0] if pool else i] * PER_POOL
        return drawn


#: The release hardcodes these index tuples for a 42-row block: seven rows
#: (anchor plus six companions) for each of the six projections. Copied verbatim
#: -- they encode which pairs count as positive and which as negative.
ANCHOR1 = [0, 0, 7, 7, 14, 14, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
POSITIVE = [1, 2, 8, 9, 15, 16, 7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20]
ANCHOR2 = [0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
           0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
NEGATIVE = [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
            21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27, 34, 41]
PAIR_LABELS = [0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
               5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9]


@register_model("confede")
class ConFEDE(MSAModel):
    def __init__(
        self,
        vision_dim: int = 20,
        audio_dim: int = 5,
        vision_length: int = 500,
        audio_length: int = 375,
        encoder_dim: int = 768,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.5,
        temperature: float = 0.5,
        pretrained: str = "bert-base-uncased",
        rebuild_every: int = 2,
        pretrain_seed: int | None = None,
    ) -> None:
        super().__init__()
        from pytorch_metric_learning.losses import NTXentLoss

        self.rebuild_every = rebuild_every
        self.text_encoder = TextEncoder(pretrained)
        self.vision_encoder = SequenceEncoder(
            vision_dim, encoder_dim, vision_length, heads, layers, dropout)
        self.audio_encoder = SequenceEncoder(
            audio_dim, encoder_dim, audio_length, heads, layers, dropout)

        width = encoder_dim // 2
        self.T_simi_proj = Projector(encoder_dim, width)
        self.V_simi_proj = Projector(encoder_dim, width)
        self.A_simi_proj = Projector(encoder_dim, width)
        self.T_dissimi_proj = Projector(encoder_dim, width)
        self.V_dissimi_proj = Projector(encoder_dim, width)
        self.A_dissimi_proj = Projector(encoder_dim, width)

        hidden = [width * 2, width, width // 2, width // 4]
        self.TVA_decoder = BaseClassifier(width * 6, hidden, 1)
        self.mono_decoder = BaseClassifier(width, hidden[2:], 1)
        self.ntxent_loss = NTXentLoss(temperature=temperature)

        # BERT is frozen for the entire fusion stage; see the module docstring.
        for parameter in self.text_encoder.extractor.parameters():
            parameter.requires_grad = False

        if pretrain_seed is not None:
            self.load_pretrained_encoders(pretrain_seed)

        self.pools: SimilarityPools | None = None
        self._train_tensors: dict[str, object] | None = None
        self._rng = random.Random(0)
        self._indices: torch.Tensor | None = None

    def on_run_start(self, seed: int) -> None:
        """Produce this seed's stage one if absent, load it, and drop the rest.

        The release pretrains and fuses under one seed, so a seed here means a
        whole pipeline and stage one is part of what varies -- sharing one
        pretraining across ten fusion runs would report a spread narrower than
        the method actually has. Ten text encoders at 418MB do not fit on this
        disk, so they are made and discarded one at a time instead.
        """
        import subprocess
        import sys
        from pathlib import Path as _Path

        from msa.config import OUTPUT_ROOT

        root = OUTPUT_ROOT / "confede_pretrain"
        current = root / f"seed{seed}"
        needed = [current / f"{m}_encoder.pt" for m in ("text", "vision", "audio")]
        if not all(path.exists() for path in needed):
            script = _Path(__file__).resolve().parents[3] / "scripts" / "pretrain_confede.py"
            print(f"  stage one for seed {seed} is missing; running {script.name}",
                  flush=True)
            subprocess.run([sys.executable, str(script), "--seed", str(seed)], check=True)
        self.load_pretrained_encoders(seed)
        # Bound the disk at one seed's worth: the previous seed's weights are
        # reproducible from the script and nothing downstream reads them again.
        for other in sorted(root.glob("seed*")):
            if other != current:
                for stale in other.glob("*.pt"):
                    stale.unlink()

    def load_pretrained_encoders(self, seed: int) -> None:
        """Load stage one, which is the release's `load_pretrain=True`.

        Refuses rather than warns when the weights are absent. Skipping stage one
        is not a small degradation to shrug at: without it the ten-seed run came
        out at MAE 1.1549, worse than this repository's LSTM baseline, because
        the vision and audio transformers never leave their random
        initialisation. A missing file must stop the run, not quietly produce a
        number that looks like ConFEDE and is not.
        """
        from msa.config import OUTPUT_ROOT

        root = OUTPUT_ROOT / "confede_pretrain" / f"seed{seed}"
        for modality, module in (("text", self.text_encoder),
                                 ("vision", self.vision_encoder),
                                 ("audio", self.audio_encoder)):
            path = root / f"{modality}_encoder.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"no pretrained {modality} encoder at {path}. Run\n"
                    f"    python scripts/pretrain_confede.py --seed {seed}\n"
                    f"first -- ConFEDE without stage one is not ConFEDE.")
            module.load_state_dict(torch.load(path, map_location="cpu"))

    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        """Construct, and load the training split this model draws companions from.

        Loading here rather than through a new trainer hook is deliberate. The
        `build(spec)` contract already exists for models whose construction
        depends on the data -- Self-MM takes `train_size` through it -- and the
        decision recorded in docs/decisions.md is that ConFEDE carries its extra
        needs internally rather than the shared loop growing a feature for one
        model. It reads the same pickle `msa.data` reads, named by the same spec.
        """
        kwargs.setdefault("vision_dim", spec.vision_dim)
        kwargs.setdefault("audio_dim", spec.audio_dim)
        model = cls(**kwargs)
        model.attach_training_set(spec.unaligned_pkl)
        return model

    def attach_training_set(self, pickle_path) -> None:
        """Hold the training split so companions can be gathered.

        Kept as tensors on the CPU and moved per draw. The text alone is ~197MB;
        it is referenced, not copied.
        """
        import pickle

        with open(pickle_path, "rb") as handle:
            train = pickle.load(handle)["train"]
        vision = torch.as_tensor(train["vision"]).float()
        audio = torch.as_tensor(train["audio"]).float()
        self._train_tensors = {
            "raw_text": list(train["raw_text"]),
            "vision": vision,
            "audio": audio,
            "vision_padding_mask": padding_mask(vision),
            "audio_padding_mask": padding_mask(audio),
            "labels": torch.as_tensor(train["regression_labels"]).float(),
        }
        self.pools = SimilarityPools(np.asarray(train["regression_labels"]))

    def on_train_epoch_start(self, epoch: int) -> None:
        """Rebuild the pools on odd epochs, as the release does."""
        if self.pools is None or self._train_tensors is None:
            return
        if epoch % self.rebuild_every != 1:
            return
        self._rng = random.Random(epoch)
        self.pools.rebuild(self._encode_training_set())

    @torch.no_grad()
    def _encode_training_set(self, chunk: int = 32) -> torch.Tensor:
        was_training = self.training
        self.eval()
        tensors = self._train_tensors
        assert tensors is not None
        device = next(self.vision_encoder.parameters()).device
        size = len(tensors["raw_text"])
        features = []
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            features.append(torch.cat(self._encode(
                list(tensors["raw_text"][start:stop]),
                tensors["vision"][start:stop].to(device).float(),
                tensors["audio"][start:stop].to(device).float(),
                tensors["vision_padding_mask"][start:stop].to(device),
                tensors["audio_padding_mask"][start:stop].to(device),
            ), dim=-1).cpu())
        if was_training:
            self.train()
        return torch.cat(features, dim=0)

    def _encode(self, text: list[str], vision: torch.Tensor, audio: torch.Tensor,
                vision_mask: torch.Tensor, audio_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.text_encoder(text),
                self.vision_encoder(vision, vision_mask),
                self.audio_encoder(audio, audio_mask))

    def _project(self, embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                 ) -> list[torch.Tensor]:
        text, vision, audio = embeddings
        return [self.T_simi_proj(text), self.V_simi_proj(vision), self.A_simi_proj(audio),
                self.T_dissimi_proj(text), self.V_dissimi_proj(vision),
                self.A_dissimi_proj(audio)]

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Two groups, as the release's AdamW has: no decay on bias or LayerNorm.

        Only parameters that require a gradient are handed over -- BERT is frozen
        here, and passing frozen tensors to an optimiser with weight decay would
        decay them anyway.
        """
        no_decay = ("bias", "LayerNorm.weight")
        decayed, plain = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (plain if any(n in name for n in no_decay) else decayed).append(parameter)
        return [{"params": decayed, "lr": lr, "weight_decay": weight_decay},
                {"params": plain, "lr": lr, "weight_decay": 0.0}]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vision, audio = batch["vision"].float(), batch["audio"].float()
        anchor = self._project(self._encode(
            batch["raw_text"], vision, audio,
            padding_mask(vision), padding_mask(audio)))
        out = {"M": torch.cat(anchor, dim=-1)}
        out["prediction"] = self.TVA_decoder(out["M"]).squeeze(-1)
        out["views"] = torch.stack(anchor)

        companions = self._companions(batch)
        if companions is not None:
            drawn, labels = companions
            out["companion_views"] = torch.stack(drawn)
            out["companion_labels"] = labels
            out["companion_prediction"] = self.TVA_decoder(
                torch.cat(drawn, dim=-1)).squeeze(-1)
        out["M"] = out["prediction"]
        return out

    def _companions(self, batch: dict[str, torch.Tensor]):
        if not self.training or self.pools is None or not self.pools.pools:
            return None
        tensors = self._train_tensors
        assert tensors is not None
        device = batch["vision"].device
        indices = self.pools.draw(batch["index"].tolist(), self._rng)
        drawn = self._project(self._encode(
            [tensors["raw_text"][i] for i in indices],
            tensors["vision"][indices].to(device).float(),
            tensors["audio"][indices].to(device).float(),
            tensors["vision_padding_mask"][indices].to(device),
            tensors["audio_padding_mask"][indices].to(device),
        ))
        labels = tensors["labels"][indices].to(device).float()
        return drawn, labels

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        label = batch["label"]
        views = outputs["views"]

        if "companion_views" not in outputs:
            prediction = torch.cat([outputs["prediction"]])
            mono = self.mono_decoder(views.reshape(-1, views.size(-1))).squeeze(-1)
            return (F.mse_loss(prediction, label)
                    + 0.01 * F.mse_loss(mono, label.repeat(views.size(0))))

        companion_views = outputs["companion_views"]
        companion_label = outputs["companion_labels"]

        # Every projection is scored, anchors and companions alike.
        prediction = torch.cat([outputs["prediction"], outputs["companion_prediction"]])
        prediction_loss = F.mse_loss(prediction, torch.cat([label, companion_label]))

        mono_views = torch.cat([views.reshape(-1, views.size(-1)),
                                companion_views.reshape(-1, companion_views.size(-1))])
        mono_labels = torch.cat([label.repeat(views.size(0)),
                                 companion_label.repeat(companion_views.size(0))])
        mono_loss = F.mse_loss(self.mono_decoder(mono_views).squeeze(-1), mono_labels)

        device = views.device
        indices_tuple = tuple(torch.tensor(x, device=device)
                              for x in (ANCHOR1, POSITIVE, ANCHOR2, NEGATIVE))
        pair_labels = torch.tensor(PAIR_LABELS, device=device)
        contrastive = views.new_zeros(())
        for i in range(views.size(1)):
            block = torch.cat([
                torch.cat((views[v, i].unsqueeze(0),
                           companion_views[v, COMPANIONS * i:COMPANIONS * (i + 1)]), dim=0)
                for v in range(views.size(0))], dim=0)
            contrastive = contrastive + self.ntxent_loss(
                block, pair_labels, indices_tuple=indices_tuple)
        contrastive = contrastive / views.size(1)

        return prediction_loss + 0.1 * contrastive + 0.01 * mono_loss
