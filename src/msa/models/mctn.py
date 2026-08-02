"""MCTN — Multimodal Cyclic Translation Network (Pham et al., AAAI 2019).

Every model before this one fuses the three modalities and needs all three at
test time. MCTN asks a different question: can a model *learn* from the
non-verbal modalities during training and then not need them at all? Its answer
is cyclic translation — translate language into vision, translate the result
back, and keep the intermediate representation. Paper equations 20-21: at
inference only the source modality is required.

**Structure and objective come from the paper** (AAAI 2019, pp.6892-6899,
arXiv:1812.07809); the hyper-parameters do not, because the paper states none.
They come from MMSA's MOSI configuration, with the authors' implementation
(`hainow/MCTN`) as the tie-breaker. Every departure below was found by reading
the paper against MMSA line by line; see docs/spec_mctn_mfm.md.

`paper_faithful=False` (the default, used by the acceptance group) reproduces
MMSA so the numbers stay comparable with the reference table. `True` follows the
paper where the two disagree:

* **Translation width.** MMSA zero-pads audio (5) and vision (20) to the text
  width (768) so one Seq2Seq can translate anything into anything. The MSE
  translation loss is then computed in a space where 99.3% of the audio target
  is zeros that exist only because of the padding. The paper translates between
  modalities at their own widths.
* **Teacher forcing.** MMSA's is a no-op: `dec_input = trg[t] if teacher_force
  else top1` with `top1 = trg[t,:]`, so both branches are the same value and the
  decoder always consumes ground truth. `random.random()` is drawn and discarded,
  which still perturbs the RNG stream.
* **Loss weights.** The paper leaves lambda_t and lambda_c as hyper-parameters
  (eq. 18); MMSA hard-codes both to 0.1.

Two further notes recorded rather than implemented:

* The paper decodes with beam search (eq. 12). It is not implemented here and
  not needed: the prediction path reads the *encoder* output, and the
  translation losses are computed against ground-truth targets, so beam search
  would change neither. Documented so nobody assumes it was overlooked.
* Selection here is this repository's default (validation MAE), while MMSA's
  `KeyEval` for MCTN is the composite loss. A known protocol difference, decided
  2026-08-02.

MMSA's `update_epochs`, `contrast`, `mem_size` and `add_va` are dead keys — its
own trainer never reads them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel


class _Encoder(nn.Module):
    """f_theta_e: a sequence to (all hidden states, summary state). Eq. 9-10."""

    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, (h, _) = self.rnn(x)
        # Sum the two directions rather than concatenating, as MMSA does; the
        # paper says only "a recurrent network".
        joint = self.dropout(outputs[:, :, : self.hidden] + outputs[:, :, self.hidden :])
        summary = torch.tanh(self.fc(h[-1] + h[-2]))
        return joint, summary


class _Attention(nn.Module):
    """Bahdanau attention over the encoder states."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, state: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        steps = joint.shape[1]
        state = state.unsqueeze(1).expand(-1, steps, -1)
        energy = torch.tanh(self.attn(torch.cat((state, joint), dim=2)))
        return F.softmax(self.v(energy).squeeze(2), dim=1)


class _Decoder(nn.Module):
    """f_theta_d: one target frame per step, attending over the source. Eq. 11."""

    def __init__(self, output_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hidden = hidden
        self.attention = _Attention(hidden)
        self.rnn = nn.LSTM(output_dim + hidden, hidden, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden * 2, output_dim)

    def forward(
        self, frame: torch.Tensor, state: torch.Tensor, joint: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.attention(state, joint).unsqueeze(1)
        context = torch.bmm(weights, joint)                       # (batch, 1, hidden)
        step_in = torch.cat((frame.unsqueeze(1), context), dim=2)
        output, (h, _) = self.rnn(step_in)
        output = output[:, :, : self.hidden] + output[:, :, self.hidden :]
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, (h[-1] + h[-2])


class _Seq2Seq(nn.Module):
    """One translation direction. Returns the joint embedding and the translation."""

    def __init__(self, encoder: _Encoder, decoder: _Decoder) -> None:
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, teacher_forcing: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint, state = self.encoder(source)
        steps = target.shape[1]
        outputs = torch.zeros(
            target.shape[0], steps, self.decoder.output_dim,
            device=target.device, dtype=target.dtype,
        )
        frame = target[:, 0]
        for step in range(1, steps):
            prediction, state = self.decoder(frame, state, joint)
            outputs[:, step] = prediction
            if teacher_forcing >= 1.0:
                frame = target[:, step]
            else:
                # Sampling on the *whole batch* keeps the decision a function of
                # the seeded RNG rather than of batch composition.
                use_truth = torch.rand((), device=target.device) < teacher_forcing
                frame = target[:, step] if use_truth else prediction.detach()
        return joint, outputs


@register_model("mctn")
class MultimodalCyclicTranslationNetwork(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden: int = 32,
        dropout: float = 0.0,
        lambda_t: float = 0.1,
        lambda_c: float = 0.1,
        teacher_forcing: float = 1.0,
        paper_faithful: bool = False,
        max_seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.paper_faithful = paper_faithful
        self.max_seq_len = max_seq_len
        self.lambda_t, self.lambda_c = lambda_t, lambda_c
        # MMSA always feeds ground truth (its teacher-forcing branch is a no-op),
        # which is teacher_forcing=1.0 here. The paper's Seq2Seq samples.
        self.teacher_forcing = 0.5 if paper_faithful and teacher_forcing >= 1.0 else teacher_forcing

        # Widths the translations operate at. MMSA pads everything to the text
        # width; the paper keeps each modality's own.
        self.vision_width = vision_dim if paper_faithful else text_dim
        self.audio_width = audio_dim if paper_faithful else text_dim
        self.text_width = text_dim

        self.encoder_lv = _Encoder(text_dim, hidden, dropout)
        self.decoder_v = _Decoder(self.vision_width, hidden, dropout)
        self.encoder_a = _Encoder(hidden, hidden, dropout)
        self.decoder_a = _Decoder(self.audio_width, hidden, dropout)
        # Eq. 6-8 reuse one encoder/decoder pair for both directions of the
        # cycle, which only type-checks when source and target share a width.
        # The paper does not say how to reconcile that with modalities of
        # different widths, and the two available references resolve it
        # differently: MMSA zero-pads everything to 768, and the authors' Keras
        # code calls the same encoder object on the decoded tensor
        # (seq2seq/mctn_models.py:92-93), which needs matching widths too.
        #
        # Padding makes 97% of the vision target and 99% of the audio target
        # zeros that exist only because of the padding, so L_t and L_c stop
        # measuring translation quality. The faithful mode instead gives the
        # back-direction its own encoder, keeping both losses in the modalities'
        # real spaces. Weight sharing is what gives way; the losses are what the
        # paper's equations are about. Recorded as an under-specification
        # resolved by us -- see docs/spec_mctn_mfm.md.
        self.decoder_l = _Decoder(text_dim, hidden, dropout) if paper_faithful else self.decoder_v
        self.encoder_vl = (
            _Encoder(self.vision_width, hidden, dropout) if paper_faithful else self.encoder_lv
        )

        self.seq2seq_lv = _Seq2Seq(self.encoder_lv, self.decoder_v)
        self.seq2seq_vl = _Seq2Seq(self.encoder_vl, self.decoder_l)
        self.seq2seq_a = _Seq2Seq(self.encoder_a, self.decoder_a)

        self.regression = _Regression(hidden, dropout)

    def _prepare(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        if not self.paper_faithful:
            vision = F.pad(vision, (0, self.text_width - vision.shape[-1]))
            audio = F.pad(audio, (0, self.text_width - audio.shape[-1]))
        limit = self.max_seq_len
        return text[:, :limit], audio[:, :limit], vision[:, :limit]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text, audio, vision = self._prepare(batch)
        forcing = self.teacher_forcing if self.training else 1.0

        # Level 1: language -> vision, then back (eq. 6-8).
        joint, vision_hat = self.seq2seq_lv(text, vision, forcing)
        _, text_hat = self.seq2seq_vl(vision_hat, text, forcing)
        # Level 2: the joint representation -> acoustics (Figure 3).
        joint2, audio_hat = self.seq2seq_a(joint, audio, forcing)

        return {
            "M": self.regression(joint2).squeeze(-1),
            "vision_hat": vision_hat, "vision_target": vision,
            "text_hat": text_hat, "text_target": text,
            "audio_hat": audio_hat, "audio_target": audio,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Eq. 18: L = lambda_t * L_t + lambda_c * L_c + L_p. MMSA weights all
        # three translations by 0.1 and the prediction by 1.0; the forward
        # translations are L_t and the back-translation into text is L_c.
        translation = (
            F.mse_loss(outputs["vision_hat"], outputs["vision_target"])
            + F.mse_loss(outputs["audio_hat"], outputs["audio_target"])
        )
        cycle = F.mse_loss(outputs["text_hat"], outputs["text_target"])
        prediction = F.l1_loss(outputs["M"], batch["label"])
        return self.lambda_t * translation + self.lambda_c * cycle + prediction


class _Regression(nn.Module):
    """g_w: attention-pooled prediction head over the joint representation."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations, _ = self.rnn(x)
        weights = F.softmax(torch.tanh(self.score(activations)).squeeze(2), dim=1)
        pooled = (activations * weights.unsqueeze(2)).sum(dim=1)
        return self.out(pooled)
