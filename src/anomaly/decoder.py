"""Reconstruction-error anomaly component (CLAUDE.md §6 step 11, Day 10).

Hangs a small decoder off the FROZEN Tier A LC-only encoder
(`outputs/train/checkpoint.pt`, `src.models.lc_encoder.LCEncoder` -- "the
Tier A encoder": trained on all 843,617 objects, has seen negative flux,
unlike the Tier B fine-tune -- see this session's chat report for why this
specific checkpoint, not the LC+static one, was chosen: the reconstruction
target here is the light curve alone, and `LCEncoder` is literally the
encoder, as opposed to `LCStaticFusionModel`, a wrapper around one).

-----------------------------------------------------------------------------
Why "predict x_t from h_{t-1}", not "encode the whole sequence then decode it"
-----------------------------------------------------------------------------
A textbook seq2seq autoencoder (bottleneck = final hidden state over the
*entire* sequence, decoder reconstructs every timestep from that one vector)
would let early-timestep reconstructions see the whole future through the
bottleneck -- exactly what CLAUDE.md §2.3 rules out ("no attention mask that
permits attending to future timesteps," stated as absolute, not scoped to
the classifier alone). It would also make reconstruction close to trivial:
`LCEncoder`'s own hidden state at t already contains x_t (the GRU just
consumed it), so reconstructing x_t from a summary that necessarily includes
h_t is answering a question the model already knows the answer to --
reconstruction error would sit near zero for everyone, normal and anomalous
alike, and carry no signal.

Design used instead: predict each timestep's flux-derived features
(`flux_norm`, `flux_abs_norm`, `flux_err_norm`) from the encoder's hidden
state as of the *previous* timestep (`h_{t-1}`, zeros at t=0 -- "nothing
known yet," the same state a live GRU starts from) plus the current
timestep's exogenous, schedule-level fields (`delta_t_days`,
`wavelength_norm`, `is_detection` -- known in advance, since they describe
*when* and *in what band* an observation happens, not what flux it reads).
This is a one-step-ahead predictive-coding model: "given everything up to
t-1, and knowing we are about to be observed at time t in this band, what
flux do we expect for an ordinary Ia/Ib-c/II?" It is causal by construction
(h_{t-1} depends only on x_{<=t-1}) and cannot trivially cheat (h_{t-1} never
contains x_t), so error on genuinely unusual light-curve behavior should
stand out from error on ordinary ones -- exactly what an anomaly score needs.

Trained on the 3 normal classes only (2222/2223/2224), per CLAUDE.md's
instruction ("trained to reconstruct light curves of the three normal
classes only") -- an object whose flux behavior an ordinary-SN-trained
decoder cannot predict is the anomaly signal.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.lc_encoder import LCEncoder

# Indices into src.features.tensorize.FEATURE_NAMES = (delta_t_days,
# flux_norm, flux_abs_norm, flux_err_norm, wavelength_norm, is_detection).
_COND_IDX = (0, 4, 5)  # delta_t_days, wavelength_norm, is_detection
_TARGET_IDX = (1, 2, 3)  # flux_norm, flux_abs_norm, flux_err_norm
NUM_COND_FEATURES = len(_COND_IDX)
NUM_TARGET_FEATURES = len(_TARGET_IDX)


@torch.no_grad()
def encoder_trunk_hidden_states(
    encoder: LCEncoder, features: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Replicates `LCEncoder.forward` up to (and including) its `layer_norm`
    + `dropout`, stopping short of `encoder.head` -- the per-timestep hidden
    state the reconstruction decoder actually consumes. `encoder` is always
    frozen/eval when this is called (see `ReconstructionDecoder` docstring),
    hence `@torch.no_grad()` here rather than relying on callers to remember.
    """
    encoder.eval()
    packed = pack_padded_sequence(
        features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
    )
    packed_out, _ = encoder.gru(packed)
    out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=features.shape[1])
    out = encoder.layer_norm(out)
    out = encoder.dropout(out)
    return out


def shift_right(hidden: torch.Tensor) -> torch.Tensor:
    """`(B, T, H) -> (B, T, H)`: position t holds position t-1's value;
    position 0 is zeros ("nothing known yet", the same state a fresh GRU
    starts from)."""
    shifted = torch.zeros_like(hidden)
    shifted[:, 1:, :] = hidden[:, :-1, :]
    return shifted


class ReconstructionDecoder(nn.Module):
    """MLP: `[h_{t-1} (encoder_hidden_size,); delta_t; wavelength_norm;
    is_detection]` -> predicted `[flux_norm, flux_abs_norm, flux_err_norm]`.

    Not itself a GRU -- there is no need for the decoder to carry its own
    recurrent state; all the causal history it needs already lives in
    `h_{t-1}`, computed once by the frozen encoder trunk.
    """

    def __init__(
        self, encoder_hidden_size: int = 128, hidden_size: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden_size + NUM_COND_FEATURES, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_TARGET_FEATURES),
        )

    def forward(self, prev_hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """`prev_hidden`: `(B, T, encoder_hidden_size)`. `cond`: `(B, T,
        NUM_COND_FEATURES)`. Returns `(B, T, NUM_TARGET_FEATURES)`."""
        return self.mlp(torch.cat([prev_hidden, cond], dim=-1))


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Scalar mean squared error over every real (unpadded) timestep,
    averaged across all `NUM_TARGET_FEATURES` channels equally."""
    diff2 = (pred - target) ** 2  # (B, T, NUM_TARGET_FEATURES)
    per_timestep = diff2.mean(dim=-1)  # (B, T)
    return per_timestep[mask].mean()


@torch.no_grad()
def per_object_reconstruction_error(
    encoder: LCEncoder,
    decoder: ReconstructionDecoder,
    features: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """`(B,)` mean squared reconstruction error per object, over that
    object's own real (unpadded) timesteps only -- the reconstruction-error
    anomaly score. Higher = more anomalous (worse predicted)."""
    hidden = encoder_trunk_hidden_states(encoder, features, lengths)
    prev_hidden = shift_right(hidden)
    cond = features[:, :, _COND_IDX]
    target = features[:, :, _TARGET_IDX]
    pred = decoder(prev_hidden, cond)
    diff2 = (pred - target) ** 2
    per_timestep = diff2.mean(dim=-1)  # (B, T)

    per_timestep = per_timestep.masked_fill(~mask, 0.0)
    valid_counts = mask.sum(dim=1).clamp(min=1)
    return per_timestep.sum(dim=1) / valid_counts


@dataclass(frozen=True)
class DecoderTrainConfig:
    num_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    encoder_hidden_size: int = 128
    decoder_hidden_size: int = 64
    dropout: float = 0.2
    seed: int = 42
