"""LC encoder + static encoder fusion model (CLAUDE.md §6 step 8, Day 7):
"MLP over the static vector, concatenated with the GRU output before the
head."

The GRU trunk here is architecturally identical to
`src.models.lc_encoder.LCEncoder`'s (same hyperparameters, same hardcoded
`bidirectional=False` per CLAUDE.md §2.3) but is constructed directly in
this module rather than by wrapping `LCEncoder` itself. `LCEncoder` already
has two completed, checkpointed runs riding on it (the flat step-6 model and
the WHXE step-7 model) -- reusing it here would mean either instantiating
a redundant, never-trained `head` submodule (wasted parameters sitting in
this model's own checkpoint) or refactoring `LCEncoder` to expose its trunk
separately, which touches a file neither of those two runs' reproducibility
needs touched. Duplicating the ~6 lines of trunk construction is the lower-
risk choice; "same encoder" means the same architecture and hyperparameters,
not a shared, aliased `nn.Module` instance -- the flat and WHXE runs were
already two separately-initialized encoders of the same architecture, not
one shared set of weights.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.static_encoder import StaticEncoder


class LCStaticFusionModel(nn.Module):
    """Causal GRU trunk (identical construction to `LCEncoder`'s) + a
    `StaticEncoder` branch, concatenated before a flat linear head.

    Args:
        lc_input_size: light-curve feature dimension per timestep
            (`src.features.tensorize.NUM_FEATURES`).
        lc_hidden_size, lc_num_layers, lc_dropout: same meaning as
            `LCEncoder`'s equivalents -- kept at the same values for a
            controlled comparison (CLAUDE.md Day-7: "same encoder").
        static_hidden_size, static_output_size: `StaticEncoder`'s MLP
            hidden width and output width.
        num_classes: flat head width (3 for Ia/Ib-c/II, same as the step-6
            model -- step 8 is "flat head, same encoder... controlled
            comparison against 0.6981", not a hierarchical head).
    """

    def __init__(
        self,
        lc_input_size: int,
        lc_hidden_size: int = 128,
        lc_num_layers: int = 2,
        lc_dropout: float = 0.2,
        static_hidden_size: int = 32,
        static_output_size: int = 16,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=lc_input_size,
            hidden_size=lc_hidden_size,
            num_layers=lc_num_layers,
            batch_first=True,
            bidirectional=False,  # CLAUDE.md §2.3 -- absolute, never a config option
            dropout=lc_dropout if lc_num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(lc_hidden_size)
        self.dropout = nn.Dropout(lc_dropout)

        self.static_encoder = StaticEncoder(
            output_size=static_output_size, hidden_size=static_hidden_size, dropout=lc_dropout
        )
        self.head = nn.Linear(lc_hidden_size + static_output_size, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: `(B, T, lc_input_size)`, zero-padded.
            lengths: `(B,)` real (unpadded) sequence length per item.
            static_features: `(B, 2*NUM_STATIC_FIELDS)`, constant per object
                (the same vector is implicitly broadcast to every timestep
                below -- host context doesn't vary with light-curve phase).

        Returns:
            `(B, T, num_classes)` logits at every timestep, including padded
            positions -- callers MUST mask, same requirement as `LCEncoder`.
        """
        packed = pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        lc_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=features.shape[1]
        )
        lc_out = self.layer_norm(lc_out)
        lc_out = self.dropout(lc_out)

        static_out = self.static_encoder(static_features)  # (B, static_output_size)
        static_broadcast = static_out.unsqueeze(1).expand(-1, lc_out.shape[1], -1)

        fused = torch.cat([lc_out, static_broadcast], dim=-1)
        return self.head(fused)


class LCStaticDRBFusionModel(nn.Module):
    """`LCStaticFusionModel` plus a small extra branch for `real_bogus_score`
    (CLAUDE.md §5.1: 100%-missing at Tier A, so excluded from
    `src.models.static_encoder.STATIC_FIELDS` entirely; populated at Tier B,
    95.1% non-null). Day 9 experiment testing whether it helps once real
    data actually provides it.

    Deliberately its own class rather than a modification of
    `LCStaticFusionModel` (same "duplicate the trunk" reasoning that module
    already states: `LCStaticFusionModel` has a completed, checkpointed Tier
    A run riding on it, and this model's head has a different input width --
    `lc_hidden_size + static_output_size + drb_output_size` -- so the head
    could not be shared even via subclassing without either wasting a
    same-shaped-but-differently-trained head or breaking that checkpoint's
    `state_dict` shape).

    Warm-starting: `gru`/`layer_norm`/`static_encoder` are named identically
    to `LCStaticFusionModel`'s attributes of the same name, so
    `load_shared_submodules` below can copy a Tier A `LCStaticFusionModel`
    checkpoint's matching-shape weights into a fresh instance of this class;
    `head` and `drb_branch` are new and always start randomly initialized.
    """

    def __init__(
        self,
        lc_input_size: int,
        lc_hidden_size: int = 128,
        lc_num_layers: int = 2,
        lc_dropout: float = 0.2,
        static_hidden_size: int = 32,
        static_output_size: int = 16,
        drb_hidden_size: int = 8,
        drb_output_size: int = 4,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=lc_input_size,
            hidden_size=lc_hidden_size,
            num_layers=lc_num_layers,
            batch_first=True,
            bidirectional=False,  # CLAUDE.md §2.3 -- absolute, never a config option
            dropout=lc_dropout if lc_num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(lc_hidden_size)
        self.dropout = nn.Dropout(lc_dropout)

        self.static_encoder = StaticEncoder(
            output_size=static_output_size, hidden_size=static_hidden_size, dropout=lc_dropout
        )
        # [value; missingness indicator] -- 2 input units, same zero-fill +
        # indicator convention as StaticEncoder (see src.models.static_encoder).
        self.drb_branch = nn.Sequential(
            nn.Linear(2, drb_hidden_size),
            nn.ReLU(),
            nn.Dropout(lc_dropout),
            nn.Linear(drb_hidden_size, drb_output_size),
        )
        self.head = nn.Linear(
            lc_hidden_size + static_output_size + drb_output_size, num_classes
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        static_features: torch.Tensor,
        drb_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: `(B, T, lc_input_size)`, zero-padded.
            lengths: `(B,)` real (unpadded) sequence length per item.
            static_features: `(B, 2*NUM_STATIC_FIELDS)`.
            drb_features: `(B, 2)` -- `[z-scored real_bogus_score, missingness
                indicator]`, constant per object.

        Returns:
            `(B, T, num_classes)` logits at every timestep, including padded
            positions -- callers MUST mask, same requirement as `LCEncoder`.
        """
        packed = pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        lc_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=features.shape[1]
        )
        lc_out = self.layer_norm(lc_out)
        lc_out = self.dropout(lc_out)

        static_out = self.static_encoder(static_features)  # (B, static_output_size)
        drb_out = self.drb_branch(drb_features)  # (B, drb_output_size)
        combined = torch.cat([static_out, drb_out], dim=-1)
        combined_broadcast = combined.unsqueeze(1).expand(-1, lc_out.shape[1], -1)

        fused = torch.cat([lc_out, combined_broadcast], dim=-1)
        return self.head(fused)


def load_shared_submodules(
    model: "LCStaticDRBFusionModel", pretrained_state_dict: dict
) -> None:
    """Copy `gru`/`layer_norm`/`static_encoder` weights from a Tier A
    `LCStaticFusionModel` checkpoint's `state_dict` into `model`, leaving
    `head`/`drb_branch` at their fresh random initialization. Only exact
    shape matches are copied; every copied/skipped key is accounted for so a
    silent architecture drift (e.g. a hidden-size mismatch) is loud, not a
    quietly-half-warm-started model.
    """
    own_state = model.state_dict()
    warm_started, skipped = [], []
    for key, value in pretrained_state_dict.items():
        if not (key.startswith("gru.") or key.startswith("layer_norm.") or key.startswith("static_encoder.")):
            continue
        if key not in own_state:
            skipped.append(key)
            continue
        if own_state[key].shape != value.shape:
            raise ValueError(
                f"load_shared_submodules: shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} model={tuple(own_state[key].shape)}"
            )
        own_state[key] = value
        warm_started.append(key)
    model.load_state_dict(own_state)
    if skipped:
        raise ValueError(
            f"load_shared_submodules: {len(skipped)} shared-prefix key(s) in the "
            f"checkpoint had no matching parameter in the model: {skipped}"
        )
