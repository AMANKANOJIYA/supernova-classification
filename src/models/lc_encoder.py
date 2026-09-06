"""Causal GRU light-curve encoder + flat 3-class head (CLAUDE.md §6 step 6:
"LC encoder + flat 3-class head <- simplest thing that trains").

Step 7 ("taxonomy.py + WHXE loss <- swap head, keep everything else") will
replace `self.head` with a hierarchical WHXE head; the GRU trunk
(`self.gru`/`self.layer_norm`/`self.dropout`) is deliberately untouched by
that swap, so it lives separately from the head rather than being fused into
one opaque block.

CLAUDE.md §2.3 ("Causal models only... bidirectional=False everywhere"): a
GRU is causal by construction — each timestep's hidden state is a function
only of inputs up to and including that timestep, with no attention
mechanism here that could reach into the future. `bidirectional=False` is
hardcoded (never a config option) because §2.3 calls it absolute, not a
tunable choice. `tests/test_gate_b.py::TestCausality` verifies this
end-to-end (truncating the real input and checking the prediction at t is
bit-identical), which is the test that actually matters — the constructor
flag alone would not catch e.g. an accidentally-added attention layer.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LCEncoder(nn.Module):
    """Causal GRU trunk + flat linear head, emitting per-timestep logits.

    Args:
        input_size: feature dimension per timestep
            (`src.features.tensorize.NUM_FEATURES`).
        hidden_size: GRU hidden size.
        num_layers: number of stacked GRU layers.
        dropout: dropout probability. Applied between GRU layers via
            `nn.GRU`'s own `dropout` argument (only takes effect when
            `num_layers > 1` — PyTorch warns otherwise, so it is zeroed for
            `num_layers == 1`) and again after the trunk's `LayerNorm`,
            before the head.
        num_classes: number of output classes (3 for the flat Ia/Ib-c/II
            head — see `src.taxonomy.TARGET_CLASS_IDS` for the class-index
            ordering callers must use).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # CLAUDE.md §2.3 — absolute, never a config option
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Emit logits at every timestep, not just the last.

        Args:
            features: `(B, T, input_size)`, zero-padded beyond each
                sequence's own length (`src.train.dataset.TrainBatch.features`).
            lengths: `(B,)` real (unpadded) sequence length per item, e.g.
                `batch.mask.sum(dim=1)`. Packing on these lengths is what
                keeps the GRU from ever processing a padded timestep, for
                any sequence, regardless of where it sits in the batch.

        Returns:
            `(B, T, num_classes)` logits at every timestep, including padded
            positions (`pad_packed_sequence` zero-fills those — callers MUST
            mask using the same lengths/mask before any loss or metric, the
            same requirement `src.train.dataset.TrainBatch.mask` already
            places on its own consumers).
        """
        packed = pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=features.shape[1])
        out = self.layer_norm(out)
        out = self.dropout(out)
        return self.head(out)
