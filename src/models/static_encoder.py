"""Static (host-galaxy + extinction) context encoder (CLAUDE.md §6 step 8,
Day 7: "MLP over the static vector, concatenated with the GRU output before
the head").

Missingness is handled explicitly, never silently imputed (CLAUDE.md §2.5's
spirit applied to features, not just rows): each field gets its own
missing-value indicator alongside a zero-filled, z-scored value.
**Zero-fill, not a learned fill, chosen and stated rather than picked
silently** (CLAUDE.md §12) -- a learned per-field "missing" embedding is
marginally more expressive, but the indicator already tells the downstream
MLP a value is a fill, not a measurement; the MLP is free to learn to
disregard or reinterpret it via the indicator, so a second set of learned
parameters buys little for a one-run, no-hyperparameter-search comparison
(Day-7 instruction). Zero-fill on z-scored data lands roughly at the
training-set mean, a neutral placeholder in the untransformed scale too.

`real_bogus_score` is omitted from `STATIC_FIELDS`: CLAUDE.md §5.1 documents
it as 100%-`None` at Tier A (no image-level classifier exists for pure
ELAsTiCC2 simulation) -- confirmed again in the Day-6 audit's leakage hunt.
A field with zero non-missing values everywhere is not a feature, it is a
constant; the slot is reserved for Tier B (BTSbot), where it is populated.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.schema import CanonicalAlert, StaticContext

STATIC_FIELDS: tuple[str, ...] = (
    "host_photoz",
    "host_photoz_err",
    "host_offset_arcsec",
    "host_ellipticity",
    "host_mag_r",
    "mw_ebv",
    "gal_lat",
)
NUM_STATIC_FIELDS: int = len(STATIC_FIELDS)

# Day 10 diagnostic (CLAUDE.md instruction): test whether host_photoz/_err
# drove Day 7's +0.0189 LC+static gain, since both are structurally absent
# at Tier B (§5.1's "three of the static branch's 7 inputs are permanently
# missing-flagged at Tier B"). Every function below takes an optional
# `fields` argument (default `STATIC_FIELDS`) so this ablation reuses the
# exact same normalizer/vector-building code path rather than a forked copy.
ABLATION_NO_PHOTOZ_FIELDS: tuple[str, ...] = tuple(
    f for f in STATIC_FIELDS if f not in ("host_photoz", "host_photoz_err")
)

# CLAUDE.md's Day-6 audit: the reader (src/readers/elasticc2.py) converts
# these raw SNANA sentinels to None before a StaticContext is ever
# constructed. A raw sentinel surviving into this module's input would mean
# that conversion was skipped somewhere -- checked for, not assumed, by
# `assert_no_sentinel_values` below and by the real-data test in
# tests/test_static_encoder.py.
_KNOWN_SENTINELS: tuple[float, ...] = (-9.0, -9999.0, 999.0)


def _raw_static_row(static: StaticContext, fields: Sequence[str] = STATIC_FIELDS) -> np.ndarray:
    """`(len(fields),)` float64, `NaN` where `None`."""
    return np.array(
        [
            getattr(static, field) if getattr(static, field) is not None else np.nan
            for field in fields
        ],
        dtype=np.float64,
    )


def assert_no_sentinel_values(
    alerts: Sequence[CanonicalAlert], fields: Sequence[str] = STATIC_FIELDS
) -> None:
    """Raise if any non-missing static field value in `alerts` equals one of
    the raw SNANA sentinels the reader is supposed to have already converted
    to `None`. CLAUDE.md Day-7 instruction: "verify no sentinel values
    survive into the encoder" -- a real check, not an assumption.
    """
    offending: list[tuple[str, str, float]] = []
    for alert in alerts:
        row = _raw_static_row(alert.static, fields)
        for field, value in zip(fields, row):
            if not np.isnan(value) and float(value) in _KNOWN_SENTINELS:
                offending.append((alert.object_id, field, float(value)))
    if offending:
        raise ValueError(
            f"{len(offending)} static field value(s) equal a known raw sentinel "
            f"despite not being None -- reader-level sentinel conversion was "
            f"skipped somewhere. First few: {offending[:5]}"
        )


@dataclass(frozen=True)
class StaticFeatureNormalizer:
    """Per-field z-score (mean/std over non-missing values only). Fit once
    on the training split, applied identically to val/test -- ordinary,
    non-leaking preprocessing: `StaticContext` doesn't vary with time, so
    there is no causal-order concern the way there is for flux features in
    `src.features.tensorize` (this is why `tensorize.py`'s wavelength
    normalization uses a *fixed* global constant rather than dataset
    statistics, and why this can safely use train-set statistics instead:
    the two are solving different problems, not the same one two ways).
    """

    mean: np.ndarray  # (len(fields),)
    std: np.ndarray  # (len(fields),), never 0

    @classmethod
    def fit(
        cls, alerts: Sequence[CanonicalAlert], fields: Sequence[str] = STATIC_FIELDS
    ) -> "StaticFeatureNormalizer":
        rows = np.stack([_raw_static_row(a.static, fields) for a in alerts])
        mean = np.nanmean(rows, axis=0)
        std = np.nanstd(rows, axis=0)
        std = np.where(std > 0, std, 1.0)  # guard: a degenerate zero-variance field
        return cls(mean=mean, std=std)

    def transform(self, row: np.ndarray) -> np.ndarray:
        return (row - self.mean) / self.std


def build_static_vector(
    static: StaticContext,
    normalizer: StaticFeatureNormalizer,
    fields: Sequence[str] = STATIC_FIELDS,
) -> np.ndarray:
    """`(2 * len(fields),)` float32: z-scored values (0.0 where
    missing) concatenated with per-field missingness indicators (1.0 =
    missing). See module docstring for why zero-fill, not a learned fill.
    `fields` must match whatever `normalizer` was fit on (same order/count).
    """
    raw = _raw_static_row(static, fields)
    is_missing = np.isnan(raw)
    normalized = normalizer.transform(raw)
    filled = np.where(is_missing, 0.0, normalized)
    return np.concatenate([filled, is_missing.astype(np.float64)]).astype(np.float32)


class StaticEncoder(nn.Module):
    """MLP over `[values; missingness indicators]`, `2*NUM_STATIC_FIELDS`
    input width."""

    def __init__(
        self,
        output_size: int,
        hidden_size: int = 32,
        dropout: float = 0.2,
        num_static_fields: int = NUM_STATIC_FIELDS,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_static_fields, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        """`static_features`: `(B, 2*NUM_STATIC_FIELDS)` -> `(B, output_size)`."""
        return self.mlp(static_features)
