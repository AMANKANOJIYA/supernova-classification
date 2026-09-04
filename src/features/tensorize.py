"""CanonicalAlert -> padded tensor path (CLAUDE.md §6 step 5).

Built ahead of src/readers/ (step 4) — deliberately, not a build-order
violation. This module's only contract is `src.schema.CanonicalAlert`,
which is frozen and fully specified independent of any reader. Nothing here
depends on a reader existing.

-----------------------------------------------------------------------------
NORMALIZATION — read this before touching flux_norm / flux_err_norm.
-----------------------------------------------------------------------------
This is the single most leakage-prone piece of the whole pipeline (per-alert
peak/mean-based normalization would let a model see "how bright this got
relative to its own future peak" at every early timestep, which does not
exist at inference time on a live stream — and CLAUDE.md §2.3's causal-model
test (Gate B) would not catch it, because that test checks the model
architecture, not preprocessing computed once per object before the model
ever runs).

The scheme chosen here computes NO statistic over the light curve at all —
not even a causal running one. Every per-timestep feature is a **pointwise**
function of that single observation's own (flux, flux_err, lam_eff,
is_detection) fields:

    flux_norm      = signed_log1p(flux / flux_err)
    flux_abs_norm  = signed_log1p(flux)
    flux_err_norm  = log1p(flux_err)
    wavelength_norm = fixed linear map of lam_eff into [-1, 1], using the
                       global LAM_EFF constant range (schema.py) — not this
                       alert's own observed wavelengths.
    is_detection    passthrough, 0.0/1.0.

`signed_log1p(x) := sign(x) * log1p(|x|)`. Stated explicitly because it
matters: a plain `log1p(x)` on a negative `x` (routine here — CLAUDE.md §2.2,
difference-image flux is negative early and often) returns NaN. Extracting
the sign first and taking `log1p` of the *magnitude* is what keeps this
finite and defined for every real `x`, while staying odd/sign-preserving —
`signed_log1p(-x) == -signed_log1p(x)` — so a formally negative flux still
reads as formally negative after compression, not clipped or folded onto
its positive counterpart.

Why `flux_norm` (the ratio) and not a running/causal per-object statistic
(e.g. a running max of |flux| seen so far, updated incrementally):

1. flux_err > 0 is a schema invariant (`CanonicalAlert.validate()`), true
   for every single observation independent of any other observation. Using
   it as the divisor for `flux` means the denominator can never be zero —
   safe on all-negative flux (ratio is just negative) and on near-zero-
   amplitude light curves (ratio is just near zero) without an epsilon
   floor, a clipped minimum, or any other survey-specific magic constant.
2. `flux / flux_err` is exactly signal-to-noise ratio — a standard,
   physically meaningful, scale-invariant quantity (this mirrors how
   photometric classifiers like ALeRCE's lc_classifier, referenced in
   CLAUDE.md §8, commonly work with SNR-like ratios rather than raw flux).
   It can range from ~0 up to several thousand for bright, well-measured
   detections; `signed_log1p` compresses that into a numerically bounded,
   NN-friendly range while staying strictly monotonic (order-preserving —
   the light curve's rise/decline *shape* survives the compression) and
   finite everywhere (`log1p` of any finite non-negative number is finite;
   `flux/flux_err` is finite whenever `flux_err > 0`, which is guaranteed).
3. This is provably leak-free in the strongest possible sense — stronger
   than "causal, using only the past": it doesn't even use the past. Each
   row's features depend on nothing but that row's own fields. There is
   no window, no running state, no "as of time t" bookkeeping to get
   subtly wrong. This directly satisfies the truncation-invariance test
   below by construction, not by careful accounting.
4. It sidesteps a second, easy-to-miss leakage vector: **implementation
   skew between batch tensorization (here) and streaming inference**
   (CLAUDE.md §6 steps 12-13, replay mode before live mode, same consumer
   code). A running-statistic design must be computed identically whether
   you're iterating an archived Parquet file in one shot or updating state
   one Kafka message at a time — any off-by-one in "does the running max
   include the current point" is a live bug that would only surface once
   the live stream is running. A pointwise scheme has no such state to
   keep in sync between the two code paths at all.

Why `flux_abs_norm` was added on top of `flux_norm`, deliberately redundant:
absolute peak brightness is real, physically meaningful signal for the
Ia-vs-Ib/c boundary specifically (Ia peaks cluster tightly near M ≈ -19.3;
Ib/c is fainter and more scattered) — CLAUDE.md's whole reason for treating
Ib/c recall as the hard, expected-to-be-mediocre metric (§11). That
information is technically recoverable from `flux_norm` and `flux_err_norm`
together (`flux = (flux/flux_err) * flux_err`), but only as a **product** of
two already-nonlinearly-compressed features. GRUs and MLPs are poor at
learning multiplicative interactions between two of their own input
channels — the information is present in an information-theoretic sense but
practically inaccessible without extra depth/capacity spent reconstructing
it. Feeding `signed_log1p(flux)` directly removes that burden. This adds no
new leakage risk (still pointwise, still stateless, still uses no light-
curve-derived statistic) and no new NaN/inf risk (same `signed_log1p`
function, already proven finite everywhere above) — it's the same
information already in the tensor, made directly accessible instead of
implicit.

`flux_abs_norm` and `flux_err_norm` are safe to compare/combine across
surveys and tiers without any per-survey correction: CLAUDE.md §3 now fixes
`flux` to the SNANA `FLUXCAL` convention (AB zeropoint 27.5) as a schema-
level contract — every reader is responsible for converting its source's
photometry to this scale before a `CanonicalAlert` is ever constructed, so
by the time flux reaches this module, one consistent zeropoint is a
guarantee, not an assumption this module has to hedge against. (An earlier
version of this docstring flagged this as an unresolved, deferred cross-
survey caveat on `flux_err_norm` — that was written against an underspecified
prior version of §3 that only said "calibrated flux" without naming a
zeropoint. §3 was amended to close that gap upstream, at the schema/reader
boundary, rather than downstream in every consumer of flux. This module
must never apply a per-survey correction of its own — CLAUDE.md §3 is
explicit that only readers do that.)

Tradeoff, stated explicitly rather than buried: `flux_norm`,
`flux_abs_norm`, and `flux_err_norm` are all pointwise, so none of them can
carry cross-observation context an object's own light-curve *shape* might
suggest (e.g. "this is unusually bright relative to its own earlier
points") — that kind of relative-to-history signal is left for the model
itself to learn from the sequence of pointwise values, not precomputed
here.

No tension found with CLAUDE.md §2.3 (causal models only): this scheme
doesn't touch model architecture at all, and being pointwise rather than
causal-with-state, it's if anything a strictly easier property to keep
correct than what §2.3 asks of the model itself.

`Δt` (days since first detection) is the one feature that isn't purely
pointwise — it's `obs.mjd - first_detection_mjd`, where
`first_detection_mjd = min(mjd for obs in alert.observations if
obs.is_detection)`. This is still leak-free, but for a different, worth-
stating-explicitly reason than the flux features: "first detection" is a
**first-occurrence** statistic, not a **growing/evolving** one like peak or
mean. It is fully determined the instant the first detection happens and
provably identical whether computed over the complete light curve or over
any causal prefix from that point onward — unlike peak flux, which
*changes* (can only increase) as more of the future is revealed. There is
therefore no distinction between "computed causally" and "computed over the
full light curve" for this particular quantity; they are mathematically the
same number. CLAUDE.md §5.1's note that forced photometry can reach back
before first detection is exactly why this can be negative for some rows —
matches the negative-cutoff convention already used in
`src/eval/phase_curves.py`.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from src.schema import LAM_EFF, CanonicalAlert

FEATURE_NAMES: tuple[str, ...] = (
    "delta_t_days",
    "flux_norm",
    "flux_abs_norm",
    "flux_err_norm",
    "wavelength_norm",
    "is_detection",
)
NUM_FEATURES = len(FEATURE_NAMES)

# Fixed, global constants — deliberately NOT derived from any alert's own
# observations. See module docstring: this is what keeps wavelength_norm
# pointwise and leak-free, and what makes ZTF r / LSST r land near the same
# encoded value regardless of which survey (or which subset of bands a
# particular object happened to be observed in) produced the alert.
_LAM_EFF_MIN = min(LAM_EFF.values())
_LAM_EFF_MAX = max(LAM_EFF.values())


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """`sign(x) * log1p(|x|)`. Finite for every finite `x`. See module
    docstring for why this is used to compress the flux/flux_err ratio."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _wavelength_norm(lam_eff: float) -> float:
    """Map a filter effective wavelength (nm) linearly into `[-1, 1]` using
    the fixed global `LAM_EFF` range — never a band index (CLAUDE.md §3)."""
    return 2.0 * (lam_eff - _LAM_EFF_MIN) / (_LAM_EFF_MAX - _LAM_EFF_MIN) - 1.0


@dataclass(frozen=True)
class TensorizedAlert:
    """One object's tensorized light curve.

    Attributes:
        object_id: from the source `CanonicalAlert`.
        features: `(T, NUM_FEATURES)` float32, one row per observation, in
            the same order as `alert.observations` (ascending mjd).
        class_id: passthrough from the source `CanonicalAlert`; `None` if
            unlabelled.
    """

    object_id: str
    features: torch.Tensor
    class_id: Optional[int]


def tensorize_alert(alert: CanonicalAlert) -> TensorizedAlert:
    """Convert one `CanonicalAlert` into a `(T, NUM_FEATURES)` tensor.

    Assumes `alert` satisfies the invariants `CanonicalAlert.validate()`
    enforces (ascending mjd, `flux_err > 0`) — this function does not
    re-validate, per the project convention of validating only at system
    boundaries (`validate()` on the schema itself is that boundary).

    Args:
        alert: the object to tensorize.

    Returns:
        `TensorizedAlert` with one feature row per observation, in input
        order (see module docstring for why row order among equal-mjd
        observations cannot change any individual row's own feature
        values).

    Raises:
        ValueError: if `alert.observations` is empty, or contains no
            detection (`is_detection=True`). CLAUDE.md §5.1: every real
            alert originates from a detection, so an alert with none is
            malformed input, not a normal edge case to silently paper over.
    """
    if not alert.observations:
        raise ValueError(f"object_id={alert.object_id!r}: no observations to tensorize")

    detection_mjds = [obs.mjd for obs in alert.observations if obs.is_detection]
    if not detection_mjds:
        raise ValueError(
            f"object_id={alert.object_id!r}: no detections among "
            f"{len(alert.observations)} observation(s); Δt has no reference point"
        )
    first_detection_mjd = min(detection_mjds)

    # Δt computed in plain Python float (float64) arithmetic *before* any
    # tensor is built, and only from the already-small (mjd - reference)
    # difference — never from the raw ~60000-magnitude MJD value itself.
    # float32 has ~7 significant digits: applied directly to a raw MJD, that
    # is only day-level precision, nowhere near enough for sub-day cadence.
    # Applied to an already-subtracted small delta, it's more than enough.
    # This is exactly the "MJD versus Δt" bug class CLAUDE.md §12 calls out
    # by name — the fix is ordering the subtraction before the cast, not
    # avoiding float32 altogether.
    rows = [
        (
            obs.mjd - first_detection_mjd,
            obs.flux,
            obs.flux_err,
            _wavelength_norm(obs.lam_eff),
            1.0 if obs.is_detection else 0.0,
        )
        for obs in alert.observations
    ]

    raw = torch.tensor(rows, dtype=torch.float64)
    delta_t, flux, flux_err, wavelength_norm, is_detection = raw.unbind(dim=1)

    flux_norm = _signed_log1p(flux / flux_err)
    flux_abs_norm = _signed_log1p(flux)
    flux_err_norm = torch.log1p(flux_err)

    features = torch.stack(
        [delta_t, flux_norm, flux_abs_norm, flux_err_norm, wavelength_norm, is_detection], dim=1
    ).to(torch.float32)

    return TensorizedAlert(object_id=alert.object_id, features=features, class_id=alert.class_id)


@dataclass(frozen=True)
class Batch:
    """A padded, masked batch, ready for a model.

    Attributes:
        object_id: length `B`.
        features: `(B, T_max, NUM_FEATURES)` float32, zero-padded.
        mask: `(B, T_max)` bool; `True` = real observation, `False` =
            padding. Padded feature rows happen to be `0.0`, but nothing
            downstream may rely on that — consumers MUST gate on `mask`,
            not on padded values happening to already be zero (see
            `tests/test_tensorize.py`, which asserts padded rows can hold
            arbitrary garbage without changing a properly masked
            computation's result).
        class_id: length `B`, passthrough (may contain `None`).
    """

    object_id: list[str]
    features: torch.Tensor
    mask: torch.Tensor
    class_id: list[Optional[int]]


def collate_fn(items: Sequence[TensorizedAlert]) -> Batch:
    """`DataLoader`-compatible collate function: pads every item in `items`
    to this batch's max length and builds the corresponding boolean mask.

    Raises:
        ValueError: if `items` is empty.
    """
    if not items:
        raise ValueError("collate_fn: empty batch")

    lengths = [item.features.shape[0] for item in items]
    t_max = max(lengths)
    batch_size = len(items)

    features = torch.zeros((batch_size, t_max, NUM_FEATURES), dtype=torch.float32)
    mask = torch.zeros((batch_size, t_max), dtype=torch.bool)

    for i, item in enumerate(items):
        t = item.features.shape[0]
        features[i, :t] = item.features
        mask[i, :t] = True

    return Batch(
        object_id=[item.object_id for item in items],
        features=features,
        mask=mask,
        class_id=[item.class_id for item in items],
    )
