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

-----------------------------------------------------------------------------
WINDOWING — train/serve distribution mismatch (Day 4 addendum).
-----------------------------------------------------------------------------
The archived ELAsTiCC2 SNANA training dump and the live LSST/ZTF alert stream
are different artifacts with different schemas, not just different volumes of
the same thing: the training dump is each object's *complete* simulated
observation history (verified this session — real objects show up to
~1100-day time spans, spanning multiple observing seasons before and after
the actual transient), while a live alert packet only ever carries forced
photometry reaching back a bounded window before first detection (CLAUDE.md
§5.1's 30-day clause — which describes the alert packet, not the archived
dump). At a measured mean of 7.9 detections against 139.2 forced-photometry
points per object, ~95% of every unwindowed training sequence is baseline a
live stream would never produce — including entire pre-explosion seasons. A
causal GRU trained on that distribution meets a materially different one at
serving time, and CLAUDE.md §2.3's causality test (Gate B) does not catch
this: that test checks the *model* (does truncating at t change the
prediction at t), not the *input distribution* the model was trained on.

The fix lives here, in tensorize, not in `src/readers/elasticc2.py`: the
reader's job is a faithful parse of what's actually in the file (CLAUDE.md
§2.5), and keeping windowing here means it stays an ablatable preprocessing
choice — pass `pre_detection_window_days=None` to recover the full,
unwindowed archived history (e.g. for comparison experiments, or for a future
long-timescale anomaly class like TDE/SLSN where the pre-explosion baseline
itself may be informative) without touching the parse.

`tensorize_alert` truncates to `[first_detection_mjd - pre_detection_window_days,
last_detection_mjd]` — approximating what a live/replay stream produces
*cumulatively* as of the object's own last real detection. Both endpoints are
properties of the observations actually being tensorized (not of some global
future the alert wouldn't have seen yet), so this composes correctly with
per-timestep truncation: tensorizing `alert.observations[:t]` recomputes its
own first/last detection from that prefix and windows against those, exactly
mirroring what a live pipeline would compute incrementally at each new alert.
`TestTruncationInvariance` (tests/test_tensorize.py) already covers this
composition — windowing does not need a separate causality-style test in
addition to the leak-free-normalization one it already passes.

This can never remove a detection: `first_detection_mjd` is by definition the
minimum, and `last_detection_mjd` the maximum, of `obs.mjd` over
`is_detection=True` rows, so every detection's own `mjd` already lies in
`[first_detection_mjd - pre_detection_window_days, last_detection_mjd]`
(the window only ever *widens* past a detection, via the pre-detection
margin). Only excess forced photometry outside that range is dropped.

-----------------------------------------------------------------------------
RIGHT-EDGE CAP — `max_span_days` (Day 12 addendum).
-----------------------------------------------------------------------------
Everything above bounds only the *left* edge. Day 10's anomaly-branch finding
(CLAUDE.md §11) is that nothing bounds the right edge: for a fading transient
`last_detection_mjd` arrives naturally close to `first_detection_mjd` because
the object drops below detection threshold, but for a periodic/continuously-
variable source (recurring variable stars, AGN) nothing fades, so
`last_detection_mjd` sits wherever the survey simulation happened to end —
CLAGN measured at a 792.8-day mean detection span, TDE at 40.9. A causal GRU
trained on Tier A's short-lived SN population meets a sequence spanning years
at inference time: measured Day 10, this does not make the model uncertain,
it makes it *confidently wrong* (CLAGN mean entropy 0.082 vs. Tier A normal's
0.574 — higher confidence than the model's own in-distribution population).

`max_span_days` (default `None`, preserving prior behaviour exactly) caps
this: the window becomes
`[first_detection_mjd - pre_detection_window_days,
  min(last_detection_mjd, first_detection_mjd + max_span_days)]`.
Unlike `pre_detection_window_days`, **this can remove a detection** — a
real, late detection arriving after `first_detection_mjd + max_span_days` is
dropped, by design: the whole point is to stop feeding the encoder a sequence
longer than what it was trained on, and a long-lived source's detections
past that point are exactly the excess this is meant to cut. This is a
deliberate behaviour change from the invariant documented above, which only
ever held for `pre_detection_window_days`.

This does not, by itself, give a correct *serving-time* policy for a
long-lived recurring source still generating new alerts — capping relative
to `first_detection_mjd` means such an object's tensorized window stops
advancing once `max_span_days` is reached, forever, rather than sliding to
stay centered on "now" (CLAUDE.md's own Day 10 note already flags this
distinction: a live pipeline needs "a fixed recent-window cap, not
since-first-detection"). What `max_span_days` fixes is the *input-length
distribution mismatch* relative to training — matching how long a real Tier
A supernova's tensorized sequence actually gets, per the ablation in
`src/train/span_ablation.py` — not the separate rolling-window question for
indefinitely-recurring sources, which is a serving-layer (Redis state)
concern, not a `tensorize_alert` one.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from src.schema import LAM_EFF, CanonicalAlert

# Default pre-first-detection lookback window, days — matches CLAUDE.md
# §5.1's documented alert-packet convention ("forced photometry reaching back
# to 30 days before first detection"). A named, overridable config parameter
# (see `tensorize_alert`'s `pre_detection_window_days`), never a constant
# hardcoded inline in the truncation logic itself.
DEFAULT_PRE_DETECTION_WINDOW_DAYS: float = 30.0

# Day 12 span ablation (`src/train/span_ablation.py`,
# `reports/span_ablation/span_ablation_report.json`), measured against the
# existing Tier A LC-only checkpoint, inference only: max_span_days=200 gave
# the best full-cutoff macro-F1 among {100, 200} on the Tier A test split
# (0.6979 vs 100's 0.6975), a negligible -0.0002 from uncapped (0.6981),
# while leaving well under 1% of every class's objects truncated at all
# (fraction_affected: Ia 0.03%, Ib/c 0.44%, II 0.32%) and cutting CLAGN's
# Day-10 confidently-wrong entropy inversion roughly in half (mean entropy
# 0.082 -> 0.282, vs. Tier A normal's 0.574 baseline — still short of that
# baseline, but a real step toward it). This is the value serving code
# (`src/serve/`) should pass explicitly as `max_span_days` — NOT
# `tensorize_alert`'s own parameter default above, which stays `None` so
# existing training/eval callers (and this checkpoint's own training run,
# which used no right-edge cap) are unaffected.
RECOMMENDED_MAX_SPAN_DAYS: float = 200.0

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


def tensorize_alert(
    alert: CanonicalAlert,
    pre_detection_window_days: Optional[float] = DEFAULT_PRE_DETECTION_WINDOW_DAYS,
    max_span_days: Optional[float] = None,
) -> TensorizedAlert:
    """Convert one `CanonicalAlert` into a `(T, NUM_FEATURES)` tensor.

    Assumes `alert` satisfies the invariants `CanonicalAlert.validate()`
    enforces (ascending mjd, `flux_err > 0`) — this function does not
    re-validate, per the project convention of validating only at system
    boundaries (`validate()` on the schema itself is that boundary).

    Args:
        alert: the object to tensorize.
        pre_detection_window_days: if not `None` (the default — see module
            docstring, "WINDOWING"), observations before
            `first_detection_mjd - pre_detection_window_days` are dropped
            before tensorizing, to match what a live alert stream would
            actually have accumulated as of this object's own last
            detection. Pass `None` to disable the left-edge bound entirely.
        max_span_days: if not `None` (the default, which preserves prior
            behaviour exactly), caps the window's *right* edge at
            `first_detection_mjd + max_span_days` — see module docstring,
            "RIGHT-EDGE CAP". Unlike `pre_detection_window_days`, this can
            remove a genuine late detection: a long-lived recurring source
            (AGN, periodic variable) whose detections continue well past
            this point has those later ones dropped, by design.

    Returns:
        `TensorizedAlert` with one feature row per surviving observation (all
        of them, if both `pre_detection_window_days` and `max_span_days` are
        `None`), in input order (see module docstring for why row order among
        equal-mjd observations cannot change any individual row's own
        feature values).

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

    if pre_detection_window_days is None and max_span_days is None:
        observations = alert.observations
    else:
        last_detection_mjd = max(detection_mjds)
        window_start = (
            first_detection_mjd - pre_detection_window_days
            if pre_detection_window_days is not None
            else float("-inf")
        )
        window_end = (
            min(last_detection_mjd, first_detection_mjd + max_span_days)
            if max_span_days is not None
            else last_detection_mjd
        )
        observations = [
            obs for obs in alert.observations if window_start <= obs.mjd <= window_end
        ]

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
        for obs in observations
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


# Day 6: the padded/masked-batch collate step lives only in
# `src.train.dataset.collate_examples`/`TrainBatch` (the one actually used in
# production) -- this module used to carry its own independent
# `collate_fn`/`Batch` that nothing imported, a second implementation that
# could silently drift from the one actually in use. Deleted rather than
# kept as untested-but-parallel code (CLAUDE.md §12: "prefer deleting code to
# adding flags"). `TensorizedAlert` above is still what
# `dataset.LightCurveDataset.__getitem__` returns per-object; batching is
# `dataset.collate_examples`'s job.
