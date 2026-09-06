"""Isolation-forest anomaly component over `[latent || engineered features]`
(CLAUDE.md §6 step 11, Day 10), via `coniferest` (already CLAUDE.md §8's
named anomaly package).

CLAUDE.md's instruction names "PineForest / Isolation Forest" -- read as
"either is acceptable," not "both required." `coniferest.pineforest
.PineForest` is used here in pure unsupervised scoring mode (`fit` once,
`score_samples`, no active-learning session/labels) -- a documented,
deliberate choice: PineForest is a strict superset of isolation forest (an
isolation forest that can *optionally* regenerate trees under supervision),
so running it with no supervision loop is a legitimate, supported use, and
it is coniferest's flagship contribution over plain `IsolationForest`. If
`PineForest.fit`/`score_samples` turn out to require a labeling session this
session doesn't have, `IsolationForestScorer` below is the documented
fallback -- see the training script's log output for which one actually ran.

Engineered features are a light-curve summary computed directly from each
`CanonicalAlert`'s own observations (not from the padded/windowed tensorized
form, so no batch-padding artifact ever leaks in): total observation count,
detection fraction, time span, and simple flux-magnitude/scatter summaries.
This project's `light-curve` (Rust-backed, CLAUDE.md §8) library is not yet
wired into any pipeline stage (still a stack-table entry, not a landed
integration) -- these are a deliberately small, directly-computed stand-in
for it, not an attempt to reproduce its full feature set.
"""

import numpy as np

from src.schema import CanonicalAlert

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "n_obs",
    "n_detections",
    "detection_fraction",
    "time_span_days",
    "max_abs_flux_norm",
    "mean_flux_err_norm",
    "std_flux_norm",
    "n_bands",
)
NUM_ENGINEERED_FEATURES = len(ENGINEERED_FEATURE_NAMES)


def _signed_log1p(x: float) -> float:
    return (1.0 if x >= 0 else -1.0) * np.log1p(abs(x))


def engineered_features(alert: CanonicalAlert) -> np.ndarray:
    """`(NUM_ENGINEERED_FEATURES,)` float64 light-curve summary. Raises if
    `alert.observations` is empty -- CLAUDE.md §5.1: every real alert
    originates from a detection, so an empty light curve is malformed input,
    not a normal edge case (same convention `tensorize_alert` already uses)."""
    obs = alert.observations
    if not obs:
        raise ValueError(f"object_id={alert.object_id!r}: no observations")

    mjds = [o.mjd for o in obs]
    n_obs = len(obs)
    n_det = sum(1 for o in obs if o.is_detection)
    flux_norms = [_signed_log1p(o.flux / o.flux_err) for o in obs]
    flux_abs_norms = [_signed_log1p(o.flux) for o in obs]
    flux_err_norms = [np.log1p(o.flux_err) for o in obs]
    n_bands = len({o.lam_eff for o in obs})

    return np.array(
        [
            float(n_obs),
            float(n_det),
            n_det / n_obs,
            max(mjds) - min(mjds),
            max(abs(v) for v in flux_abs_norms),
            float(np.mean(flux_err_norms)),
            float(np.std(flux_norms)),
            float(n_bands),
        ],
        dtype=np.float64,
    )


class EngineeredFeatureNormalizer:
    """Per-field z-score, fit on the training population only (same
    train-only-statistics discipline as
    `src.models.static_encoder.StaticFeatureNormalizer`)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, rows: np.ndarray) -> "EngineeredFeatureNormalizer":
        mean = rows.mean(axis=0)
        std = rows.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return cls(mean=mean, std=std)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return (rows - self.mean) / self.std


class IsolationForestScorer:
    """Fallback if `PineForestScorer` cannot run unsupervised -- plain
    `coniferest.isoforest.IsolationForest`, same fit/score contract."""

    def __init__(self, random_seed: int = 42) -> None:
        from coniferest.isoforest import IsolationForest

        self._model = IsolationForest(random_seed=random_seed)

    def fit(self, X: np.ndarray) -> "IsolationForestScorer":
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. `coniferest`'s own `score_samples`
        convention is the opposite (lower = more anomalous, verified
        empirically against a synthetic normal-vs-outlier sample this
        session -- matches sklearn's `IsolationForest.score_samples`), so
        this negates it for consistency with every other component here."""
        return -self._model.score_samples(X)


class PineForestScorer:
    """`coniferest.pineforest.PineForest`, unsupervised (no active-learning
    session) -- see module docstring."""

    def __init__(self, random_seed: int = 42) -> None:
        from coniferest.pineforest import PineForest

        self._model = PineForest(random_seed=random_seed)

    def fit(self, X: np.ndarray) -> "PineForestScorer":
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)
