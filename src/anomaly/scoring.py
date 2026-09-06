"""Entropy / Mahalanobis / combination scoring for the anomaly branch
(CLAUDE.md §6 step 11, Day 10). Reconstruction error lives in
`src.anomaly.decoder`; the isolation-forest component lives in
`src.anomaly.pineforest`. This module is the one place all four get
combined and turned into ROC-AUC numbers.

One-way import from `src.eval` (never the reverse), same convention
`src.diagnostics` already follows: `src/eval/` is frozen (CLAUDE.md §2.1)
and judges *classifier* performance; this module reuses its already-tested
rank-based ROC-AUC implementation rather than re-deriving Mann-Whitney U a
second time, but is not itself part of the frozen harness.
"""

import numpy as np

from src.eval.metrics import roc_auc as _multiclass_roc_auc


def entropy_score(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of `probs` (`(N, K)`, rows summing to 1),
    normalized by `log(K)` so the result sits in `[0, 1]` regardless of `K`.
    "A model saying 33/33/33 is telling you something" (CLAUDE.md Day 10) --
    higher = more anomalous (less confident)."""
    probs = np.clip(probs, 1e-12, 1.0)
    raw = -(probs * np.log(probs)).sum(axis=1)
    return raw / np.log(probs.shape[1])


class MahalanobisScorer:
    """Mahalanobis distance of a latent vector from a fitted Gaussian.
    Covariance is ridge-regularized (`+= eps * I`) before inversion -- the
    latent dimension (128, `LCEncoder`'s hidden size) is much larger than
    what a naive sample covariance can safely invert against a training
    population of the size used here, and an ill-conditioned inverse would
    turn "distance from normal" into numerical noise rather than signal.
    """

    def __init__(self, mean: np.ndarray, precision: np.ndarray) -> None:
        self.mean = mean
        self.precision = precision

    @classmethod
    def fit(cls, latents: np.ndarray, ridge: float = 1.0e-3) -> "MahalanobisScorer":
        mean = latents.mean(axis=0)
        centered = latents - mean
        cov = (centered.T @ centered) / max(len(latents) - 1, 1)
        cov = cov + ridge * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        precision = np.linalg.inv(cov)
        return cls(mean=mean, precision=precision)

    def score(self, latents: np.ndarray) -> np.ndarray:
        centered = latents - self.mean
        return np.einsum("ij,jk,ik->i", centered, self.precision, centered)


def binary_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC for "does a higher `score` predict `y_true==1` (anomalous)",
    via `src.eval.metrics.roc_auc`'s existing rank-based (Mann-Whitney U)
    implementation. That function only ever reads column `k` of `y_prob` to
    score class `k` (verified against its source -- purely rank-based, needs
    no properly-normalized probabilities), so a 2-column stack of
    `[-score, score]` reuses it correctly for an arbitrary continuous score
    without re-deriving the rank-sum identity here.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    y_prob = np.stack([-score, score], axis=1)
    auc = _multiclass_roc_auc(y_true, y_prob, num_classes=2)
    return float(auc[1])


def fit_zscore(reference: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(reference))
    std = float(np.std(reference))
    return mean, (std if std > 0 else 1.0)


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) / std


def combine_equal_weight(z_scored_components: list[np.ndarray]) -> np.ndarray:
    """Equal-weighted average of already-z-scored component scores (each
    z-scored against its own fit-population statistics -- see `fit_zscore`).
    Simple, standard, and defensible for a first pass with no labeled
    anomalies to tune weights against (there are none: both eval sets are
    used only for scoring, never for fitting anything)."""
    return np.mean(np.stack(z_scored_components, axis=0), axis=0)
