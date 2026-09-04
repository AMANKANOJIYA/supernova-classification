"""Baseline classifiers — "the number to beat" (CLAUDE.md §6 step 3). FROZEN.

No dependency beyond numpy: CLAUDE.md §8 does not list scikit-learn, and §2.4
requires asking before adding a new dependency, so the majority-class and
multinomial logistic regression baselines below are implemented directly
rather than pulling in sklearn for what is, for these two models, a small
amount of code.
"""

from dataclasses import dataclass

import numpy as np


class MajorityClassBaseline:
    """Predicts whatever class was most common in the training labels.

    This is deliberately not competitive — it exists so that every other
    number in the project has something trivial to beat, and so that the
    accuracy/macro-F1 divergence on an imbalanced class distribution is
    visible from day one (CLAUDE.md §11: an always-predict-Ia classifier
    scores well on accuracy and poorly on macro-F1).
    """

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.num_classes_: int | None = None

    def fit(self, y_train: np.ndarray, num_classes: int) -> "MajorityClassBaseline":
        """Fit on training labels.

        Args:
            y_train: `(N,)` integer array, true class index in `[0, num_classes)`.
            num_classes: number of classes `K`.
        """
        y_train = np.asarray(y_train)
        counts = np.bincount(y_train, minlength=num_classes)
        self.majority_class_ = int(np.argmax(counts))
        self.num_classes_ = num_classes
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        """Return `(n_samples,)` array of the fitted majority class."""
        if self.majority_class_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.full(n_samples, self.majority_class_, dtype=np.int64)

    def predict_proba(self, n_samples: int) -> np.ndarray:
        """Return `(n_samples, num_classes)` one-hot probabilities."""
        if self.majority_class_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        proba = np.zeros((n_samples, self.num_classes_))
        proba[:, self.majority_class_] = 1.0
        return proba


class MulticlassLogisticRegression:
    """Multinomial (softmax) logistic regression, fit by batch gradient descent.

    Intended for the 5 hand-computed features in `extract_five_features` —
    small `N`, small feature count — not as a general-purpose classifier.
    Features are standardized internally (the 5 features have wildly
    different natural scales: flux counts vs. days) using statistics from the
    training data only.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.1,
        n_iter: int = 2000,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.seed = seed
        self.weights_: np.ndarray | None = None  # (n_features + 1, num_classes), bias last row folded in via augmented X
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feature_mean) / self._feature_std

    def _augment(self, X_std: np.ndarray) -> np.ndarray:
        bias = np.ones((X_std.shape[0], 1))
        return np.concatenate([X_std, bias], axis=1)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassLogisticRegression":
        """Fit on `X (N, n_features)`, `y (N,)` integer class indices."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0  # constant feature guard

        X_aug = self._augment(self._standardize(X))
        one_hot = np.zeros((n_samples, self.num_classes))
        one_hot[np.arange(n_samples), y] = 1.0

        rng = np.random.RandomState(self.seed)
        self.weights_ = rng.normal(scale=0.01, size=(n_features + 1, self.num_classes))

        for _ in range(self.n_iter):
            logits = X_aug @ self.weights_
            probs = self._softmax(logits)
            grad = X_aug.T @ (probs - one_hot) / n_samples
            grad[:-1] += self.l2 * self.weights_[:-1]  # no L2 on the bias row
            self.weights_ -= self.learning_rate * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        X = np.asarray(X, dtype=np.float64)
        X_aug = self._augment(self._standardize(X))
        return self._softmax(X_aug @ self.weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


FEATURE_NAMES: tuple[str, ...] = (
    "peak_flux",
    "rise_time_days",
    "decline_time_days",
    "colour_at_peak",
    "total_duration_days",
)


def extract_five_features(
    mjd: np.ndarray,
    flux: np.ndarray,
    lam_eff: np.ndarray,
    is_detection: np.ndarray,
    colour_window_days: float = 5.0,
) -> np.ndarray:
    """Hand-computed 5-feature summary of one light curve, for the logistic
    regression baseline. Order matches `FEATURE_NAMES`.

    Args:
        mjd: `(M,)` Modified Julian Date of each observation.
        flux: `(M,)` calibrated flux (may be negative — see CLAUDE.md §2.2).
        lam_eff: `(M,)` filter effective wavelength, nanometres.
        is_detection: `(M,)` boolean, True for real detections.
        colour_window_days: half-width, in days, of the window around peak
            MJD searched for a blue/red pair when computing `colour_at_peak`.

    Returns:
        `(5,)` float array: `[peak_flux, rise_time_days, decline_time_days,
        colour_at_peak, total_duration_days]`.

    Raises:
        ValueError: if there are zero detections (there is no light curve to
            summarize).

    `colour_at_peak` is a pseudo-magnitude difference (`-2.5*log10(flux)`)
    between the bluest and reddest detections within `colour_window_days` of
    peak MJD. This is the one place this function converts flux to magnitude
    (CLAUDE.md §2.2 allows it for the purpose of producing a colour, an
    inherently magnitude-space quantity) — it only uses detections near peak,
    where flux is expected to be positive. If fewer than two distinct
    wavelengths are available in the window, or any candidate flux is <= 0,
    `colour_at_peak` falls back to `0.0` (documented limitation, not silently
    wrong: a 0.0 colour on too little data is a known-uninformative value,
    not a plausible measurement).
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    lam_eff = np.asarray(lam_eff, dtype=np.float64)
    is_detection = np.asarray(is_detection, dtype=bool)

    if not is_detection.any():
        raise ValueError("extract_five_features: no detections in this light curve")

    det_mjd = mjd[is_detection]
    det_flux = flux[is_detection]
    det_lam = lam_eff[is_detection]

    peak_idx = int(np.argmax(det_flux))
    peak_flux = float(det_flux[peak_idx])
    peak_mjd = float(det_mjd[peak_idx])

    first_mjd = float(det_mjd.min())
    last_mjd = float(det_mjd.max())
    rise_time = peak_mjd - first_mjd
    decline_time = last_mjd - peak_mjd
    total_duration = last_mjd - first_mjd

    window_mask = np.abs(det_mjd - peak_mjd) <= colour_window_days
    colour_at_peak = 0.0
    if window_mask.sum() >= 2:
        win_lam = det_lam[window_mask]
        win_flux = det_flux[window_mask]
        unique_lams = np.unique(win_lam)
        if len(unique_lams) >= 2:
            blue_lam, red_lam = unique_lams[0], unique_lams[-1]
            # When a band has multiple points in the window, use its brightest
            # (highest-flux) point — the one closest to that band's own peak —
            # rather than an arbitrary window point, so the result doesn't
            # depend on input ordering.
            blue_flux = float(win_flux[win_lam == blue_lam].max())
            red_flux = float(win_flux[win_lam == red_lam].max())
            if blue_flux > 0 and red_flux > 0:
                mag_blue = -2.5 * np.log10(blue_flux)
                mag_red = -2.5 * np.log10(red_flux)
                colour_at_peak = float(mag_blue - mag_red)

    return np.array(
        [peak_flux, rise_time, decline_time, colour_at_peak, total_duration],
        dtype=np.float64,
    )
