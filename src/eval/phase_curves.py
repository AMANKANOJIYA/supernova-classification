"""Time-resolved evaluation: metrics as a function of days since first
detection (CLAUDE.md §6 step 4, §7 Gate A/C, §11). FROZEN.

Early-time classification is the entire point of this project (CLAUDE.md §2.2,
§5.1), so a single end-of-light-curve macro-F1 hides the number that matters
most. This module truncates per-timestep predictions at fixed phase cutoffs
and re-scores at each one — the primary reported figure is macro-F1 vs. days
since first detection, per CLAUDE.md §11.

Compute only — this module does NOT import matplotlib (or anything else
outside numpy and `src.eval.metrics`). Rendering lives in `src.eval.plots`,
which consumes the `PhaseCutoffResult` values this module produces;
`src.eval.report` orchestrates compute-then-render. Keeping this file
plot-free means it (and its return values) stay usable anywhere a figure
isn't wanted or possible — e.g. a headless training loop logging metrics to
MLflow every epoch. No src/models/ or src/train/ import either, so this
stays runnable independently of any trained model.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.eval.metrics import (
    CalibrationResult,
    ConfusionMatrix,
    PerClassPRF1,
    confusion_matrix,
    expected_calibration_error,
    per_class_pr_auc,
    per_class_precision_recall_f1,
    roc_auc,
)

# Cutoff = days since first detection. None means "full" — no truncation, use
# the latest available prediction for each object. A negative cutoff (e.g.
# -5) is meaningful here: CLAUDE.md §5.1 notes forced photometry reaches back
# up to 30 days before first detection from the second night's alert onward,
# so "-5" means "using only data through 5 days before first detection".
DEFAULT_CUTOFFS: tuple[float | None, ...] = (-5.0, 0.0, 5.0, 15.0, 30.0, None)


@dataclass(frozen=True)
class PhasePrediction:
    """One model prediction for one object, made using only data available up
    to `days_since_first_detection`.

    Attributes:
        object_id: which object this prediction is for.
        days_since_first_detection: days, may be negative (pre-first-detection
            forced photometry) or fractional.
        true_class: ground-truth class index in `[0, num_classes)`.
        probs: `(num_classes,)` predicted class probabilities.
    """

    object_id: str
    days_since_first_detection: float
    true_class: int
    probs: np.ndarray


def cutoff_label(cutoff: float | None) -> str:
    """Human-readable label for a cutoff: `"full"` or e.g. `"-5"`, `"15"`."""
    if cutoff is None:
        return "full"
    return str(int(cutoff)) if float(cutoff).is_integer() else str(cutoff)


def select_at_cutoff(
    predictions: Sequence[PhasePrediction], cutoff: float | None
) -> tuple[list[PhasePrediction], int]:
    """For each object, pick its latest prediction at or before `cutoff` days
    since first detection (or, if `cutoff` is None, the latest prediction
    available at all — "full").

    An object with no prediction at or before `cutoff` (e.g. a day-30 cutoff
    but the object was only observed through day 10) is excluded rather than
    guessed at. Per CLAUDE.md §2.5's spirit of never silently dropping data,
    the exclusion count is returned rather than swallowed.

    Returns:
        `(selected, n_excluded)`: one `PhasePrediction` per included object
        (order not guaranteed), and how many distinct objects had no eligible
        prediction.
    """
    by_object: dict[str, list[PhasePrediction]] = {}
    for p in predictions:
        by_object.setdefault(p.object_id, []).append(p)

    selected: list[PhasePrediction] = []
    n_excluded = 0
    for object_id, group in by_object.items():
        if cutoff is None:
            eligible = group
        else:
            eligible = [p for p in group if p.days_since_first_detection <= cutoff]

        if not eligible:
            n_excluded += 1
            continue

        selected.append(max(eligible, key=lambda p: p.days_since_first_detection))

    return selected, n_excluded


@dataclass(frozen=True)
class PhaseCutoffResult:
    """Evaluation results at a single phase cutoff."""

    cutoff: float | None
    n_objects: int
    n_excluded: int
    macro_f1: float
    per_class: PerClassPRF1
    confusion: ConfusionMatrix
    pr_auc: np.ndarray
    roc_auc: np.ndarray
    calibration: CalibrationResult


def evaluate_at_cutoffs(
    predictions: Sequence[PhasePrediction],
    num_classes: int,
    cutoffs: Sequence[float | None] = DEFAULT_CUTOFFS,
    ece_num_bins: int = 15,
) -> dict[str, PhaseCutoffResult]:
    """Truncate at each cutoff and compute the full metric suite.

    Args:
        predictions: every available per-timestep prediction, across all
            objects and all phases (this function does the truncating).
        num_classes: number of classes `K`.
        cutoffs: phase cutoffs to evaluate at; see `DEFAULT_CUTOFFS`.
        ece_num_bins: bin count for expected calibration error.

    Returns:
        Mapping from `cutoff_label(cutoff)` to `PhaseCutoffResult`.
    """
    results: dict[str, PhaseCutoffResult] = {}

    for cutoff in cutoffs:
        selected, n_excluded = select_at_cutoff(predictions, cutoff)

        if selected:
            y_true = np.array([p.true_class for p in selected])
            y_prob = np.stack([p.probs for p in selected])
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_true = np.zeros(0, dtype=np.int64)
            y_prob = np.zeros((0, num_classes))
            y_pred = np.zeros(0, dtype=np.int64)

        results[cutoff_label(cutoff)] = PhaseCutoffResult(
            cutoff=cutoff,
            n_objects=len(selected),
            n_excluded=n_excluded,
            macro_f1=(
                float(np.mean(per_class_precision_recall_f1(y_true, y_pred, num_classes).f1))
                if selected
                else float("nan")
            ),
            per_class=per_class_precision_recall_f1(y_true, y_pred, num_classes),
            confusion=confusion_matrix(y_true, y_pred, num_classes),
            pr_auc=per_class_pr_auc(y_true, y_prob, num_classes),
            roc_auc=roc_auc(y_true, y_prob, num_classes),
            calibration=expected_calibration_error(y_true, y_prob, ece_num_bins)
            if selected
            else expected_calibration_error(
                np.zeros(0, dtype=np.int64), np.zeros((0, num_classes)), ece_num_bins
            ),
        )

    return results
