"""Classification metrics for the transient classifier. FROZEN — CLAUDE.md §2.1.

Deliberately does not import from src/models/ or src/train/ (or anything
outside numpy), so this harness can be run against any set of predictions —
a baseline, a checkpoint, a competitor's output — independently of how they
were produced.

No bare `accuracy()` function is provided on purpose. CLAUDE.md §11 is
explicit that plain accuracy is not a reportable metric here: on the real
class balance (§7 Gate A expects ~3700 Ia to ~255 Ib/c), an always-predict-Ia
classifier scores high accuracy while being useless, which is exactly the
failure mode macro-F1 exists to catch. If a future need for accuracy shows up
(e.g. as one diagnostic column among many), add it then with the specific
justification, rather than by default here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion matrix, rows = true class, columns = predicted class.

    Attributes:
        raw: `(K, K)` integer counts.
        normalized: `(K, K)` float, each row divided by that row's sum (i.e.
            recall-style normalization: `normalized[i, j]` is the fraction of
            true-class-`i` examples predicted as class `j`). Rows with zero
            support are all-zero rather than NaN.
    """

    raw: np.ndarray
    normalized: np.ndarray


@dataclass(frozen=True)
class PerClassPRF1:
    """Per-class precision / recall / F1, each shape `(K,)`, indexed by class.

    A class with zero true support has precision/recall/F1 defined as 0.0
    (not NaN), by convention, so it drags down macro-F1 rather than silently
    dropping out of the average — CLAUDE.md's whole point in requiring
    macro-F1 is that rare/absent classes must be visible, not averaged away.
    """

    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> ConfusionMatrix:
    """Build a `num_classes x num_classes` confusion matrix.

    Args:
        y_true: `(N,)` integer array, true class index in `[0, num_classes)`.
        y_pred: `(N,)` integer array, predicted class index, same range.
        num_classes: number of classes `K`.

    Returns:
        `ConfusionMatrix` with `raw` counts and row-`normalized` fractions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    raw = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(raw, (y_true, y_pred), 1)

    row_sums = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(row_sums > 0, raw / np.maximum(row_sums, 1), 0.0)

    return ConfusionMatrix(raw=raw, normalized=normalized)


def per_class_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> PerClassPRF1:
    """Precision, recall, F1, and support for each of `num_classes` classes.

    Args:
        y_true: `(N,)` integer array, true class index in `[0, num_classes)`.
        y_pred: `(N,)` integer array, predicted class index, same range.
        num_classes: number of classes `K`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes, dtype=np.int64)

    for k in range(num_classes):
        true_k = y_true == k
        pred_k = y_pred == k
        tp = int(np.sum(true_k & pred_k))
        fp = int(np.sum(~true_k & pred_k))
        fn = int(np.sum(true_k & ~pred_k))
        support[k] = int(np.sum(true_k))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[k] = p
        recall[k] = r
        f1[k] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return PerClassPRF1(precision=precision, recall=recall, f1=f1, support=support)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1: unweighted mean of per-class F1 over all `num_classes`
    classes (zero-support classes count as F1 = 0 — see `PerClassPRF1`).

    This is the primary reportable metric per CLAUDE.md §11; plain accuracy
    is not provided (see module docstring).
    """
    prf1 = per_class_precision_recall_f1(y_true, y_pred, num_classes)
    return float(np.mean(prf1.f1))


def per_class_pr_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> np.ndarray:
    """One-vs-rest average precision (area under the precision-recall curve)
    for each class.

    Args:
        y_true: `(N,)` integer array, true class index in `[0, num_classes)`.
        y_prob: `(N, num_classes)` float array of predicted probabilities.
        num_classes: number of classes `K`.

    Returns:
        `(K,)` array of average precision. A class with zero positive
        examples in `y_true` gets `nan` (undefined, not zero — it must not
        silently pull an average down).

    Uses the standard average-precision definition: AP = mean of the
    precision achieved at each point a true positive is encountered, when
    ranking all examples by descending predicted score for that class. This
    is equivalent to the area under the (non-interpolated, step-function)
    precision-recall curve.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    ap = np.full(num_classes, np.nan)

    for k in range(num_classes):
        labels = (y_true == k).astype(np.int64)
        n_positive = int(labels.sum())
        if n_positive == 0:
            continue

        scores = y_prob[:, k]
        order = np.argsort(-scores, kind="stable")
        sorted_labels = labels[order]

        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)
        precision_at_i = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap[k] = float(np.mean(precision_at_i[sorted_labels == 1]))

    return ap


@dataclass(frozen=True)
class CalibrationResult:
    """Output of `expected_calibration_error`, including reliability-diagram data.

    Attributes:
        ece: scalar expected calibration error.
        bin_edges: `(num_bins + 1,)` confidence bin edges, in `[0, 1]`.
        bin_confidence: `(num_bins,)` mean top-1 predicted confidence per bin,
            `nan` for empty bins.
        bin_accuracy: `(num_bins,)` mean top-1 accuracy per bin, `nan` for
            empty bins.
        bin_count: `(num_bins,)` number of examples per bin.
    """

    ece: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15
) -> CalibrationResult:
    """Expected calibration error (ECE) over top-1 predicted confidence.

    Args:
        y_true: `(N,)` integer array, true class index.
        y_prob: `(N, num_classes)` float array of predicted probabilities.
        num_bins: number of equal-width confidence bins in `[0, 1]`.

    Returns:
        `CalibrationResult`. ECE is the confidence-weighted mean absolute gap
        between top-1 confidence and top-1 accuracy across bins:
        `sum_b (count_b / N) * |accuracy_b - confidence_b|`.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    confidence = y_prob.max(axis=1)
    prediction = y_prob.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confidence = np.full(num_bins, np.nan)
    bin_accuracy = np.full(num_bins, np.nan)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    # Rightmost bin is closed on both ends so confidence == 1.0 lands in it.
    bin_idx = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, num_bins - 1)

    ece = 0.0
    for b in range(num_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_count[b] = count
        if count == 0:
            continue
        bin_confidence[b] = float(confidence[mask].mean())
        bin_accuracy[b] = float(correct[mask].mean())
        ece += (count / n) * abs(bin_accuracy[b] - bin_confidence[b])

    return CalibrationResult(
        ece=float(ece),
        bin_edges=bin_edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
    )
