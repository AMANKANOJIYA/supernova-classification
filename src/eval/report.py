"""Single entry point: predictions + truth -> comparable figures + metrics.json
(CLAUDE.md §6 step 4 deliverable). FROZEN.

Every experiment (a baseline, a checkpoint at epoch N, a competitor's output)
should go through this one function so results are comparable by construction
— no experiment gets its own bespoke plotting code.
"""

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from src.eval.phase_curves import (
    DEFAULT_CUTOFFS,
    PhaseCutoffResult,
    PhasePrediction,
    evaluate_at_cutoffs,
    plot_confusion_matrices,
    plot_macro_f1_vs_phase,
)


def _sanitize(value):
    """Recursively convert numpy arrays/scalars to JSON-safe Python types,
    mapping NaN/Inf to None (bare `NaN` is not valid JSON, only a Python
    convention `json.dump` happens to allow by default)."""
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _result_to_dict(result: PhaseCutoffResult) -> dict:
    return {
        "cutoff": result.cutoff,
        "n_objects": result.n_objects,
        "n_excluded": result.n_excluded,
        "macro_f1": _sanitize(result.macro_f1),
        "per_class": {
            "precision": _sanitize(result.per_class.precision),
            "recall": _sanitize(result.per_class.recall),
            "f1": _sanitize(result.per_class.f1),
            "support": _sanitize(result.per_class.support),
        },
        "confusion": {
            "raw": _sanitize(result.confusion.raw),
            "normalized": _sanitize(result.confusion.normalized),
        },
        "pr_auc": _sanitize(result.pr_auc),
        "calibration": {
            "ece": _sanitize(result.calibration.ece),
            "bin_edges": _sanitize(result.calibration.bin_edges),
            "bin_confidence": _sanitize(result.calibration.bin_confidence),
            "bin_accuracy": _sanitize(result.calibration.bin_accuracy),
            "bin_count": _sanitize(result.calibration.bin_count),
        },
    }


def run_eval_report(
    predictions: Sequence[PhasePrediction],
    num_classes: int,
    class_names: Sequence[str],
    output_dir: str,
    cutoffs: Sequence[float | None] = DEFAULT_CUTOFFS,
    ece_num_bins: int = 15,
) -> dict:
    """Evaluate `predictions` at every phase cutoff and write a standard
    report to `output_dir`.

    Args:
        predictions: every available per-timestep prediction (see
            `src.eval.phase_curves.PhasePrediction`).
        num_classes: number of classes `K`; must match `len(class_names)`.
        class_names: display name for each class index, `[0, num_classes)`.
        output_dir: directory to write `metrics.json`,
            `macro_f1_vs_phase.png`, and `confusion_matrices.png` into.
            Created if it does not exist.
        cutoffs: phase cutoffs to evaluate at.
        ece_num_bins: bin count for expected calibration error.

    Returns:
        The same nested dict that is written to `metrics.json`, keyed by
        `cutoff_label` (e.g. `"-5"`, `"0"`, ..., `"full"`), for programmatic
        comparison between experiments without re-reading the file.
    """
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but num_classes={num_classes}"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = evaluate_at_cutoffs(predictions, num_classes, cutoffs, ece_num_bins)

    metrics_dict = {label: _result_to_dict(r) for label, r in results.items()}
    (out / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    fig1 = plot_macro_f1_vs_phase(results, class_names=class_names, cutoff_order=cutoffs)
    fig1.savefig(out / "macro_f1_vs_phase.png", dpi=150)

    fig2 = plot_confusion_matrices(results, class_names=class_names, cutoff_order=cutoffs)
    fig2.savefig(out / "confusion_matrices.png", dpi=150)

    import matplotlib.pyplot as plt

    plt.close(fig1)
    plt.close(fig2)

    return metrics_dict
