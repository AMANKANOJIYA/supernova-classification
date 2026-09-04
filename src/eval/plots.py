"""Rendering for the eval harness. FROZEN.

The only file under src/eval/ that draws figures. `metrics.py` and
`phase_curves.py` are pure compute, returning plain dataclasses/arrays; this
module turns those return values into `matplotlib.figure.Figure` objects,
and `report.py` orchestrates compute-then-render (it also imports
`matplotlib.pyplot`, but only to call `plt.close()` on the figures this
module hands back after saving them — it does not draw anything itself).
Splitting it this way means the compute side stays usable in contexts a
figure isn't wanted (e.g. logging metrics to MLflow every epoch from inside
a training loop) without paying for or even importing a plotting backend
there.
"""

from typing import Sequence

import matplotlib.figure
import numpy as np

from src.eval.phase_curves import DEFAULT_CUTOFFS, PhaseCutoffResult, cutoff_label


def plot_macro_f1_vs_phase(
    results: dict[str, PhaseCutoffResult],
    class_names: Sequence[str] | None = None,
    cutoff_order: Sequence[float | None] = DEFAULT_CUTOFFS,
) -> matplotlib.figure.Figure:
    """The primary figure (CLAUDE.md §11): macro-F1 vs. days since first
    detection, with one additional line per class if `class_names` is given.
    """
    import matplotlib.pyplot as plt

    labels = [cutoff_label(c) for c in cutoff_order]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    macro = [results[label].macro_f1 for label in labels]
    ax.plot(x, macro, marker="o", linewidth=2.5, color="black", label="macro-F1")

    if class_names is not None:
        for k, cname in enumerate(class_names):
            per_class_f1 = [results[label].per_class.f1[k] for label in labels]
            ax.plot(x, per_class_f1, marker="o", linestyle="--", alpha=0.7, label=cname)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("days since first detection")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 vs. phase")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    results: dict[str, PhaseCutoffResult],
    class_names: Sequence[str],
    cutoff_order: Sequence[float | None] = DEFAULT_CUTOFFS,
    normalize: bool = True,
) -> matplotlib.figure.Figure:
    """Grid of confusion matrices, one per phase cutoff."""
    import matplotlib.pyplot as plt

    labels = [cutoff_label(c) for c in cutoff_order]
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, label in zip(axes[0], labels):
        cm = results[label].confusion
        matrix = cm.normalized if normalize else cm.raw
        im = ax.imshow(matrix, vmin=0, vmax=1 if normalize else None, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_title(f"t={label}d (n={results[label].n_objects})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    return fig
