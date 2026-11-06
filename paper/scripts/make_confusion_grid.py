"""Figure 5: row-normalized confusion matrices at each phase cutoff.

Top row: Tier A test (LC+static, outputs/train_static/eval/metrics.json).

Bottom row: Tier B test, the DEPLOYED configuration -- LC+static fine-tune on
zeropoint-corrected Tier B, seed 42 -- read from that run's own evaluation,
reports/zeropoint_fix_2026-11-05/finetune_seed42/with_static/eval/metrics.json.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "paper/figures/fig5_confusion_grid.pdf"
CLASS_NAMES = ["Ia", "Ib/c", "II"]
CUTOFFS = ["0", "5", "15", "30", "full"]
CUTOFF_LABELS = ["0d", "5d", "15d", "30d", "full"]


def draw_matrix(ax, norm: np.ndarray, raw: np.ndarray, title: str) -> None:
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            frac = norm[i, j]
            color = "white" if frac > 0.55 else "black"
            ax.text(j, i, f"{frac:.2f}\n({int(raw[i, j])})", ha="center", va="center", fontsize=6.3, color=color)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, fontsize=6.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6.5)
    ax.set_title(title, fontsize=7.5)


def main() -> None:
    tier_a = json.load(open("outputs/train_static/eval/metrics.json"))
    tier_b = json.load(open("reports/zeropoint_fix_2026-11-05/finetune_seed42/with_static/eval/metrics.json"))

    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.4))

    for col, cutoff in enumerate(CUTOFFS):
        d = tier_a[cutoff]
        draw_matrix(
            axes[0, col],
            np.array(d["confusion"]["normalized"]),
            np.array(d["confusion"]["raw"]),
            f"Tier A, {CUTOFF_LABELS[col]}\nmacro-F1={d['macro_f1']:.3f}",
        )
    for col, cutoff in enumerate(CUTOFFS):
        d = tier_b[cutoff]
        draw_matrix(
            axes[1, col],
            np.array(d["confusion"]["normalized"]),
            np.array(d["confusion"]["raw"]),
            f"Tier B, {CUTOFF_LABELS[col]}\nmacro-F1={d['macro_f1']:.3f}",
        )

    axes[0, 0].set_ylabel("True class", fontsize=8)
    axes[1, 0].set_ylabel("True class", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("Predicted", fontsize=7)

    fig.text(0.012, 0.75, "Tier A (LC+static)", fontsize=9, rotation=90, va="center", ha="center")
    fig.text(0.012, 0.28, "Tier B (LC+static, deployed)", fontsize=9, rotation=90, va="center", ha="center")

    fig.tight_layout(rect=(0.045, 0, 1, 1))
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
