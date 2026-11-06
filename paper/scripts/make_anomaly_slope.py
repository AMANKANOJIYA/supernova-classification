"""Figure 7: anomaly-component behaviour under domain shift -- a slope chart,
Tier A (simulated exotics) on the left axis, Tier B (real BTS "weird"
objects) on the right, one line per component. Tier A values come from
outputs/anomaly/anomaly_results.json (simulated data, no zeropoint
conversion, unaffected by the 2026-11-05 fix). Tier B values come from the
zeropoint-corrected re-score, reports/anomaly_tier_b_zpfix_2026-11-06/
(same persisted scorers, no refit). The 3-component ("entropy dropped")
combined score is recomputed from the per-object score files on both tiers,
the same way paper/scripts/collect_numbers.py does.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "paper/figures/fig7_anomaly_slope.pdf"
TIER_B_DIR = "reports/anomaly_tier_b_zpfix_2026-11-06"

COMPONENT_LABELS = {
    "reconstruction_error": "Reconstruction error",
    "mahalanobis": "Mahalanobis distance",
    "isoforest": "Isolation forest",
    "entropy": "Classifier entropy",
}
COMPONENT_COLORS = {
    "reconstruction_error": "#3E8E5A",
    "mahalanobis": "#2E5C8A",
    "isoforest": "#D98324",
    "entropy": "#8A2E5C",
}
COMPONENT_STYLES = {
    "reconstruction_error": "-",
    "mahalanobis": "-",
    "isoforest": "-",
    "entropy": "--",
}


def zscore(x, mean, std):
    return (np.asarray(x, dtype=np.float64) - mean) / std


def combine3(pop, ref):
    comps = ["reconstruction_error", "mahalanobis", "isoforest"]
    z = [zscore(pop[c], *ref[c]) for c in comps]
    return np.mean(np.stack(z, axis=0), axis=0)


def binary_auc(normal_scores, anomalous_scores):
    y = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    s = np.concatenate([normal_scores, anomalous_scores])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank = (sums / counts)[inv]
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    r1 = avg_rank[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n0 * n1))


def main() -> None:
    results = json.load(open("outputs/anomaly/anomaly_results.json"))
    tier_a = results["tier_a_roc_auc"]
    tier_b = json.load(open(f"{TIER_B_DIR}/tier_b_roc_auc.json"))["tier_b_roc_auc"]

    ref = results["reference_zscore_stats"]
    tier_a_normal = json.load(open("outputs/anomaly/scores_tier_a_normal_test.json"))
    tier_a_exotic = json.load(open("outputs/anomaly/scores_tier_a_exotic.json"))
    tier_b_normal = json.load(open(f"{TIER_B_DIR}/scores_tier_b_normal_test.json"))
    tier_b_weird = json.load(open(f"{TIER_B_DIR}/scores_tier_b_weird.json"))
    combined3_a = binary_auc(combine3(tier_a_normal, ref), combine3(tier_a_exotic, ref))
    combined3_b = binary_auc(combine3(tier_b_normal, ref), combine3(tier_b_weird, ref))

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    x = [0, 1]
    all_lines = []  # (comp_key, label_text, color, bold, y_left, y_right)
    for comp in ["reconstruction_error", "mahalanobis", "isoforest", "entropy"]:
        y = [tier_a[comp], tier_b[comp]]
        ax.plot(
            x,
            y,
            marker="o",
            ms=6,
            lw=2.2,
            color=COMPONENT_COLORS[comp],
            linestyle=COMPONENT_STYLES[comp],
            label=COMPONENT_LABELS[comp],
        )
        all_lines.append((COMPONENT_COLORS[comp], False, y[0], y[1]))

    # 3-component combination (entropy dropped) -- the practical output.
    y_combined = [combined3_a, combined3_b]
    ax.plot(x, y_combined, marker="s", ms=7, lw=2.6, color="0.15", label="Combined (entropy dropped)")
    all_lines.append(("0.15", True, y_combined[0], y_combined[1]))

    def place_labels(side_index: int, xytext_x: int, ha: str) -> None:
        """Place one value label per line, nudging vertically (in points) to
        avoid overlap when two lines land close together in data space."""
        items = sorted(
            [(entry[2 + side_index], entry[0], entry[1]) for entry in all_lines],
            key=lambda t: t[0],
            reverse=True,
        )
        min_gap_pts = 13.0
        placed_pts = []  # points-space y already used, in placement order
        for value, color, bold in items:
            base_pts = ax.transData.transform((0, value))[1]
            dy = 0.0
            # push down only far enough to clear every already-placed label
            for used_base, used_dy in placed_pts:
                if abs((base_pts + dy) - (used_base + used_dy)) < min_gap_pts:
                    dy = (used_base + used_dy) - base_pts - min_gap_pts
            placed_pts.append((base_pts, dy))
            ax.annotate(
                f"{value:.3f}",
                (side_index, value),
                xytext=(xytext_x, dy),
                textcoords="offset points",
                va="center",
                ha=ha,
                fontsize=8.5 if bold else 8,
                fontweight="bold" if bold else "normal",
                color=color,
            )

    place_labels(0, -8, "right")
    place_labels(1, 8, "left")

    ax.axhline(0.5, color="0.8", lw=0.8, ls=":", zorder=0)
    ax.text(0.5, 0.505, "chance", color="0.6", fontsize=7, ha="center")

    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Tier A\n(simulated exotics, $n$=" + f"{tier_a['n_normal']}" + "/" + f"{tier_a['n_anomalous']})",
         "Tier B\n(real BTS \"weird\", $n$=" + f"{tier_b['n_normal']}" + "/" + f"{tier_b['n_anomalous']})"]
    )
    ax.set_ylabel("ROC-AUC (normal vs. anomalous)")
    ax.set_ylim(0.15, 0.95)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print("combined3_a", combined3_a, "combined3_b", combined3_b)


if __name__ == "__main__":
    main()
