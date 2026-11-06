"""Figure 4: macro-F1 vs. days since first detection, Tier A (LC+static,
full cutoff set including the pre-detection -5d point) vs. Tier B.

Tier B curve: the DEPLOYED configuration -- the LC+static fine-tune on
zeropoint-corrected Tier B, seed 42, the checkpoint serving runs
(outputs/finetune_bts_zpfix/with_static_checkpoint.pt). Every Tier B point,
including the full cutoff, is read from that run's own evaluation,
reports/zeropoint_fix_2026-11-05/finetune_seed42/with_static/eval/metrics.json.
Its full-cutoff value is the single deployed seed; the paper's Tier B table
reports the 5-seed mean for the same configuration.

Tier A curve: outputs/train_static/eval/metrics.json (LC+static, step 8);
Tier A needs no zeropoint conversion and is unchanged by the fix.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = "paper/figures/fig4_phase_curve.pdf"
TIER_B_DEPLOYED = "reports/zeropoint_fix_2026-11-05/finetune_seed42/with_static/eval/metrics.json"


def main() -> None:
    tier_a = json.load(open("outputs/train_static/eval/metrics.json"))
    tier_b = json.load(open(TIER_B_DEPLOYED))

    tier_a_x = [-5, 0, 5, 15, 30, 40]  # "full" plotted at a nominal position past 30
    tier_a_y = [tier_a[c]["macro_f1"] for c in ("-5", "0", "5", "15", "30", "full")]

    tier_b_x = [0, 5, 15, 30, 40]
    tier_b_y = [tier_b[c]["macro_f1"] for c in ("0", "5", "15", "30", "full")]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(tier_a_x, tier_a_y, "o-", color="#2E5C8A", lw=2, ms=6, label="Tier A test (LC+static)")
    ax.plot(tier_b_x, tier_b_y, "s-", color="#D98324", lw=2, ms=6, label="Tier B test (LC+static fine-tune, deployed)")

    ax.set_xticks([-5, 0, 5, 15, 30, 40])
    ax.set_xticklabels(["-5", "0", "5", "15", "30", "full"])
    ax.axvspan(-7, -2.5, color="0.93", zorder=0)
    ax.text(-5, 0.30, "pre-detection\n(Tier B: N/A)", fontsize=6.5, ha="center", color="0.4")
    ax.set_xlabel("Days since first detection")
    ax.set_ylabel("Macro-F1 (3-class)")
    ax.set_ylim(0.25, 0.75)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print("Tier B (LC+static, deployed seed 42) by cutoff:")
    for c, y in zip(["0", "5", "15", "30", "full"], tier_b_y):
        print(f"  {c}: {y:.4f}")


if __name__ == "__main__":
    main()
