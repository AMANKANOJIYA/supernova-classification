"""Figure 4: macro-F1 vs. days since first detection, Tier A (LC+static,
full cutoff set including the pre-detection -5d point) vs. Tier B.

Tier B curve: the LC+static fine-tune variant's own phase_curve, stored
intact in reports/gate_c/gate_c_report.json ("with_static"."phase_curve") --
NOT the deployed LC-only variant. The deployed LC-only variant's own
intermediate-cutoff numbers exist only inside
outputs/finetune_bts/lc_only/eval/metrics.json, which this project's own
GUIDE.md records was accidentally overwritten by a non-deterministic re-run
this session (full-cutoff value drifted from the original, git-tracked
0.6715774 to 0.6727254). Using that file here would silently mix an
intact, original number with re-run numbers on the same curve. The
LC+static variant's phase_curve was never touched by that re-run (only
lc_only's eval outputs were regenerated), so it is used instead, and this
substitution is stated in the caption text (paper/sections/results.tex),
not hidden.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = "paper/figures/fig4_phase_curve.pdf"


def main() -> None:
    tier_a = json.load(open("outputs/train_static/eval/metrics.json"))
    gate_c = json.load(open("reports/gate_c/gate_c_report.json"))

    tier_a_cutoffs = [-5, 0, 5, 15, 30, "full"]
    tier_a_x = [-5, 0, 5, 15, 30, 40]  # "full" plotted at a nominal position past 30
    tier_a_y = [tier_a[str(c)]["macro_f1"] for c in tier_a_cutoffs[:-1]] + [tier_a["full"]["macro_f1"]]

    tier_b_phase = gate_c["with_static"]["phase_curve"]
    tier_b_cutoffs = ["0", "5", "15", "30", "full"]
    tier_b_x = [0, 5, 15, 30, 40]
    tier_b_y = [tier_b_phase[c] for c in tier_b_cutoffs]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(tier_a_x, tier_a_y, "o-", color="#2E5C8A", lw=2, ms=6, label="Tier A test (LC+static)")
    ax.plot(tier_b_x, tier_b_y, "s-", color="#D98324", lw=2, ms=6, label="Tier B test (LC+static fine-tune)")

    ax.set_xticks([-5, 0, 5, 15, 30, 40])
    ax.set_xticklabels(["-5", "0", "5", "15", "30", "full"])
    ax.axvspan(-7, -2.5, color="0.93", zorder=0)
    ax.text(-5, ax.get_ylim()[0] if False else 0.30, "pre-detection\n(Tier B: N/A)", fontsize=6.5, ha="center", color="0.4")
    ax.set_xlabel("Days since first detection")
    ax.set_ylabel("Macro-F1 (3-class)")
    ax.set_ylim(0.25, 0.75)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
