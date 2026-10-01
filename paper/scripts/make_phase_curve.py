"""Figure 4: macro-F1 vs. days since first detection, Tier A (LC+static,
full cutoff set including the pre-detection -5d point) vs. Tier B.

Tier B curve: the DEPLOYED LC-only fine-tune variant, not LC+static.

Sourcing, per point, and why it isn't a single flat read of one file:
- Full cutoff: reports/gate_c/gate_c_report.json ("lc_only"."macro_f1_full"),
  the intact, git-tracked scalar quoted everywhere else in this project as
  the deployed model's headline number (0.6716).
- Cutoffs 0/5/15/30: outputs/finetune_bts/lc_only/eval/metrics.json, the
  only place these intermediate-cutoff LC-only numbers exist at all. This
  file is itself the product of a later, non-deterministic re-run of the
  fine-tuning script (GUIDE.md Sec 5) -- its own "full" value has drifted to
  0.6727254, 0.0011 away from the original 0.6715774 used everywhere else.
  That's a real, small, already-documented run-to-run discrepancy (see
  paper/NUMBERS_GAPS.md and the Figure 5 caption, which hits the identical
  situation), not a typo and not something this script papers over: the
  intermediate points come from that re-run because nothing else has them,
  but the plotted endpoint is pinned to the original, authoritative
  full-cutoff scalar so this figure's drop matches the number quoted
  everywhere else, rather than silently reproducing the re-run's own
  0.0011 drift.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = "paper/figures/fig4_phase_curve.pdf"


def main() -> None:
    tier_a = json.load(open("outputs/train_static/eval/metrics.json"))
    gate_c = json.load(open("reports/gate_c/gate_c_report.json"))
    lc_only_cutoffs = json.load(open("outputs/finetune_bts/lc_only/eval/metrics.json"))

    tier_a_cutoffs = [-5, 0, 5, 15, 30, "full"]
    tier_a_x = [-5, 0, 5, 15, 30, 40]  # "full" plotted at a nominal position past 30
    tier_a_y = [tier_a[str(c)]["macro_f1"] for c in tier_a_cutoffs[:-1]] + [tier_a["full"]["macro_f1"]]

    tier_b_x = [0, 5, 15, 30, 40]
    tier_b_y = [
        lc_only_cutoffs["0"]["macro_f1"],
        lc_only_cutoffs["5"]["macro_f1"],
        lc_only_cutoffs["15"]["macro_f1"],
        lc_only_cutoffs["30"]["macro_f1"],
        gate_c["lc_only"]["macro_f1_full"],  # authoritative full-cutoff scalar, not the re-run file's own
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(tier_a_x, tier_a_y, "o-", color="#2E5C8A", lw=2, ms=6, label="Tier A test (LC+static)")
    ax.plot(tier_b_x, tier_b_y, "s-", color="#D98324", lw=2, ms=6, label="Tier B test (LC-only fine-tune, deployed)")

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
    print("Tier B (LC-only, deployed) by cutoff:")
    for c, y in zip(["0", "5", "15", "30", "full"], tier_b_y):
        print(f"  {c}: {y:.4f}")
    print(f"Tier A full: {tier_a_y[-1]:.4f}  Tier B full: {tier_b_y[-1]:.4f}  gap: {tier_a_y[-1] - tier_b_y[-1]:.4f}")


if __name__ == "__main__":
    main()
