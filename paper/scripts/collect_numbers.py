"""Generate paper/numbers.tex — every numerical value used in the report,
as a `\\newcommand`, read directly from the JSON report/output artifacts
already on disk (reports/, outputs/) plus the small derived-stats JSON files
this same paper/scripts/ directory produces from that same on-disk data
(paper/data_derived/*.json — see compute_exotic_spans.py and
compute_windowing_stats.py).

Hard rule (paper spec): every number in the .tex body must come from a macro
defined here, and every macro here must be traceable to a stored JSON file.
Nothing in this script is invented, estimated, or copied from CLAUDE.md /
RESULTS.md prose. Where a number the report wants does not exist on disk,
this script simply does not define a macro for it — the missing macro is
listed in `paper/NUMBERS_GAPS.md` (also written by this script) and the .tex
sections say so in place of a number.

Run: python paper/scripts/collect_numbers.py  (from repo root; needs the
project venv — `.venv/bin/python3 paper/scripts/collect_numbers.py`).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_TEX = ROOT / "paper" / "numbers.tex"
GAPS_MD = ROOT / "paper" / "NUMBERS_GAPS.md"
DERIVED = ROOT / "paper" / "data_derived"


def load(relpath: str):
    with open(ROOT / relpath) as f:
        return json.load(f)


macros: dict[str, str] = {}
gaps: list[str] = []


def num(name: str, value, fmt: str = "{:.4f}") -> None:
    """Register \\newcommand{\\<name>}{<formatted value>}. `name` must be
    all-letters (LaTeX \\newcommand names cannot contain digits/underscores)."""
    if not name.isalpha():
        raise ValueError(
            f"macro name {name!r} contains a non-letter character -- LaTeX "
            "\\newcommand names must be all-letters (no digits/underscores)"
        )
    if name in macros:
        raise ValueError(f"duplicate macro name {name!r}")
    if isinstance(value, str):
        macros[name] = value
    elif isinstance(value, int) and fmt == "{:.4f}":
        macros[name] = str(value)
    else:
        macros[name] = fmt.format(value)


def pct(name: str, fraction: float, decimals: int = 1) -> None:
    num(name, fraction * 100.0, fmt="{:." + str(decimals) + "f}")


def gap(description: str) -> None:
    gaps.append(description)


# ---------------------------------------------------------------------------
# Section 2/4: Gate A -- Tier A (ELAsTiCC2) per-directory stats
# ---------------------------------------------------------------------------
gate_a = load("reports/gate_a/gate_a_report.json")["per_directory"]

tier_a_total_objects = sum(v["n_objects_read"] for v in gate_a.values())
num("TierATotalObjects", tier_a_total_objects, "{:d}")
num("TierANumModelDirs", len(gate_a), "{:d}")

neg_flux_fracs = {k: v["negative_flux_fraction"] for k, v in gate_a.items()}
num("TierANegFluxMin", min(neg_flux_fracs.values()) * 100, "{:.1f}")
num("TierANegFluxMax", max(neg_flux_fracs.values()) * 100, "{:.1f}")
min_dir = min(neg_flux_fracs, key=neg_flux_fracs.get)
max_dir = max(neg_flux_fracs, key=neg_flux_fracs.get)
num("TierANegFluxMinDir", min_dir.replace("_", r"\_"), "{}")
num("TierANegFluxMaxDir", max_dir.replace("_", r"\_"), "{}")

# Per-class object totals (class_id keys: 2222=Ia, 2223=Ibc, 2224=II) --
# summed across whichever directories map to that class_id.
class_totals = {2222: 0, 2223: 0, 2224: 0}
for v in gate_a.values():
    for cid_str, count in v["class_counts_by_object"].items():
        cid = int(cid_str)
        if cid in class_totals:
            class_totals[cid] += count
num("TierACountIa", class_totals[2222], "{:d}")
num("TierACountIbc", class_totals[2223], "{:d}")
num("TierACountII", class_totals[2224], "{:d}")
pct("TierAFracIa", class_totals[2222] / tier_a_total_objects)
pct("TierAFracIbc", class_totals[2223] / tier_a_total_objects)
pct("TierAFracII", class_totals[2224] / tier_a_total_objects)
num("TierAImbalanceRatio", class_totals[2224] / class_totals[2222], "{:.2f}")

# Number of raw SNANA model directories per target class (Table 2 / Sec 3)
n_dirs_ia = sum(1 for v in gate_a.values() if "2222" in v["class_counts_by_object"])
n_dirs_ibc = sum(1 for v in gate_a.values() if "2223" in v["class_counts_by_object"])
n_dirs_ii = sum(1 for v in gate_a.values() if "2224" in v["class_counts_by_object"])
num("TierANumDirsIa", n_dirs_ia, "{:d}")
num("TierANumDirsIbc", n_dirs_ibc, "{:d}")
num("TierANumDirsII", n_dirs_ii, "{:d}")

# ---------------------------------------------------------------------------
# Gate A -- Tier B (BTSbot) stats
# ---------------------------------------------------------------------------
gate_a_bts = load("reports/gate_a_btsbot/gate_a_btsbot_report.json")
num("TierBNObjTrain", gate_a_bts["n_objects"]["train"], "{:d}")
num("TierBNObjVal", gate_a_bts["n_objects"]["val"], "{:d}")
num("TierBNObjTest", gate_a_bts["n_objects"]["test"], "{:d}")
num(
    "TierBNObjTotal",
    gate_a_bts["n_objects"]["train"] + gate_a_bts["n_objects"]["val"] + gate_a_bts["n_objects"]["test"],
    "{:d}",
)
pct("TierBNegFluxFrac", gate_a_bts["negative_flux_fraction"], decimals=1)

cc = gate_a_bts["class_id_counts_bts_join"]
num("TierBJoinCountIa", cc.get("2222", 0), "{:d}")
num("TierBJoinCountIbc", cc.get("2223", 0), "{:d}")
num("TierBJoinCountII", cc.get("2224", 0), "{:d}")
num("TierBJoinRatio", cc.get("2222", 0) / cc.get("2223", 0), "{:.1f}")
num("TierBObsBandR", gate_a_bts["observations_per_band"]["ztf_r"], "{:d}")
num("TierBObsBandG", gate_a_bts["observations_per_band"]["ztf_g"], "{:d}")

# ---------------------------------------------------------------------------
# Gate C -- Tier B, ZEROPOINT-CORRECTED (2026-11-05 fix; see
# reports/zeropoint_fix_2026-11-05/README.md). The pre-fix Tier B numbers
# (reports/gate_c/gate_c_report.json) were measured on fluxes stored ~10x too
# faint and are superseded; they are kept below only as PreFix* / OldFlux*
# macros so the paper can state what the conversion error produced.
#
# Fine-tuned variants: mean (and SD) over the 5-seed sweep (seeds 42/1/2/3/4,
# seed_sweep.json "fixed_flux_seed*"). Zero-shot is inference-only and
# seed-independent. The DEPLOYED checkpoint is the seed-42 LC+static run
# (finetune_seed42/), which serving loads; Deployed* macros are that one seed.
# ---------------------------------------------------------------------------
FIX = "reports/zeropoint_fix_2026-11-05"
gate_c = load(f"{FIX}/finetune_seed42/gate_c_report.json")
sweep = load(f"{FIX}/seed_sweep.json")
sweep_summary = load(f"{FIX}/seed_sweep_summary.json")
fixed_runs = [sweep[k] for k in sorted(sweep) if k.startswith("fixed_flux_seed")]
num("TierBNSeeds", len(fixed_runs), "{:d}")

num("TierAFullMacroF", gate_c["tier_a_test_macro_f1_full"])
tier_b_mean: dict[str, float] = {}
for key, label in [("zero_shot", "ZeroShot"), ("with_static", "WithStatic"),
                    ("lc_only", "LCOnly"), ("with_static_drb", "WithStaticDRB")]:
    macro = np.array([r[key]["macro_f1"] for r in fixed_runs])
    summary = sweep_summary[f"fixed_flux/{key}"]
    assert abs(macro.mean() - summary["macro_f1_mean"]) < 1e-12
    tier_b_mean[key] = float(macro.mean())
    num(f"{label}MacroF", macro.mean())
    if key != "zero_shot":
        num(f"{label}MacroFSD", summary["macro_f1_sd"])
    num(f"{label}RocAucIa", float(np.mean([r[key]["roc"] for r in fixed_runs])))
    f1s = np.mean([r[key]["f1"] for r in fixed_runs], axis=0)
    for i, cls in enumerate(["Ia", "Ibc", "II"]):
        num(f"{label}F{cls}", f1s[i])

zs_lc = load(f"{FIX}/inference_only_evals/eval_extra.json")["zero_shot_lc_only"]
num("ZeroShotLCOnlyMacroF", zs_lc["macro_f1_full"])
num("ZeroShotLCOnlyRocAucIa", zs_lc["ia_vs_rest_roc_auc"])
for i, cls in enumerate(["Ia", "Ibc", "II"]):
    num(f"ZeroShotLCOnlyF{cls}", zs_lc["per_class_f1_full"][i])

for i, cls in enumerate(["Ia", "Ibc", "II"]):
    num(f"TierBSupport{cls}", gate_c["zero_shot"]["per_class_support_full"][i], "{:d}")

deployed = gate_c["with_static"]
num("DeployedMacroF", deployed["macro_f1_full"])
num("DeployedRocAucIa", deployed["ia_vs_rest_roc_auc"])
for i, cls in enumerate(["Ia", "Ibc", "II"]):
    num(f"DeployedF{cls}", deployed["per_class_f1_full"][i])
deployed_eval = load(f"{FIX}/finetune_seed42/with_static/eval/metrics.json")
assert abs(deployed_eval["full"]["macro_f1"] - deployed["macro_f1_full"]) < 1e-12
for cutoff_key, label in [("0", "Zero"), ("5", "Five"), ("15", "Fifteen"), ("30", "Thirty"), ("full", "Full")]:
    num(f"DeployedPhase{label}", deployed_eval[cutoff_key]["macro_f1"])
num("DeployedPhaseZeroPredIaFrac",
    sum(row[0] for row in deployed_eval["0"]["confusion"]["raw"]) / deployed_eval["0"]["n_objects"] * 100, "{:.0f}")

# Matched-configuration sim-to-real gaps (LC+static vs LC+static, LC-only vs
# LC-only) and like-for-like transfer gains. Tier A values are defined in the
# Tier A section below; computed here from the same metrics files.
_tier_a_static_full = load("outputs/train_static/eval/metrics.json")["full"]["macro_f1"]
_tier_a_lc_full = load("outputs/train/eval/metrics.json")["full"]["macro_f1"]
num("SimToRealGapStatic", _tier_a_static_full - tier_b_mean["with_static"])
num("SimToRealGapLCOnly", _tier_a_lc_full - tier_b_mean["lc_only"])
num("TransferGain", tier_b_mean["with_static"] - tier_b_mean["zero_shot"])
num("TransferGainLCOnly", tier_b_mean["lc_only"] - zs_lc["macro_f1_full"])
num("StaticGainTierB", tier_b_mean["with_static"] - tier_b_mean["lc_only"])
num("StaticGainOverSD",
    (tier_b_mean["with_static"] - tier_b_mean["lc_only"])
    / max(sweep_summary["fixed_flux/with_static"]["macro_f1_sd"], sweep_summary["fixed_flux/lc_only"]["macro_f1_sd"]),
    "{:.1f}")

# Pre-fix values, for stating what the conversion error produced. OldFlux* is
# the controlled comparison (same code, same object order, old fluxes, 5 seeds);
# PreFix* is what the paper originally reported (reports/gate_c, git-tracked).
for key, label in [("with_static", "WithStatic"), ("lc_only", "LCOnly")]:
    num(f"OldFlux{label}MacroF", sweep_summary[f"old_flux_same_order/{key}"]["macro_f1_mean"])
num("OldFluxStaticInversion",
    sweep_summary["old_flux_same_order/lc_only"]["macro_f1_mean"]
    - sweep_summary["old_flux_same_order/with_static"]["macro_f1_mean"])
prefix_gate_c = load("reports/gate_c/gate_c_report.json")
num("PreFixLCOnlyMacroF", prefix_gate_c["lc_only"]["macro_f1_full"])
num("PreFixWithStaticMacroF", prefix_gate_c["with_static"]["macro_f1_full"])
num("PreFixLCOnlyFIbc", prefix_gate_c["lc_only"]["per_class_f1_full"][1])
num("PreFixWithStaticFIbc", prefix_gate_c["with_static"]["per_class_f1_full"][1])

# Baselines (majority class, 5-feature logistic regression), both tiers, from
# the frozen src/eval/baselines.py + report.py harness; Tier B on corrected flux.
baselines = load(f"{FIX}/baselines/baseline_report.json")
for tier, tlabel in [("tier_a", "TierA"), ("tier_b", "TierB")]:
    for model, mlabel in [("majority", "Maj"), ("logreg", "LR")]:
        full = baselines[tier]["results"][model]["full"]
        num(f"Baseline{mlabel}{tlabel}MacroF", full["macro_f1"])
        num(f"Baseline{mlabel}{tlabel}RocAucIa", full["ia_vs_rest_roc_auc"])
        for i, cls in enumerate(["Ia", "Ibc", "II"]):
            num(f"Baseline{mlabel}{tlabel}F{cls}", full["per_class_f1"][i])

# Pre-registered anchors (CLAUDE.md §11 / Gate B) -- constants fixed before any
# result, defined here so the prose carries no literal numbers.
num("IbcBandLow", 0.40, "{:.2f}")
num("IbcBandHigh", 0.60, "{:.2f}")
num("RocCeilingLow", 0.93, "{:.2f}")
num("RocCeilingHigh", 0.96, "{:.2f}")
num("GateDArtifactAnchorPct", 68, "{:d}")

js = gate_c["join_stats"]
for split in ("train", "val", "test"):
    num(f"TierBJoinN{split.capitalize()}Total", sum(js[split].values()), "{:d}")
n3 = gate_c["n_3class_labeled"]
num("TierBLabeledTrain", n3["train"], "{:d}")
num("TierBLabeledVal", n3["val"], "{:d}")
num("TierBLabeledTest", n3["test"], "{:d}")

# ---------------------------------------------------------------------------
# Tier A eval reports (train / train_static / train_static_no_photoz / train_whxe)
# -- these outputs/*/eval/metrics.json files were never touched by the
# session's accidental re-run (only outputs/finetune_bts/* was), so their
# "full" cutoff values ARE the originally-reported ones (train_static's own
# full macro_f1, 0.716945209621613, matches gate_c_report.json's
# tier_a_test_macro_f1_full to 12 significant figures -- cross-checked here,
# not assumed).
# ---------------------------------------------------------------------------
metrics_paths = {
    "LCOnly": "outputs/train/eval/metrics.json",
    "LCStatic": "outputs/train_static/eval/metrics.json",
    "LCStaticNoPhotoz": "outputs/train_static_no_photoz/eval/metrics.json",
    "WHXE": "outputs/train_whxe/eval/metrics.json",
}
tier_a_metrics = {k: load(v) for k, v in metrics_paths.items()}

# cross-check the frozen-harness invariant this script relies on
assert abs(tier_a_metrics["LCStatic"]["full"]["macro_f1"] - gate_c["tier_a_test_macro_f1_full"]) < 1e-9

cutoff_label_map = {"-5": "MinusFive", "0": "Zero", "5": "Five", "15": "Fifteen", "30": "Thirty", "full": "Full"}
for model_label, d in tier_a_metrics.items():
    for cutoff_key, suffix in cutoff_label_map.items():
        if cutoff_key not in d:
            continue
        cd = d[cutoff_key]
        num(f"Tier{model_label}MacroF{suffix}", cd["macro_f1"])
        if cutoff_key == "full":
            num(f"Tier{model_label}NObjectsFull", cd["n_objects"], "{:d}")
            num(f"Tier{model_label}NExcludedFull", cd["n_excluded"], "{:d}")
            for i, cls in enumerate(["Ia", "Ibc", "II"]):
                num(f"Tier{model_label}F{cls}Full", cd["per_class"]["f1"][i])
                num(f"Tier{model_label}RocAuc{cls}Full", cd["roc_auc"][i])
                num(f"Tier{model_label}Support{cls}Full", cd["per_class"]["support"][i], "{:d}")
                num(f"Tier{model_label}Precision{cls}Full", cd["per_class"]["precision"][i])
                num(f"Tier{model_label}Recall{cls}Full", cd["per_class"]["recall"][i])
            # full confusion matrix, raw + normalized (row = true, col = pred)
            for i, cls_i in enumerate(["Ia", "Ibc", "II"]):
                for j, cls_j in enumerate(["Ia", "Ibc", "II"]):
                    num(f"Tier{model_label}Conf{cls_i}{cls_j}Raw", cd["confusion"]["raw"][i][j], "{:d}")
                    num(f"Tier{model_label}Conf{cls_i}{cls_j}Norm", cd["confusion"]["normalized"][i][j])
        if cutoff_key == "-5":
            num(f"Tier{model_label}NObjectsMinusFive", cd["n_objects"], "{:d}")
            num(f"Tier{model_label}NExcludedMinusFive", cd["n_excluded"], "{:d}")

# photo-z ablation gain accounting (Day 10, refuted-hypothesis note)
gain_with_photoz = tier_a_metrics["LCStatic"]["full"]["macro_f1"] - tier_a_metrics["LCOnly"]["full"]["macro_f1"]
gain_without_photoz = tier_a_metrics["LCStaticNoPhotoz"]["full"]["macro_f1"] - tier_a_metrics["LCOnly"]["full"]["macro_f1"]
num("PhotozGainWith", gain_with_photoz)
num("PhotozGainWithout", gain_without_photoz)
pct("PhotozGainFractionSurviving", gain_without_photoz / gain_with_photoz)
pct("PhotozGainFractionAttributable", 1 - gain_without_photoz / gain_with_photoz)

# WHXE node accuracy
node_acc = load("outputs/train_whxe/eval/node_accuracy.json")
num("WHXENodeTargetId", node_acc["target_node_id"], "{:d}")
num("WHXENodeAccuracy", node_acc["accuracy"])
num("WHXENodeNObjects", node_acc["n_objects"], "{:d}")
num("WHXEMacroFDelta", tier_a_metrics["WHXE"]["full"]["macro_f1"] - tier_a_metrics["LCOnly"]["full"]["macro_f1"])

# ---------------------------------------------------------------------------
# Span ablation (right-edge cap)
# ---------------------------------------------------------------------------
span = load("reports/span_ablation/span_ablation_report.json")
for k, label in [("100", "Hundred"), ("200", "TwoHundred"), ("full", "Uncapped")]:
    s = span["span_settings"][k]
    num(f"Span{label}MacroF", s["macro_f1_full"])
    num(f"Span{label}ElapsedSeconds", s["elapsed_seconds"], "{:.1f}")
    if k != "full":
        for cls, key in [("Ia", "Ia"), ("Ibc", "Ib/c"), ("II", "II")]:
            frac = s["fraction_affected_per_class"][key]["fraction_affected"]
            pct(f"Span{label}Affected{cls}", frac, decimals=2)
num("SpanChosenDefault", span["chosen_default"], "{:.0f}")
num("SpanTwoHundredCost", span["span_settings"]["full"]["macro_f1_full"] - span["span_settings"]["200"]["macro_f1_full"])
num("SpanTwoHundredMaxAffected",
    max(v["fraction_affected"] for v in span["span_settings"]["200"]["fraction_affected_per_class"].values()) * 100,
    "{:.2f}")

clagn = span["clagn_entropy_check"]
num("ClagnNObjects", clagn["n_clagn_objects"], "{:d}")
num("ClagnEntropyUncapped", clagn["baseline_max_span_days_none"]["mean_entropy"])
num("ClagnMaxProbUncapped", clagn["baseline_max_span_days_none"]["mean_max_softmax_prob"])
num("ClagnSeqLenUncapped", clagn["baseline_max_span_days_none"]["mean_seq_length"], "{:.1f}")
num("ClagnEntropyCapped", clagn["capped"]["mean_entropy"])
num("ClagnMaxProbCapped", clagn["capped"]["mean_max_softmax_prob"])
num("ClagnSeqLenCapped", clagn["capped"]["mean_seq_length"], "{:.1f}")

# ---------------------------------------------------------------------------
# Anomaly branch
# ---------------------------------------------------------------------------
# Tier A: outputs/anomaly/anomaly_results.json (simulated data, no zeropoint
# conversion, unaffected by the fix). Tier B: the zeropoint-corrected re-score
# with the same persisted scorers, reports/anomaly_tier_b_zpfix_2026-11-06/
# (score_tier_b.py; it refuses to write unless the 3-component ROC-AUC
# reproduces the fix pass's recorded 0.7793).
anomaly = load("outputs/anomaly/anomaly_results.json")
ANOMALY_B = "reports/anomaly_tier_b_zpfix_2026-11-06"
anomaly_b = load(f"{ANOMALY_B}/tier_b_roc_auc.json")["tier_b_roc_auc"]
for d, label in [(anomaly["tier_a_roc_auc"], "TierA"), (anomaly_b, "TierB")]:
    for comp in ["reconstruction_error", "entropy", "mahalanobis", "isoforest", "combined"]:
        cname = "".join(w.capitalize() for w in comp.split("_"))
        num(f"Anomaly{label}{cname}", d[comp])
    num(f"Anomaly{label}NNormal", d["n_normal"], "{:d}")
    num(f"Anomaly{label}NAnomalous", d["n_anomalous"], "{:d}")
num("AnomalyTierBEntropyDropGain", anomaly_b["combined_3component"] - anomaly_b["combined"])

# Recompute the "entropy dropped" 3-component combined score directly from
# the per-object score files, independently of anomaly_results.json's own
# 4-component "combined" field (same method GUIDE.md Sec 7 verifies with).

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
    # tie correction via average ranks
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank = (sums / counts)[inv]
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    r1 = avg_rank[y == 1].sum()
    auc = (r1 - n1 * (n1 + 1) / 2) / (n0 * n1)
    return float(auc)


ref_stats = anomaly["reference_zscore_stats"]
tier_a_normal = load("outputs/anomaly/scores_tier_a_normal_test.json")
tier_a_exotic = load("outputs/anomaly/scores_tier_a_exotic.json")
tier_b_normal = load(f"{ANOMALY_B}/scores_tier_b_normal_test.json")
tier_b_weird = load(f"{ANOMALY_B}/scores_tier_b_weird.json")

combined3_tier_a = binary_auc(combine3(tier_a_normal, ref_stats), combine3(tier_a_exotic, ref_stats))
combined3_tier_b = binary_auc(combine3(tier_b_normal, ref_stats), combine3(tier_b_weird, ref_stats))
num("AnomalyTierACombinedThreeRecomputed", combined3_tier_a)
num("AnomalyTierBCombinedThreeRecomputed", combined3_tier_b)

# Pre-fix Tier B combined score (same Day-10 per-object files the paper used
# before the fix), for the old-vs-new statement.
prefix_b_normal = load("outputs/anomaly/scores_tier_b_normal_test.json")
prefix_b_weird = load("outputs/anomaly/scores_tier_b_weird.json")
num("PreFixAnomalyTierBCombinedThree",
    binary_auc(combine3(prefix_b_normal, ref_stats), combine3(prefix_b_weird, ref_stats)))

# entropy means per population (raw entropy stored per-object; no
# max-softmax-probability is stored for these 4 populations except CLAGN
# above -- see gap note below)
num("EntropyTierANormal", float(np.mean(tier_a_normal["entropy"])))
num("EntropyTierAExotic", float(np.mean(tier_a_exotic["entropy"])))
num("EntropyTierBNormal", float(np.mean(tier_b_normal["entropy"])))
num("EntropyTierBWeird", float(np.mean(tier_b_weird["entropy"])))
gap(
    "Mean max-softmax-probability for the four main anomaly populations "
    "(Tier A normal/exotic, Tier B normal/weird) is not stored anywhere on "
    "disk -- only raw entropy is saved per object in "
    "outputs/anomaly/scores_*.json. Only CLAGN's mean max-softmax-probability "
    "(before/after span capping) survives, in "
    "reports/span_ablation/span_ablation_report.json. The Discussion section "
    "therefore reports entropy only for the four-population comparison, and "
    "max-softmax-probability only for CLAGN specifically."
)

# per-exotic-class entropy breakdown (Tier A), reproducing CLAUDE.md's Day-10
# table directly from scores_tier_a_exotic.json's stored class_id/entropy
# columns -- not copied from CLAUDE.md prose.
_EXOTIC_NAMES = {2232: "KN", 2235: "uLens", 2242: "SLSN", 2243: "TDE",
                 2244: "ILOT", 2245: "CART", 2246: "PISN", 2332: "CLAGN"}
by_class: dict[int, list[float]] = {}
for cid, ent in zip(tier_a_exotic["class_id"], tier_a_exotic["entropy"]):
    by_class.setdefault(cid, []).append(ent)
for cid, name in _EXOTIC_NAMES.items():
    vals = by_class.get(cid, [])
    if vals:
        num(f"ExoticEntropy{name}", float(np.mean(vals)))
        num(f"ExoticN{name}", len(vals), "{:d}")

# ---------------------------------------------------------------------------
# Gate D -- re-derived on zeropoint-corrected Tier B with the persisted
# (deployed) scorers: reports/gate_d_zpfix_2026-11-06/. The pre-fix top 50
# (reports/gate_d/) is kept for the composition comparison.
# ---------------------------------------------------------------------------
gate_d = load("reports/gate_d_zpfix_2026-11-06/gate_d_top50.json")
assert abs(gate_d["roc_auc_tier_b"]["combined_3component"] - anomaly_b["combined_3component"]) < 1e-12


def _gate_d_counts(objects: list[dict], prefix: str) -> None:
    num(f"{prefix}NTotal", len(objects), "{:d}")
    for cat, cname in [("artifact_like", "ArtifactLike"), ("genuine_anomaly", "GenuineAnomaly"),
                       ("short_sparse", "ShortSparse")]:
        num(f"{prefix}N{cname}", sum(1 for o in objects if o["category"] == cat), "{:d}")
    num(f"{prefix}NNoTreeNode", sum(1 for o in objects if o["label"].startswith("no_tree_node")), "{:d}")
    for lab, lname in [("AGN", "AGN"), ("91bg", "NinetyOneBG"), ("TDE", "TDE")]:
        num(f"{prefix}N{lname}", sum(1 for o in objects if o["label"] == lab), "{:d}")
    num(f"{prefix}NFromNormal", sum(1 for o in objects if o["source"] == "normal_test"), "{:d}")


_gate_d_counts(gate_d["objects"], "GateD")
_gate_d_counts(load("reports/gate_d/gate_d_top50.json")["objects"], "GateDPreFix")
num("GateDOverlapPreFix", gate_d["overlap_with_pre_fix_top50"], "{:d}")

# ---------------------------------------------------------------------------
# Live Fink ingestion. Day-13 runs (pre-fix model; kept for the CCD-edge
# cutout bug they exposed) and the 2026-11-06 runs with the corrected,
# deployed LC+static model.
# ---------------------------------------------------------------------------
ztf1 = load("reports/live_fink/ztf_run1.json")
ztf2 = load("reports/live_fink/ztf_run2_fixed.json")

num("LiveZtfRunOneNReceived", ztf1["n_received"], "{:d}")
num("LiveZtfRunOneNProduced", ztf1["n_produced"], "{:d}")
num("LiveZtfRunOneNSkipped", ztf1["n_skipped"], "{:d}")
pct("LiveZtfRunOneNegFluxFrac", ztf1["negative_flux_fraction"], decimals=1)
num("LiveZtfRunOneNDetObs", ztf1["n_detection_observations"], "{:d}")
num("LiveZtfRunTwoNReceived", ztf2["n_received"], "{:d}")
num("LiveZtfRunTwoNProduced", ztf2["n_produced"], "{:d}")
num("LiveZtfRunTwoNSkipped", ztf2["n_skipped"], "{:d}")
pct("LiveZtfRunTwoNegFluxFrac", ztf2["negative_flux_fraction"], decimals=1)

LIVE = "reports/live_2026-11-06"
for survey, slabel in [("ztf", "Ztf"), ("lsst", "Lsst")]:
    run = load(f"{LIVE}/live_{survey}_run.json")
    dist = load(f"{LIVE}/live_{survey}_classdist.json")
    prefix = f"LiveNov{slabel}"
    num(f"{prefix}NReceived", run["n_received"], "{:d}")
    num(f"{prefix}NProduced", run["n_produced"], "{:d}")
    num(f"{prefix}NSkipped", run["n_skipped"], "{:d}")
    num(f"{prefix}NDetObs", run["n_detection_observations"], "{:d}")
    pct(f"{prefix}NegFluxFrac", run["negative_flux_fraction"], decimals=1)
    num(f"{prefix}ElapsedMin", run["elapsed_seconds"] / 60.0, "{:.0f}")
    num(f"{prefix}NObjects", dist["n_objects"], "{:d}")
    for cls, cname in [("Ia", "Ia"), ("Ib/c", "Ibc"), ("II", "II")]:
        num(f"{prefix}Raw{cname}", dist["raw_pct"][cls], "{:.1f}")
        num(f"{prefix}ElasticcCorr{cname}", dist["elasticc_prior"]["midpoint"][cls], "{:.1f}")
        num(f"{prefix}TierBCorr{cname}", dist["tier_b_effective_prior"]["midpoint"][cls], "{:.1f}")
    for corr, clabel in [("elasticc_prior", "Elasticc"), ("tier_b_effective_prior", "TierB")]:
        lo, hi = dist[corr]["ibc_range_pct"]
        num(f"{prefix}{clabel}IbcRangeLow", lo, "{:.1f}")
        num(f"{prefix}{clabel}IbcRangeHigh", hi, "{:.1f}")
tb_prior = load(f"{LIVE}/live_ztf_classdist.json")["tier_b_effective_prior"]["train_prior"]
for i, cname in enumerate(["Ia", "Ibc", "II"]):
    pct(f"TierBEffectivePrior{cname}", tb_prior[i], decimals=1)
lsst_archive = load(f"{LIVE}/lsst_archive_reader_check.json")
num("LsstArchiveNPackets", lsst_archive["n_packets"], "{:d}")
num("LsstArchiveNConverted", lsst_archive["n_converted"], "{:d}")
_mant, _exp = f"{lsst_archive['max_abs_dmag_trigger']:.0e}".split("e")
num("LsstArchiveMaxDmag", f"${_mant}\\times 10^{{{int(_exp)}}}$")
num("LsstArchiveStampMin", lsst_archive["stamp_size_px_min"], "{:d}")
num("LsstArchiveStampMax", lsst_archive["stamp_size_px_max"], "{:d}")

# ---------------------------------------------------------------------------
# Serving benchmark (deployed LC+static model, uncontended full Tier B test
# replay, 2026-11-05) and isolated ONNX timing (2026-11-06).
# ---------------------------------------------------------------------------
bench = load(f"{FIX}/serving_lc_static/replay_benchmark_uncontended.json")["runs"][0]
num("ServingNAlerts", bench["n_rows"], "{:d}")
num("ServingNObjects", bench["n_objects"], "{:d}")
num("ServingLatencyPFifty", bench["latency_ms"]["p50"], "{:.1f}")
num("ServingLatencyPNinetyFive", bench["latency_ms"]["p95"], "{:.1f}")
num("ServingLatencyPNinetyNine", bench["latency_ms"]["p99"], "{:.1f}")
num("ServingThroughput", bench["throughput_alerts_per_sec_busy"], "{:.0f}")
num("ReplayMacroF", bench["macro_f1_latest"])
num("ReplayOfflineGap", bench["macro_f1_latest"] - deployed["macro_f1_full"])
onnx = load("reports/serving_2026-11-06/bench_onnx_lc_static.json")["onnx_isolated"]
num("OnnxMsPerAlertBatchOne", onnx["1"]["per_alert_ms_p50"], "{:.3f}")
num("OnnxMsPerAlertBatchThirtyTwo", onnx["32"]["per_alert_ms_p50"], "{:.3f}")

# ---------------------------------------------------------------------------
# Flux scale per tier (paper/scripts/compute_flux_scale.py)
# ---------------------------------------------------------------------------
flux = json.load(open(DERIVED / "flux_scale.json"))
num("FluxMedianTierA", flux["tier_a_median_detection_fluxcal"], "{:.0f}")
num("FluxMedianTierB", flux["tier_b_corrected_median_detection_fluxcal"], "{:.0f}")
num("FluxMedianTierBPreFix", flux["tier_b_prefix_median_detection_fluxcal"], "{:.0f}")
num("FluxBrightnessRatio", flux["brightness_ratio_tier_b_corrected_over_tier_a"], "{:.0f}")
num("FluxBugFactor", flux["row_ratio_corrected_over_prefix"]["median"], "{:.1f}")
num("FluxBugFactorLow", flux["row_ratio_corrected_over_prefix"]["p05"], "{:.1f}")
num("FluxBugFactorHigh", flux["row_ratio_corrected_over_prefix"]["p95"], "{:.1f}")

# ---------------------------------------------------------------------------
# Prediction panels (paper/scripts/make_prediction_panels.py)
# ---------------------------------------------------------------------------
panels_json = json.load(open(DERIVED / "prediction_panels.json"))
panels = panels_json["panels"]
num("PanelMinObs", panels_json["min_obs_well_sampled"], "{:d}")
num("PanelMisclassConfidencePct", panels["misclassified"]["confidence"] * 100, "{:.0f}")
num("PanelMisclassNObs", panels["misclassified"]["n_obs"], "{:d}")

# ---------------------------------------------------------------------------
# Test-suite counts around the zeropoint fix
# (reports/zeropoint_fix_2026-11-05/test_counts.json)
# ---------------------------------------------------------------------------
tc = load(f"{FIX}/test_counts.json")
num("TestsPassedPreFix", tc["prefix_suite"]["passed"], "{:d}")
num("TestsFailedInvarianceOnOldCode", tc["new_tests_on_prefix_code"]["failed"], "{:d}")
# Factor by which the inverted conversion mis-scaled LSST (nJy, AB zp 31.4):
# it multiplied by 10**((31.4-27.5)/2.5) where it should have divided by it.
num("LsstInversionFactor", 10 ** (2 * (31.4 - 27.5) / 2.5), "{:.0f}")

# ---------------------------------------------------------------------------
# Derived stats (produced by other paper/scripts/*.py from raw ELAsTiCC2 data)
# ---------------------------------------------------------------------------
exotic_spans_path = DERIVED / "exotic_spans.json"
if exotic_spans_path.exists():
    exotic = json.load(open(exotic_spans_path))
    for name in ("TDE", "EB", "CLAGN"):
        d = exotic[name]
        num(f"ExoticSpanMean{name}", d["mean_detection_span_days"], "{:.1f}")
        num(f"ExoticSpanMedian{name}", d["median_detection_span_days"], "{:.1f}")
        num(f"ExoticNobsMean{name}", d["mean_nobs_per_object"], "{:.1f}")
        num(f"ExoticNObjects{name}", d["n_objects_read"], "{:d}")
    for name in ("EB", "CLAGN"):
        num(f"ExoticSpanRatio{name}",
            exotic[name]["mean_detection_span_days"] / exotic["TDE"]["mean_detection_span_days"], "{:.0f}")
        num(f"ExoticNobsRatio{name}",
            exotic[name]["mean_nobs_per_object"] / exotic["TDE"]["mean_nobs_per_object"], "{:.1f}")
else:
    gap(
        "TDE/EB/CLAGN mean/median detection-span and NOBS/object figures "
        "(Sec 4 windowing subsection) require reading the raw ELAsTiCC2 "
        "exotic model directories via compute_exotic_spans.py; that script "
        "had not finished running when numbers.tex was last generated -- "
        "see paper/data_derived/exotic_spans.json."
    )

windowing_path = DERIVED / "windowing_stats.json"
if windowing_path.exists():
    w = json.load(open(windowing_path))
    num("WindowMeanDetections", w["mean_detections_per_object"], "{:.1f}")
    num("WindowMeanForcedPhot", w["mean_forced_photometry_per_object"], "{:.1f}")
    num("WindowMeanSeqLenUnwindowed", w["mean_seq_len_unwindowed"], "{:.1f}")
    num("WindowMeanSeqLenWindowed", w["mean_seq_len_windowed"], "{:.1f}")
    pct("WindowFracRemoved", w["fraction_observations_removed"], decimals=1)
    num("WindowNSampleObjects", w["n_sample_objects"], "{:d}")
    num("WindowForcedToDetRatio", w["mean_forced_photometry_per_object"] / w["mean_detections_per_object"], "{:.0f}")
else:
    gap(
        "Mean detections-per-object vs. forced-photometry-per-object and the "
        "windowed/unwindowed sequence-length distributions (Sec 4 windowing "
        "subsection, Fig. 3) require compute_windowing_stats.py, which reads "
        "a sample of Tier A objects from the interim parquet; see "
        "paper/data_derived/windowing_stats.json."
    )

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TEX, "w") as f:
    f.write("% AUTO-GENERATED by paper/scripts/collect_numbers.py -- do not edit by hand.\n")
    f.write("% Every value here is read from a JSON file under reports/ or outputs/, or\n")
    f.write("% from paper/data_derived/*.json (itself produced from raw project data by\n")
    f.write("% another paper/scripts/*.py script). Regenerate with:\n")
    f.write("%   .venv/bin/python3 paper/scripts/collect_numbers.py\n")
    for name in sorted(macros):
        f.write(f"\\newcommand{{\\{name}}}{{{macros[name]}}}\n")

with open(GAPS_MD, "w") as f:
    f.write("# Numbers requested by the paper spec that are not on disk\n\n")
    f.write("Generated by paper/scripts/collect_numbers.py. Each item below has no\n")
    f.write("corresponding macro in numbers.tex; the relevant .tex section states the\n")
    f.write("gap in place of a number, per the paper spec's absolute rule on numbers.\n\n")
    for g in gaps:
        f.write(f"- {g}\n\n")

print(f"Wrote {len(macros)} macros to {OUT_TEX}")
print(f"Wrote {len(gaps)} gap notes to {GAPS_MD}")
