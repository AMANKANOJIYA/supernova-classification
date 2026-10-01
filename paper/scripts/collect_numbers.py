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
# Gate C -- Tier B fine-tune report (reports/gate_c/gate_c_report.json is
# the git-tracked, ORIGINAL run; outputs/finetune_bts/*/eval/*.json were
# accidentally regenerated once this session (per GUIDE.md Sec 5/9) and are
# NOT used for any headline number below -- only for the deployed model's
# fully-intact Tier A comparator numbers, which were never touched).
# ---------------------------------------------------------------------------
gate_c = load("reports/gate_c/gate_c_report.json")

num("TierAFullMacroF", gate_c["tier_a_test_macro_f1_full"])
num("SimToRealGapStatic", gate_c["sim_to_real_gap"])
num("ZeroShotMacroF", gate_c["zero_shot"]["macro_f1_full"])
num("WithStaticMacroF", gate_c["with_static"]["macro_f1_full"])
num("LCOnlyMacroF", gate_c["lc_only"]["macro_f1_full"])
num("WithStaticDRBMacroF", gate_c["with_static_drb"]["macro_f1_full"])
num("TransferGain", gate_c["with_static"]["macro_f1_full"] - gate_c["zero_shot"]["macro_f1_full"])
num("SimToRealGapLCOnly", gate_c["tier_a_test_macro_f1_full"] - gate_c["lc_only"]["macro_f1_full"])

num("ZeroShotRocAucIa", gate_c["zero_shot"]["ia_vs_rest_roc_auc"])
num("WithStaticRocAucIa", gate_c["with_static"]["ia_vs_rest_roc_auc"])
num("LCOnlyRocAucIa", gate_c["lc_only"]["ia_vs_rest_roc_auc"])
num("WithStaticDRBRocAucIa", gate_c["with_static_drb"]["ia_vs_rest_roc_auc"])

for key, label in [("zero_shot", "ZeroShot"), ("with_static", "WithStatic"),
                    ("lc_only", "LCOnly"), ("with_static_drb", "WithStaticDRB")]:
    f1s = gate_c[key]["per_class_f1_full"]
    supports = gate_c[key]["per_class_support_full"]
    for i, cls in enumerate(["Ia", "Ibc", "II"]):
        num(f"{label}F{cls}", f1s[i])
    if key == "zero_shot":
        for i, cls in enumerate(["Ia", "Ibc", "II"]):
            num(f"TierBSupport{cls}", supports[i], "{:d}")

wstatic_phase = gate_c["with_static"]["phase_curve"]
for cutoff_key, label in [("0", "Zero"), ("5", "Five"), ("15", "Fifteen"), ("30", "Thirty"), ("full", "Full")]:
    num(f"TierBWithStaticPhase{label}", wstatic_phase[cutoff_key])

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
anomaly = load("outputs/anomaly/anomaly_results.json")
for tier, label in [("tier_a_roc_auc", "TierA"), ("tier_b_roc_auc", "TierB")]:
    d = anomaly[tier]
    for comp in ["reconstruction_error", "entropy", "mahalanobis", "isoforest", "combined"]:
        cname = "".join(w.capitalize() for w in comp.split("_"))
        num(f"Anomaly{label}{cname}", d[comp])
    num(f"Anomaly{label}NNormal", d["n_normal"], "{:d}")
    num(f"Anomaly{label}NAnomalous", d["n_anomalous"], "{:d}")

persisted = load("outputs/anomaly/persisted_scorers_meta.json")
num("AnomalyTierBCombinedThreeVerified", persisted["verification_tier_b_roc_auc"]["combined_3component"])
num("AnomalyTierBReconVerified", persisted["verification_tier_b_roc_auc"]["reconstruction_error"])
num("AnomalyTierBMahalVerified", persisted["verification_tier_b_roc_auc"]["mahalanobis"])
num("AnomalyTierBIsoVerified", persisted["verification_tier_b_roc_auc"]["isoforest"])

# Recompute the "entropy dropped" 3-component combined score directly from
# the per-object score files, independently of anomaly_results.json's own
# 4-component "combined" field (same method GUIDE.md Sec 7 verifies with).
import numpy as np  # noqa: E402


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
tier_b_normal = load("outputs/anomaly/scores_tier_b_normal_test.json")
tier_b_weird = load("outputs/anomaly/scores_tier_b_weird.json")

combined3_tier_a = binary_auc(combine3(tier_a_normal, ref_stats), combine3(tier_a_exotic, ref_stats))
combined3_tier_b = binary_auc(combine3(tier_b_normal, ref_stats), combine3(tier_b_weird, ref_stats))
num("AnomalyTierACombinedThreeRecomputed", combined3_tier_a)
num("AnomalyTierBCombinedThreeRecomputed", combined3_tier_b)

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
# Gate D
# ---------------------------------------------------------------------------
gate_d = load("reports/gate_d/gate_d_top50.json")
bd = gate_d["breakdown"]
num("GateDNTotal", len(gate_d["objects"]), "{:d}")
num("GateDNArtifactLike", bd.get("artifact_like", 0), "{:d}")
num("GateDNGenuineAnomaly", bd.get("genuine_anomaly", 0), "{:d}")
num("GateDNShortSparse", bd.get("short_sparse", 0), "{:d}")
n_no_tree_node = sum(1 for o in gate_d["objects"] if o["label"].startswith("no_tree_node"))
n_agn = sum(1 for o in gate_d["objects"] if o["label"] == "AGN")
n_91bg = sum(1 for o in gate_d["objects"] if o["label"] == "91bg")
num("GateDNNoTreeNode", n_no_tree_node, "{:d}")
num("GateDNAGN", n_agn, "{:d}")
num("GateDNNinetyOneBG", n_91bg, "{:d}")

# ---------------------------------------------------------------------------
# Live Fink ingestion
# ---------------------------------------------------------------------------
ztf1 = load("reports/live_fink/ztf_run1.json")
ztf2 = load("reports/live_fink/ztf_run2_fixed.json")
lsst1 = load("reports/live_fink/lsst_run1.json")

num("LiveZtfRunOneNReceived", ztf1["n_received"], "{:d}")
num("LiveZtfRunOneNProduced", ztf1["n_produced"], "{:d}")
num("LiveZtfRunOneNSkipped", ztf1["n_skipped"], "{:d}")
pct("LiveZtfRunOneNegFluxFrac", ztf1["negative_flux_fraction"], decimals=1)
num("LiveZtfRunOneNDetObs", ztf1["n_detection_observations"], "{:d}")
num("LiveZtfRunTwoNReceived", ztf2["n_received"], "{:d}")
num("LiveZtfRunTwoNProduced", ztf2["n_produced"], "{:d}")
num("LiveZtfRunTwoNSkipped", ztf2["n_skipped"], "{:d}")
pct("LiveZtfRunTwoNegFluxFrac", ztf2["negative_flux_fraction"], decimals=1)
num(
    "LiveZtfTotalAlerts",
    ztf1["n_produced"] + ztf2["n_produced"],
    "{:d}",
)
num("LiveLsstNReceived", lsst1["n_received"], "{:d}")

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
else:
    gap(
        "Mean detections-per-object vs. forced-photometry-per-object and the "
        "windowed/unwindowed sequence-length distributions (Sec 4 windowing "
        "subsection, Fig. 3) require compute_windowing_stats.py, which reads "
        "a sample of Tier A objects from the interim parquet; see "
        "paper/data_derived/windowing_stats.json."
    )

# ---------------------------------------------------------------------------
# Serving benchmarks -- NOT ON DISK. GUIDE.md Sec 7 itself states these
# (p50/p99 latency, throughput) are reported only in CLAUDE.md prose from a
# Day-12 benchmark, with no stored JSON artifact; GUIDE.md's own reproduction
# attempt this session could not re-verify them either (Postgres port
# occupied). No macro is defined for these -- Sec 5/Table 8 must say so.
# ---------------------------------------------------------------------------
gap(
    "Serving latency (p50/p99/mean/max) and throughput (peak alerts/sec) are "
    "reported only in CLAUDE.md Day-12 prose. No JSON artifact stores them "
    "(confirmed: grepped reports/ and outputs/ for p50/p99/throughput, no "
    "hits), and GUIDE.md's own attempt to re-verify them this session failed "
    "(Postgres port occupied). No macro is defined for these numbers; Table "
    "8 and the abstract say so explicitly rather than repeating the prose "
    "figures as if they were freshly verified."
)
gap(
    "Majority-class and logistic-regression baseline rows (Table 6) were "
    "never run to completion or persisted anywhere on disk in this project "
    "-- src/eval/baselines.py exists but reports/ and outputs/ contain no "
    "baseline_report.json or equivalent. Producing one now would mean "
    "running a new baseline classifier fit, which the paper-writing brief's "
    "'no retraining' rule was read to forbid (baseline fitting is a form of "
    "model training, not figure generation) -- see paper/NUMBERS_GAPS.md. "
    "Table 6 marks these rows as unavailable rather than fabricating a "
    "plausible baseline number."
)
gap(
    "The Day-14 live-ZTF prior-correction analysis (raw vs. corrected "
    "class-distribution percentages, e.g. 'Ib/c 34.6% -> 13.9%') was "
    "computed once, live, against a Postgres query over that session's "
    "prediction rows, and was never persisted to a JSON file. It cannot be "
    "reproduced from any artifact in reports/ or outputs/ this session. The "
    "Discussion section describes the qualitative finding (training-prior "
    "leakage into output probabilities) without restating the specific "
    "percentages as disk-verified numbers."
)
gap(
    "The deployed Tier B LC-only model's intermediate phase-cutoff macro-F1 "
    "values (cutoffs 0/5/15/30) exist only in "
    "outputs/finetune_bts/lc_only/eval/metrics.json, which this session's "
    "accidental re-run of finetune_bts.py overwrote (GUIDE.md Sec 5): its "
    "own 'full' value, 0.6727254, differs from the git-tracked, originally- "
    "reported 'full' value in reports/gate_c/gate_c_report.json, 0.6715774, "
    "by 0.0011 -- a real, unresolved run-to-run non-determinism, not a typo. "
    "Figure 4 (phase curve) plots the deployed LC-only variant using this "
    "file for cutoffs 0/5/15/30 (the only place they exist) but pins its "
    "full-cutoff point to the original, git-tracked scalar "
    "(reports/gate_c/gate_c_report.json's lc_only.macro_f1_full, 0.6716) "
    "rather than this file's own drifted 'full' value, so the plotted curve "
    "matches the deployed model's headline number used everywhere else in "
    "the paper. The figure's caption states this sourcing and the 0.0011 "
    "discrepancy explicitly."
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
