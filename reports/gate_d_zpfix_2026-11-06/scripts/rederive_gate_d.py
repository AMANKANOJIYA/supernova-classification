"""Re-derive Gate D (CLAUDE.md §7) on the zeropoint-corrected Tier B parquets.

Uses the persisted, deployed anomaly scorers exactly as `src.serve.model.AnomalyScorer`
loads them (no refit): Tier A LC-only encoder + reconstruction decoder,
`mahalanobis.npz`, `engineered_normalizer.npz`, `forest.pkl`, z-scored against
`anomaly_results.json`'s `reference_zscore_stats`; combined = equal-weight z-score
average of reconstruction error, Mahalanobis and isolation forest (entropy dropped).

Before ranking anything it checks that this reproduces the recorded corrected
Tier B 3-component ROC-AUC (0.7793, reports/zeropoint_fix_2026-11-05/anomaly_tier_b/).
Categorisation reuses `src.train.gate_d_inspection._categorize` unchanged, so the
composition is comparable with the pre-fix `reports/gate_d/gate_d_top50.json`.

Run from the repo root: `.venv/bin/python reports/gate_d_zpfix_2026-11-06/scripts/rederive_gate_d.py`
"""

import collections
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.anomaly.pineforest import EngineeredFeatureNormalizer
from src.anomaly.scoring import MahalanobisScorer, binary_roc_auc, combine_equal_weight, zscore
from src.diagnostics import plot_light_curve
from src.train.gate_d_inspection import TOP_N, _categorize
from src.train.persist_anomaly_scorers import (
    ANOMALY_RESULTS_JSON,
    ENGINEERED_NORMALIZER_NPZ,
    FOREST_PKL,
    MAHALANOBIS_NPZ,
    _load_decoder,
    _set_seed,
)
from src.train.train_anomaly import load_tier_a_encoder, load_tier_b_weird, score_dataset

OUT = Path("reports/gate_d_zpfix_2026-11-06")
PRE_FIX = Path("reports/gate_d/gate_d_top50.json")
EXPECTED_COMBINED_AUC = 0.7793109466121784
COMPONENTS = ("reconstruction_error", "mahalanobis", "isoforest")


def main() -> None:
    _set_seed()
    (OUT / "light_curves").mkdir(parents=True, exist_ok=True)

    with open(ANOMALY_RESULTS_JSON) as f:
        ref_stats = json.load(f)["reference_zscore_stats"]
    encoder = load_tier_a_encoder()
    decoder = _load_decoder()
    maha_npz = np.load(MAHALANOBIS_NPZ)
    mahalanobis = MahalanobisScorer(mean=maha_npz["mean"], precision=maha_npz["precision"])
    eng_npz = np.load(ENGINEERED_NORMALIZER_NPZ)
    normalizer = EngineeredFeatureNormalizer(mean=eng_npz["mean"], std=eng_npz["std"])
    with open(FOREST_PKL, "rb") as f:
        forest = pickle.load(f)

    def score(alerts) -> dict:
        scores, engineered = score_dataset(encoder, decoder, alerts)
        pine_x = np.concatenate([scores.latent, normalizer.transform(engineered)], axis=1)
        comps = {
            "reconstruction_error": scores.reconstruction_error,
            "mahalanobis": mahalanobis.score(scores.latent),
            "isoforest": forest.score(pine_x),
        }
        z = [zscore(comps[k], *ref_stats[k]) for k in COMPONENTS]
        return {"components": comps, "combined": combine_equal_weight(z)}

    tier_b = load_tier_b_weird()
    normal = score(tier_b.normal_test)
    weird = score(tier_b.weird)

    y = np.concatenate([np.zeros(len(normal["combined"])), np.ones(len(weird["combined"]))])
    aucs = {k: binary_roc_auc(y, np.concatenate([normal["components"][k], weird["components"][k]])) for k in COMPONENTS}
    aucs["combined_3component"] = binary_roc_auc(y, np.concatenate([normal["combined"], weird["combined"]]))
    print("ROC-AUC:", aucs)
    if abs(aucs["combined_3component"] - EXPECTED_COMBINED_AUC) > 1e-6:
        raise SystemExit(f"combined ROC-AUC {aucs['combined_3component']} != recorded {EXPECTED_COMBINED_AUC}; not ranking")

    pooled = []
    for alerts, result, source in ((tier_b.weird, weird, "weird"), (tier_b.normal_test, normal, "normal_test")):
        for i, alert in enumerate(alerts):
            pooled.append((alert, float(result["combined"][i]), source, {k: float(result["components"][k][i]) for k in COMPONENTS}))
    pooled.sort(key=lambda r: r[1], reverse=True)

    rows = []
    for rank, (alert, combined, source, comps) in enumerate(pooled[:TOP_N], start=1):
        label = tier_b.label_by_object_id.get(alert.object_id, "unknown")
        category = _categorize(alert)
        rows.append({
            "rank": rank, "object_id": alert.object_id, "combined": combined, "components": comps,
            "source": source, "label": label, "category": category, "n_obs": len(alert.observations),
        })
        fig = plot_light_curve(alert, class_name=f"#{rank} {label} (combined={combined:.2f})")
        fig.savefig(OUT / "light_curves" / f"{rank:02d}_{alert.object_id}.png", dpi=110)
        plt.close(fig)

    with open(PRE_FIX) as f:
        pre = json.load(f)["objects"]
    pre_ids = {r["object_id"] for r in pre}
    new_ids = {r["object_id"] for r in rows}

    def composition(rs) -> dict:
        return {
            "by_label": dict(collections.Counter(r["label"] for r in rs).most_common()),
            "by_source": dict(collections.Counter(r["source"] for r in rs)),
            "by_category": dict(collections.Counter(r["category"] for r in rs)),
        }

    report = {
        "definition": "combined = equal-weight z-score average of reconstruction_error, mahalanobis, isoforest "
                      "(persisted scorers, no refit); Tier B normal_test (n=658) + weird (n=1045) pooled, top 50",
        "roc_auc_tier_b": aucs,
        "n_normal": len(normal["combined"]),
        "n_weird": len(weird["combined"]),
        "post_fix": composition(rows),
        "pre_fix": composition(pre),
        "overlap_with_pre_fix_top50": len(pre_ids & new_ids),
        "entered_top50": sorted(new_ids - pre_ids),
        "left_top50": sorted(pre_ids - new_ids),
        "objects": rows,
    }
    with open(OUT / "gate_d_top50.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("post_fix", "pre_fix", "overlap_with_pre_fix_top50")}, indent=2))


if __name__ == "__main__":
    main()
