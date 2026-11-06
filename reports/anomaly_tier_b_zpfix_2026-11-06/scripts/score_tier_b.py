"""Per-object Tier B anomaly scores on the zeropoint-corrected parquets, all four
components, persisted in the same format as `outputs/anomaly/scores_tier_b_*.json`.

The 2026-11-05 fix pass recorded only the corrected Tier B ROC-AUC summary for
three components (`reports/zeropoint_fix_2026-11-05/anomaly_tier_b/`). The paper's
anomaly table also reports classifier entropy and the 4-component combination, and
the per-object scores are needed to recompute everything from disk. This script
uses the persisted, deployed scorers (no refit), exactly as
`reports/gate_d_zpfix_2026-11-06/scripts/rederive_gate_d.py` does, and z-scores
every component against `anomaly_results.json`'s `reference_zscore_stats` -- the
same definition Day 10 used for both the 4-component and 3-component combinations.

It stops if the 3-component combined ROC-AUC does not reproduce the recorded 0.7793.

Run from the repo root: `.venv/bin/python reports/anomaly_tier_b_zpfix_2026-11-06/scripts/score_tier_b.py`
"""

import json
import pickle
from pathlib import Path

import numpy as np

from src.anomaly.pineforest import EngineeredFeatureNormalizer
from src.anomaly.scoring import MahalanobisScorer, binary_roc_auc, combine_equal_weight, zscore
from src.train.persist_anomaly_scorers import (
    ANOMALY_RESULTS_JSON,
    ENGINEERED_NORMALIZER_NPZ,
    FOREST_PKL,
    MAHALANOBIS_NPZ,
    _load_decoder,
    _set_seed,
)
from src.train.train_anomaly import load_tier_a_encoder, load_tier_b_weird, score_dataset

OUT = Path("reports/anomaly_tier_b_zpfix_2026-11-06")
EXPECTED_COMBINED3_AUC = 0.7793109466121784
ALL4 = ("reconstruction_error", "entropy", "mahalanobis", "isoforest")
THREE = ("reconstruction_error", "mahalanobis", "isoforest")


def main() -> None:
    _set_seed()
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
            "entropy": scores.entropy,
            "mahalanobis": mahalanobis.score(scores.latent),
            "isoforest": forest.score(pine_x),
        }
        z = {k: zscore(comps[k], *ref_stats[k]) for k in ALL4}
        return {
            "object_id": [a.object_id for a in alerts],
            "class_id": [a.class_id for a in alerts],
            "combined": combine_equal_weight([z[k] for k in ALL4]).tolist(),
            "combined_3component": combine_equal_weight([z[k] for k in THREE]).tolist(),
            **{k: np.asarray(v).tolist() for k, v in comps.items()},
        }

    tier_b = load_tier_b_weird()
    normal = score(tier_b.normal_test)
    weird = score(tier_b.weird)

    y = np.concatenate([np.zeros(len(normal["object_id"])), np.ones(len(weird["object_id"]))])
    roc = {
        k: binary_roc_auc(y, np.concatenate([normal[k], weird[k]]))
        for k in (*ALL4, "combined", "combined_3component")
    }
    if abs(roc["combined_3component"] - EXPECTED_COMBINED3_AUC) > 1e-6:
        raise SystemExit(f"combined_3component {roc['combined_3component']} != recorded {EXPECTED_COMBINED3_AUC}")
    roc["n_normal"] = len(normal["object_id"])
    roc["n_anomalous"] = len(weird["object_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores_tier_b_normal_test.json").write_text(json.dumps(normal))
    (OUT / "scores_tier_b_weird.json").write_text(json.dumps(weird))
    (OUT / "tier_b_roc_auc.json").write_text(json.dumps({
        "definition": "persisted Tier-A-fitted scorers, no refit; every component z-scored against "
                      "outputs/anomaly/anomaly_results.json reference_zscore_stats; 'combined' = 4-component "
                      "equal-weight average (Day-10 definition), 'combined_3component' drops entropy",
        "tier_b_roc_auc": roc,
        "mean_entropy": {"normal_test": float(np.mean(normal["entropy"])), "weird": float(np.mean(weird["entropy"]))},
    }, indent=2))
    print(json.dumps(roc, indent=2))


if __name__ == "__main__":
    main()
