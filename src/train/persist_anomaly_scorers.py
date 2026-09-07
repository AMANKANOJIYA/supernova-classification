"""Persist the fitted Mahalanobis + isolation-forest anomaly scorers (Day 13
preliminary item 2 -- CLAUDE.md §11 Day 12's own flagged gap: "Persisting the
fitted scorer objects themselves... is what would let the full 3-component
combination serve -- flagged here, not fixed today").

`src/train/train_anomaly.py` fits these two components on the SAME
`DECODER_FIT_SUBSAMPLE=60,000`-object subsample of Tier A train it uses to
train the reconstruction decoder (same `SEED=42`, same shuffle-then-slice
logic) -- but never saved anything beyond their SCORES on the four eval
populations (`outputs/anomaly/anomaly_results.json`) and the decoder's own
weights (`decoder_checkpoint.pt`). Recovering the fitted scorer objects
therefore does NOT require retraining anything: the frozen Tier A encoder
and the already-trained decoder are loaded as-is, the identical fit
population is reconstructed (same seed -> same 60,000 objects, same order),
and only a forward pass (encoder + decoder, no gradient/optimizer step) plus
the two scorers' own `.fit()` calls are re-run.

-----------------------------------------------------------------------------
Verification -- two tiers, deliberately asymmetric in cost
-----------------------------------------------------------------------------
Cheap, done for BOTH tiers: recompute the fit-population z-score reference
stats (`fit_zscore` on reconstruction_error/mahalanobis/isoforest over the
same 60,000-object fit population) and diff against `anomaly_results.json`'s
already-recorded `reference_zscore_stats`. `MahalanobisScorer.fit`/
`PineForestScorer.fit`/`EngineeredFeatureNormalizer.fit` are deterministic
given the same input array in the same order, so a match here is strong
evidence this session's fit population, encoder, and decoder are all
identical to Day 10's -- not just similar.

Full ROC-AUC recompute, done for Tier B ONLY: Tier B's eval populations
(658 normal-test + 1045 weird = 1,703 objects) are cheap to re-score in
full, matching this branch's own established convention ("every EVALUATION
population is used in full, uncapped" -- Day 10 note). Tier A's exotic eval
population is 442,303 objects across 12 separately-read model directories
(Day 10: ~164-182s of read time PER directory alone) -- re-scoring it in
full here would cost as much compute as `train_anomaly.py`'s own eval pass
for no new information beyond what the cheap z-score check already confirms.
Tier A's full 3-component ROC-AUC is therefore NOT recomputed this session;
the individual per-component numbers already in `anomaly_results.json` are
unaffected regardless (they never depended on persisting these objects).

Output (new, this session): `outputs/anomaly/mahalanobis.npz` (mean,
precision), `outputs/anomaly/engineered_normalizer.npz` (mean, std),
`outputs/anomaly/forest.pkl` (the fitted PineForest/IsolationForest object,
whichever `train_anomaly.py`'s own fallback logic selected), and
`outputs/anomaly/persisted_scorers_meta.json` (forest_kind, the z-score
verification diff, and Tier B's real 3-component ROC-AUC).

Entry point: `python -m src.train.persist_anomaly_scorers`
"""

import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from src.anomaly.decoder import ReconstructionDecoder
from src.anomaly.pineforest import EngineeredFeatureNormalizer, IsolationForestScorer, PineForestScorer
from src.anomaly.scoring import binary_roc_auc, combine_equal_weight, fit_zscore, zscore, MahalanobisScorer
from src.train.dataset import load_alerts, load_object_class_ids, split_by_object_id
from src.train.train_anomaly import (
    DECODER_FIT_SUBSAMPLE,
    OUTPUT_DIR,
    SEED,
    TIER_A_PARQUET,
    load_tier_a_encoder,
    load_tier_b_weird,
    score_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DECODER_CHECKPOINT = OUTPUT_DIR / "decoder_checkpoint.pt"
ANOMALY_RESULTS_JSON = OUTPUT_DIR / "anomaly_results.json"

MAHALANOBIS_NPZ = OUTPUT_DIR / "mahalanobis.npz"
ENGINEERED_NORMALIZER_NPZ = OUTPUT_DIR / "engineered_normalizer.npz"
FOREST_PKL = OUTPUT_DIR / "forest.pkl"
PERSISTED_META_JSON = OUTPUT_DIR / "persisted_scorers_meta.json"


def _set_seed() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def _load_decoder() -> ReconstructionDecoder:
    ckpt = torch.load(DECODER_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    decoder = ReconstructionDecoder(
        encoder_hidden_size=cfg["encoder_hidden_size"],
        hidden_size=cfg["decoder_hidden_size"],
        dropout=cfg["dropout"],
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    return decoder


def main() -> None:
    _set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANOMALY_RESULTS_JSON) as f:
        prior_results = json.load(f)

    encoder = load_tier_a_encoder()
    decoder = _load_decoder()

    # Cheap pass first (object_id/class_id only, per dataset.py's own
    # documented pattern -- reconstructing every CanonicalAlert in the full
    # 843,617-object corpus takes ~50 minutes; this needs only ~186,542 of
    # them (60,000 fit + ~126,542 test), so computing the split on the
    # lightweight id-only rows and then loading ONLY those ids' full
    # CanonicalAlerts avoids reconstructing the ~657,000 objects (train
    # minus the fit subsample, plus val) neither the fit nor the
    # verification step ever needs.
    logger.info("Loading Tier A object_id/class_id only from %s (cheap pass)", TIER_A_PARQUET)
    class_ids = load_object_class_ids(TIER_A_PARQUET)
    split = split_by_object_id(class_ids, seed=SEED, train_frac=0.7, val_frac=0.15)
    # Same order dataset.load_alerts(...) would have produced (same parquet
    # row order, same filter-by-membership) -- shuffle-then-slice below
    # reproduces train_anomaly.py's own fit population exactly, since
    # random.Random(seed).shuffle depends only on sequence length and seed,
    # not on element type/identity.
    train_ids_in_order = [c.object_id for c in class_ids if c.object_id in split.train]
    del class_ids

    rng = random.Random(SEED)
    rng.shuffle(train_ids_in_order)
    fit_ids = frozenset(train_ids_in_order[:DECODER_FIT_SUBSAMPLE])
    del train_ids_in_order
    logger.info("Reconstructed fit population ids: %d objects (matches train_anomaly.py's own)", len(fit_ids))

    logger.info("Loading fit population CanonicalAlerts (id-filtered)...")
    fit_alerts = load_alerts(TIER_A_PARQUET, object_ids=fit_ids)

    logger.info("Scoring fit population (encoder+decoder forward pass only, no training)...")
    fit_scores, fit_engineered = score_dataset(encoder, decoder, fit_alerts)
    del fit_alerts

    logger.info("Fitting MahalanobisScorer + EngineeredFeatureNormalizer...")
    mahalanobis_scorer = MahalanobisScorer.fit(fit_scores.latent)
    engineered_normalizer = EngineeredFeatureNormalizer.fit(fit_engineered)
    fit_pine_X = np.concatenate([fit_scores.latent, engineered_normalizer.transform(fit_engineered)], axis=1)

    try:
        forest = PineForestScorer(random_seed=SEED).fit(fit_pine_X)
        forest_kind = "PineForest"
    except Exception as exc:
        logger.warning("PineForest.fit failed (%s: %s) -- falling back to IsolationForest", type(exc).__name__, exc)
        forest = IsolationForestScorer(random_seed=SEED).fit(fit_pine_X)
        forest_kind = "IsolationForest"
    logger.info("Isolation-forest component: using %s (prior run recorded: %s)", forest_kind, prior_results["forest_kind"])

    # --- persist ---
    np.savez(MAHALANOBIS_NPZ, mean=mahalanobis_scorer.mean, precision=mahalanobis_scorer.precision)
    np.savez(ENGINEERED_NORMALIZER_NPZ, mean=engineered_normalizer.mean, std=engineered_normalizer.std)
    with open(FOREST_PKL, "wb") as f:
        pickle.dump(forest, f)
    logger.info("Wrote %s, %s, %s", MAHALANOBIS_NPZ, ENGINEERED_NORMALIZER_NPZ, FOREST_PKL)

    # --- cheap verification: fit-population z-score stats vs. prior ---
    ref_stats = prior_results["reference_zscore_stats"]
    fit_maha = mahalanobis_scorer.score(fit_scores.latent)
    fit_iso = forest.score(fit_pine_X)
    recomputed_ref_stats = {
        "reconstruction_error": fit_zscore(fit_scores.reconstruction_error),
        "mahalanobis": fit_zscore(fit_maha),
        "isoforest": fit_zscore(fit_iso),
    }
    logger.info("=== Cheap verification: recomputed fit z-score stats vs. anomaly_results.json ===")
    for key, (mean, std) in recomputed_ref_stats.items():
        prior_mean, prior_std = ref_stats[key]
        logger.info(
            "  %s: recomputed=(%.4f, %.4f) prior=(%.4f, %.4f) diff=(%.2e, %.2e)",
            key, mean, std, prior_mean, prior_std, mean - prior_mean, std - prior_std,
        )

    # --- full ROC-AUC recompute, Tier B only (cheap, small eval sets) ---
    def _score_population(alerts) -> dict:
        scores, engineered = score_dataset(encoder, decoder, alerts)
        maha = mahalanobis_scorer.score(scores.latent)
        pine_X = np.concatenate([scores.latent, engineered_normalizer.transform(engineered)], axis=1)
        iso = forest.score(pine_X)
        components = {"reconstruction_error": scores.reconstruction_error, "mahalanobis": maha, "isoforest": iso}
        z = [zscore(components[k], *ref_stats[k]) for k in components]
        return {"components": components, "combined": combine_equal_weight(z)}

    logger.info("Loading + scoring Tier B normal test + weird (full, uncapped)...")
    tier_b = load_tier_b_weird()
    tier_b_normal_result = _score_population(tier_b.normal_test)
    tier_b_weird_result = _score_population(tier_b.weird)

    def _auc_report(normal_result, anomalous_result) -> dict:
        n_normal = len(normal_result["combined"])
        n_anom = len(anomalous_result["combined"])
        y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
        report = {}
        for key in ("reconstruction_error", "mahalanobis", "isoforest"):
            score = np.concatenate([normal_result["components"][key], anomalous_result["components"][key]])
            report[key] = binary_roc_auc(y_true, score)
        combined_score = np.concatenate([normal_result["combined"], anomalous_result["combined"]])
        report["combined_3component"] = binary_roc_auc(y_true, combined_score)
        report["n_normal"] = n_normal
        report["n_anomalous"] = n_anom
        return report

    verify_tier_b = _auc_report(tier_b_normal_result, tier_b_weird_result)
    logger.info("=== Tier B (full, uncapped) 3-component ROC-AUC === %s", verify_tier_b)
    for key in ("reconstruction_error", "mahalanobis", "isoforest"):
        logger.info(
            "  tier_b.%s: recomputed=%.4f prior=%.4f diff=%.4f",
            key, verify_tier_b[key], prior_results["tier_b_roc_auc"][key],
            verify_tier_b[key] - prior_results["tier_b_roc_auc"][key],
        )

    meta = {
        "forest_kind": forest_kind,
        "fit_zscore_stats_diff": {
            k: {"mean_diff": recomputed_ref_stats[k][0] - ref_stats[k][0], "std_diff": recomputed_ref_stats[k][1] - ref_stats[k][1]}
            for k in recomputed_ref_stats
        },
        "verification_tier_b_roc_auc": verify_tier_b,
        "prior_tier_b_roc_auc": prior_results["tier_b_roc_auc"],
        "tier_a_note": (
            "Tier A's full 3-component ROC-AUC was NOT recomputed this session "
            "(442,303-object exotic eval set -- deemed too expensive for a "
            "verification pass; see module docstring). The cheap fit-population "
            "z-score check above covers Tier A equally (same fit population "
            "backs both tiers) -- only the expensive per-eval-set ROC-AUC number "
            "is Tier-B-only here."
        ),
        "combined_definition": (
            "reconstruction_error + mahalanobis + isoforest, equal-weight z-score "
            "average (entropy dropped, per CLAUDE.md Sec 11 Day 10 decision) -- "
            "this is the combination src.serve.model.AnomalyScorer now serves "
            "when these persisted artifacts are present."
        ),
    }
    with open(PERSISTED_META_JSON, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    logger.info("Wrote %s", PERSISTED_META_JSON)


if __name__ == "__main__":
    main()
