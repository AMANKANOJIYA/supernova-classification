"""Day 10 anomaly branch (CLAUDE.md §6 step 11): train the reconstruction
decoder, fit Mahalanobis + isolation-forest components on the frozen Tier A
LC-only encoder's latents, then evaluate all four components (+ their
combination) on two independent held-out sets neither seen anywhere in this
pipeline: Tier A exotics (newly downloaded model directories) and Tier B
real weird objects (BTS non-SN/AGN/SLSN/TDE/CSM-interacting).

Not Hydra-driven, like `src.train.finetune_bts` -- one script sharing one
encoder load, one Tier A split, and one Tier B join across all four
components and both eval sets, for the same "silent drift across separate
runs" reason that script's docstring gives.

Entry point: `python -m src.train.train_anomaly`
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src import taxonomy
from src.anomaly.decoder import (
    DecoderTrainConfig,
    ReconstructionDecoder,
    encoder_trunk_hidden_states,
    masked_mse_loss,
    per_object_reconstruction_error,
    shift_right,
)
from src.anomaly.pineforest import (
    EngineeredFeatureNormalizer,
    IsolationForestScorer,
    PineForestScorer,
    engineered_features,
)
from src.anomaly.scoring import (
    MahalanobisScorer,
    binary_roc_auc,
    combine_equal_weight,
    entropy_score,
    fit_zscore,
    zscore,
)
from src.features import tensorize
from src.models.lc_encoder import LCEncoder
from src.readers import bts_labels, elasticc2
from src.schema import CanonicalAlert
from src.train.dataset import collate_examples, load_alerts, split_by_object_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEED = 42
TIER_A_ENCODER_CHECKPOINT = Path("outputs/train/checkpoint.pt")
TIER_A_PARQUET = "data/interim/elasticc2/train_02_full_3class.parquet"
ELASTICC2_RAW_DIR = "data/raw/elasticc2"
GENTYPE_CSV = os.path.join(ELASTICC2_RAW_DIR, "elasticc2_gentypeofclassid.csv")

# The 18 exotic model directories fetched Day 10 (CLAUDE.md instruction) --
# NOT the 12 SN Ia/Ib-c/II directories already in TIER_A_PARQUET. Held out
# from every training population in this script, by construction.
EXOTIC_MODEL_DIR_NAMES: tuple[str, ...] = (
    "TDE",
    "SLSN-I+host",
    "SLSN-I_no_host",
    "CART",
    "ILOT",
    "PISN",
    "KN_B19",
    "KN_K17",
    "CLAGN",
    "uLens-Binary",
    "uLens-Single-GenLens",
    "uLens-Single_PyLIMA",
)

EXCLUDED_PERIODIC_DIR_NAMES: tuple[str, ...] = (
    "Cepheid",
    "RRL",
    "EB",
    "d-Sct",
    "dwarf-nova",
    "Mdwarf-flare",
)

BTSBOT_INTERIM_DIR = Path("data/interim")

# Time-boxing decisions (CLAUDE.md §12 precedent, e.g. Day 9's static-branch
# ablation note): full-scale (843,617-object) decoder training would cost
# hours per epoch on this machine on top of the concurrent photo-z-ablation
# run already in flight. Subsampling the FITTING population only -- every
# EVALUATION population below is used in full, uncapped, since ROC-AUC
# integrity depends on the eval set, not the fitting set.
DECODER_FIT_SUBSAMPLE = 60_000
DECODER_NUM_EPOCHS = 5
TIER_A_EXOTIC_EVAL_CAP: Optional[int] = None  # no cap -- exotics are the rare resource here

OUTPUT_DIR = Path("outputs/anomaly")
REPORT_DIR = Path("reports/gate_d")


def _set_seed() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Generic (not-3-class-filtered) tensorized dataset -- src.train.dataset's
# LightCurveDataset deliberately drops anything outside TARGET_CLASS_IDS;
# the anomaly branch needs exotic/non-SN objects to survive tensorization
# with their real class_id intact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyExample:
    object_id: str
    features: torch.Tensor  # (T, NUM_FEATURES)
    class_id: Optional[int]


class AnyClassDataset(Dataset):
    def __init__(self, alerts: list[CanonicalAlert], pre_detection_window_days: float = 30.0) -> None:
        self._examples: list[AnomalyExample] = []
        n_skipped = 0
        for alert in alerts:
            try:
                t = tensorize.tensorize_alert(
                    alert, pre_detection_window_days=pre_detection_window_days
                )
            except ValueError:
                n_skipped += 1
                continue
            self._examples.append(
                AnomalyExample(object_id=alert.object_id, features=t.features, class_id=alert.class_id)
            )
        if n_skipped:
            logger.warning(
                "AnyClassDataset: skipped %d/%d alert(s) that failed to tensorize "
                "(no observations or no detection)",
                n_skipped,
                len(alerts),
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> AnomalyExample:
        return self._examples[idx]


@dataclass(frozen=True)
class AnomalyBatch:
    object_id: list[str]
    features: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    class_id: list[Optional[int]]


def collate_any_class(items: list[AnomalyExample]) -> AnomalyBatch:
    lengths = [item.features.shape[0] for item in items]
    t_max = max(lengths)
    batch_size = len(items)
    num_features = items[0].features.shape[1]
    features = torch.zeros((batch_size, t_max, num_features), dtype=torch.float32)
    mask = torch.zeros((batch_size, t_max), dtype=torch.bool)
    for i, item in enumerate(items):
        t = item.features.shape[0]
        features[i, :t] = item.features
        mask[i, :t] = True
    return AnomalyBatch(
        object_id=[item.object_id for item in items],
        features=features,
        mask=mask,
        lengths=torch.tensor(lengths, dtype=torch.int64),
        class_id=[item.class_id for item in items],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tier_a_encoder() -> LCEncoder:
    ckpt = torch.load(TIER_A_ENCODER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LCEncoder(input_size=6, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("Loaded frozen Tier A LC-only encoder from %s", TIER_A_ENCODER_CHECKPOINT)
    return model


def read_one_exotic_dir(name: str, gentype_to_classid: dict) -> list[CanonicalAlert]:
    """Read a single exotic model directory. Kept separate from a
    "load all exotics" function so the caller (`main`) can score and
    discard each directory's alerts one at a time -- holding all 18
    directories' raw `CanonicalAlert` lists in memory simultaneously risked
    OOM this session (periodic-variable classes alone forced the scope cut
    recorded above; even the non-recurring classes kept here total several
    hundred thousand objects)."""
    model_dir = os.path.join(ELASTICC2_RAW_DIR, f"ELASTICC2_TRAIN_02_{name}")
    if not os.path.isdir(model_dir):
        logger.warning("Tier A exotics: %s not found on disk, skipping", model_dir)
        return []
    alerts, stats = elasticc2.read_model_directory(model_dir, gentype_to_classid)
    if stats.n_head_files_found != stats.n_phot_files_found or stats.n_pairs_read_ok != stats.n_head_files_found:
        logger.warning(
            "Tier A exotics: %s incomplete-looking read (head=%d phot=%d pairs_ok=%d) -- "
            "proceeding with what was read, not raising (expected file count for exotic "
            "directories was never asserted the way the 12 SN dirs' 40+40 convention was)",
            model_dir,
            stats.n_head_files_found,
            stats.n_phot_files_found,
            stats.n_pairs_read_ok,
        )
    class_counts: dict = {}
    for a in alerts:
        class_counts[a.class_id] = class_counts.get(a.class_id, 0) + 1
    logger.info(
        "Tier A exotics: %s -> %d objects read, %d discarded (%s), class_id counts=%s",
        name,
        stats.n_objects_read,
        stats.n_objects_discarded,
        stats.discard_reasons,
        class_counts,
    )
    return alerts


@dataclass(frozen=True)
class TierBWeird:
    normal_test: list[CanonicalAlert]  # confirmed Ia/Ib-c/II, test split only
    weird: list[CanonicalAlert]  # mapped_other_leaf | no_tree_node | excluded_csm_interacting, pooled
    category_counts: dict[str, int]
    # object_id -> a human-readable label for Gate D plot titles/breakdowns,
    # since `weird` alerts with category in {no_tree_node,
    # excluded_csm_interacting} carry class_id=None (no taxonomy leaf exists
    # for CV/nova/LBV/LRN or for Ibn/Icn -- see src.readers.bts_labels) and
    # would otherwise show up unlabeled.
    label_by_object_id: dict[str, str]


def load_tier_b_weird() -> TierBWeird:
    bts_sample = bts_labels.fetch_bts_sample()
    classification_by_id = bts_labels.build_object_id_to_classification(bts_sample)

    normal_test: list[CanonicalAlert] = []
    weird: list[CanonicalAlert] = []
    category_counts: dict[str, int] = {}
    label_by_object_id: dict[str, str] = {}

    for split in ("train", "val", "test"):
        alerts = load_alerts(str(BTSBOT_INTERIM_DIR / f"btsbot_{split}.parquet"))
        for alert in alerts:
            classification = classification_by_id.get(alert.object_id)
            if classification is None:
                category_counts["not_in_bts_sample"] = category_counts.get("not_in_bts_sample", 0) + 1
                continue
            category = classification.category
            if category == "mapped" and classification.class_id in taxonomy.TARGET_CLASS_IDS:
                if split == "test":
                    normal_test.append(dataclasses_replace(alert, class_id=classification.class_id))
                    label_by_object_id[alert.object_id] = taxonomy.name(classification.class_id)
                category_counts[f"mapped_{classification.class_id}"] = (
                    category_counts.get(f"mapped_{classification.class_id}", 0) + 1
                )
                continue
            if category == "mapped":
                weird.append(dataclasses_replace(alert, class_id=classification.class_id))
                label_by_object_id[alert.object_id] = taxonomy.name(classification.class_id)
                category_counts["mapped_other_leaf"] = category_counts.get("mapped_other_leaf", 0) + 1
            elif category == "no_tree_node":
                weird.append(alert)  # class_id stays None -- confirmed real, no taxonomy leaf
                label_by_object_id[alert.object_id] = "no_tree_node (CV/nova/LBV/LRN)"
                category_counts["no_tree_node"] = category_counts.get("no_tree_node", 0) + 1
            elif category == "excluded_csm_interacting":
                weird.append(alert)
                label_by_object_id[alert.object_id] = "excluded_csm_interacting (Ibn/Icn)"
                category_counts["excluded_csm_interacting"] = (
                    category_counts.get("excluded_csm_interacting", 0) + 1
                )
            else:
                category_counts[category] = category_counts.get(category, 0) + 1

    logger.info("Tier B category counts across train+val+test: %s", category_counts)
    logger.info(
        "Tier B eval sets: normal_test(3-class, test split only)=%d weird(pooled)=%d",
        len(normal_test),
        len(weird),
    )
    return TierBWeird(
        normal_test=normal_test,
        weird=weird,
        category_counts=category_counts,
        label_by_object_id=label_by_object_id,
    )


# ---------------------------------------------------------------------------
# Decoder training
# ---------------------------------------------------------------------------


def train_decoder(encoder: LCEncoder, train_ds: AnyClassDataset, cfg: DecoderTrainConfig) -> ReconstructionDecoder:
    decoder = ReconstructionDecoder(
        encoder_hidden_size=cfg.encoder_hidden_size,
        hidden_size=cfg.decoder_hidden_size,
        dropout=cfg.dropout,
    )
    optimizer = AdamW(decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_any_class)

    for epoch in range(cfg.num_epochs):
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in loader:
            hidden = encoder_trunk_hidden_states(encoder, batch.features, batch.lengths)
            prev_hidden = shift_right(hidden)
            cond = batch.features[:, :, (0, 4, 5)]
            target = batch.features[:, :, (1, 2, 3)]

            pred = decoder(prev_hidden, cond)
            loss = masked_mse_loss(pred, target, batch.mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        logger.info(
            "decoder epoch %d/%d: loss=%.5f (%.1fs)",
            epoch + 1,
            cfg.num_epochs,
            epoch_loss / max(n_batches, 1),
            time.time() - t0,
        )
    return decoder


# ---------------------------------------------------------------------------
# Scoring: run every component over a dataset, return per-object arrays.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentScores:
    object_id: list[str]
    class_id: list[Optional[int]]
    reconstruction_error: np.ndarray
    entropy: np.ndarray
    mahalanobis: np.ndarray
    isoforest: np.ndarray
    latent: np.ndarray  # (N, hidden_size) -- kept for Gate D inspection


@torch.no_grad()
def score_dataset(
    encoder: LCEncoder,
    decoder: ReconstructionDecoder,
    alerts: list[CanonicalAlert],
    batch_size: int = 256,
) -> tuple[ComponentScores, np.ndarray]:
    ds = AnyClassDataset(alerts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_any_class)

    object_ids: list[str] = []
    class_ids: list[Optional[int]] = []
    recon_errors: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    engineered_rows: list[np.ndarray] = []

    alerts_by_id = {a.object_id: a for a in alerts}

    for batch in loader:
        recon = per_object_reconstruction_error(encoder, decoder, batch.features, batch.lengths, batch.mask)
        recon_errors.append(recon.numpy())

        logits = encoder(batch.features, batch.lengths)
        idx = (batch.lengths - 1).clamp(min=0)
        last_logits = logits[torch.arange(logits.shape[0]), idx]
        probs = torch.softmax(last_logits, dim=-1).numpy()
        entropies.append(entropy_score(probs))

        hidden = encoder_trunk_hidden_states(encoder, batch.features, batch.lengths)
        last_hidden = hidden[torch.arange(hidden.shape[0]), idx].numpy()
        latents.append(last_hidden)

        for oid in batch.object_id:
            engineered_rows.append(engineered_features(alerts_by_id[oid]))

        object_ids.extend(batch.object_id)
        class_ids.extend(batch.class_id)

    return ComponentScores(
        object_id=object_ids,
        class_id=class_ids,
        reconstruction_error=np.concatenate(recon_errors),
        entropy=np.concatenate(entropies),
        mahalanobis=np.zeros(len(object_ids)),  # filled in by caller (needs a fitted scorer)
        isoforest=np.zeros(len(object_ids)),  # filled in by caller
        latent=np.concatenate(latents, axis=0),
    ), np.stack(engineered_rows, axis=0)


def main() -> None:
    _set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    encoder = load_tier_a_encoder()

    logger.info("Loading Tier A alerts from %s", TIER_A_PARQUET)
    tier_a_alerts = load_alerts(TIER_A_PARQUET)
    logger.info("Loaded %d Tier A alerts", len(tier_a_alerts))
    split = split_by_object_id(tier_a_alerts, seed=SEED, train_frac=0.7, val_frac=0.15)
    tier_a_train = [a for a in tier_a_alerts if a.object_id in split.train]
    tier_a_test = [a for a in tier_a_alerts if a.object_id in split.test]
    logger.info("Tier A split: train=%d test=%d (val unused here)", len(tier_a_train), len(tier_a_test))
    del tier_a_alerts  # frees the ~126,543-object val split, never referenced again

    rng = random.Random(SEED)
    fit_pool = tier_a_train[:]
    rng.shuffle(fit_pool)
    fit_alerts = fit_pool[:DECODER_FIT_SUBSAMPLE]
    logger.info(
        "Decoder/Mahalanobis/PineForest FIT population: %d objects (subsampled from %d Tier A "
        "train objects -- time-boxing decision, see module docstring)",
        len(fit_alerts),
        len(tier_a_train),
    )

    fit_ds = AnyClassDataset(fit_alerts)
    decoder_cfg = DecoderTrainConfig(num_epochs=DECODER_NUM_EPOCHS)
    logger.info("Training reconstruction decoder: %s", decoder_cfg)
    decoder = train_decoder(encoder, fit_ds, decoder_cfg)
    decoder.eval()

    torch.save({"decoder_state_dict": decoder.state_dict(), "config": decoder_cfg.__dict__},
               OUTPUT_DIR / "decoder_checkpoint.pt")

    logger.info("Scoring FIT population (to fit Mahalanobis + isolation forest)...")
    fit_scores, fit_engineered = score_dataset(encoder, decoder, fit_alerts)

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
    logger.info("Isolation-forest component: using %s", forest_kind)

    # Reference (z-score) statistics for combining components -- fit on the
    # SAME fit population used for training/fitting every component, never on
    # an eval set, so the combination step introduces no additional leakage.
    fit_maha = mahalanobis_scorer.score(fit_scores.latent)
    fit_iso = forest.score(fit_pine_X)
    ref_stats = {
        "reconstruction_error": fit_zscore(fit_scores.reconstruction_error),
        "entropy": fit_zscore(fit_scores.entropy),
        "mahalanobis": fit_zscore(fit_maha),
        "isoforest": fit_zscore(fit_iso),
    }
    logger.info("Reference (fit-population) z-score stats: %s", ref_stats)

    def _score_and_combine(alerts: list[CanonicalAlert]) -> dict:
        scores, engineered = score_dataset(encoder, decoder, alerts)
        maha = mahalanobis_scorer.score(scores.latent)
        pine_X = np.concatenate([scores.latent, engineered_normalizer.transform(engineered)], axis=1)
        iso = forest.score(pine_X)

        components = {
            "reconstruction_error": scores.reconstruction_error,
            "entropy": scores.entropy,
            "mahalanobis": maha,
            "isoforest": iso,
        }
        z = [zscore(components[k], *ref_stats[k]) for k in components]
        combined = combine_equal_weight(z)
        return {
            "object_id": scores.object_id,
            "class_id": scores.class_id,
            "components": components,
            "combined": combined,
            "latent": scores.latent,
        }

    logger.info("Scoring Tier A test (normal reference, %d objects)...", len(tier_a_test))
    tier_a_normal_result = _score_and_combine(tier_a_test)
    del tier_a_test

    def _concat_results(results: list[dict]) -> dict:
        return {
            "object_id": sum((r["object_id"] for r in results), []),
            "class_id": sum((r["class_id"] for r in results), []),
            "components": {
                k: np.concatenate([r["components"][k] for r in results])
                for k in results[0]["components"]
            },
            "combined": np.concatenate([r["combined"] for r in results]),
            "latent": np.concatenate([r["latent"] for r in results], axis=0),
        }

    logger.info(
        "Loading + scoring Tier A exotics one directory at a time (%d dirs kept, %d excluded "
        "as periodic/recurring -- see EXCLUDED_PERIODIC_DIR_NAMES)...",
        len(EXOTIC_MODEL_DIR_NAMES),
        len(EXCLUDED_PERIODIC_DIR_NAMES),
    )
    gentype_to_classid = taxonomy.load_gentype_of_classid(GENTYPE_CSV)
    exotic_dir_results = []
    for name in EXOTIC_MODEL_DIR_NAMES:
        dir_alerts = read_one_exotic_dir(name, gentype_to_classid)
        if not dir_alerts:
            continue
        exotic_dir_results.append(_score_and_combine(dir_alerts))
        del dir_alerts  # score-and-discard: never hold more than one directory's raw alerts
    tier_a_exotic_result = _concat_results(exotic_dir_results)
    del exotic_dir_results
    logger.info("Scored %d Tier A exotic objects total", len(tier_a_exotic_result["object_id"]))

    logger.info("Loading Tier B weird-object join...")
    tier_b = load_tier_b_weird()
    logger.info("Scoring Tier B normal test (%d objects)...", len(tier_b.normal_test))
    tier_b_normal_result = _score_and_combine(tier_b.normal_test)
    logger.info("Scoring Tier B weird (%d objects)...", len(tier_b.weird))
    tier_b_weird_result = _score_and_combine(tier_b.weird)

    def _auc_report(normal_result: dict, anomalous_result: dict) -> dict:
        n_normal = len(normal_result["object_id"])
        n_anom = len(anomalous_result["object_id"])
        y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
        report = {}
        for key in ("reconstruction_error", "entropy", "mahalanobis", "isoforest"):
            score = np.concatenate([normal_result["components"][key], anomalous_result["components"][key]])
            report[key] = binary_roc_auc(y_true, score)
        combined_score = np.concatenate([normal_result["combined"], anomalous_result["combined"]])
        report["combined"] = binary_roc_auc(y_true, combined_score)
        report["n_normal"] = n_normal
        report["n_anomalous"] = n_anom
        return report

    tier_a_auc = _auc_report(tier_a_normal_result, tier_a_exotic_result)
    tier_b_auc = _auc_report(tier_b_normal_result, tier_b_weird_result)

    logger.info("=== Tier A exotics ROC-AUC (normal test vs exotics) === %s", tier_a_auc)
    logger.info("=== Tier B weird ROC-AUC (normal test vs weird) === %s", tier_b_auc)

    results = {
        "forest_kind": forest_kind,
        "decoder_fit_subsample": DECODER_FIT_SUBSAMPLE,
        "decoder_num_epochs": DECODER_NUM_EPOCHS,
        "reference_zscore_stats": ref_stats,
        "tier_a_roc_auc": tier_a_auc,
        "tier_b_roc_auc": tier_b_auc,
        "tier_b_category_counts": tier_b.category_counts,
    }
    with open(OUTPUT_DIR / "anomaly_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Wrote %s", OUTPUT_DIR / "anomaly_results.json")

    # Persist per-object scores for every scored population, for Gate D.
    def _save_population(name: str, result: dict) -> None:
        rows = {
            "object_id": result["object_id"],
            "class_id": result["class_id"],
            "combined": result["combined"].tolist(),
            **{k: v.tolist() for k, v in result["components"].items()},
        }
        with open(OUTPUT_DIR / f"scores_{name}.json", "w") as f:
            json.dump(rows, f)
        np.save(OUTPUT_DIR / f"latent_{name}.npy", result["latent"])

    _save_population("tier_a_normal_test", tier_a_normal_result)
    _save_population("tier_a_exotic", tier_a_exotic_result)
    _save_population("tier_b_normal_test", tier_b_normal_result)
    _save_population("tier_b_weird", tier_b_weird_result)

    with open(OUTPUT_DIR / "tier_b_label_by_object_id.json", "w") as f:
        json.dump(tier_b.label_by_object_id, f)

    logger.info("Done. See %s and %s for per-object scores.", OUTPUT_DIR, REPORT_DIR)


if __name__ == "__main__":
    main()
