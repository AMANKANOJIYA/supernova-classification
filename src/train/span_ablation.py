"""Day 12 task 1: right-edge windowing ablation.

`src.features.tensorize.tensorize_alert` bounded only the left edge before
today (`pre_detection_window_days`). Day 10's anomaly-branch finding (see
CLAUDE.md §11) is that nothing bounded the right edge: a fading transient's
`last_detection_mjd` arrives naturally close to `first_detection_mjd`, but a
periodic/continuously-variable source (AGN, periodic variable) keeps being
detected for years, and the Tier A causal GRU -- trained exclusively on
short-lived SN light curves -- turned out to be *confidently wrong* rather
than uncertain on those long sequences (CLAGN mean entropy 0.082, vs. 0.574
for Tier A's own normal test population).

This script measures, rather than assumes, what capping that right edge via
the new `max_span_days` parameter does:

1. Tier A test-split macro-F1 at `max_span_days` in {100, 200, None}, using
   the EXISTING Tier A LC-only checkpoint (`outputs/train/checkpoint.pt`) --
   inference only, no retraining. This is an inference-time question: the
   checkpoint's own training used `pre_detection_window_days=30.0`
   (configs/train/default.yaml) and untouched right edges, so this ablation
   asks how ROBUST that checkpoint already is to a differently-windowed
   input, not whether retraining under a cap would do better.
2. Fraction of Tier A test objects, per class, whose OWN detection span
   (`last_detection_mjd - first_detection_mjd`) exceeds the cap -- i.e.
   whose tensorized sequence is actually truncated by it. Computed directly
   from the raw alerts, not from the model, since it's a property of the
   input distribution, not the encoder.
3. The chosen default's effect on CLAGN's Day-10 entropy inversion: rereads
   the raw CLAGN model directory (held out from every training population,
   same as Day 10), tensorizes it under `max_span_days=None` (reproducing
   the Day-10 setup, as a sanity check against the recorded 0.082 figure)
   and under the chosen default, and reports mean entropy / mean
   max-softmax-probability for both.

Entry point: `python -m src.train.span_ablation`
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import taxonomy
from src.anomaly.scoring import entropy_score
from src.eval.report import run_eval_report
from src.features import tensorize
from src.models.lc_encoder import LCEncoder
from src.readers import elasticc2
from src.train.dataset import load_alerts, load_object_class_ids, split_by_object_id
from src.train.pretrain import CLASS_NAMES, collect_phase_predictions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEED = 42  # matches configs/train/default.yaml / src.train.pretrain -- same split as training
TIER_A_PARQUET = "data/interim/elasticc2/train_02_full_3class.parquet"
TIER_A_ENCODER_CHECKPOINT = Path("outputs/train/checkpoint.pt")
PRE_DETECTION_WINDOW_DAYS = 30.0  # unchanged from training; only the right edge is ablated here

ELASTICC2_RAW_DIR = "data/raw/elasticc2"
GENTYPE_CSV = os.path.join(ELASTICC2_RAW_DIR, "elasticc2_gentypeofclassid.csv")
CLAGN_MODEL_DIR_NAME = "CLAGN"
CLAGN_CLASS_ID = 2332  # AGN -- gentype 60 -> classid 2332, elasticc2_classidofgentype.csv

SPAN_SETTINGS: tuple[Optional[float], ...] = (100.0, 200.0, None)

_CLASS_ID_TO_INDEX: dict[int, int] = {cid: i for i, cid in enumerate(taxonomy.TARGET_CLASS_IDS)}

OUTPUT_DIR = Path("reports/span_ablation")


def _span_label(max_span_days: Optional[float]) -> str:
    return "full" if max_span_days is None else str(int(max_span_days))


def load_tier_a_encoder() -> LCEncoder:
    ckpt = torch.load(TIER_A_ENCODER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LCEncoder(input_size=6, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("Loaded frozen Tier A LC-only encoder from %s", TIER_A_ENCODER_CHECKPOINT)
    return model


# ---------------------------------------------------------------------------
# Part 1+2: Tier A test-split macro-F1 and per-class fraction-affected.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Example:
    object_id: str
    features: torch.Tensor
    class_index: int
    delta_t_days: torch.Tensor


class _SpanDataset(Dataset):
    """Same contract as `src.train.dataset.LightCurveDataset` (pre-tensorized,
    windowing already applied), plus the ablated `max_span_days` cap this
    script exists to measure -- kept local rather than added to
    `LightCurveDataset` itself, since that dataset backs the frozen
    production training/eval path and this is a one-off inference-time
    ablation."""

    def __init__(
        self,
        alerts,
        object_ids: frozenset[str],
        max_span_days: Optional[float],
    ) -> None:
        self._examples: list[_Example] = []
        for alert in alerts:
            if alert.object_id not in object_ids:
                continue
            if alert.class_id not in _CLASS_ID_TO_INDEX:
                continue
            t = tensorize.tensorize_alert(
                alert,
                pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS,
                max_span_days=max_span_days,
            )
            self._examples.append(
                _Example(
                    object_id=alert.object_id,
                    features=t.features,
                    class_index=_CLASS_ID_TO_INDEX[alert.class_id],
                    delta_t_days=t.features[:, 0].clone(),
                )
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> _Example:
        return self._examples[idx]


def _detection_span_days(alert) -> float:
    detection_mjds = [o.mjd for o in alert.observations if o.is_detection]
    return max(detection_mjds) - min(detection_mjds)


def fraction_affected_per_class(alerts, max_span_days: float) -> dict[str, dict[str, float]]:
    """Fraction of objects, per class, whose OWN detection span exceeds
    `max_span_days` -- i.e. whose tensorized window is actually truncated
    (a real, late detection dropped) by this cap. A property of the raw
    alerts, computed without touching the model."""
    counts: dict[int, list[int]] = {}
    for alert in alerts:
        if alert.class_id not in _CLASS_ID_TO_INDEX:
            continue
        idx = _CLASS_ID_TO_INDEX[alert.class_id]
        counts.setdefault(idx, [0, 0])
        counts[idx][0] += 1
        if _detection_span_days(alert) > max_span_days:
            counts[idx][1] += 1
    return {
        CLASS_NAMES[idx]: {
            "n_total": n_total,
            "n_affected": n_affected,
            "fraction_affected": n_affected / n_total if n_total else float("nan"),
        }
        for idx, (n_total, n_affected) in counts.items()
    }


def run_span_ablation() -> dict:
    encoder = load_tier_a_encoder()
    device = torch.device("cpu")

    logger.info("Loading Tier A object_id/class_id (cheap columns) to compute the split...")
    object_class_ids = load_object_class_ids(TIER_A_PARQUET)
    split = split_by_object_id(object_class_ids, seed=SEED, train_frac=0.7, val_frac=0.15)
    logger.info("Test split: %d objects", len(split.test))

    logger.info("Reconstructing test-split alerts only...")
    test_alerts = load_alerts(TIER_A_PARQUET, object_ids=split.test)
    logger.info("Reconstructed %d test alerts", len(test_alerts))

    report: dict = {"span_settings": {}, "chosen_default": None, "chosen_default_reason": None}

    for max_span_days in SPAN_SETTINGS:
        label = _span_label(max_span_days)
        logger.info("=== max_span_days=%s ===", label)

        fraction_affected = (
            fraction_affected_per_class(test_alerts, max_span_days)
            if max_span_days is not None
            else {
                CLASS_NAMES[idx]: {"n_total": 0, "n_affected": 0, "fraction_affected": 0.0}
                for idx in range(len(CLASS_NAMES))
            }
        )
        logger.info("fraction_affected[%s] = %s", label, fraction_affected)

        t0 = time.time()
        ds = _SpanDataset(test_alerts, split.test, max_span_days)
        predictions = collect_phase_predictions(encoder, ds, device)
        elapsed = time.time() - t0
        logger.info(
            "max_span_days=%s: tensorized+scored %d objects (%d predictions) in %.1fs",
            label, len(ds), len(predictions), elapsed,
        )

        metrics = run_eval_report(
            predictions=predictions,
            num_classes=len(CLASS_NAMES),
            class_names=CLASS_NAMES,
            output_dir=str(OUTPUT_DIR / f"span_{label}"),
        )
        macro_f1_full = metrics["full"]["macro_f1"]
        logger.info("max_span_days=%s: full-cutoff macro_f1=%.4f", label, macro_f1_full)

        report["span_settings"][label] = {
            "max_span_days": max_span_days,
            "macro_f1_full": macro_f1_full,
            "n_objects": metrics["full"]["n_objects"],
            "n_excluded": metrics["full"]["n_excluded"],
            "fraction_affected_per_class": fraction_affected,
            "elapsed_seconds": elapsed,
        }

    return report


# ---------------------------------------------------------------------------
# Part 3: CLAGN entropy inversion, before/after the chosen default.
# ---------------------------------------------------------------------------


def _read_clagn_alerts():
    gentype_to_classid = taxonomy.load_gentype_of_classid(GENTYPE_CSV)
    model_dir = os.path.join(ELASTICC2_RAW_DIR, f"ELASTICC2_TRAIN_02_{CLAGN_MODEL_DIR_NAME}")
    logger.info("Reading CLAGN model directory: %s", model_dir)
    t0 = time.time()
    alerts, stats = elasticc2.read_model_directory(model_dir, gentype_to_classid)
    logger.info(
        "Read %d CLAGN objects (%d discarded, %s) in %.1fs",
        stats.n_objects_read, stats.n_objects_discarded, stats.discard_reasons, time.time() - t0,
    )
    non_clagn = [a for a in alerts if a.class_id != CLAGN_CLASS_ID]
    if non_clagn:
        logger.warning(
            "%d objects in the CLAGN directory did not map to class_id=%d as expected -- "
            "excluding them from the entropy check",
            len(non_clagn), CLAGN_CLASS_ID,
        )
    return [a for a in alerts if a.class_id == CLAGN_CLASS_ID]


@dataclass(frozen=True)
class _AnyExample:
    object_id: str
    features: torch.Tensor


class _AnySpanDataset(Dataset):
    def __init__(self, alerts, max_span_days: Optional[float]) -> None:
        self._examples: list[_AnyExample] = []
        n_skipped = 0
        for alert in alerts:
            try:
                t = tensorize.tensorize_alert(
                    alert,
                    pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS,
                    max_span_days=max_span_days,
                )
            except ValueError:
                n_skipped += 1
                continue
            self._examples.append(_AnyExample(object_id=alert.object_id, features=t.features))
        if n_skipped:
            logger.warning("_AnySpanDataset: skipped %d alert(s) that failed to tensorize", n_skipped)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> _AnyExample:
        return self._examples[idx]


def _collate_any(items: list[_AnyExample]):
    lengths = [item.features.shape[0] for item in items]
    t_max = max(lengths)
    num_features = items[0].features.shape[1]
    features = torch.zeros((len(items), t_max, num_features), dtype=torch.float32)
    for i, item in enumerate(items):
        t = item.features.shape[0]
        features[i, :t] = item.features
    return (
        [item.object_id for item in items],
        features,
        torch.tensor(lengths, dtype=torch.int64),
    )


@torch.no_grad()
def _mean_entropy_and_confidence(
    encoder: LCEncoder, alerts, max_span_days: Optional[float], batch_size: int = 256
) -> dict:
    ds = _AnySpanDataset(alerts, max_span_days)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_any)
    entropies: list[np.ndarray] = []
    max_probs: list[np.ndarray] = []
    seq_lengths: list[int] = []
    for _object_ids, features, lengths in loader:
        logits = encoder(features, lengths)
        idx = (lengths - 1).clamp(min=0)
        last_logits = logits[torch.arange(logits.shape[0]), idx]
        probs = torch.softmax(last_logits, dim=-1).numpy()
        entropies.append(entropy_score(probs))
        max_probs.append(probs.max(axis=1))
        seq_lengths.extend(lengths.tolist())
    return {
        "n_objects": len(ds),
        "mean_entropy": float(np.mean(np.concatenate(entropies))),
        "mean_max_softmax_prob": float(np.mean(np.concatenate(max_probs))),
        "mean_seq_length": float(np.mean(seq_lengths)) if seq_lengths else float("nan"),
    }


def run_clagn_entropy_check(encoder: LCEncoder, chosen_default: Optional[float]) -> dict:
    clagn_alerts = _read_clagn_alerts()
    logger.info("Scoring CLAGN (%d objects) at max_span_days=None (Day-10 baseline reproduction)...",
                len(clagn_alerts))
    baseline = _mean_entropy_and_confidence(encoder, clagn_alerts, max_span_days=None)
    logger.info("CLAGN baseline (max_span_days=None): %s", baseline)

    logger.info("Scoring CLAGN at chosen default max_span_days=%s...", chosen_default)
    capped = _mean_entropy_and_confidence(encoder, clagn_alerts, max_span_days=chosen_default)
    logger.info("CLAGN capped (max_span_days=%s): %s", chosen_default, capped)

    return {
        "n_clagn_objects": len(clagn_alerts),
        "baseline_max_span_days_none": baseline,
        "capped_max_span_days": chosen_default,
        "capped": capped,
        "day10_recorded_mean_entropy": 0.082,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = run_span_ablation()

    # Chosen default: highest full-cutoff macro-F1 among the two finite
    # candidates (100, 200), unless the difference from "full" (uncapped) is
    # negative enough to suggest the cap actively hurts -- see the report's
    # own narrative in the session summary for the reasoning applied to
    # whatever numbers come out here.
    finite_settings = [s for s in SPAN_SETTINGS if s is not None]
    best = max(finite_settings, key=lambda s: report["span_settings"][_span_label(s)]["macro_f1_full"])
    report["chosen_default"] = best
    report["chosen_default_reason"] = (
        f"max_span_days={best} gave the best full-cutoff macro-F1 among the finite settings "
        f"tested ({[report['span_settings'][_span_label(s)]['macro_f1_full'] for s in finite_settings]} "
        f"for {finite_settings}), vs. uncapped "
        f"{report['span_settings']['full']['macro_f1_full']:.4f}."
    )
    logger.info("Chosen default: max_span_days=%s -- %s", best, report["chosen_default_reason"])

    encoder = load_tier_a_encoder()
    report["clagn_entropy_check"] = run_clagn_entropy_check(encoder, best)

    with open(OUTPUT_DIR / "span_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Wrote %s", OUTPUT_DIR / "span_ablation_report.json")


if __name__ == "__main__":
    main()
