"""Day 9: fine-tune on Tier B (BTSbot), then Gate C (CLAUDE.md §6 step 9,
§7 Gate C). Entry point: `python -m src.train.finetune_bts`.

Deliberately NOT Hydra-driven, unlike every other `src/train/` entry point.
This script runs four evaluation conditions against the SAME Tier A
checkpoints and the SAME Tier B join in one process and produces one
consolidated Gate C report:

  1. zero-shot   -- Tier A LC+static weights, unmodified, on Tier B test.
  2. with_static -- Tier A LC+static weights, fine-tuned on Tier B train.
  3. lc_only     -- Tier A LC-only (flat, step-6) weights, fine-tuned on
                     Tier B train. The "without the four available host
                     fields" ablation arm for Gate C report item #6.
  4. with_static_drb -- like (2), plus a small extra branch for
                     `real_bogus_score` (populated at Tier B, always-null at
                     Tier A -- CLAUDE.md §5.1). Answers the separate "does
                     real_bogus_score help" question, independent of the
                     host-fields ablation.

Four separate Hydra invocations could silently drift out of sync (different
join, different split, different eval cutoffs); one script sharing one join
and one `LightCurveStaticDataset`/`LightCurveDataset` construction per
variant cannot.

All hyperparameters below are logged constants, not swept -- CLAUDE.md's
Day-9 instruction is "report your LR and freezing schedule and justify
them," not tune them to hit a target number. See LOW-LR / FREEZING
SCHEDULE below for the justification.

-----------------------------------------------------------------------------
LOW-LR / FREEZING SCHEDULE -- justification
-----------------------------------------------------------------------------
Tier A is 36.6% negative flux; Tier B is 0% (CLAUDE.md §5.1's `isdiffpos`
finding). The GRU trunk is the only part of this model that has ever seen a
negative flux value, and it is also the most expensive part to have learned
(15 epochs over 590,532 objects). Tier B's positive-only signal must be
allowed to adapt the model to real ZTF cadence/noise/host statistics without
being allowed to overwrite what the trunk learned about negative-flux
inputs it will never see again during this fine-tune but will see again the
moment this model serves live Tier C traffic (§5.1's Tier B -> Tier C
divergence note).

Two-phase schedule, applied identically to every fine-tuned variant for
comparability:
  Phase 1 (epochs 1-2): GRU trunk (`gru`+`layer_norm`) FROZEN
    (`requires_grad=False`). Only the head (+ static_encoder / drb_branch,
    where present) trains, at LR=3e-4 -- 3x below Tier A's flat 1e-3, since
    Tier B train is ~37x smaller than Tier A train (5,407 vs 590,532
    labeled objects for the with_static/with_static_drb variants) and a
    small real dataset overfits a freshly-adapting head faster than a huge
    simulated one does.
  Phase 2 (epochs 3-8): trunk unfrozen, but at LR=3e-6 -- 100x below the
    head's LR -- via discriminative per-parameter-group learning rates in a
    fresh AdamW (Adam's moment estimates for the trunk start clean at the
    unfreeze point, standard practice for gradual unfreezing). The head
    (+static_encoder/drb_branch) continues at 3e-4. Cosine-annealed to ~0
    over these 6 epochs.
Total 8 epochs (vs Tier A's 15) -- Tier B train is small enough that more
epochs mainly buys overfitting, not signal.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.eval.phase_curves import PhasePrediction, cutoff_label, select_at_cutoff
from src.eval.metrics import roc_auc as roc_auc_metric
from src.eval.report import run_eval_report
from src.models.fusion import LCStaticDRBFusionModel, LCStaticFusionModel, load_shared_submodules
from src.models.lc_encoder import LCEncoder
from src.models.static_encoder import StaticFeatureNormalizer, build_static_vector
from src.readers import bts_labels
from src.train.dataset import LightCurveDataset, collate_examples, load_alerts
from src.train.dataset_static import LightCurveStaticDataset, collate_static_examples
from src.train.pretrain import CLASS_NAMES, compute_class_weights, masked_cross_entropy
from src.train.pretrain_static import collect_phase_predictions_static

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- logged, not swept. See module docstring for the freeze/LR
# justification.
# ---------------------------------------------------------------------------
SEED = 42
FREEZE_EPOCHS = 2
UNFREEZE_EPOCHS = 6
TOTAL_EPOCHS = FREEZE_EPOCHS + UNFREEZE_EPOCHS
HEAD_LR = 3e-4
TRUNK_LR_UNFROZEN = 3e-6
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 128
CLASS_WEIGHTING = True
PRE_DETECTION_WINDOW_DAYS = 30.0  # matches Tier A; a documented no-op on Tier B (§5.1)
# Tier B has no pre-detection data (§5.1): -5 is dropped from every Tier B
# cutoff set below. `None` == "full".
TIER_B_CUTOFFS: tuple = (0.0, 5.0, 15.0, 30.0, None)

BTSBOT_INTERIM_DIR = Path("data/interim")
TIER_A_STATIC_CHECKPOINT = Path("outputs/train_static/checkpoint.pt")
TIER_A_LC_ONLY_CHECKPOINT = Path("outputs/train/checkpoint.pt")
TIER_A_STATIC_METRICS = Path("outputs/train_static/eval/metrics.json")
OUTPUT_ROOT = Path("outputs/finetune_bts")
REPORT_DIR = Path("reports/gate_c")


def _set_seed() -> None:
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Tier B data: load + join + split (splits ship with the release, per
# CLAUDE.md -- never re-split).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierBSplits:
    train: list
    val: list
    test: list
    join_stats: dict


def load_tier_b(bts_sample) -> TierBSplits:
    join_stats: dict = {}
    splits = {}
    for split in ("train", "val", "test"):
        alerts = load_alerts(str(BTSBOT_INTERIM_DIR / f"btsbot_{split}.parquet"))
        labeled, stats = bts_labels.apply_classifications_to_alerts(alerts, bts_sample)
        join_stats[split] = stats
        splits[split] = labeled
        logger.info("Tier B %s: %d objects read, join outcomes=%s", split, len(alerts), stats)

    n_3class = {
        split: sum(1 for a in splits[split] if a.class_id in (2222, 2223, 2224))
        for split in splits
    }
    logger.info("Tier B 3-class labeled object counts: %s", n_3class)

    return TierBSplits(train=splits["train"], val=splits["val"], test=splits["test"], join_stats=join_stats)


# ---------------------------------------------------------------------------
# real_bogus_score normalizer (not part of src.models.static_encoder --
# that module's STATIC_FIELDS deliberately excludes it, since it is
# 100%-null at Tier A; see that module's docstring). Same zero-fill +
# missingness-indicator convention, one field only.
# ---------------------------------------------------------------------------


def _fit_drb_normalizer(alerts: list) -> tuple:
    values = np.array(
        [a.static.real_bogus_score if a.static.real_bogus_score is not None else np.nan for a in alerts]
    )
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    std = std if std > 0 else 1.0
    return mean, std


def _build_drb_vector(real_bogus_score, mean: float, std: float) -> np.ndarray:
    if real_bogus_score is None:
        return np.array([0.0, 1.0], dtype=np.float32)
    z = (float(real_bogus_score) - mean) / std
    return np.array([z, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class DRBExample:
    object_id: str
    features: torch.Tensor
    class_index: int
    delta_t_days: torch.Tensor
    static_features: torch.Tensor
    drb_features: torch.Tensor


@dataclass(frozen=True)
class DRBTrainBatch:
    object_id: list
    features: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    class_index: torch.Tensor
    delta_t_days: torch.Tensor
    static_features: torch.Tensor
    drb_features: torch.Tensor


class LightCurveStaticDRBDataset(Dataset):
    """Like `LightCurveStaticDataset`, plus a `[value, missingness]`
    `real_bogus_score` vector per example. Kept local to this script rather
    than added to `src/train/dataset_static.py` -- it exists only to support
    the Day 9 real_bogus_score ablation, not as a permanent dataset class."""

    def __init__(self, alerts, object_ids, pre_detection_window_days, normalizer, drb_mean, drb_std):
        from src import taxonomy
        from src.features import tensorize

        class_id_to_index = {cid: i for i, cid in enumerate(taxonomy.TARGET_CLASS_IDS)}
        self._examples: list[DRBExample] = []
        for alert in alerts:
            if alert.object_id not in object_ids or alert.class_id not in class_id_to_index:
                continue
            t = tensorize.tensorize_alert(alert, pre_detection_window_days=pre_detection_window_days)
            static_vec = build_static_vector(alert.static, normalizer)
            drb_vec = _build_drb_vector(alert.static.real_bogus_score, drb_mean, drb_std)
            self._examples.append(
                DRBExample(
                    object_id=alert.object_id,
                    features=t.features,
                    class_index=class_id_to_index[alert.class_id],
                    delta_t_days=t.features[:, 0].clone(),
                    static_features=torch.from_numpy(static_vec),
                    drb_features=torch.from_numpy(drb_vec),
                )
            )

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]


def collate_drb_examples(items: list) -> DRBTrainBatch:
    if not items:
        raise ValueError("collate_drb_examples: empty batch")
    lengths = [item.features.shape[0] for item in items]
    t_max = max(lengths)
    batch_size = len(items)
    num_features = items[0].features.shape[1]

    features = torch.zeros((batch_size, t_max, num_features), dtype=torch.float32)
    mask = torch.zeros((batch_size, t_max), dtype=torch.bool)
    delta_t = torch.zeros((batch_size, t_max), dtype=torch.float32)
    for i, item in enumerate(items):
        t = item.features.shape[0]
        features[i, :t] = item.features
        mask[i, :t] = True
        delta_t[i, :t] = item.delta_t_days

    return DRBTrainBatch(
        object_id=[item.object_id for item in items],
        features=features,
        mask=mask,
        lengths=torch.tensor(lengths, dtype=torch.int64),
        class_index=torch.tensor([item.class_index for item in items], dtype=torch.int64),
        delta_t_days=delta_t,
        static_features=torch.stack([item.static_features for item in items]),
        drb_features=torch.stack([item.drb_features for item in items]),
    )


@torch.no_grad()
def collect_phase_predictions_drb(model, dataset, device, batch_size=256) -> list[PhasePrediction]:
    model.eval()
    predictions: list[PhasePrediction] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_drb_examples)
    for batch in loader:
        logits = model(
            batch.features.to(device), batch.lengths.to(device), batch.static_features.to(device),
            batch.drb_features.to(device),
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        lengths_np = batch.lengths.numpy()
        delta_t_np = batch.delta_t_days.numpy()
        class_index_np = batch.class_index.numpy()
        for i, object_id in enumerate(batch.object_id):
            n = int(lengths_np[i])
            for t in range(n):
                predictions.append(
                    PhasePrediction(
                        object_id=object_id,
                        days_since_first_detection=float(delta_t_np[i, t]),
                        true_class=int(class_index_np[i]),
                        probs=probs[i, t],
                    )
                )
    return predictions


# ---------------------------------------------------------------------------
# Generic two-phase freeze/discriminative-LR fine-tune loop, specialized per
# variant below (deliberately NOT one fully-generic function across the
# three different forward signatures/dataset types -- CLAUDE.md: "three
# similar lines is better than a premature abstraction").
# ---------------------------------------------------------------------------


def _log_epoch(variant: str, epoch: int, total: int, loss: float, lr: float, seconds: float) -> None:
    logger.info(
        "[%s] epoch %d/%d: loss=%.4f lr=%.2e (%.1fs)", variant, epoch, total, loss, lr, seconds
    )


def finetune_with_static(train_alerts) -> tuple[nn.Module, StaticFeatureNormalizer]:
    checkpoint = torch.load(TIER_A_STATIC_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    normalizer = StaticFeatureNormalizer.fit(train_alerts)
    logger.info("with_static: Tier-B-refit normalizer mean=%s std=%s", normalizer.mean.tolist(), normalizer.std.tolist())

    train_ds = LightCurveStaticDataset(train_alerts, frozenset(a.object_id for a in train_alerts), PRE_DETECTION_WINDOW_DAYS, normalizer)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_static_examples)

    model = LCStaticFusionModel(
        lc_input_size=cfg["model"]["lc_input_size"],
        lc_hidden_size=cfg["model"]["lc_hidden_size"],
        lc_num_layers=cfg["model"]["lc_num_layers"],
        lc_dropout=cfg["model"]["lc_dropout"],
        static_hidden_size=cfg["model"]["static_hidden_size"],
        static_output_size=cfg["model"]["static_output_size"],
        num_classes=cfg["model"]["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = compute_class_weights(train_ds, cfg["model"]["num_classes"], CLASS_WEIGHTING)
    logger.info("with_static: class_weights=%s", class_weights.tolist())

    for p in model.gru.parameters():
        p.requires_grad = False
    for p in model.layer_norm.parameters():
        p.requires_grad = False

    head_params = list(model.static_encoder.parameters()) + list(model.head.parameters())
    optimizer = AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(FREEZE_EPOCHS):
        _train_one_epoch_static(model, loader, optimizer, class_weights, "with_static[frozen]", epoch + 1, FREEZE_EPOCHS)

    for p in model.gru.parameters():
        p.requires_grad = True
    for p in model.layer_norm.parameters():
        p.requires_grad = True
    optimizer = AdamW(
        [
            {"params": list(model.gru.parameters()) + list(model.layer_norm.parameters()), "lr": TRUNK_LR_UNFROZEN},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCHS)
    for epoch in range(UNFREEZE_EPOCHS):
        _train_one_epoch_static(model, loader, optimizer, class_weights, "with_static[unfrozen]", epoch + 1, UNFREEZE_EPOCHS)
        scheduler.step()

    return model, normalizer


def _train_one_epoch_static(model, loader, optimizer, class_weights, tag, epoch, total):
    import time

    model.train()
    epoch_loss, n_batches, t0 = 0.0, 0, time.time()
    for batch in loader:
        logits = model(batch.features, batch.lengths, batch.static_features)
        loss = masked_cross_entropy(logits, batch.class_index, batch.mask, class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    _log_epoch(tag, epoch, total, epoch_loss / max(n_batches, 1), optimizer.param_groups[-1]["lr"], time.time() - t0)


def finetune_lc_only(train_alerts) -> nn.Module:
    checkpoint = torch.load(TIER_A_LC_ONLY_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    train_ds = LightCurveDataset(train_alerts, frozenset(a.object_id for a in train_alerts), PRE_DETECTION_WINDOW_DAYS)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_examples)

    model = LCEncoder(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = compute_class_weights(train_ds, cfg["model"]["num_classes"], CLASS_WEIGHTING)
    logger.info("lc_only: class_weights=%s", class_weights.tolist())

    for p in model.gru.parameters():
        p.requires_grad = False
    for p in model.layer_norm.parameters():
        p.requires_grad = False

    head_params = list(model.head.parameters())
    optimizer = AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(FREEZE_EPOCHS):
        _train_one_epoch_lc_only(model, loader, optimizer, class_weights, "lc_only[frozen]", epoch + 1, FREEZE_EPOCHS)

    for p in model.gru.parameters():
        p.requires_grad = True
    for p in model.layer_norm.parameters():
        p.requires_grad = True
    optimizer = AdamW(
        [
            {"params": list(model.gru.parameters()) + list(model.layer_norm.parameters()), "lr": TRUNK_LR_UNFROZEN},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCHS)
    for epoch in range(UNFREEZE_EPOCHS):
        _train_one_epoch_lc_only(model, loader, optimizer, class_weights, "lc_only[unfrozen]", epoch + 1, UNFREEZE_EPOCHS)
        scheduler.step()

    return model


def _train_one_epoch_lc_only(model, loader, optimizer, class_weights, tag, epoch, total):
    import time

    model.train()
    epoch_loss, n_batches, t0 = 0.0, 0, time.time()
    for batch in loader:
        logits = model(batch.features, batch.lengths)
        loss = masked_cross_entropy(logits, batch.class_index, batch.mask, class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    _log_epoch(tag, epoch, total, epoch_loss / max(n_batches, 1), optimizer.param_groups[-1]["lr"], time.time() - t0)


def finetune_with_static_drb(train_alerts) -> tuple[nn.Module, StaticFeatureNormalizer, float, float]:
    checkpoint = torch.load(TIER_A_STATIC_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    normalizer = StaticFeatureNormalizer.fit(train_alerts)
    drb_mean, drb_std = _fit_drb_normalizer(train_alerts)
    logger.info("with_static_drb: drb_mean=%.4f drb_std=%.4f", drb_mean, drb_std)

    train_ds = LightCurveStaticDRBDataset(
        train_alerts, frozenset(a.object_id for a in train_alerts), PRE_DETECTION_WINDOW_DAYS, normalizer, drb_mean, drb_std
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_drb_examples)

    model = LCStaticDRBFusionModel(
        lc_input_size=cfg["model"]["lc_input_size"],
        lc_hidden_size=cfg["model"]["lc_hidden_size"],
        lc_num_layers=cfg["model"]["lc_num_layers"],
        lc_dropout=cfg["model"]["lc_dropout"],
        static_hidden_size=cfg["model"]["static_hidden_size"],
        static_output_size=cfg["model"]["static_output_size"],
        num_classes=cfg["model"]["num_classes"],
    )
    load_shared_submodules(model, checkpoint["model_state_dict"])

    class_weights = compute_class_weights(train_ds, cfg["model"]["num_classes"], CLASS_WEIGHTING)
    logger.info("with_static_drb: class_weights=%s", class_weights.tolist())

    for p in model.gru.parameters():
        p.requires_grad = False
    for p in model.layer_norm.parameters():
        p.requires_grad = False

    head_params = (
        list(model.static_encoder.parameters()) + list(model.drb_branch.parameters()) + list(model.head.parameters())
    )
    optimizer = AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(FREEZE_EPOCHS):
        _train_one_epoch_drb(model, loader, optimizer, class_weights, "with_static_drb[frozen]", epoch + 1, FREEZE_EPOCHS)

    for p in model.gru.parameters():
        p.requires_grad = True
    for p in model.layer_norm.parameters():
        p.requires_grad = True
    optimizer = AdamW(
        [
            {"params": list(model.gru.parameters()) + list(model.layer_norm.parameters()), "lr": TRUNK_LR_UNFROZEN},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCHS)
    for epoch in range(UNFREEZE_EPOCHS):
        _train_one_epoch_drb(model, loader, optimizer, class_weights, "with_static_drb[unfrozen]", epoch + 1, UNFREEZE_EPOCHS)
        scheduler.step()

    return model, normalizer, drb_mean, drb_std


def _train_one_epoch_drb(model, loader, optimizer, class_weights, tag, epoch, total):
    import time

    model.train()
    epoch_loss, n_batches, t0 = 0.0, 0, time.time()
    for batch in loader:
        logits = model(batch.features, batch.lengths, batch.static_features, batch.drb_features)
        loss = masked_cross_entropy(logits, batch.class_index, batch.mask, class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    _log_epoch(tag, epoch, total, epoch_loss / max(n_batches, 1), optimizer.param_groups[-1]["lr"], time.time() - t0)


# ---------------------------------------------------------------------------
# Report composition
# ---------------------------------------------------------------------------


def _ia_vs_rest_roc_auc(predictions: list[PhasePrediction]) -> float:
    selected, _ = select_at_cutoff(predictions, cutoff=None)
    y_true = np.array([p.true_class for p in selected])
    y_prob = np.stack([p.probs for p in selected])
    return float(roc_auc_metric(y_true, y_prob, num_classes=3)[0])  # index 0 == Ia


def _plot_phase_comparison(tier_a_metrics: dict, tier_b_metrics: dict, output_path: Path) -> None:
    cutoffs = [0.0, 5.0, 15.0, 30.0, None]
    labels = [cutoff_label(c) for c in cutoffs]
    tier_a_vals = [tier_a_metrics.get(l, {}).get("macro_f1") for l in labels]
    tier_b_vals = [tier_b_metrics.get(l, {}).get("macro_f1") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.plot(x, tier_a_vals, marker="o", label="Tier A test (sim, LC+static)")
    ax.plot(x, tier_b_vals, marker="o", label="Tier B test (real, fine-tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("days since first detection (cutoff)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Gate C phase curve, cutoffs >= 0 only\n(Tier B has no pre-detection data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_seed()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching/loading cached BTS Sample Explorer sample...")
    bts_sample = bts_labels.fetch_bts_sample()

    logger.info("Loading + joining Tier B splits...")
    tb = load_tier_b(bts_sample)

    n_3class_train = sum(1 for a in tb.train if a.class_id in (2222, 2223, 2224))
    n_3class_val = sum(1 for a in tb.val if a.class_id in (2222, 2223, 2224))
    n_3class_test = sum(1 for a in tb.test if a.class_id in (2222, 2223, 2224))
    logger.info(
        "Tier B 3-class labeled objects: train=%d val=%d test=%d", n_3class_train, n_3class_val, n_3class_test
    )

    report: dict = {
        "join_stats": tb.join_stats,
        "n_3class_labeled": {"train": n_3class_train, "val": n_3class_val, "test": n_3class_test},
    }

    # -------------------------------------------------------------------
    # 1. Zero-shot: Tier A LC+static weights, unmodified, on Tier B test.
    #    Uses Tier A's OWN normalizer -- this measures how Tier A weights,
    #    as-is, generalize, including any static-feature distribution
    #    shift as part of the honest sim-to-real gap (not papered over).
    # -------------------------------------------------------------------
    logger.info("=== Zero-shot: Tier A LC+static weights on Tier B test ===")
    tier_a_checkpoint = torch.load(TIER_A_STATIC_CHECKPOINT, map_location="cpu", weights_only=False)
    tier_a_cfg = tier_a_checkpoint["config"]
    zero_shot_model = LCStaticFusionModel(
        lc_input_size=tier_a_cfg["model"]["lc_input_size"],
        lc_hidden_size=tier_a_cfg["model"]["lc_hidden_size"],
        lc_num_layers=tier_a_cfg["model"]["lc_num_layers"],
        lc_dropout=tier_a_cfg["model"]["lc_dropout"],
        static_hidden_size=tier_a_cfg["model"]["static_hidden_size"],
        static_output_size=tier_a_cfg["model"]["static_output_size"],
        num_classes=tier_a_cfg["model"]["num_classes"],
    )
    zero_shot_model.load_state_dict(tier_a_checkpoint["model_state_dict"])
    tier_a_normalizer = StaticFeatureNormalizer(
        mean=np.array(tier_a_checkpoint["static_normalizer_mean"]),
        std=np.array(tier_a_checkpoint["static_normalizer_std"]),
    )
    zero_shot_test_ds = LightCurveStaticDataset(
        tb.test, frozenset(a.object_id for a in tb.test), PRE_DETECTION_WINDOW_DAYS, tier_a_normalizer
    )
    zero_shot_predictions = collect_phase_predictions_static(zero_shot_model, zero_shot_test_ds, torch.device("cpu"))
    zero_shot_metrics = run_eval_report(
        zero_shot_predictions, num_classes=3, class_names=CLASS_NAMES,
        output_dir=str(OUTPUT_ROOT / "zero_shot" / "eval"), cutoffs=TIER_B_CUTOFFS,
    )
    report["zero_shot"] = {
        "macro_f1_full": zero_shot_metrics["full"]["macro_f1"],
        "ia_vs_rest_roc_auc": _ia_vs_rest_roc_auc(zero_shot_predictions),
        "per_class_f1_full": zero_shot_metrics["full"]["per_class"]["f1"],
        "per_class_support_full": zero_shot_metrics["full"]["per_class"]["support"],
    }
    logger.info("zero_shot: %s", report["zero_shot"])

    # -------------------------------------------------------------------
    # 2. Fine-tune with_static (the flagship variant, matches Tier A's own
    #    architecture) -- headline Gate C numbers come from this.
    # -------------------------------------------------------------------
    logger.info("=== Fine-tuning: with_static ===")
    with_static_model, with_static_normalizer = finetune_with_static(tb.train)
    torch.save(
        {"model_state_dict": with_static_model.state_dict(),
         "static_normalizer_mean": with_static_normalizer.mean.tolist(),
         "static_normalizer_std": with_static_normalizer.std.tolist()},
        OUTPUT_ROOT / "with_static_checkpoint.pt",
    )
    with_static_test_ds = LightCurveStaticDataset(
        tb.test, frozenset(a.object_id for a in tb.test), PRE_DETECTION_WINDOW_DAYS, with_static_normalizer
    )
    with_static_predictions = collect_phase_predictions_static(with_static_model, with_static_test_ds, torch.device("cpu"))
    with_static_metrics = run_eval_report(
        with_static_predictions, num_classes=3, class_names=CLASS_NAMES,
        output_dir=str(OUTPUT_ROOT / "with_static" / "eval"), cutoffs=TIER_B_CUTOFFS,
    )
    report["with_static"] = {
        "macro_f1_full": with_static_metrics["full"]["macro_f1"],
        "ia_vs_rest_roc_auc": _ia_vs_rest_roc_auc(with_static_predictions),
        "per_class_f1_full": with_static_metrics["full"]["per_class"]["f1"],
        "per_class_support_full": with_static_metrics["full"]["per_class"]["support"],
        "phase_curve": {label: with_static_metrics[label]["macro_f1"] for label in [cutoff_label(c) for c in TIER_B_CUTOFFS]},
    }
    logger.info("with_static: %s", report["with_static"])

    # -------------------------------------------------------------------
    # 3. Fine-tune lc_only (host-fields ablation "without" arm).
    # -------------------------------------------------------------------
    logger.info("=== Fine-tuning: lc_only (host-fields ablation, 'without') ===")
    lc_only_model = finetune_lc_only(tb.train)
    torch.save({"model_state_dict": lc_only_model.state_dict()}, OUTPUT_ROOT / "lc_only_checkpoint.pt")
    lc_only_test_ds = LightCurveDataset(tb.test, frozenset(a.object_id for a in tb.test), PRE_DETECTION_WINDOW_DAYS)
    from src.train.pretrain import collect_phase_predictions as collect_phase_predictions_lc_only

    lc_only_predictions = collect_phase_predictions_lc_only(lc_only_model, lc_only_test_ds, torch.device("cpu"))
    lc_only_metrics = run_eval_report(
        lc_only_predictions, num_classes=3, class_names=CLASS_NAMES,
        output_dir=str(OUTPUT_ROOT / "lc_only" / "eval"), cutoffs=TIER_B_CUTOFFS,
    )
    report["lc_only"] = {
        "macro_f1_full": lc_only_metrics["full"]["macro_f1"],
        "ia_vs_rest_roc_auc": _ia_vs_rest_roc_auc(lc_only_predictions),
        "per_class_f1_full": lc_only_metrics["full"]["per_class"]["f1"],
        "per_class_support_full": lc_only_metrics["full"]["per_class"]["support"],
    }
    logger.info("lc_only: %s", report["lc_only"])

    # -------------------------------------------------------------------
    # 4. Fine-tune with_static_drb (real_bogus_score ablation).
    # -------------------------------------------------------------------
    logger.info("=== Fine-tuning: with_static_drb (real_bogus_score ablation) ===")
    drb_model, drb_normalizer, drb_mean, drb_std = finetune_with_static_drb(tb.train)
    torch.save(
        {"model_state_dict": drb_model.state_dict(),
         "static_normalizer_mean": drb_normalizer.mean.tolist(),
         "static_normalizer_std": drb_normalizer.std.tolist(),
         "drb_mean": drb_mean, "drb_std": drb_std},
        OUTPUT_ROOT / "with_static_drb_checkpoint.pt",
    )
    drb_test_ds = LightCurveStaticDRBDataset(
        tb.test, frozenset(a.object_id for a in tb.test), PRE_DETECTION_WINDOW_DAYS, drb_normalizer, drb_mean, drb_std
    )
    drb_predictions = collect_phase_predictions_drb(drb_model, drb_test_ds, torch.device("cpu"))
    drb_metrics = run_eval_report(
        drb_predictions, num_classes=3, class_names=CLASS_NAMES,
        output_dir=str(OUTPUT_ROOT / "with_static_drb" / "eval"), cutoffs=TIER_B_CUTOFFS,
    )
    report["with_static_drb"] = {
        "macro_f1_full": drb_metrics["full"]["macro_f1"],
        "ia_vs_rest_roc_auc": _ia_vs_rest_roc_auc(drb_predictions),
        "per_class_f1_full": drb_metrics["full"]["per_class"]["f1"],
        "per_class_support_full": drb_metrics["full"]["per_class"]["support"],
    }
    logger.info("with_static_drb: %s", report["with_static_drb"])

    # -------------------------------------------------------------------
    # Gate C headline: Tier A test macro-F1 vs Tier B test macro-F1.
    # -------------------------------------------------------------------
    tier_a_metrics = json.loads(TIER_A_STATIC_METRICS.read_text())
    report["tier_a_test_macro_f1_full"] = tier_a_metrics["full"]["macro_f1"]
    report["sim_to_real_gap"] = report["tier_a_test_macro_f1_full"] - report["with_static"]["macro_f1_full"]

    _plot_phase_comparison(tier_a_metrics, with_static_metrics, REPORT_DIR / "phase_curve_comparison.png")

    (REPORT_DIR / "gate_c_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    logger.info("Wrote Gate C report to %s", REPORT_DIR / "gate_c_report.json")


if __name__ == "__main__":
    main()
