"""Training loop for the causal GRU LC encoder + static (host-galaxy/
extinction) MLP fusion model (CLAUDE.md §6 step 8, Day 7). Controlled
comparison against the flat LC-only checkpoint's 0.6981 full-cutoff
macro-F1 (src/train/pretrain.py) -- same encoder architecture
(src.models.fusion.LCStaticFusionModel reuses LCEncoder's trunk
construction, see that module's docstring), same data, same split, same
seed, same class weighting, flat (not hierarchical) head.

Entry point: `python -m src.train.pretrain_static` (Hydra-driven,
configs/config_static.yaml + configs/model/lc_static_fusion.yaml +
configs/train/static.yaml).

Evaluates exclusively through the frozen `src/eval/` harness, exactly like
src/train/pretrain.py -- this model's output is already a plain 3-class
softmax (a flat head, per CLAUDE.md's Day-7 instruction), so no
renormalization step is needed the way src/train/pretrain_whxe.py needs one
for its hierarchical head.
"""

import logging
import os
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.eval.phase_curves import PhasePrediction
from src.eval.report import run_eval_report
from src.features import tensorize
from src.models.static_encoder import StaticFeatureNormalizer, assert_no_sentinel_values
from src.train.dataset import load_alerts, split_by_object_id
from src.train.dataset_static import LightCurveStaticDataset, collate_static_examples
from src.train.pretrain import CLASS_NAMES, compute_class_weights, masked_cross_entropy

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_phase_predictions_static(
    model: nn.Module, dataset: LightCurveStaticDataset, device: torch.device, batch_size: int = 256
) -> list[PhasePrediction]:
    """Like src.train.pretrain.collect_phase_predictions, but the model's
    forward signature also takes static_features."""
    model.eval()
    predictions: list[PhasePrediction] = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_static_examples
    )
    for batch in loader:
        features = batch.features.to(device)
        lengths = batch.lengths.to(device)
        static_features = batch.static_features.to(device)
        logits = model(features, lengths, static_features)
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


@hydra.main(config_path="../../configs", config_name="config_static", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.train.seed)

    assert cfg.model.lc_input_size == tensorize.NUM_FEATURES, (
        f"configs/model/lc_static_fusion.yaml lc_input_size={cfg.model.lc_input_size} != "
        f"src.features.tensorize.NUM_FEATURES={tensorize.NUM_FEATURES}"
    )

    logger.info("Loading alerts from %s", cfg.train.data_parquet)
    alerts = load_alerts(cfg.train.data_parquet)
    logger.info("Loaded %d alerts", len(alerts))

    logger.info("Verifying no sentinel values survive into the static encoder...")
    assert_no_sentinel_values(alerts)
    logger.info("Sentinel check passed: 0 raw sentinel values found across %d alerts", len(alerts))

    split = split_by_object_id(
        alerts,
        seed=cfg.train.seed,
        train_frac=cfg.train.split.train_frac,
        val_frac=cfg.train.split.val_frac,
    )
    logger.info(
        "Split sizes: train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test)
    )

    # Normalizer fit on the TRAIN split only -- val/test statistics must
    # never leak into it (CLAUDE.md's usual train-only-statistics discipline,
    # same reasoning as compute_class_weights using train_ds only).
    train_alerts_only = [a for a in alerts if a.object_id in split.train]
    normalizer = StaticFeatureNormalizer.fit(train_alerts_only)
    logger.info("Fit static feature normalizer on %d train alerts", len(train_alerts_only))
    logger.info("normalizer.mean=%s", normalizer.mean.tolist())
    logger.info("normalizer.std=%s", normalizer.std.tolist())

    pre_window = cfg.train.pre_detection_window_days
    train_ds = LightCurveStaticDataset(alerts, split.train, pre_window, normalizer)
    val_ds = LightCurveStaticDataset(alerts, split.val, pre_window, normalizer)
    test_ds = LightCurveStaticDataset(alerts, split.test, pre_window, normalizer)
    logger.info(
        "Tensorized dataset sizes: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_static_examples,
        num_workers=cfg.train.num_workers,
    )

    device = torch.device("cpu")
    model = hydra.utils.instantiate(cfg.model).to(device)

    class_weights = compute_class_weights(
        train_ds, cfg.model.num_classes, cfg.train.class_weighting
    ).to(device)
    logger.info("class_weighting=%s class_weights=%s", cfg.train.class_weighting, class_weights.tolist())

    optimizer = AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    t_max = cfg.train.cosine_t_max or cfg.train.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            features = batch.features.to(device)
            lengths = batch.lengths.to(device)
            mask = batch.mask.to(device)
            targets = batch.class_index.to(device)
            static_features = batch.static_features.to(device)

            logits = model(features, lengths, static_features)
            loss = masked_cross_entropy(logits, targets, mask, class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        logger.info(
            "epoch %d/%d: loss=%.4f lr=%.2e (%.1fs)",
            epoch + 1,
            cfg.train.num_epochs,
            epoch_loss / max(n_batches, 1),
            scheduler.get_last_lr()[0],
            time.time() - t0,
        )

    out_dir = cfg.train.output_dir
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg),
            "static_normalizer_mean": normalizer.mean.tolist(),
            "static_normalizer_std": normalizer.std.tolist(),
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)

    logger.info("Collecting test-split predictions at every timestep...")
    test_predictions = collect_phase_predictions_static(model, test_ds, device)
    logger.info(
        "Collected %d predictions across %d test objects", len(test_predictions), len(test_ds)
    )

    metrics = run_eval_report(
        predictions=test_predictions,
        num_classes=cfg.model.num_classes,
        class_names=CLASS_NAMES,
        output_dir=os.path.join(out_dir, "eval"),
        cutoffs=tuple(cfg.eval.phase_cutoffs),
        ece_num_bins=cfg.eval.ece_num_bins,
    )
    logger.info("Wrote eval report to %s", os.path.join(out_dir, "eval"))
    for label, r in metrics.items():
        logger.info(
            "cutoff=%s macro_f1=%s n_objects=%s n_excluded=%s",
            label,
            r["macro_f1"],
            r["n_objects"],
            r["n_excluded"],
        )


if __name__ == "__main__":
    main()
