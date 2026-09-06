"""Day 10 diagnostic: does removing `host_photoz`/`host_photoz_err` from the
static branch erase most of Day 7's LC+static gain over LC-only (0.7169 vs
0.6981, +0.0189 full-cutoff macro-F1)?

Motivation (CLAUDE.md, Day 9's static-inversion finding): the Tier B
fine-tune found LC+static *underperforming* LC-only, and one plausible
mechanism was that `host_photoz` -- structurally absent at Tier B (§5.1) --
was doing most of Day 7's work, since Ia are standard candles and comparing
apparent brightness against M ~= -19.3 requires a distance estimate. This
script tests that mechanism directly on Tier A, where photo-z *is* present:
same data/split/seed/architecture/class-weighting as
`src.train.pretrain_static`, with `host_photoz`/`host_photoz_err` dropped
from `STATIC_FIELDS` (`src.models.static_encoder.ABLATION_NO_PHOTOZ_FIELDS`,
5 fields instead of 7). If the +0.0189 gain mostly survives without those
two fields, the diagnosis is wrong (something else in the static branch was
doing the work); if it mostly disappears, the diagnosis is confirmed.

Entry point: `python -m src.train.pretrain_static_ablation` (Hydra-driven,
configs/config_static_no_photoz.yaml).
"""

import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.eval.report import run_eval_report
from src.features import tensorize
from src.models.static_encoder import (
    ABLATION_NO_PHOTOZ_FIELDS,
    StaticFeatureNormalizer,
    assert_no_sentinel_values,
)
from src.train.dataset import load_alerts, split_by_object_id
from src.train.dataset_static import collate_static_examples, LightCurveStaticDataset
from src.train.pretrain import CLASS_NAMES, compute_class_weights, masked_cross_entropy
from src.train.pretrain_static import collect_phase_predictions_static

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config_static_no_photoz", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.train.seed)

    assert cfg.model.lc_input_size == tensorize.NUM_FEATURES, (
        f"configs/model/lc_static_fusion_no_photoz.yaml lc_input_size={cfg.model.lc_input_size} "
        f"!= src.features.tensorize.NUM_FEATURES={tensorize.NUM_FEATURES}"
    )
    assert cfg.model.num_static_fields == len(ABLATION_NO_PHOTOZ_FIELDS), (
        f"configs/model/lc_static_fusion_no_photoz.yaml num_static_fields="
        f"{cfg.model.num_static_fields} != "
        f"len(ABLATION_NO_PHOTOZ_FIELDS)={len(ABLATION_NO_PHOTOZ_FIELDS)}"
    )
    logger.info("Ablation static fields (photo-z removed): %s", ABLATION_NO_PHOTOZ_FIELDS)

    logger.info("Loading alerts from %s", cfg.train.data_parquet)
    alerts = load_alerts(cfg.train.data_parquet)
    logger.info("Loaded %d alerts", len(alerts))

    logger.info("Verifying no sentinel values survive into the static encoder...")
    assert_no_sentinel_values(alerts, ABLATION_NO_PHOTOZ_FIELDS)
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

    train_alerts_only = [a for a in alerts if a.object_id in split.train]
    normalizer = StaticFeatureNormalizer.fit(train_alerts_only, ABLATION_NO_PHOTOZ_FIELDS)
    logger.info("Fit static feature normalizer (no photo-z) on %d train alerts", len(train_alerts_only))
    logger.info("normalizer.mean=%s", normalizer.mean.tolist())
    logger.info("normalizer.std=%s", normalizer.std.tolist())

    pre_window = cfg.train.pre_detection_window_days
    train_ds = LightCurveStaticDataset(
        alerts, split.train, pre_window, normalizer, ABLATION_NO_PHOTOZ_FIELDS
    )
    val_ds = LightCurveStaticDataset(
        alerts, split.val, pre_window, normalizer, ABLATION_NO_PHOTOZ_FIELDS
    )
    test_ds = LightCurveStaticDataset(
        alerts, split.test, pre_window, normalizer, ABLATION_NO_PHOTOZ_FIELDS
    )
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
            "static_fields": list(ABLATION_NO_PHOTOZ_FIELDS),
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
