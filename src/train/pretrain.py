"""Training loop for the causal GRU LC encoder + flat 3-class head
(CLAUDE.md §6 step 6: "simplest thing that trains").

Entry point: `python -m src.train.pretrain` (Hydra-driven, configs/config.yaml
+ configs/model/lc_encoder.yaml + configs/train/default.yaml).

Evaluates exclusively through the frozen `src/eval/` harness (CLAUDE.md
§2.1) -- this module computes no metrics of its own beyond a running
training-loss average logged per epoch for progress visibility. The
reported macro-F1 / confusion-matrix / phase-curve numbers all come from
`src.eval.report.run_eval_report`.
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
from src.train.dataset import (
    LightCurveDataset,
    class_distribution,
    collate_examples,
    load_alerts,
    split_by_object_id,
)

logger = logging.getLogger(__name__)

# Index order matches src.taxonomy.TARGET_CLASS_IDS = (2222, 2223, 2224).
CLASS_NAMES = ["Ia", "Ib/c", "II"]


def compute_class_weights(train_ds: LightCurveDataset, num_classes: int, enabled: bool) -> torch.Tensor:
    """Sqrt-dampened inverse-frequency weights, normalized to mean 1 --
    deliberately "mild" (CLAUDE.md §11 Day-4: natural ELAsTiCC composition,
    no oversampling; mild class weighting only if Ia underperforms), not a
    full inverse-frequency rebalancing. Returns all-ones when `enabled` is
    False."""
    if not enabled:
        return torch.ones(num_classes)
    counts = torch.zeros(num_classes)
    for i in range(len(train_ds)):
        counts[train_ds[i].class_index] += 1
    n = counts.sum()
    balanced = n / (num_classes * counts.clamp(min=1))
    weights = balanced.sqrt()
    return weights / weights.mean()


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over every real (unpadded) timestep, treating each
    timestep as its own training example with the object's single true
    class as the target (the object's class does not change over time)."""
    batch_size, seq_len, _ = logits.shape
    logits_flat = logits[mask]  # (n_valid, num_classes)
    targets_flat = targets.unsqueeze(1).expand(batch_size, seq_len)[mask]  # (n_valid,)
    return nn.functional.cross_entropy(logits_flat, targets_flat, weight=class_weights)


@torch.no_grad()
def collect_phase_predictions(
    model: nn.Module, dataset: LightCurveDataset, device: torch.device, batch_size: int = 256
) -> list[PhasePrediction]:
    """Run `model` once per object and emit one `PhasePrediction` per real
    (unpadded) timestep -- CLAUDE.md Day-5: "Emit predictions at every
    timestep... gives the phase-vs-F1 curves for free," consumed here by
    the frozen `src.eval.phase_curves`/`report` harness (§2.1)."""
    model.eval()
    predictions: list[PhasePrediction] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    for batch in loader:
        features = batch.features.to(device)
        lengths = batch.lengths.to(device)
        logits = model(features, lengths)
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


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.train.seed)

    assert cfg.model.input_size == tensorize.NUM_FEATURES, (
        f"configs/model/lc_encoder.yaml input_size={cfg.model.input_size} != "
        f"src.features.tensorize.NUM_FEATURES={tensorize.NUM_FEATURES}"
    )

    logger.info("Loading alerts from %s", cfg.train.data_parquet)
    alerts = load_alerts(cfg.train.data_parquet)
    logger.info("Loaded %d alerts", len(alerts))

    split = split_by_object_id(
        alerts,
        seed=cfg.train.seed,
        train_frac=cfg.train.split.train_frac,
        val_frac=cfg.train.split.val_frac,
    )
    logger.info(
        "Split sizes: train=%d val=%d test=%d",
        len(split.train),
        len(split.val),
        len(split.test),
    )
    for name, ids in [("train", split.train), ("val", split.val), ("test", split.test)]:
        logger.info("%s class distribution: %s", name, class_distribution(alerts, ids))

    pre_window = cfg.train.pre_detection_window_days
    train_ds = LightCurveDataset(alerts, split.train, pre_window)
    val_ds = LightCurveDataset(alerts, split.val, pre_window)
    test_ds = LightCurveDataset(alerts, split.test, pre_window)
    logger.info(
        "Tensorized dataset sizes: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
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

            logits = model(features, lengths)
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
        {"model_state_dict": model.state_dict(), "config": OmegaConf.to_container(cfg)},
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)

    logger.info("Collecting test-split predictions at every timestep...")
    test_predictions = collect_phase_predictions(model, test_ds, device)
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
