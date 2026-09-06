"""Training loop for the causal GRU LC encoder + WHXE hierarchical head
(CLAUDE.md §6 step 7: "taxonomy.py + WHXE loss <- swap head, keep everything
else"). Day 6: a controlled comparison against the flat 3-class checkpoint's
0.698 full-cutoff macro-F1 (src/train/pretrain.py) -- same encoder
(src.models.lc_encoder.LCEncoder, unchanged), same data
(configs/train/whxe.yaml is a verbatim copy of configs/train/default.yaml's
values except output_dir), same split, same seed, same class weighting.

Entry point: `python -m src.train.pretrain_whxe` (Hydra-driven,
configs/config_whxe.yaml + configs/model/whxe_head.yaml +
configs/train/whxe.yaml).

Evaluates leaf-level metrics exclusively through the frozen `src/eval/`
harness, exactly like src/train/pretrain.py, by renormalizing the
hierarchical head's output down to a 3-class probability triple
(src.models.whxe.leaf_probabilities) -- this keeps macro-F1/per-class
F1/phase curves/ROC-AUC directly comparable to the flat model's own numbers.
2220-node ("SN-like") accuracy is a hierarchy-specific metric with no
frozen-harness home (src/eval/ has no notion of a taxonomy node), so it is
computed and reported separately, alongside but outside that harness.
"""

import json
import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src import taxonomy
from src.eval.phase_curves import PhasePrediction
from src.eval.report import run_eval_report
from src.features import tensorize
from src.models import whxe
from src.models.lc_encoder import LCEncoder
from src.train.dataset import (
    LightCurveDataset,
    class_distribution,
    collate_examples,
    load_alerts,
    split_by_object_id,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Ia", "Ib/c", "II"]  # same leaf order as src.taxonomy.TARGET_CLASS_IDS
SN_LIKE_NODE_ID = 2220


def compute_node_weights(train_ds: LightCurveDataset, enabled: bool) -> dict[int, float]:
    """Per-node W(c) (src.models.whxe.compute_node_class_weights) from the
    train split's own leaf class distribution -- mirrors
    src.train.pretrain.compute_class_weights's use of train_ds only (never
    the full corpus), for the same reason: weights come from what the model
    actually trains on. Returns all-1.0 when `enabled` is False, matching
    compute_class_weights's own convention."""
    leaf_counts: dict[int, int] = {}
    for i in range(len(train_ds)):
        leaf_id = taxonomy.TARGET_CLASS_IDS[train_ds[i].class_index]
        leaf_counts[leaf_id] = leaf_counts.get(leaf_id, 0) + 1

    if not enabled:
        touched_nodes = set(leaf_counts)
        for leaf_id in leaf_counts:
            touched_nodes.update(whxe.ancestor_path(leaf_id))
        return {node_id: 1.0 for node_id in touched_nodes}

    return whxe.compute_node_class_weights(leaf_counts)


@torch.no_grad()
def collect_phase_predictions_and_node_accuracy(
    model: torch.nn.Module,
    dataset: LightCurveDataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[list[PhasePrediction], list[bool]]:
    """Like src.train.pretrain.collect_phase_predictions, but for the
    hierarchical head: `PhasePrediction.probs` is the renormalized 3-leaf
    probability triple (whxe.leaf_probabilities), for direct comparability
    with the flat model through the frozen eval harness. Additionally
    returns, per object, whether the model's cascading top-down prediction
    (at that object's own last real timestep -- the "full cutoff", same
    convention as src.eval.phase_curves.select_at_cutoff(cutoff=None))
    reaches SN_LIKE_NODE_ID -- the 2220-node accuracy CLAUDE.md's Day-6
    instruction asks for, which has no home in the frozen eval harness
    (src/eval/ has no notion of a taxonomy node)."""
    model.eval()
    predictions: list[PhasePrediction] = []
    node_correct: list[bool] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)

    for batch in loader:
        features = batch.features.to(device)
        lengths = batch.lengths.to(device)
        logits = model(features, lengths)  # (B, T, NUM_TAXONOMY_NODES)

        leaf_probs = whxe.leaf_probabilities(logits, taxonomy.TARGET_CLASS_IDS).cpu().numpy()
        lengths_np = batch.lengths.numpy()
        delta_t_np = batch.delta_t_days.numpy()
        class_index_np = batch.class_index.numpy()

        last_idx = (batch.lengths - 1).to(device)
        last_logits = logits[torch.arange(logits.shape[0], device=device), last_idx]
        matches = whxe.predicted_node_matches(last_logits, SN_LIKE_NODE_ID).cpu().tolist()
        node_correct.extend(matches)

        for i, object_id in enumerate(batch.object_id):
            n = int(lengths_np[i])
            for t in range(n):
                predictions.append(
                    PhasePrediction(
                        object_id=object_id,
                        days_since_first_detection=float(delta_t_np[i, t]),
                        true_class=int(class_index_np[i]),
                        probs=leaf_probs[i, t],
                    )
                )
    return predictions, node_correct


@hydra.main(config_path="../../configs", config_name="config_whxe", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.train.seed)

    assert cfg.model.input_size == tensorize.NUM_FEATURES, (
        f"configs/model/whxe_head.yaml input_size={cfg.model.input_size} != "
        f"src.features.tensorize.NUM_FEATURES={tensorize.NUM_FEATURES}"
    )
    assert cfg.model.num_classes == whxe.NUM_TAXONOMY_NODES, (
        f"configs/model/whxe_head.yaml num_classes={cfg.model.num_classes} != "
        f"src.models.whxe.NUM_TAXONOMY_NODES={whxe.NUM_TAXONOMY_NODES}"
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
        "Split sizes: train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test)
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

    node_weights = compute_node_weights(train_ds, cfg.train.class_weighting)
    logger.info("class_weighting=%s node_weights=%s", cfg.train.class_weighting, node_weights)

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
            loss = whxe.whxe_loss(
                logits, targets, mask, taxonomy.TARGET_CLASS_IDS, node_weights
            )

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
    test_predictions, node_correct = collect_phase_predictions_and_node_accuracy(
        model, test_ds, device
    )
    logger.info(
        "Collected %d predictions across %d test objects", len(test_predictions), len(test_ds)
    )

    # run_eval_report has no notion of "the head has NUM_TAXONOMY_NODES output
    # slots but we've renormalized to 3" -- num_classes here must match
    # len(CLASS_NAMES) (leaf_probabilities' actual output width), not
    # cfg.model.num_classes (the raw head width).
    metrics = run_eval_report(
        predictions=test_predictions,
        num_classes=len(CLASS_NAMES),
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

    node_accuracy = sum(node_correct) / max(len(node_correct), 1)
    logger.info(
        "2220 (SN-like) node accuracy at full cutoff: %.6f (%d/%d objects)",
        node_accuracy,
        sum(node_correct),
        len(node_correct),
    )
    node_accuracy_path = os.path.join(out_dir, "eval", "node_accuracy.json")
    with open(node_accuracy_path, "w") as f:
        json.dump(
            {
                "target_node_id": SN_LIKE_NODE_ID,
                "target_node_name": "SN-like",
                "cutoff": "full",
                "n_objects": len(node_correct),
                "n_correct": sum(node_correct),
                "accuracy": node_accuracy,
            },
            f,
            indent=2,
        )
    logger.info("Wrote node-accuracy report to %s", node_accuracy_path)


if __name__ == "__main__":
    main()
