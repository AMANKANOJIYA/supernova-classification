"""Static-context-carrying light curve dataset (CLAUDE.md §6 step 8, Day 7).

Deliberately NOT a modification of `src.train.dataset.LightCurveDataset` --
that class (and `Example`/`TrainBatch`/`collate_examples`) is exactly what
the flat (step 6) and WHXE (step 7) runs already depend on and have
completed checkpoints against. Duplicating its construction here rather
than extending it in place keeps those two runs' reproducibility completely
untouched. The GRU-facing fields (`features`/`mask`/`lengths`/
`delta_t_days`/`class_index`) are built identically to
`src.train.dataset.LightCurveDataset`/`collate_examples` -- only
`static_features` is new.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src import taxonomy
from src.features import tensorize
from src.models.static_encoder import STATIC_FIELDS, StaticFeatureNormalizer, build_static_vector
from src.schema import CanonicalAlert

_CLASS_ID_TO_INDEX: dict[int, int] = {
    class_id: i for i, class_id in enumerate(taxonomy.TARGET_CLASS_IDS)
}


@dataclass(frozen=True)
class StaticExample:
    object_id: str
    features: torch.Tensor  # (T, NUM_FEATURES)
    class_index: int
    delta_t_days: torch.Tensor  # (T,)
    static_features: torch.Tensor  # (2*NUM_STATIC_FIELDS,)


class LightCurveStaticDataset(Dataset):
    """Like `src.train.dataset.LightCurveDataset`, plus a precomputed
    `static_features` vector per example (CLAUDE.md's usual "tensorize once
    up front" reasoning applies identically here: normalization is a fixed
    function of the alert's own static context and the (already-fit)
    normalizer, so nothing here needs recomputing per epoch)."""

    def __init__(
        self,
        alerts: list[CanonicalAlert],
        object_ids: frozenset[str],
        pre_detection_window_days: Optional[float],
        normalizer: StaticFeatureNormalizer,
        static_fields: Sequence[str] = STATIC_FIELDS,
    ) -> None:
        self._examples: list[StaticExample] = []
        for alert in alerts:
            if alert.object_id not in object_ids:
                continue
            if alert.class_id not in _CLASS_ID_TO_INDEX:
                continue
            t = tensorize.tensorize_alert(alert, pre_detection_window_days=pre_detection_window_days)
            static_vec = build_static_vector(alert.static, normalizer, static_fields)
            self._examples.append(
                StaticExample(
                    object_id=alert.object_id,
                    features=t.features,
                    class_index=_CLASS_ID_TO_INDEX[alert.class_id],
                    delta_t_days=t.features[:, 0].clone(),
                    static_features=torch.from_numpy(static_vec),
                )
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> StaticExample:
        return self._examples[idx]


@dataclass(frozen=True)
class StaticTrainBatch:
    object_id: list[str]
    features: torch.Tensor  # (B, T_max, NUM_FEATURES)
    mask: torch.Tensor  # (B, T_max) bool
    lengths: torch.Tensor  # (B,) int64
    class_index: torch.Tensor  # (B,) int64
    delta_t_days: torch.Tensor  # (B, T_max)
    static_features: torch.Tensor  # (B, 2*NUM_STATIC_FIELDS)


def collate_static_examples(items: list[StaticExample]) -> StaticTrainBatch:
    if not items:
        raise ValueError("collate_static_examples: empty batch")
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

    return StaticTrainBatch(
        object_id=[item.object_id for item in items],
        features=features,
        mask=mask,
        lengths=torch.tensor(lengths, dtype=torch.int64),
        class_index=torch.tensor([item.class_index for item in items], dtype=torch.int64),
        delta_t_days=delta_t,
        static_features=torch.stack([item.static_features for item in items]),
    )
