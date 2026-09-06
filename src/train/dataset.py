"""Real-data loading, splitting, and Dataset/DataLoader glue for
`src/train/pretrain.py` (CLAUDE.md §6 step 6).

Reconstructs `CanonicalAlert` objects from the cached interim parquet
(`data/interim/elasticc2/train_02_full_3class.parquet`) rather than
re-reading FITS -- the parquet is a lossless flattened round-trip of what
`src/readers/elasticc2.py` already produced and Gate A already verified, and
reading it is far faster.

Splitting is by `object_id`. This session verified (direct read of the SNID
column from every `_HEAD.FITS` file across all 12 model directories) that
`object_id` is globally unique -- zero collisions, within or across
directories -- so no composite key is needed.
"""

import random
from dataclasses import dataclass
from typing import Optional

import polars as pl
import torch
from torch.utils.data import Dataset

from src import taxonomy
from src.features import tensorize
from src.schema import CanonicalAlert, Observation, StaticContext, Survey

_CLASS_ID_TO_INDEX: dict[int, int] = {
    class_id: i for i, class_id in enumerate(taxonomy.TARGET_CLASS_IDS)
}


def _row_to_alert(row: dict) -> CanonicalAlert:
    observations = [
        Observation(mjd=m, lam_eff=l, flux=f, flux_err=fe, is_detection=d)
        for m, l, f, fe, d in zip(
            row["mjd"], row["lam_eff"], row["flux"], row["flux_err"], row["is_detection"]
        )
    ]
    static = StaticContext(
        host_photoz=row["host_photoz"],
        host_photoz_err=row["host_photoz_err"],
        host_offset_arcsec=row["host_offset_arcsec"],
        host_ellipticity=row["host_ellipticity"],
        host_mag_r=row["host_mag_r"],
        mw_ebv=row["mw_ebv"],
        gal_lat=row["gal_lat"],
        real_bogus_score=row["real_bogus_score"],
    )
    return CanonicalAlert(
        object_id=row["object_id"],
        survey=Survey(row["survey"]),
        observations=observations,
        static=static,
        cutouts=None,
        class_id=row["class_id"],
    )


def load_alerts(
    parquet_path: str, object_ids: Optional[frozenset[str]] = None
) -> list[CanonicalAlert]:
    """Read the cached interim parquet back into `CanonicalAlert` objects.

    Args:
        parquet_path: path to the interim parquet.
        object_ids: if given, only rows whose `object_id` is in this set are
            reconstructed. Reconstructing a `CanonicalAlert` per row (nested
            `Observation` objects, ~147 per object on average) is the
            expensive step -- on the full 843,617-object corpus this takes
            ~50 minutes (measured Day 5), which makes any test or script that
            only needs one split (e.g. the test set, ~15% of objects)
            needlessly slow if it loads everything first. Pass the target
            split's ids (computed cheaply via `load_object_class_ids` +
            `split_by_object_id`, which only ever touch `object_id`/
            `class_id`) to reconstruct only what's needed. `None` (the
            default) preserves the original full-corpus behavior.
    """
    df = pl.read_parquet(parquet_path)
    if object_ids is not None:
        df = df.filter(pl.col("object_id").is_in(object_ids))
    return [_row_to_alert(row) for row in df.iter_rows(named=True)]


def load_object_class_ids(parquet_path: str) -> list:
    """Cheap `(object_id, class_id)`-only read of the interim parquet, for
    computing a split without reconstructing every alert's full observation
    history first. `split_by_object_id` only ever reads `.object_id`/
    `.class_id` off its input, so lightweight stand-ins are sufficient."""
    df = pl.read_parquet(parquet_path, columns=["object_id", "class_id"])
    return [
        _ObjectClassId(object_id=oid, class_id=cid)
        for oid, cid in zip(df["object_id"].to_list(), df["class_id"].to_list())
    ]


@dataclass(frozen=True)
class _ObjectClassId:
    """Lightweight stand-in for a `CanonicalAlert` exposing only the two
    attributes `split_by_object_id`/`class_distribution` actually use."""

    object_id: str
    class_id: Optional[int]


@dataclass(frozen=True)
class SplitAssignment:
    """`object_id` sets for each split. Frozen once built by `split_by_object_id`."""

    train: frozenset[str]
    val: frozenset[str]
    test: frozenset[str]


def split_by_object_id(
    alerts: list[CanonicalAlert],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitAssignment:
    """Deterministic train/val/test split by `object_id`, stratified per
    class so each split's class balance tracks the overall composition
    (CLAUDE.md §11's Day-4 recorded decision: no oversampling/reweighting of
    the natural ELAsTiCC composition -- stratifying the *split* preserves
    that composition in every split rather than letting random assignment
    drift it).

    Raises:
        ValueError: if any `object_id` is duplicated within `alerts` --
            this session confirmed real ELAsTiCC2 data has none, but a
            duplicate here would make split assignment ambiguous (which
            copy "owns" the split?), so it is checked rather than assumed.
    """
    object_ids = [a.object_id for a in alerts]
    if len(object_ids) != len(set(object_ids)):
        seen = set()
        dupes = {oid for oid in object_ids if oid in seen or seen.add(oid)}
        raise ValueError(
            f"split_by_object_id: {len(dupes)} duplicate object_id(s) found, "
            f"e.g. {sorted(dupes)[:5]} -- split assignment would be ambiguous"
        )

    by_class: dict[Optional[int], list[str]] = {}
    for a in alerts:
        by_class.setdefault(a.class_id, []).append(a.object_id)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for class_id in sorted(by_class, key=lambda c: (c is None, c)):
        ids = sorted(by_class[class_id])  # sort first for determinism independent of file order
        random.Random(seed).shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    return SplitAssignment(
        train=frozenset(train_ids), val=frozenset(val_ids), test=frozenset(test_ids)
    )


def class_distribution(alerts: list[CanonicalAlert], ids: frozenset[str]) -> dict[int, int]:
    """Object counts by `class_id`, restricted to `ids`."""
    counts: dict[int, int] = {}
    for a in alerts:
        if a.object_id in ids:
            counts[a.class_id] = counts.get(a.class_id, 0) + 1
    return counts


@dataclass(frozen=True)
class Example:
    object_id: str
    features: torch.Tensor  # (T, NUM_FEATURES)
    class_index: int  # 0/1/2, see taxonomy.TARGET_CLASS_IDS
    delta_t_days: torch.Tensor  # (T,) -- passthrough for phase-cutoff eval


class LightCurveDataset(Dataset):
    """Pre-tensorized (with windowing already applied) in-memory dataset.
    Tensorizing once up front -- rather than in `__getitem__` every epoch --
    is safe because windowing is a deterministic function of each alert's
    own observations and the fixed `pre_detection_window_days`, and it
    avoids repeating that work every epoch.
    """

    def __init__(
        self,
        alerts: list[CanonicalAlert],
        object_ids: frozenset[str],
        pre_detection_window_days: Optional[float],
    ) -> None:
        self._examples: list[Example] = []
        for alert in alerts:
            if alert.object_id not in object_ids:
                continue
            if alert.class_id not in _CLASS_ID_TO_INDEX:
                continue  # defensive; the 3-class parquet should only ever contain these
            t = tensorize.tensorize_alert(alert, pre_detection_window_days=pre_detection_window_days)
            self._examples.append(
                Example(
                    object_id=alert.object_id,
                    features=t.features,
                    class_index=_CLASS_ID_TO_INDEX[alert.class_id],
                    delta_t_days=t.features[:, 0].clone(),
                )
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


@dataclass(frozen=True)
class TrainBatch:
    object_id: list[str]
    features: torch.Tensor  # (B, T_max, NUM_FEATURES)
    mask: torch.Tensor  # (B, T_max) bool
    lengths: torch.Tensor  # (B,) int64
    class_index: torch.Tensor  # (B,) int64
    delta_t_days: torch.Tensor  # (B, T_max), 0.0 in padded positions


def collate_examples(items: list[Example]) -> TrainBatch:
    if not items:
        raise ValueError("collate_examples: empty batch")
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

    return TrainBatch(
        object_id=[item.object_id for item in items],
        features=features,
        mask=mask,
        lengths=torch.tensor(lengths, dtype=torch.int64),
        class_index=torch.tensor([item.class_index for item in items], dtype=torch.int64),
        delta_t_days=delta_t,
    )
