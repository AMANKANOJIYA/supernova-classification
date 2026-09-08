"""Derived-statistics script for the paper (Sec. 4, "Windowing" subsection,
Fig. 3): mean detections vs. forced-photometry per object, and the
sequence-length distribution before/after `tensorize_alert`'s left-edge
windowing -- computed from a stratified sample of the ALREADY-EXISTING Tier A
interim parquet (data/interim/elasticc2/train_02_full_3class.parquet) via
the existing, unmodified `src.train.dataset.load_alerts` /
`src.diagnostics` / `src.features.tensorize` functions.

A full-corpus reconstruction (843,617 objects) costs ~50 minutes (see
`load_alerts`'s own docstring) -- this script instead draws a stratified
random sample (default 3,000 objects, 1,000 per target class, seed 42) via
the cheap `load_object_class_ids` path, matching the "object-count-only"
scan pattern the codebase already uses elsewhere for the same reason
(e.g. `finetune_bts.py`'s persist_anomaly_scorers.py fast-load path).

Not a retrain, not a metric recomputation, not a src/ change -- this reads
stored data through frozen/unmodified reader and tensorize code to produce
one aggregate statistic and one histogram's worth of numbers for a figure.
"""

import json
import random
from collections import Counter
from pathlib import Path

from src import diagnostics
from src.features import tensorize
from src.train.dataset import load_alerts, load_object_class_ids

PARQUET = "data/interim/elasticc2/train_02_full_3class.parquet"
OUT_PATH = Path("paper/data_derived/windowing_stats.json")
SEED = 42
N_PER_CLASS = 1000
TARGET_CLASSES = (2222, 2223, 2224)


def main() -> None:
    rng = random.Random(SEED)
    object_class_ids = load_object_class_ids(PARQUET)
    by_class: dict[int, list[str]] = {c: [] for c in TARGET_CLASSES}
    for row in object_class_ids:
        if row.class_id in by_class:
            by_class[row.class_id].append(row.object_id)

    sample_ids: set[str] = set()
    for cid in TARGET_CLASSES:
        pool = by_class[cid]
        rng.shuffle(pool)
        sample_ids.update(pool[:N_PER_CLASS])

    print(f"Sampled {len(sample_ids)} object ids across {len(TARGET_CLASSES)} classes", flush=True)
    alerts = load_alerts(PARQUET, object_ids=frozenset(sample_ids))
    print(f"Reconstructed {len(alerts)} alerts", flush=True)

    detections = diagnostics.detections_per_object(alerts)
    forced = diagnostics.forced_photometry_per_object(alerts)

    seq_len_unwindowed = []
    seq_len_windowed = []
    for a in alerts:
        seq_len_unwindowed.append(
            tensorize.tensorize_alert(a, pre_detection_window_days=None, max_span_days=None).features.shape[0]
        )
        seq_len_windowed.append(
            tensorize.tensorize_alert(
                a,
                pre_detection_window_days=tensorize.DEFAULT_PRE_DETECTION_WINDOW_DAYS,
                max_span_days=None,
            ).features.shape[0]
        )

    mean_unwindowed = sum(seq_len_unwindowed) / len(seq_len_unwindowed)
    mean_windowed = sum(seq_len_windowed) / len(seq_len_windowed)

    result = {
        "n_sample_objects": len(alerts),
        "mean_detections_per_object": sum(detections) / len(detections),
        "mean_forced_photometry_per_object": sum(forced) / len(forced),
        "mean_seq_len_unwindowed": mean_unwindowed,
        "mean_seq_len_windowed": mean_windowed,
        "fraction_observations_removed": 1.0 - mean_windowed / mean_unwindowed,
        "seq_len_unwindowed_histogram": dict(Counter(seq_len_unwindowed // 20 * 20 for seq_len_unwindowed in seq_len_unwindowed)),
        "seq_len_windowed_histogram": dict(Counter(seq_len_windowed // 20 * 20 for seq_len_windowed in seq_len_windowed)),
        "seq_len_unwindowed_raw": seq_len_unwindowed,
        "seq_len_windowed_raw": seq_len_windowed,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print({k: v for k, v in result.items() if not k.endswith("_raw") and not k.endswith("_histogram")})


if __name__ == "__main__":
    main()
