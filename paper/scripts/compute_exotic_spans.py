"""One-off derived-statistics script for the paper (NOT part of src/, not a
training or metrics-recomputation script). Reads the already-downloaded raw
ELAsTiCC2 exotic model directories (data/raw/elasticc2/ELASTICC2_TRAIN_02_{TDE,EB,CLAGN})
through the existing, frozen `src.readers.elasticc2` reader and the existing
`src.diagnostics` helpers, and writes a small derived-stats JSON the paper's
Sec. 4 windowing subsection cites (Table: detection span by class).

This computes a NEW aggregate statistic from already-existing raw data via
already-existing, unmodified reader code -- it does not retrain any model,
recompute any existing eval metric, or change any src/ file. Per the paper
spec's "absolute rule on numbers": every value used in the report must be
either read from disk or computed by a script under paper/scripts/ that
reads from disk. This script is that computation for the TDE/EB/CLAGN
detection-span numbers, which CLAUDE.md only ever recorded as prose.
"""

import json
import time
from pathlib import Path

from src import taxonomy
from src.readers import elasticc2

RAW_DIR = Path("data/raw/elasticc2")
OUT_PATH = Path("paper/data_derived/exotic_spans.json")

MODELS = {
    "TDE": "ELASTICC2_TRAIN_02_TDE",
    "EB": "ELASTICC2_TRAIN_02_EB",
    "CLAGN": "ELASTICC2_TRAIN_02_CLAGN",
}


def detection_span_days(alert) -> float | None:
    mjds = [obs.mjd for obs in alert.observations if obs.is_detection]
    if not mjds:
        return None
    return max(mjds) - min(mjds)


def nobs(alert) -> int:
    return len(alert.observations)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def main() -> None:
    gentype_to_classid = taxonomy.load_gentype_of_classid(
        str(RAW_DIR / "elasticc2_gentypeofclassid.csv")
    )
    results = {}
    for name, dirname in MODELS.items():
        t0 = time.time()
        alerts, stats = elasticc2.read_model_directory(str(RAW_DIR / dirname), gentype_to_classid)
        elapsed = time.time() - t0
        spans = [s for s in (detection_span_days(a) for a in alerts) if s is not None]
        nobs_list = [nobs(a) for a in alerts]
        results[name] = {
            "n_objects_read": stats.n_objects_read,
            "n_objects_discarded": stats.n_objects_discarded,
            "mean_detection_span_days": mean(spans),
            "median_detection_span_days": median(spans),
            "mean_nobs_per_object": mean(nobs_list),
            "median_nobs_per_object": median(nobs_list),
            "elapsed_seconds": elapsed,
        }
        print(f"{name}: {results[name]}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
