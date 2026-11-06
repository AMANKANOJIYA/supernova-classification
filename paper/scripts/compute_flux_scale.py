"""Median detection flux per tier, before and after the 2026-11-05 zeropoint fix.

Writes paper/data_derived/flux_scale.json, which collect_numbers.py reads for the
limitations paragraph ("real BTS objects are ~N x brighter than the simulated
training set; the bug hid ~M x of that"). All fluxes are FLUXCAL (AB zp 27.5),
detections only (`is_detection`), pooled over every split of each tier.

  - Tier A: data/interim/elasticc2/train_02_full_3class.parquet (no conversion,
    unaffected by the fix).
  - Tier B corrected: data/interim/btsbot_{train,val,test}.parquet.
  - Tier B pre-fix: data/interim/btsbot_{train,val,test}.parquet.pre_zpfix.

Also the per-row corrected/pre-fix flux ratio, matched on (object_id, mjd,
lam_eff), since the two Tier B builds do not store objects in the same order.

Run from the repo root: `.venv/bin/python paper/scripts/compute_flux_scale.py`
"""

import json
from pathlib import Path

import polars as pl

OUT = Path("paper/data_derived/flux_scale.json")
TIER_A = "data/interim/elasticc2/train_02_full_3class.parquet"
SPLITS = ("train", "val", "test")


def detections(paths: list[str]) -> pl.DataFrame:
    return (
        pl.concat([pl.scan_parquet(p, glob=False) for p in paths])
        .select("object_id", "mjd", "lam_eff", "flux", "is_detection")
        .explode("mjd", "lam_eff", "flux", "is_detection")
        .filter(pl.col("is_detection"))
        .select("object_id", "mjd", "lam_eff", "flux")
        .collect()
    )


def main() -> None:
    tier_a_median = (
        pl.scan_parquet(TIER_A)
        .select("flux", "is_detection")
        .explode("flux", "is_detection")
        .filter(pl.col("is_detection"))
        .select(pl.col("flux").median())
        .collect()
        .item()
    )
    fixed = detections([f"data/interim/btsbot_{s}.parquet" for s in SPLITS])
    old = detections([f"data/interim/btsbot_{s}.parquet.pre_zpfix" for s in SPLITS])
    joined = fixed.join(old, on=["object_id", "mjd", "lam_eff"], how="inner", suffix="_old")
    ratio = joined["flux"] / joined["flux_old"]

    report = {
        "tier_a_median_detection_fluxcal": tier_a_median,
        "tier_b_corrected_median_detection_fluxcal": fixed["flux"].median(),
        "tier_b_prefix_median_detection_fluxcal": old["flux"].median(),
        "n_detections_tier_b_corrected": fixed.height,
        "n_detections_tier_b_prefix": old.height,
        "n_rows_matched": joined.height,
        "row_ratio_corrected_over_prefix": {
            "median": ratio.median(),
            "p05": ratio.quantile(0.05),
            "p95": ratio.quantile(0.95),
        },
    }
    report["brightness_ratio_tier_b_corrected_over_tier_a"] = (
        report["tier_b_corrected_median_detection_fluxcal"] / tier_a_median
    )
    report["brightness_ratio_tier_b_prefix_over_tier_a"] = report["tier_b_prefix_median_detection_fluxcal"] / tier_a_median
    report["notes"] = [
        "n_rows_matched exceeds the detection count because a few (object_id, mjd, lam_eff) keys repeat; "
        "the ratio quantiles are insensitive to this and match reports/zeropoint_fix_2026-11-05/README.md.",
        "CLAUDE.md's earlier prose figures (Tier A ~62, so ~41x) do not reproduce: the Tier A median "
        "detection FLUXCAL is 73.2-73.7 in every split. Tier B train alone reproduces its 2578 / 236.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
