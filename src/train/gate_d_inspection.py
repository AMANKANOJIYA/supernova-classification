"""Gate D (CLAUDE.md §7): manually inspect the top 50 combined-score
anomalies from `src.train.train_anomaly`'s Tier B run and categorize them.

Run AFTER `python -m src.train.train_anomaly` has written
`outputs/anomaly/scores_tier_b_*.json` and `tier_b_label_by_object_id.json`.
Entry point: `python -m src.train.gate_d_inspection`

-----------------------------------------------------------------------------
A real limitation, stated up front rather than glossed over
-----------------------------------------------------------------------------
CLAUDE.md §7 Gate D's own artifact-rate expectation (68%, from a published
2.25M-object ZTF run) is about IMAGE-level subtraction artifacts and
blending -- a judgment made by looking at cutout pixels. This pipeline has
no cutout pixel data anywhere: ELAsTiCC2 (Tier A) has none by construction
(§5.1), BTSbot's `images_v10` has never been loaded into any reader
(`src/readers/btsbot.py`'s own comment: "images_v10 is Day 10" -- and this
session's user explicitly chose to build the anomaly branch instead of the
image encoder, see CLAUDE.md §10's Day-10 reorder note). So this Gate D
pass is necessarily PHOTOMETRY-ONLY: light-curve shape/cadence heuristics
stand in for an eye-on-the-cutout judgment. This is real, load-bearing gap,
not a stylistic caveat -- a proper image-based Gate D can only happen once
cutouts are loaded. Categorization below is reported as a heuristic proxy,
not a substitute for actual visual inspection of subtraction images.

Three buckets (CLAUDE.md's own wording): "genuine astrophysical anomaly,
artifact, or short/sparse light curve."
  - short_sparse: n_obs < SHORT_SPARSE_MIN_OBS or time_span_days <
    SHORT_SPARSE_MIN_SPAN_DAYS. Directly checkable, no proxy needed.
  - artifact_like: NOT short/sparse, but dominated by a single extreme
    outlier point relative to its neighbors in flux_norm (a discontinuous
    single-point jump is the photometric signature closest to what a
    subtraction artifact or bad-pixel spike would leave in the light curve,
    though a real fast transient -- KN, M-dwarf flare -- can look similar;
    this is exactly the ambiguity a cutout would resolve and this heuristic
    cannot).
  - genuine_anomaly: everything else -- a light curve that is unusual in
    shape/duration/cadence but not attributable to one isolated bad point.
"""

import json
import logging
from pathlib import Path

from src.diagnostics import plot_light_curve
from src.train.train_anomaly import load_tier_b_weird, OUTPUT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORT_DIR = Path("reports/gate_d")
TOP_N = 50
SHORT_SPARSE_MIN_OBS = 8
SHORT_SPARSE_MIN_SPAN_DAYS = 3.0
OUTLIER_RATIO_THRESHOLD = 4.0  # max |flux_norm - median| vs median absolute deviation of the rest


def _load_scores(name: str) -> dict:
    with open(OUTPUT_DIR / f"scores_{name}.json") as f:
        return json.load(f)


def _categorize(alert) -> str:
    obs = alert.observations
    n_obs = len(obs)
    time_span = max(o.mjd for o in obs) - min(o.mjd for o in obs) if obs else 0.0
    if n_obs < SHORT_SPARSE_MIN_OBS or time_span < SHORT_SPARSE_MIN_SPAN_DAYS:
        return "short_sparse"

    import numpy as np

    flux = np.array([o.flux for o in obs])
    flux_err = np.array([o.flux_err for o in obs])
    flux_norm = np.sign(flux / flux_err) * np.log1p(np.abs(flux / flux_err))
    for i in range(len(flux_norm)):
        rest = np.delete(flux_norm, i)
        mad = np.median(np.abs(rest - np.median(rest))) or 1e-6
        if abs(flux_norm[i] - np.median(rest)) / mad > OUTLIER_RATIO_THRESHOLD:
            return "artifact_like"
    return "genuine_anomaly"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    light_curve_dir = REPORT_DIR / "light_curves"
    light_curve_dir.mkdir(parents=True, exist_ok=True)

    weird_scores = _load_scores("tier_b_weird")
    normal_scores = _load_scores("tier_b_normal_test")

    pooled = []
    for scores, source in ((weird_scores, "weird"), (normal_scores, "normal_test")):
        for oid, combined in zip(scores["object_id"], scores["combined"]):
            pooled.append((oid, combined, source))
    pooled.sort(key=lambda x: x[1], reverse=True)
    top = pooled[:TOP_N]
    top_ids = {oid for oid, _, _ in top}

    logger.info("Loading Tier B alerts to match top-%d object_ids against...", TOP_N)
    tier_b = load_tier_b_weird()
    alerts_by_id = {a.object_id: a for a in (tier_b.weird + tier_b.normal_test)}

    breakdown: dict[str, int] = {"short_sparse": 0, "artifact_like": 0, "genuine_anomaly": 0, "not_found": 0}
    rows = []
    for oid, combined, source in top:
        alert = alerts_by_id.get(oid)
        if alert is None:
            breakdown["not_found"] += 1
            rows.append({"object_id": oid, "combined": combined, "source": source, "category": "not_found"})
            continue
        category = _categorize(alert)
        breakdown[category] += 1
        label = tier_b.label_by_object_id.get(oid, "SN-like (Ia/Ib-c/II)" if source == "normal_test" else "unknown")
        rows.append(
            {
                "object_id": oid,
                "combined": combined,
                "source": source,
                "label": label,
                "category": category,
                "n_obs": len(alert.observations),
            }
        )
        fig = plot_light_curve(alert, class_name=f"{label} (Gate D rank, combined={combined:.2f})")
        fig.savefig(light_curve_dir / f"{oid}.png", dpi=110)
        import matplotlib.pyplot as plt

        plt.close(fig)

    logger.info("Gate D top-%d breakdown: %s", TOP_N, breakdown)
    with open(REPORT_DIR / "gate_d_top50.json", "w") as f:
        json.dump({"breakdown": breakdown, "objects": rows}, f, indent=2)
    logger.info("Wrote %s and %d light-curve plots to %s", REPORT_DIR / "gate_d_top50.json", len(rows), light_curve_dir)


if __name__ == "__main__":
    main()
