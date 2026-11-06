"""Class distribution of a live run, raw and prior-corrected, read back from the
serving Postgres (latest prediction per object, survey filter, scored_at window).

Two prior corrections, both corrected(c) ∝ p(c) * sky(c) / train(c), renormalised:

  1. `elasticc_prior` -- train = ELAsTiCC composition (20.4 / 28.5 / 51.1 %), the
     method of CLAUDE.md Day 14 and reports/audit_2026-11-05/scripts/db_audit.py.
     Kept so this run is comparable with every earlier live run.
  2. `tier_b_effective_prior` -- train = the prior the DEPLOYED model was last
     trained under: Tier B train class counts (finetune_seed42/gate_c_report.json
     join_stats) times the sqrt-dampened class weights finetune_bts.py applies
     (`src.train.pretrain.compute_class_weights`), i.e. ∝ sqrt(count). This is the
     prior the LC+static fine-tune's softmax actually carries.

Sky priors: midpoint and the two Ib/c extremes of CLAUDE.md §11's sky-rate ranges,
identical to db_audit.py. The robustness range is the spread over those three.

Usage: .venv/bin/python reports/live_2026-11-06/scripts/live_classdist.py <survey> <since_utc_iso> <until_utc_iso> <out_json>
"""

import json
import sys
from pathlib import Path

import numpy as np
import psycopg

from src.serve.db import DEFAULT_DSN

NAMES = ["Ia", "Ib/c", "II"]
ELASTICC_PRIOR = np.array([0.204, 0.285, 0.511])
SKY = {"midpoint": [0.275, 0.15, 0.575], "low_ibc": [0.30, 0.10, 0.60], "high_ibc": [0.25, 0.20, 0.55]}
GATE_C = "reports/zeropoint_fix_2026-11-05/finetune_seed42/gate_c_report.json"


def tier_b_effective_prior() -> np.ndarray:
    train = json.load(open(GATE_C))["join_stats"]["train"]
    counts = np.array([train["mapped_ia"], train["mapped_ibc"], train["mapped_ii"]], dtype=float)
    weights = np.sqrt(counts.sum() / (3 * counts))
    weights /= weights.mean()
    effective = counts * weights
    return effective / effective.sum()


def distribution(p: np.ndarray) -> dict:
    counts = np.bincount(p.argmax(1), minlength=3)
    return {n: round(100.0 * c / len(p), 1) for n, c in zip(NAMES, counts)}


def corrected(p: np.ndarray, train_prior: np.ndarray) -> dict:
    out = {}
    for key, sky in SKY.items():
        q = p * (np.array(sky) / train_prior)
        out[key] = distribution(q / q.sum(1, keepdims=True))
    ibc = [out[k]["Ib/c"] for k in SKY]
    out["ibc_range_pct"] = [min(ibc), max(ibc)]
    return out


def main() -> None:
    survey, since, until, out_path = sys.argv[1:5]
    with psycopg.connect(DEFAULT_DSN) as conn:
        rows = conn.execute(
            """select object_id, scored_at_utc, prob_ia, prob_ibc, prob_ii, seq_length, inference_latency_ms
               from predictions where survey = %s and scored_at_utc >= %s and scored_at_utc < %s
               order by scored_at_utc, id""",
            (survey, since, until),
        ).fetchall()
    latest = {}
    for r in rows:
        latest[r[0]] = r  # rows are time-ordered, so the last one wins
    p = np.array([[r[2], r[3], r[4]] for r in latest.values()], dtype=float)
    lat = np.array([r[6] for r in rows], dtype=float)
    tb_prior = tier_b_effective_prior()
    report = {
        "survey": survey,
        "window_utc": [since, until],
        "n_rows": len(rows),
        "n_objects": len(latest),
        "mean_seq_length_latest": float(np.mean([r[5] for r in latest.values()])),
        "latency_ms": {k: float(np.percentile(lat, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "raw_pct": distribution(p),
        "elasticc_prior": {"train_prior": ELASTICC_PRIOR.tolist(), **corrected(p, ELASTICC_PRIOR)},
        "tier_b_effective_prior": {"train_prior": tb_prior.tolist(), **corrected(p, tb_prior)},
        "sky_priors": SKY,
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
