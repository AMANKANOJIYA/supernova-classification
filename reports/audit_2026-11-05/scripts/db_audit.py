"""Read-only re-measurement from the serving Postgres (predictions table).
Usage: python db_audit.py <repo_root> <out_json> [<since_utc_iso>]
Splits rows into runs by gaps > 20 min in scored_at_utc, per survey.
For each run: rows, objects, latency percentiles (inference_latency_ms =
Kafka-poll receipt -> just before that row's INSERT), busy-interval
throughput, latest-prediction-per-object class distribution, and, where
labels exist, macro-F1 / Ia ROC-AUC via the frozen src.eval.metrics.
Prior correction: corrected(c) ∝ p(c) * sky(c)/train(c), as in CLAUDE.md Day 14.
"""
import json, sys
from pathlib import Path
import numpy as np, psycopg

sys.path.insert(0, sys.argv[1])
from src.eval.metrics import macro_f1, per_class_precision_recall_f1, roc_auc, confusion_matrix
from src.serve.db import DEFAULT_DSN

OUT = Path(sys.argv[2]); SINCE = sys.argv[3] if len(sys.argv) > 3 else "1970-01-01"
CIDX = {2222: 0, 2223: 1, 2224: 2}; NAMES = ["Ia", "Ib/c", "II"]
TRAIN_PRIOR = np.array([0.204, 0.285, 0.511])
SKY = {"midpoint": [0.275, 0.15, 0.575], "low_ibc": [0.30, 0.10, 0.60], "high_ibc": [0.25, 0.20, 0.55]}

with psycopg.connect(DEFAULT_DSN) as c:
    rows = c.execute("""select id, object_id, survey, scored_at_utc, extract(epoch from scored_at_utc),
        prob_ia, prob_ibc, prob_ii, true_class_id, seq_length, inference_latency_ms, alert_mjd
        from predictions where scored_at_utc >= %s order by survey, scored_at_utc, id""", (SINCE,)).fetchall()

runs, cur = [], None
for r in rows:
    if cur is None or r[2] != cur[-1][2] or float(r[4]) - float(cur[-1][4]) > 1200:
        cur = []; runs.append(cur)
    cur.append(r)

def prior_correct(P, sky):
    q = P * (np.array(sky) / TRAIN_PRIOR); return q / q.sum(1, keepdims=True)

out = []
for run in runs:
    t = np.array([float(r[4]) for r in run]); lat = np.array([r[10] for r in run])
    latest = {}
    for r in run: latest[r[1]] = r  # rows ordered by time -> last wins
    P = np.array([[r[5], r[6], r[7]] for r in latest.values()])
    dist = lambda probs: {NAMES[k]: float((probs.argmax(1) == k).mean()) for k in range(3)}
    d = {"survey": run[0][2], "start_utc": str(run[0][3]), "end_utc": str(run[-1][3]),
         "n_rows": len(run), "n_objects": len(latest),
         "busy_seconds": float(t[-1] - t[0]),
         "throughput_alerts_per_sec_busy": len(run) / float(t[-1] - t[0]) if t[-1] > t[0] else None,
         "latency_ms": {k: float(np.percentile(lat, q)) for k, q in [("p50", 50), ("p95", 95), ("p99", 99)]}
                       | {"mean": float(lat.mean()), "max": float(lat.max())},
         "mean_seq_length_latest": float(np.mean([r[9] for r in latest.values()])),
         "class_distribution_latest_raw": dist(P),
         "class_distribution_latest_prior_corrected": {k: dist(prior_correct(P, v)) for k, v in SKY.items()},
         "alert_mjd_range": [float(min(r[11] for r in run)), float(max(r[11] for r in run))]}
    lab = [(CIDX[r[8]], [r[5], r[6], r[7]]) for r in latest.values() if r[8] in CIDX]
    if lab:
        y = np.array([a for a, _ in lab]); pr = np.array([b for _, b in lab])
        d["labeled_3class_objects"] = len(y)
        d["macro_f1_latest"] = float(macro_f1(y, pr.argmax(1), 3))
        d["per_class_f1_latest"] = [float(x) for x in per_class_precision_recall_f1(y, pr.argmax(1), 3).f1]
        d["ia_vs_rest_roc_auc_latest"] = float(roc_auc(y, pr, 3)[0])
        d["confusion_latest"] = confusion_matrix(y, pr.argmax(1), 3).matrix.tolist() if hasattr(confusion_matrix(y, pr.argmax(1), 3), "matrix") else None
    out.append(d)
    print(json.dumps({k: d[k] for k in d if k not in ("class_distribution_latest_prior_corrected",)}, default=str))
OUT.write_text(json.dumps({"method": __doc__, "train_prior": TRAIN_PRIOR.tolist(), "sky_priors": SKY, "runs": out}, indent=1, default=str))
