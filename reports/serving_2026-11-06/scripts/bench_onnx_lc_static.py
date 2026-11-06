"""Isolated ONNX timing for the deployed LC+static classifier.

Same method as reports/audit_2026-11-05/serving_benchmark/bench_onnx.json (which
timed the earlier LC-only model): `ONNXClassifier.predict_probs` only, on the
Tier B test split's 658 labelled objects, full tensorized sequences built exactly
as the inference worker builds them (200-day span cap), CPU, after warm-up.
Batch sizes 1 and 32. Run with nothing else heavy on the machine.

Run from the repo root: `.venv/bin/python reports/serving_2026-11-06/scripts/bench_onnx_lc_static.py`
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

from src.features.tensorize import NUM_FEATURES, RECOMMENDED_MAX_SPAN_DAYS, tensorize_alert
from src.serve.inference_worker import PRE_DETECTION_WINDOW_DAYS
from src.serve.model import TIER_B_ONNX_PATH, ONNXClassifier
from src.train.train_anomaly import load_tier_b_weird

OUT = Path("reports/serving_2026-11-06/bench_onnx_lc_static.json")
WARMUP_PASSES = 2


def pad(tensorized) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([t.features.shape[0] for t in tensorized], dtype=np.int64)
    features = np.zeros((len(tensorized), int(lengths.max()), NUM_FEATURES), dtype=np.float32)
    for i, t in enumerate(tensorized):
        features[i, : lengths[i]] = t.features
    return features, lengths


def main() -> None:
    alerts = load_tier_b_weird().normal_test
    tensorized = [
        tensorize_alert(a, pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS, max_span_days=RECOMMENDED_MAX_SPAN_DAYS)
        for a in alerts
    ]
    clf = ONNXClassifier()
    result = {}
    for batch_size in (1, 32):
        batches = [
            (pad(tensorized[i : i + batch_size]), [a.static for a in alerts[i : i + batch_size]])
            for i in range(0, len(alerts), batch_size)
        ]
        for _ in range(WARMUP_PASSES):
            for (features, lengths), statics in batches:
                clf.predict_probs(features, lengths, statics)
        per_alert_ms = []
        for (features, lengths), statics in batches:
            t0 = time.perf_counter()
            clf.predict_probs(features, lengths, statics)
            per_alert_ms.append((time.perf_counter() - t0) * 1000.0 / len(statics))
        result[str(batch_size)] = {
            "n_batches": len(batches),
            "per_alert_ms_p50": float(np.percentile(per_alert_ms, 50)),
            "per_alert_ms_p95": float(np.percentile(per_alert_ms, 95)),
            "per_alert_ms_mean": float(np.mean(per_alert_ms)),
        }
    report = {
        "model": str(TIER_B_ONNX_PATH),
        "n_objects": len(alerts),
        "onnx_isolated": result,
        "machine": platform.platform(),
        "note": "isolated = ONNXClassifier.predict_probs only (includes building the normalised static matrix), "
                "Tier B test labelled objects, full tensorized sequences with the 200-day serving cap, CPU, warm",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
