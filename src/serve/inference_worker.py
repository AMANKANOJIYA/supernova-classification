"""The one consumer implementation (CLAUDE.md §6: "same consumer code" for
both replay and live mode -- only the producer differs, per step 13's "swap
producer; config change only"). Consumes alert packets from Kafka, merges
each into the object's Redis-backed rolling history (`src.serve.state`),
scores every packet that carries a NEW DETECTION (§5.1: never re-score on
forced photometry alone -- `src.serve.ingest.has_new_detection` gates this),
and writes one row per scored alert to Postgres.

Micro-batched: alerts are accumulated into a batch (bounded by count and by
a wait-time ceiling, whichever comes first) before one ONNX Runtime call
scores the whole batch -- CLAUDE.md's own instruction ("ONNX export of the
Tier B model, micro-batched"). A pure count-based batch would stall
indefinitely under low traffic (an object's Redis state never gets scored
until enough OTHER alerts arrive); the wait-time ceiling bounds that.

Entry point: `python -m src.serve.inference_worker [--max-messages N] [--max-seconds S]`
"""

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import redis
import torch
from confluent_kafka import Consumer, KafkaError

from src.features.tensorize import NUM_FEATURES, RECOMMENDED_MAX_SPAN_DAYS, tensorize_alert
from src.schema import CanonicalAlert
from src.serve import db, ingest, state
from src.serve.model import CLASS_NAMES, AnomalyScorer, ONNXClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "alerts"
GROUP_ID = "inference-worker"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

PRE_DETECTION_WINDOW_DAYS = 30.0  # matches every trained checkpoint's own training window

MICRO_BATCH_SIZE = 32
MICRO_BATCH_MAX_WAIT_SECONDS = 0.1
POLL_TIMEOUT_SECONDS = 0.05


@dataclass
class _PendingScore:
    alert: CanonicalAlert
    survey: str
    alert_mjd: float  # the mjd of the detection that triggered this score
    true_class_id: int | None
    recv_time: float  # time.monotonic() when this alert was pulled off Kafka


def _build_batch_tensor(pending: list[_PendingScore]) -> tuple[np.ndarray, np.ndarray]:
    tensorized = [
        tensorize_alert(
            p.alert,
            pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS,
            max_span_days=RECOMMENDED_MAX_SPAN_DAYS,
        )
        for p in pending
    ]
    lengths = np.array([t.features.shape[0] for t in tensorized], dtype=np.int64)
    t_max = int(lengths.max())
    features = np.zeros((len(pending), t_max, NUM_FEATURES), dtype=np.float32)
    for i, t in enumerate(tensorized):
        features[i, : t.features.shape[0]] = t.features.numpy()
    return features, lengths


def _score_batch(
    pending: list[_PendingScore],
    classifier: ONNXClassifier,
    anomaly_scorer: AnomalyScorer,
    pg_conn,
) -> None:
    t0 = time.monotonic()
    features, lengths = _build_batch_tensor(pending)

    probs = classifier.predict_probs(features, lengths)

    features_t = torch.from_numpy(features)
    lengths_t = torch.from_numpy(lengths)
    t_max = features.shape[1]
    mask_t = (torch.arange(t_max).unsqueeze(0) < lengths_t.unsqueeze(1))
    anomaly_scores = anomaly_scorer.score(features_t, lengths_t, mask_t)

    elapsed_ms_total = (time.monotonic() - t0) * 1000.0

    for i, p in enumerate(pending):
        predicted_class = CLASS_NAMES[int(np.argmax(probs[i]))]
        latency_ms = (time.monotonic() - p.recv_time) * 1000.0
        db.insert_prediction(
            pg_conn,
            object_id=p.alert.object_id,
            survey=p.survey,
            alert_mjd=p.alert_mjd,
            predicted_class=predicted_class,
            probs=tuple(float(x) for x in probs[i]),
            anomaly_score=float(anomaly_scores[i]),
            true_class_id=p.true_class_id,
            seq_length=int(lengths[i]),
            inference_latency_ms=latency_ms,
        )
    logger.info(
        "scored batch of %d alert(s) in %.2fms model time (%.2fms/alert)",
        len(pending), elapsed_ms_total, elapsed_ms_total / len(pending),
    )


def run(
    max_messages: int | None = None,
    max_seconds: float | None = None,
    idle_stop_seconds: float | None = None,
) -> None:
    """`idle_stop_seconds`: stop once no message has arrived for this long
    AND nothing is pending -- the natural termination condition for a
    finite replay run (a real live consumer should leave this `None` and
    run forever)."""
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    classifier = ONNXClassifier()
    anomaly_scorer = AnomalyScorer()

    with db.connect() as pg_conn:
        db.init_schema(pg_conn)

        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": GROUP_ID,
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([TOPIC])
        logger.info("Subscribed to topic=%s bootstrap=%s", TOPIC, KAFKA_BOOTSTRAP)

        n_messages = 0
        n_scored = 0
        start_time = time.monotonic()
        first_message_time: float | None = None
        last_message_time = start_time
        pending: list[_PendingScore] = []
        batch_started_at: float | None = None

        try:
            while True:
                if max_seconds is not None and time.monotonic() - start_time > max_seconds:
                    logger.info("max_seconds=%.1f reached, stopping", max_seconds)
                    break
                if max_messages is not None and n_messages >= max_messages:
                    logger.info("max_messages=%d reached, stopping", max_messages)
                    break
                if (
                    idle_stop_seconds is not None
                    and not pending
                    and n_messages > 0
                    and time.monotonic() - last_message_time > idle_stop_seconds
                ):
                    logger.info("idle for >%.1fs with nothing pending, stopping", idle_stop_seconds)
                    break

                msg = consumer.poll(POLL_TIMEOUT_SECONDS)
                if msg is not None and msg.error() is None:
                    recv_time = time.monotonic()
                    if first_message_time is None:
                        first_message_time = recv_time
                    last_message_time = recv_time
                    n_messages += 1
                    try:
                        normalized = ingest.parse_alert_packet(msg.value())
                    except ValueError:
                        logger.exception("dropping malformed alert packet")
                    else:
                        alert = state.get_and_merge(redis_client, normalized)
                        if ingest.has_new_detection(normalized):
                            trigger_mjd = max(
                                o.mjd for o in normalized.new_observations if o.is_detection
                            )
                            pending.append(
                                _PendingScore(
                                    alert=alert,
                                    survey=normalized.survey.value,
                                    alert_mjd=trigger_mjd,
                                    true_class_id=normalized.class_id,
                                    recv_time=recv_time,
                                )
                            )
                            if batch_started_at is None:
                                batch_started_at = recv_time
                elif msg is not None and msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.warning("Kafka error: %s", msg.error())

                batch_wait_expired = (
                    batch_started_at is not None
                    and time.monotonic() - batch_started_at >= MICRO_BATCH_MAX_WAIT_SECONDS
                )
                if pending and (len(pending) >= MICRO_BATCH_SIZE or batch_wait_expired):
                    _score_batch(pending, classifier, anomaly_scorer, pg_conn)
                    n_scored += len(pending)
                    pending = []
                    batch_started_at = None

            if pending:
                _score_batch(pending, classifier, anomaly_scorer, pg_conn)
                n_scored += len(pending)

            wall_elapsed = time.monotonic() - start_time
            busy_elapsed = (
                (last_message_time - first_message_time) if first_message_time is not None else 0.0
            )
            logger.info(
                "Consumed %d message(s), scored %d alert(s); wall=%.1fs busy=%.1fs "
                "(%.1f alerts/sec over busy interval, %.1f alerts/sec over wall time)",
                n_messages, n_scored, wall_elapsed, busy_elapsed,
                n_scored / busy_elapsed if busy_elapsed > 0 else float("nan"),
                n_scored / wall_elapsed if wall_elapsed > 0 else float("nan"),
            )
        finally:
            consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--idle-stop-seconds", type=float, default=None)
    args = parser.parse_args()
    run(max_messages=args.max_messages, max_seconds=args.max_seconds, idle_stop_seconds=args.idle_stop_seconds)


if __name__ == "__main__":
    main()
