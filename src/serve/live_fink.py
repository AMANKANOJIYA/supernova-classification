"""Live Fink producer (CLAUDE.md §6 step 13: "swap producer; config change
only"). Consumes real alerts from Fink's Kafka brokers via `fink-client`,
converts each into a `CanonicalAlert` via `src.readers.avro_ztf` /
`src.readers.avro_lsst`, and republishes onto the SAME internal `alerts`
Kafka topic `src.serve.replay` already produces to for Tier B -- so
`src.serve.inference_worker` (and `src.serve.api`'s `/classify`) consume live
Fink alerts with zero code changes of their own, exactly matching §6's
"same consumer code... swap producer" for steps 12->13.

-----------------------------------------------------------------------------
Pre-filter, stated per Day 13's task instruction ("Filter before inference")
-----------------------------------------------------------------------------
Chosen filter: consume Fink's OWN science-module-filtered candidate topics
(`fink_sn_candidates_ztf` for ZTF, `fink_most_likely_sn_lsst` for LSST)
rather than either survey's full/uniform alert stream. This delegates
real-bogus and SN-likelihood filtering to Fink's own deployed,
production-tested science modules (real-bogus deep-learning score, SN
classifiers, etc. -- these are exactly the kind of vetted filtering this
project would otherwise have to reimplement from scratch) rather than
building a second, redundant, untested filter here. LSST's own alert rate
(up to ~7M/night, per the task) makes this the difference between a
consumable stream and an unconsumable one; ZTF's full stream is smaller but
the same choice applies uniformly across both surveys, for the same reason.
Fink also publishes narrower topics (`fink_early_sn_candidates_ztf`,
`fink_early_snia_candidate_lsst`) for even-earlier-phase candidates, closer
to this project's own early-classification focus (§1) -- not consumed here,
for simplicity; a `--topics` override lets a future session point this
module at those instead without any code change.

-----------------------------------------------------------------------------
Cutouts
-----------------------------------------------------------------------------
Stored separately via `src.serve.cutouts` (Redis `sn:cutouts:{object_id}`),
never through the internal `AlertPacket` wire format -- see that module's
docstring for why.

-----------------------------------------------------------------------------
Credentials
-----------------------------------------------------------------------------
No password was required to consume either broker this session (verified:
listing topics and polling real alerts both succeeded with `username` +
`group.id` only, no `password` key in the Kafka config -- `fink_client.
consumer._get_kafka_config` only enables SASL when both `username` AND
`password` are present). `--username`/`--group-id` are still required CLI
arguments (or `FINK_USERNAME`/`FINK_GROUP_ID` env vars), never hardcoded
defaults, so this module doesn't bake a specific person's identifiers into
committed source.

Entry point:
    python -m src.serve.live_fink --survey ztf --username <u> --group-id <g>
        [--topics topic1 topic2] [--max-messages N] [--max-seconds S]
"""

import argparse
import logging
import os
import time

import redis
from confluent_kafka import Producer
from fink_client.consumer import AlertConsumer

from src.readers.avro_lsst import lsst_alert_to_canonical
from src.readers.avro_ztf import ztf_alert_to_canonical
from src.serve import cutouts as cutouts_store
from src.serve.schemas import AlertPacket

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INTERNAL_KAFKA_BOOTSTRAP = "localhost:9092"
INTERNAL_TOPIC = "alerts"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

FINK_SERVERS = {
    "ztf": "kafka-ztf.fink-broker.org:24499",
    "lsst": "kafka-lsst.fink-broker.org:24499",
}
# See module docstring, "Pre-filter" -- Fink's own curated candidate topics,
# not the raw/uniform stream.
DEFAULT_TOPICS = {
    "ztf": ["fink_sn_candidates_ztf"],
    "lsst": ["fink_most_likely_sn_lsst"],
}
READERS = {
    "ztf": ztf_alert_to_canonical,
    "lsst": lsst_alert_to_canonical,
}


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("delivery failed for object_id key=%s: %s", msg.key(), err)


def run(
    survey: str,
    username: str,
    group_id: str,
    topics: list[str] | None = None,
    max_messages: int | None = None,
    max_seconds: float | None = None,
    poll_timeout: float = 5.0,
) -> dict:
    """Returns a summary dict: `n_received`, `n_produced`, `n_skipped`,
    `skip_reasons` (dict[str, int]), `n_cutouts_stored`, `elapsed_seconds`.
    """
    if survey not in READERS:
        raise ValueError(f"survey must be one of {sorted(READERS)}, got {survey!r}")
    topics = topics or DEFAULT_TOPICS[survey]
    reader = READERS[survey]

    config = {
        "username": username,
        "group.id": group_id,
        "bootstrap.servers": FINK_SERVERS[survey],
    }
    consumer = AlertConsumer(topics=topics, config=config, survey=survey)
    producer = Producer({"bootstrap.servers": INTERNAL_KAFKA_BOOTSTRAP})
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    n_received = n_produced = n_skipped = n_cutouts_stored = 0
    n_detection_obs = n_negative_flux_detection_obs = 0
    skip_reasons: dict[str, int] = {}
    start = time.monotonic()

    logger.info(
        "live_fink[%s]: subscribing to %s on %s as username=%s group.id=%s",
        survey, topics, FINK_SERVERS[survey], username, group_id,
    )

    try:
        while True:
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                logger.info("live_fink[%s]: max_seconds=%.1f reached, stopping", survey, max_seconds)
                break
            if max_messages is not None and n_received >= max_messages:
                logger.info("live_fink[%s]: max_messages=%d reached, stopping", survey, max_messages)
                break

            _, alert, _ = consumer.poll(timeout=poll_timeout)
            if alert is None:
                continue
            n_received += 1

            try:
                canonical = reader(alert)
            except ValueError as exc:
                n_skipped += 1
                reason = str(exc)
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.warning("live_fink[%s]: skipping alert (%s)", survey, exc)
                continue

            # Negative-flux fraction (Day 13 task instruction) -- only over
            # real detections, since a non-detection's flux is defined as
            # exactly 0.0 by this session's own convention (see avro_ztf.py/
            # avro_lsst.py module docstrings), not a real measured value that
            # could itself be negative.
            for obs in canonical.observations:
                if obs.is_detection:
                    n_detection_obs += 1
                    if obs.flux < 0:
                        n_negative_flux_detection_obs += 1

            packet = AlertPacket.from_alert_slice(
                object_id=canonical.object_id,
                survey=canonical.survey,
                observations=canonical.observations,
                static=canonical.static,
                class_id=None,  # live stream: no spectroscopic label at ingest time
            )
            producer.produce(
                INTERNAL_TOPIC,
                key=packet.object_id.encode("utf-8"),
                value=packet.to_json(),
                callback=_delivery_callback,
            )
            n_produced += 1

            if canonical.cutouts is not None:
                cutouts_store.store_cutouts(redis_client, canonical.object_id, canonical.cutouts)
                n_cutouts_stored += 1

            if n_produced % 20 == 0:
                producer.poll(0)
                logger.info(
                    "live_fink[%s]: %d received, %d produced, %d skipped, %d cutouts stored",
                    survey, n_received, n_produced, n_skipped, n_cutouts_stored,
                )
    finally:
        producer.flush()
        consumer.close()

    elapsed = time.monotonic() - start
    summary = {
        "survey": survey,
        "topics": topics,
        "n_received": n_received,
        "n_produced": n_produced,
        "n_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "n_cutouts_stored": n_cutouts_stored,
        "n_detection_observations": n_detection_obs,
        "n_negative_flux_detection_observations": n_negative_flux_detection_obs,
        "negative_flux_fraction": (
            n_negative_flux_detection_obs / n_detection_obs if n_detection_obs else None
        ),
        "elapsed_seconds": elapsed,
    }
    logger.info("live_fink[%s]: done. %s", survey, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=sorted(READERS))
    parser.add_argument("--username", default=os.environ.get("FINK_USERNAME"))
    parser.add_argument("--group-id", default=os.environ.get("FINK_GROUP_ID"))
    parser.add_argument("--topics", nargs="+", default=None)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--report-path", default=None, help="write the run summary as JSON to this path")
    args = parser.parse_args()

    if not args.username or not args.group_id:
        parser.error("--username/--group-id required (or FINK_USERNAME/FINK_GROUP_ID env vars)")

    summary = run(
        survey=args.survey,
        username=args.username,
        group_id=args.group_id,
        topics=args.topics,
        max_messages=args.max_messages,
        max_seconds=args.max_seconds,
    )

    if args.report_path:
        import json

        os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
        with open(args.report_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
