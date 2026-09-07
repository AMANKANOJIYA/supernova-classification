"""Kafka producer replaying archived Tier B parquet in MJD order (CLAUDE.md
§6 step 12: "replay before live... same consumer code, archived data as the
producer"). No credentials, nothing that can block -- reads
`data/interim/btsbot_{split}.parquet` (already produced by
`src.readers.btsbot`), decomposes each object's accumulated observation
history into one `AlertPacket` per row, and replays every object's alerts
interleaved in GLOBAL chronological order (not object-by-object) -- a real
stream interleaves unrelated objects' alerts by when they actually happened,
and `src.serve.inference_worker`'s micro-batching behavior depends on that
interleaving, not on receiving one object's full history as a contiguous
run.

Every row in Tier B is itself a real detection (CLAUDE.md §5.1 Day-8
finding: no forced-photometry row type exists in this table at all), so
every replayed packet's own `observations` is exactly one detection --
there is no bundled forced photometry to simulate here, unlike a real
ZTF/LSST alert packet.

Entry point: `python -m src.serve.replay [--split test] [--speed-days-per-second X]`
(omit `--speed-days-per-second`, or pass 0, for unthrottled replay -- the
mode used for the throughput/latency benchmark.)
"""

import argparse
import logging
import time

from confluent_kafka import Producer

from src.readers import bts_labels
from src.serve.schemas import AlertPacket
from src.train.dataset import load_alerts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "alerts"

# CLAUDE.md §5.1's Day-8 Tier B finding -- forced photometry never enters
# this table, so replay only ever produces single-observation packets.
BTSBOT_PARQUET = "data/interim/btsbot_{split}.parquet"

# A cap on the real sleep between two consecutive replayed alerts. Real
# cadence gaps range from sub-day to many months (CLAUDE.md §11 Day 10: a
# periodic source's gaps can span a whole survey); an uncapped
# days-per-second throttle would make the replay simply hang across a large
# gap. This keeps a throttled replay watchable without ever exceeding the
# configured speed.
MAX_SLEEP_SECONDS = 2.0


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("delivery failed for object_id key=%s: %s", msg.key(), err)


def replay(split: str = "test", speed_days_per_second: float = 0.0) -> int:
    """Returns the number of alerts produced."""
    parquet_path = BTSBOT_PARQUET.format(split=split)
    logger.info("Loading Tier B %s split from %s", split, parquet_path)
    alerts = load_alerts(parquet_path)
    logger.info("Loaded %d objects", len(alerts))

    # `src.readers.btsbot` always emits class_id=None (CLAUDE.md: labels come
    # from a separate BTS Sample Explorer join, same one Day 9's Gate C
    # fine-tune used) -- attach real ground truth here so replay-mode's
    # `true_class_id` is meaningful, not silently null for every object.
    bts_sample = bts_labels.fetch_bts_sample()
    alerts, join_stats = bts_labels.apply_classifications_to_alerts(alerts, bts_sample)
    logger.info("BTS label join outcomes: %s", join_stats)

    events: list[tuple[float, AlertPacket]] = []
    for alert in alerts:
        for obs in alert.observations:
            packet = AlertPacket.from_alert_slice(
                object_id=alert.object_id,
                survey=alert.survey,
                observations=[obs],
                static=alert.static,
                class_id=alert.class_id,
            )
            events.append((obs.mjd, packet))
    events.sort(key=lambda e: e[0])
    logger.info("Replaying %d alert(s) in global MJD order", len(events))

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    n_produced = 0
    prev_mjd: float | None = None
    t_start = time.monotonic()
    for mjd, packet in events:
        if speed_days_per_second > 0 and prev_mjd is not None:
            gap_days = mjd - prev_mjd
            sleep_s = min(gap_days / speed_days_per_second, MAX_SLEEP_SECONDS)
            if sleep_s > 0:
                time.sleep(sleep_s)
        prev_mjd = mjd

        producer.produce(
            TOPIC, key=packet.object_id.encode("utf-8"), value=packet.to_json(),
            callback=_delivery_callback,
        )
        n_produced += 1
        if n_produced % 5000 == 0:
            producer.poll(0)
            logger.info("Produced %d/%d alerts...", n_produced, len(events))

    producer.flush()
    elapsed = time.monotonic() - t_start
    logger.info(
        "Done: produced %d alerts in %.1fs (%.1f alerts/sec produced)",
        n_produced, elapsed, n_produced / elapsed if elapsed > 0 else float("nan"),
    )
    return n_produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--speed-days-per-second", type=float, default=0.0,
        help="Simulated days replayed per real second. 0 (default) = unthrottled.",
    )
    args = parser.parse_args()
    replay(split=args.split, speed_days_per_second=args.speed_days_per_second)


if __name__ == "__main__":
    main()
