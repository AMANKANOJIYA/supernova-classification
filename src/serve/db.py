"""Postgres results store (CLAUDE.md §6 step 12, "results -- Postgres").
Plain `psycopg` + raw SQL -- one table, no ORM, since the access pattern is
"insert one row per scored alert, query by object_id or by anomaly score,"
nothing an ORM would make meaningfully safer or shorter here.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg

DEFAULT_DSN = (
    f"host=localhost port=5432 "
    f"dbname={os.environ.get('POSTGRES_DB', 'sn_classifier')} "
    f"user={os.environ.get('POSTGRES_USER', 'sn')} "
    f"password={os.environ.get('POSTGRES_PASSWORD', 'sn_dev_only_password')}"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    object_id TEXT NOT NULL,
    survey TEXT NOT NULL,
    alert_mjd DOUBLE PRECISION NOT NULL,
    scored_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_class TEXT NOT NULL,
    prob_ia DOUBLE PRECISION NOT NULL,
    prob_ibc DOUBLE PRECISION NOT NULL,
    prob_ii DOUBLE PRECISION NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    true_class_id INTEGER,           -- replay-mode ground truth only; NULL on a live stream
    seq_length INTEGER NOT NULL,      -- number of timesteps actually fed to the model (post-windowing)
    inference_latency_ms DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_object_id ON predictions (object_id);
CREATE INDEX IF NOT EXISTS idx_predictions_anomaly_score ON predictions (anomaly_score DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_scored_at ON predictions (scored_at_utc);
"""


@contextmanager
def connect(dsn: str = DEFAULT_DSN) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)


def insert_prediction(
    conn: psycopg.Connection,
    object_id: str,
    survey: str,
    alert_mjd: float,
    predicted_class: str,
    probs: tuple[float, float, float],
    anomaly_score: float,
    true_class_id: Optional[int],
    seq_length: int,
    inference_latency_ms: float,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (object_id, survey, alert_mjd, predicted_class,
                 prob_ia, prob_ibc, prob_ii, anomaly_score,
                 true_class_id, seq_length, inference_latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                object_id, survey, alert_mjd, predicted_class,
                probs[0], probs[1], probs[2], anomaly_score,
                true_class_id, seq_length, inference_latency_ms,
            ),
        )


def latest_prediction_for_object(conn: psycopg.Connection, object_id: str) -> Optional[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            "SELECT * FROM predictions WHERE object_id = %s ORDER BY alert_mjd DESC LIMIT 1",
            (object_id,),
        )
        return cur.fetchone()


def top_anomalies(conn: psycopg.Connection, limit: int = 50) -> list[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (object_id) *
            FROM predictions
            ORDER BY object_id, alert_mjd DESC
            """
        )
        latest_per_object = cur.fetchall()
    return sorted(latest_per_object, key=lambda r: r["anomaly_score"], reverse=True)[:limit]


def stats(conn: psycopg.Connection) -> dict:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT count(*) AS n_predictions, count(DISTINCT object_id) AS n_objects FROM predictions")
        counts = cur.fetchone()
        cur.execute(
            """
            SELECT predicted_class, count(*) AS n
            FROM (SELECT DISTINCT ON (object_id) object_id, predicted_class FROM predictions
                  ORDER BY object_id, alert_mjd DESC) latest
            GROUP BY predicted_class
            """
        )
        class_distribution = {row["predicted_class"]: row["n"] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT
                percentile_cont(0.5) WITHIN GROUP (ORDER BY inference_latency_ms) AS p50_ms,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY inference_latency_ms) AS p99_ms,
                avg(inference_latency_ms) AS mean_ms,
                max(inference_latency_ms) AS max_ms
            FROM predictions
            """
        )
        latency = cur.fetchone()
    return {
        "n_predictions": counts["n_predictions"],
        "n_objects": counts["n_objects"],
        "class_distribution": class_distribution,
        "inference_latency_ms": latency,
    }
