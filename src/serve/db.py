"""Postgres results store (CLAUDE.md §6 step 12, "results -- Postgres").
Plain `psycopg` + raw SQL -- one table, no ORM, since the access pattern is
"insert one row per scored alert, query by object_id or by anomaly score,"
nothing an ORM would make meaningfully safer or shorter here.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg

# Taxonomy ids for the 3 target leaves (src.taxonomy.TARGET_CLASS_IDS),
# duplicated here as a tiny literal rather than imported: this module has no
# other reason to depend on src.taxonomy, and the mapping is frozen by
# CLAUDE.md §4 (Ib/Ic merged, verbatim ELAsTiCC2 ids) so it will not drift.
TRUE_CLASS_NAME = {2222: "Ia", 2223: "Ib/c", 2224: "II"}

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


def list_objects(
    conn: psycopg.Connection,
    survey: Optional[str] = None,
    predicted_class: Optional[str] = None,
    true_class: Optional[str] = None,
    label_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Latest-per-object rows, for the UI's browsable object list (CLAUDE.md
    Day-13-follow-up task: "make BTSbot objects browsable"). `label_filter`:
    'labeled' (true_class_id is not null), 'unlabeled', 'correct'/'incorrect'
    (a 3-class true label -- one of `TRUE_CLASS_NAME` -- whose name does/
    doesn't match `predicted_class`; an object with a true label outside the
    3 target classes, e.g. AGN, is neither correct nor incorrect and is
    excluded from both). `true_class`: one of `TRUE_CLASS_NAME`'s values
    ('Ia' / 'Ib/c' / 'II') -- filters on the derived name, not a stored
    column, so it's applied after `true_class_name` is computed below, same
    as `label_filter`'s correct/incorrect branch. Every row gets
    `true_class_name` and `correct` (bool or None) attached, computed here
    so the API layer never re-derives the taxonomy-id mapping itself.
    correct/incorrect/true_class filtering happens BEFORE pagination, so
    `limit`/`offset` page over the filtered set, not the unfiltered one."""
    where = ["1=1"]
    params: list = []
    if survey:
        where.append("survey = %s")
        params.append(survey)
    if predicted_class:
        where.append("predicted_class = %s")
        params.append(predicted_class)
    if label_filter == "labeled":
        where.append("true_class_id IS NOT NULL")
    elif label_filter == "unlabeled":
        where.append("true_class_id IS NULL")
    if search:
        where.append("object_id ILIKE %s")
        params.append(f"%{search}%")
    where_sql = " AND ".join(where)

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT DISTINCT ON (object_id) *
                FROM predictions
                ORDER BY object_id, alert_mjd DESC
            ) latest
            WHERE {where_sql}
            ORDER BY alert_mjd DESC
            """,
            params,
        )
        all_rows = cur.fetchall()

    for row in all_rows:
        row["true_class_name"] = TRUE_CLASS_NAME.get(row["true_class_id"])
        row["correct"] = (
            (row["true_class_name"] == row["predicted_class"]) if row["true_class_name"] else None
        )

    if label_filter in ("correct", "incorrect"):
        want = label_filter == "correct"
        all_rows = [r for r in all_rows if r["correct"] is want]

    if true_class:
        all_rows = [r for r in all_rows if r["true_class_name"] == true_class]

    total = len(all_rows)
    return all_rows[offset : offset + limit], total


def recent_alerts(
    conn: psycopg.Connection,
    limit: int = 200,
    survey: Optional[str] = None,
    predicted_class: Optional[str] = None,
) -> list[dict]:
    """Every scored alert (not deduped by object), newest first -- the
    Live tab's own feed, distinct from `top_anomalies`/`list_objects`'s
    latest-per-object view."""
    where = ["1=1"]
    params: list = []
    if survey:
        where.append("survey = %s")
        params.append(survey)
    if predicted_class:
        where.append("predicted_class = %s")
        params.append(predicted_class)
    where_sql = " AND ".join(where)
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            f"SELECT * FROM predictions WHERE {where_sql} ORDER BY id DESC LIMIT %s",
            (*params, limit),
        )
        rows = cur.fetchall()
    for row in rows:
        row["true_class_name"] = TRUE_CLASS_NAME.get(row["true_class_id"])
        row["correct"] = (
            (row["true_class_name"] == row["predicted_class"]) if row["true_class_name"] else None
        )
    return rows


def alerts_per_minute(conn: psycopg.Connection, minutes: int = 30) -> list[dict]:
    """One row per minute bucket (ascending) over the trailing window, for
    the Live tab's alerts-per-minute sparkline. Buckets with zero alerts are
    simply absent -- the caller fills gaps if it wants a continuous axis."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT date_trunc('minute', scored_at_utc) AS minute, count(*) AS n
            FROM predictions
            WHERE scored_at_utc > now() - (%s || ' minutes')::interval
            GROUP BY minute
            ORDER BY minute ASC
            """,
            (minutes,),
        )
        return cur.fetchall()


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
