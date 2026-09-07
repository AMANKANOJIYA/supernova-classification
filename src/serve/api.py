"""FastAPI serving layer (CLAUDE.md §6 step 12). Read-side endpoints hit
Postgres (durable results) and Redis (rolling per-object light-curve
state); `/classify` additionally runs the same ingest -> state -> model
path `src.serve.inference_worker` uses, for synchronous on-demand scoring
without needing Kafka running.

Entry point: `uvicorn src.serve.api:app --host 0.0.0.0 --port 8000`
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.tensorize import NUM_FEATURES, RECOMMENDED_MAX_SPAN_DAYS, tensorize_alert
from src.serve import db, ingest, state
from src.serve.inference_worker import PRE_DETECTION_WINDOW_DAYS, REDIS_HOST, REDIS_PORT
from src.serve.model import CLASS_NAMES, AnomalyScorer, ONNXClassifier
from src.serve.schemas import AlertPacket, ObservationPacket, StaticContextPacket

logger = logging.getLogger(__name__)

_resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _resources["redis"] = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    _resources["classifier"] = ONNXClassifier()
    _resources["anomaly_scorer"] = AnomalyScorer()
    with db.connect() as conn:
        db.init_schema(conn)
    yield
    _resources.clear()


app = FastAPI(title="Transient Supernova Classifier — serving API", lifespan=lifespan)


class ObservationIn(BaseModel):
    mjd: float
    lam_eff: float
    flux: float
    flux_err: float
    is_detection: bool


class StaticContextIn(BaseModel):
    host_photoz: Optional[float] = None
    host_photoz_err: Optional[float] = None
    host_offset_arcsec: Optional[float] = None
    host_ellipticity: Optional[float] = None
    host_mag_r: Optional[float] = None
    mw_ebv: Optional[float] = None
    gal_lat: Optional[float] = None
    real_bogus_score: Optional[float] = None


class AlertIn(BaseModel):
    object_id: str
    survey: str
    observations: list[ObservationIn]
    static: StaticContextIn
    class_id: Optional[int] = None


@app.get("/health")
def health() -> dict:
    redis_ok, postgres_ok = False, False
    try:
        redis_ok = bool(_resources["redis"].ping())
    except Exception:
        logger.exception("redis health check failed")
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                postgres_ok = cur.fetchone() is not None
    except Exception:
        logger.exception("postgres health check failed")
    status = "ok" if (redis_ok and postgres_ok) else "degraded"
    return {"status": status, "redis": redis_ok, "postgres": postgres_ok}


@app.post("/classify")
def classify(alert_in: AlertIn) -> dict:
    """Synchronous scoring: same ingest -> state -> tensorize -> model path
    `src.serve.inference_worker` runs per micro-batch, run here for exactly
    one alert, without needing Kafka. Detection-driven, same as the
    consumer (§5.1): forced-photometry-only packets update state but return
    no new prediction."""
    packet = AlertPacket(
        object_id=alert_in.object_id,
        survey=alert_in.survey,
        observations=[ObservationPacket(**o.model_dump()) for o in alert_in.observations],
        static=StaticContextPacket(**alert_in.static.model_dump()),
        class_id=alert_in.class_id,
    )
    try:
        normalized = ingest.parse_alert_packet(packet.to_json())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    canonical = state.get_and_merge(_resources["redis"], normalized)

    if not ingest.has_new_detection(normalized):
        return {"object_id": alert_in.object_id, "scored": False, "reason": "no new detection in this packet"}

    t0 = time.monotonic()
    tensorized = tensorize_alert(
        canonical,
        pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS,
        max_span_days=RECOMMENDED_MAX_SPAN_DAYS,
    )
    features = tensorized.features.unsqueeze(0).numpy().astype(np.float32)
    lengths = np.array([tensorized.features.shape[0]], dtype=np.int64)

    probs = _resources["classifier"].predict_probs(features, lengths)[0]

    features_t = torch.from_numpy(features)
    lengths_t = torch.from_numpy(lengths)
    mask_t = torch.ones((1, features.shape[1]), dtype=torch.bool)
    anomaly_score = float(_resources["anomaly_scorer"].score(features_t, lengths_t, mask_t)[0])
    latency_ms = (time.monotonic() - t0) * 1000.0

    predicted_class = CLASS_NAMES[int(np.argmax(probs))]
    trigger_mjd = max(o.mjd for o in normalized.new_observations if o.is_detection)

    with db.connect() as conn:
        db.insert_prediction(
            conn, object_id=alert_in.object_id, survey=normalized.survey.value,
            alert_mjd=trigger_mjd, predicted_class=predicted_class,
            probs=tuple(float(x) for x in probs), anomaly_score=anomaly_score,
            true_class_id=normalized.class_id, seq_length=int(lengths[0]),
            inference_latency_ms=latency_ms,
        )

    return {
        "object_id": alert_in.object_id,
        "scored": True,
        "predicted_class": predicted_class,
        "probs": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        "anomaly_score": anomaly_score,
        "seq_length": int(lengths[0]),
        "inference_latency_ms": latency_ms,
    }


@app.get("/object/{object_id}")
def get_object(object_id: str) -> dict:
    alert = state.get_alert(_resources["redis"], object_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"no stored history for object_id={object_id!r}")

    with db.connect() as conn:
        latest = db.latest_prediction_for_object(conn, object_id)

    return {
        "object_id": object_id,
        "survey": alert.survey.value,
        "n_observations": len(alert.observations),
        "observations": [
            {"mjd": o.mjd, "lam_eff": o.lam_eff, "flux": o.flux, "flux_err": o.flux_err, "is_detection": o.is_detection}
            for o in alert.observations
        ],
        "class_id": alert.class_id,
        "latest_prediction": latest,
    }


@app.get("/top-anomalies")
def get_top_anomalies(limit: int = 50) -> dict:
    with db.connect() as conn:
        rows = db.top_anomalies(conn, limit=limit)
    return {"n": len(rows), "results": rows}


@app.get("/stats")
def get_stats() -> dict:
    with db.connect() as conn:
        return db.stats(conn)
