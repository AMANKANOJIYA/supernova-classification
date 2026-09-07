"""Kafka message schema for the replay/live serving pipeline (CLAUDE.md §6
steps 12-13). One `AlertPacket` == one real alert: the serving trigger is
detection-driven (§5.1 -- "one alert per detection"), so every packet's
`observations` list is the set of rows a survey would bundle with THAT
detection (the triggering detection itself, plus any forced photometry the
survey reaches back for). For Tier B (BTSbot), that list is always exactly
one row: §5.1's Day-8 finding is that Tier B has no forced-photometry row
type at all, every one of its rows is itself a real detection.

Plain JSON, not Avro -- `fink-client`/`confluent-kafka` both work fine with a
JSON value schema, and this keeps replay.py/ingest.py free of a schema
registry dependency for what is currently a single internal topic.

`class_id` is carried only for replay-mode bookkeeping (the API's /stats
endpoint can report accuracy-adjacent diagnostics against archived ground
truth) -- it is never read by ingest.py's normalization or by the model.
A live Fink/Tier-C stream would not have it at all, so every consumer must
treat it as optional and inference-irrelevant, never assume its presence.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional

from src.schema import CanonicalAlert, Observation, StaticContext, Survey


@dataclass(frozen=True)
class ObservationPacket:
    mjd: float
    lam_eff: float
    flux: float
    flux_err: float
    is_detection: bool

    @classmethod
    def from_observation(cls, obs: Observation) -> "ObservationPacket":
        return cls(
            mjd=obs.mjd, lam_eff=obs.lam_eff, flux=obs.flux,
            flux_err=obs.flux_err, is_detection=obs.is_detection,
        )

    def to_observation(self) -> Observation:
        return Observation(
            mjd=self.mjd, lam_eff=self.lam_eff, flux=self.flux,
            flux_err=self.flux_err, is_detection=self.is_detection,
        )


@dataclass(frozen=True)
class StaticContextPacket:
    host_photoz: Optional[float]
    host_photoz_err: Optional[float]
    host_offset_arcsec: Optional[float]
    host_ellipticity: Optional[float]
    host_mag_r: Optional[float]
    mw_ebv: Optional[float]
    gal_lat: Optional[float]
    real_bogus_score: Optional[float]

    @classmethod
    def from_static_context(cls, static: StaticContext) -> "StaticContextPacket":
        return cls(**asdict(static))

    def to_static_context(self) -> StaticContext:
        return StaticContext(**asdict(self))


@dataclass(frozen=True)
class AlertPacket:
    """One alert on the wire. `observations` is deliberately just THIS
    alert's own rows -- accumulating an object's full history across
    multiple alerts is the rolling-state layer's job (`src.serve.state`),
    not something the producer bundles up front (a real survey doesn't
    resend an object's entire history in every packet either)."""

    object_id: str
    survey: str
    observations: list[ObservationPacket]
    static: StaticContextPacket
    class_id: Optional[int]  # replay-mode ground truth only; see module docstring

    @classmethod
    def from_alert_slice(
        cls, object_id: str, survey: Survey, observations: list[Observation],
        static: StaticContext, class_id: Optional[int],
    ) -> "AlertPacket":
        return cls(
            object_id=object_id,
            survey=survey.value,
            observations=[ObservationPacket.from_observation(o) for o in observations],
            static=StaticContextPacket.from_static_context(static),
            class_id=class_id,
        )

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "AlertPacket":
        d = json.loads(raw)
        return cls(
            object_id=d["object_id"],
            survey=d["survey"],
            observations=[ObservationPacket(**o) for o in d["observations"]],
            static=StaticContextPacket(**d["static"]),
            class_id=d["class_id"],
        )
