"""Stateless alert normalization (CLAUDE.md §6 step 12). Turns a raw Kafka
message (`src.serve.schemas.AlertPacket`, JSON-encoded) into validated
`src.schema.Observation` rows, ready to be folded into an object's rolling
history (`src.serve.state`) -- but does not touch Redis, Postgres, or any
other I/O itself, so this module is exactly as easy to unit-test as
`src.features.tensorize` is, and shares nothing with the replay-vs-live
distinction (CLAUDE.md §6: "same consumer code" for both).
"""

from dataclasses import dataclass
from typing import Optional

from src.schema import Observation, StaticContext, Survey
from src.serve.schemas import AlertPacket


@dataclass(frozen=True)
class NormalizedAlert:
    """This alert's own new rows, validated and mjd-sorted -- not yet merged
    with any prior history for this object (that is `src.serve.state`'s
    job)."""

    object_id: str
    survey: Survey
    new_observations: list[Observation]
    static: StaticContext
    class_id: Optional[int]


def parse_alert_packet(raw: bytes) -> NormalizedAlert:
    """Decode + validate one Kafka message.

    Raises:
        ValueError: empty observations, or any `flux_err <= 0` -- the same
            invariant `CanonicalAlert.validate()` enforces on a full alert,
            checked here per-row since a single packet is never itself a
            full `CanonicalAlert` (see module docstring).
    """
    packet = AlertPacket.from_json(raw)

    if not packet.observations:
        raise ValueError(f"object_id={packet.object_id!r}: empty observations in alert packet")

    observations = [o.to_observation() for o in packet.observations]
    for obs in observations:
        if obs.flux_err <= 0:
            raise ValueError(
                f"object_id={packet.object_id!r}: flux_err={obs.flux_err} <= 0 at mjd={obs.mjd}"
            )
    observations.sort(key=lambda o: o.mjd)

    return NormalizedAlert(
        object_id=packet.object_id,
        survey=Survey(packet.survey),
        new_observations=observations,
        static=packet.static.to_static_context(),
        class_id=packet.class_id,
    )


def has_new_detection(normalized: NormalizedAlert) -> bool:
    """CLAUDE.md §5.1: "the serving trigger is detection-driven, not
    photometry-driven" -- forced photometry bundled with a packet must never
    itself cause a re-score. `src.serve.inference_worker` gates scoring on
    this; `src.serve.state.merge_history` still folds every row (detection
    or not) into the stored history regardless, since forced photometry
    genuinely widens what a later window can see."""
    return any(obs.is_detection for obs in normalized.new_observations)
