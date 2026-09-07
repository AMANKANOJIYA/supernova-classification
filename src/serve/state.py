"""Redis-backed rolling per-object light-curve history (CLAUDE.md §6 step
12, "state" bullet). One JSON blob per object, keyed `sn:lc:{object_id}`.

Stores an object's FULL accumulated observation history, not a fixed
30-day-trailing window -- despite §8's stack-table shorthand ("Redis
(rolling 30-day object history)"), a literal trailing-30-day trim at the
storage layer would be wrong, not just an optimization left undone:
`tensorize_alert`'s window is anchored to `first_detection_mjd`
(`[first_detection_mjd - pre_detection_window_days,
min(last_detection_mjd, first_detection_mjd + max_span_days)]` — see
`src.features.tensorize` module docstring), which can be discovered
arbitrarily far in the past relative to "now." Trimming stored rows to a
sliding recent window would make `first_detection_mjd` itself
unrecoverable after 30 days, silently corrupting every Δt computed from
that point on. The actual bound on what's fed to the model is
`max_span_days`, applied at TENSORIZATION time (by the caller, e.g.
`src.serve.inference_worker`), not at storage time.

The one property this module must get right is non-monotonic merging
(§5.1): a later alert's bundled forced photometry can have an earlier mjd
than points already stored from an earlier alert (forced photometry
between two detections enters the window only once the LATER detection's
packet arrives, not when the survey originally took that exposure). Every
merge re-sorts by mjd rather than appending, so storage order never depends
on arrival order -- `tests/test_serve_state.py` covers exactly this case.
"""

import json
from dataclasses import asdict
from typing import Optional

from src.schema import CanonicalAlert, Observation, StaticContext, Survey
from src.serve.ingest import NormalizedAlert

STATE_KEY_PREFIX = "sn:lc:"


def redis_key(object_id: str) -> str:
    return f"{STATE_KEY_PREFIX}{object_id}"


def _obs_dedup_key(obs: dict) -> tuple:
    """Identifies "the same physical measurement" for dedup purposes -- a
    real stream can resend the same forced-photometry point across
    multiple alert packets (CLAUDE.md §5.1: each detection's packet
    reaches back up to 30 days, so consecutive detections' packets
    routinely overlap)."""
    return (obs["mjd"], obs["lam_eff"], obs["flux"], obs["flux_err"], obs["is_detection"])


def merge_history(existing: Optional[dict], normalized: NormalizedAlert) -> dict:
    """Pure merge step (no Redis I/O -- see module docstring on why this is
    split out): fold `normalized`'s new rows into `existing`'s stored state
    (`None` for a never-before-seen object).
    """
    if existing is None:
        merged_by_key: dict = {}
        prior_class_id = None
    else:
        merged_by_key = {_obs_dedup_key(o): o for o in existing["observations"]}
        prior_class_id = existing.get("class_id")

    for obs in normalized.new_observations:
        row = asdict(obs)
        merged_by_key[_obs_dedup_key(row)] = row

    observations_sorted = sorted(merged_by_key.values(), key=lambda o: o["mjd"])

    return {
        "object_id": normalized.object_id,
        "survey": normalized.survey.value,
        "observations": observations_sorted,
        "static": asdict(normalized.static),
        # A live stream's static context (host features, MW E(B-V)) does not
        # change alert-to-alert; keep the first-seen value rather than let a
        # later, possibly-incomplete packet silently overwrite it with worse
        # information. class_id likewise: replay-mode ground truth is
        # constant per object, but only ever trust it once set.
        "class_id": prior_class_id if prior_class_id is not None else normalized.class_id,
    }


def state_to_canonical_alert(state: dict) -> CanonicalAlert:
    observations = [Observation(**o) for o in state["observations"]]
    static = StaticContext(**state["static"])
    alert = CanonicalAlert(
        object_id=state["object_id"],
        survey=Survey(state["survey"]),
        observations=observations,
        static=static,
        cutouts=None,
        class_id=state["class_id"],
    )
    alert.validate()
    return alert


def get_and_merge(redis_client, normalized: NormalizedAlert) -> CanonicalAlert:
    """The one I/O-touching entry point: GET the object's current state,
    merge in `normalized`'s new rows, SET it back, and return the object's
    full history as a `CanonicalAlert` ready for tensorization."""
    key = redis_key(normalized.object_id)
    raw = redis_client.get(key)
    existing = json.loads(raw) if raw is not None else None
    merged = merge_history(existing, normalized)
    redis_client.set(key, json.dumps(merged))
    return state_to_canonical_alert(merged)


def get_alert(redis_client, object_id: str) -> Optional[CanonicalAlert]:
    """Read-only lookup for the API's `/object/{id}` endpoint -- no merge,
    no write."""
    raw = redis_client.get(redis_key(object_id))
    if raw is None:
        return None
    return state_to_canonical_alert(json.loads(raw))
