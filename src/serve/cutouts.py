"""Cutout storage + rendering for the results-viewer UI (CLAUDE.md Day 13,
Task 2: "Cutouts are present in Fink alerts. Store them for the UI.").

Deliberately a separate Redis key (`sn:cutouts:{object_id}`), not threaded
through `src.serve.state`'s rolling light-curve blob: `CanonicalAlert.cutouts`
(schema.py) is a single current-snapshot field, not per-observation, and
every existing serving code path (`AlertPacket`/`ObservationPacket`/
`state.merge_history`) was built for cutouts always being `None`
(ELAsTiCC2/BTSbot both never populate them). Keeping cutout storage
independent means the live-Fink cutout addition (this session, the first
reader that ever populates `CanonicalAlert.cutouts`) touches none of that
existing, tested state-merging code.

Stores the latest cutout triplet only (overwritten on each new alert for an
object) -- the UI's "science/reference/difference triplet" requirement
(Task 2) is about showing what an object currently looks like, not a cutout
history.
"""

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.visualization import ZScaleInterval
from PIL import Image

CUTOUT_KEY_PREFIX = "sn:cutouts:"
PLANE_NAMES = ("science", "template", "difference")

# Precomputed BTSbot Tier-B cutouts (src.serve.extract_btsbot_cutouts), one
# JSON triplet per object -- the images_v10.zip archive itself is ~18GB
# compressed and cannot be decoded per-request in a live server, so this is
# a one-time-extracted, disk-backed store, not Redis. Checked as a fallback
# after Redis (live-Fink objects) in `load_cutouts` below, never the reverse:
# a live object's own just-ingested cutout should win if somehow both exist.
PRECOMPUTED_BTSBOT_DIR = Path("data/interim/btsbot_cutouts")


def redis_key(object_id: str) -> str:
    return f"{CUTOUT_KEY_PREFIX}{object_id}"


def store_cutouts(redis_client, object_id: str, cutouts: np.ndarray) -> None:
    """`cutouts`: `(3, H, W)` float32, `[science, template, difference]` --
    the same convention `CanonicalAlert.cutouts` documents (schema.py)."""
    arr = np.asarray(cutouts, dtype=np.float32)
    payload = {
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    redis_client.set(redis_key(object_id), json.dumps(payload))


def load_cutouts(redis_client, object_id: str) -> Optional[np.ndarray]:
    raw = redis_client.get(redis_key(object_id))
    if raw is None:
        return None
    obj = json.loads(raw)
    flat = np.frombuffer(base64.b64decode(obj["data_b64"]), dtype=np.float32)
    return flat.reshape(obj["shape"])


def _zscale_plane_to_png_bytes(plane: np.ndarray) -> bytes:
    """Astronomical cutouts are mostly near-black with a few bright pixels --
    linear scaling renders as nothing (Day 13 task instruction). ZScaleInterval
    (the same algorithm DS9 and other FITS viewers default to) finds the
    display range a human would actually pick by hand."""
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(plane)
    if vmax <= vmin:  # a flat/constant plane -- degenerate but must not divide by zero
        vmax = vmin + 1.0
    scaled = np.clip((plane.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    img = Image.fromarray((scaled * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_precomputed_btsbot(object_id: str, split: str = "test") -> Optional[dict[str, str]]:
    """Already-rendered `{"science": ..., ...}` base64-PNG triplet for a
    Tier-B object, if `src.serve.extract_btsbot_cutouts` was run for this
    `split` and this object was one of the ~1,999 kept (the latest-`jd` row
    per object -- see that module's docstring)."""
    path = PRECOMPUTED_BTSBOT_DIR / split / f"{object_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["images"]


def load_cutout_images(redis_client, object_id: str) -> Optional[dict[str, str]]:
    """The one entry point the API needs: a live-Fink cutout (Redis, raw
    array -> rendered here) takes priority over a precomputed Tier-B one
    (already rendered) -- a live object's own just-ingested cutout should
    never be shadowed by a stale precomputed file that happens to share its
    object_id."""
    arr = load_cutouts(redis_client, object_id)
    if arr is not None:
        return cutouts_to_png_triplet(arr)
    return load_precomputed_btsbot(object_id)


def cutouts_to_png_triplet(cutouts: np.ndarray) -> dict[str, str]:
    """`(3, H, W)` array -> `{"science": <base64 png>, "template": ..., "difference": ...}`,
    each independently ZScaled (a difference image's natural range differs
    from the science/template frames, so scaling them together would wash
    one out)."""
    return {
        name: base64.b64encode(_zscale_plane_to_png_bytes(cutouts[i])).decode("ascii")
        for i, name in enumerate(PLANE_NAMES)
    }
