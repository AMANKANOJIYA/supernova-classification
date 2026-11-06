"""Run the fixed LSST reader (`src.readers.avro_lsst`) over every live Fink LSST packet
archived on 2026-11-05 (data/raw/fink_lsst_archive/, raw packets saved by the audit's
listener before any reader ran) and persist the outcome.

Checks, per packet: conversion succeeds; the triggering diaSource's AB magnitude is
unchanged by `convert_zeropoint` (nJy at zp 31.4 -> FLUXCAL at zp 27.5); and the raw
cutout stamp size, which the frozen schema's (3, 63, 63) cutout shape cannot hold.

Run from the repo root: `.venv/bin/python reports/live_2026-11-06/scripts/lsst_archive_reader_check.py`
"""

import collections
import gzip
import io
import json
import logging
import math
import pickle
from pathlib import Path

from astropy.io import fits

from src.readers import avro_lsst

ARCHIVE = Path("data/raw/fink_lsst_archive")
OUT = Path("reports/live_2026-11-06/lsst_archive_reader_check.json")


def main() -> None:
    logging.getLogger("src.readers.avro_lsst").setLevel(logging.ERROR)
    files = sorted(ARCHIVE.glob("*.pkl.gz"))
    errors: collections.Counter = collections.Counter()
    topics: collections.Counter = collections.Counter()
    stamp_sizes: collections.Counter = collections.Counter()
    n_converted = n_detections = n_negative = 0
    max_dmag = 0.0
    for path in files:
        with gzip.open(path) as f:
            packet = pickle.load(f)
        alert = packet["alert"]
        topics[packet["topic"]] += 1
        stamp = alert.get("cutoutDifference")
        if stamp:
            with fits.open(io.BytesIO(stamp)) as hdul:
                stamp_sizes[hdul[0].data.shape[0]] += 1
        try:
            canonical = avro_lsst.lsst_alert_to_canonical(alert)
        except Exception as exc:  # noqa: BLE001 -- counted and reported, not hidden
            errors[f"{type(exc).__name__}: {exc}"] += 1
            continue
        n_converted += 1
        src = alert["diaSource"]
        trig = [o for o in canonical.observations if o.is_detection and abs(o.mjd - src["midpointMjdTai"]) < 1e-9]
        if src["psfFlux"] > 0 and trig:
            dmag = abs((31.4 - 2.5 * math.log10(src["psfFlux"])) - (27.5 - 2.5 * math.log10(trig[0].flux)))
            max_dmag = max(max_dmag, dmag)
        for o in canonical.observations:
            if o.is_detection:
                n_detections += 1
                n_negative += o.flux < 0
    report = {
        "archive": str(ARCHIVE),
        "n_packets": len(files),
        "n_converted": n_converted,
        "errors": dict(errors),
        "topics": dict(topics),
        "max_abs_dmag_trigger": max_dmag,
        "negative_flux_fraction_detections": n_negative / n_detections if n_detections else None,
        "stamp_size_px_min": min(stamp_sizes),
        "stamp_size_px_max": max(stamp_sizes),
        "n_stamps_63px": stamp_sizes.get(63, 0),
        "n_stamps": sum(stamp_sizes.values()),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
