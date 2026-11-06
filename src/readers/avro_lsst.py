"""Fink LSST live-alert reader -> CanonicalAlert (CLAUDE.md §6 step 13, Day 13).

-----------------------------------------------------------------------------
UNVERIFIED AGAINST LIVE DATA -- read this before trusting a field name below
-----------------------------------------------------------------------------
Every field name in this module is the standard Rubin/LSST DIA alert-packet
convention (the DPDD `diaSource`/`diaObject`/`prvDiaSources`/
`prvDiaForcedSources` schema), NOT something checked against a real Fink LSST
packet the way `src/readers/avro_ztf.py` was checked against a live
`fink_sn_candidates_ztf` alert this same session.

Checked directly, this session (2026-09-07): every Fink LSST livestream topic
(`fink_most_likely_sn_lsst`, `fink_uniform_sample_lsst`,
`fink_early_snia_candidate_lsst`, `fink_sn_near_galaxy_candidate_lsst`,
`fink_extragalactic_new_candidate_lsst`, `fink_extragalactic_lt20mag_candidate_lsst`,
`fink_extragalactic_svom_lsst`, `fink_faint_trails_lsst`,
`fink_hostless_candidate_lsst`, `fink_in_tns_lsst`,
`fink_remove_unlikely_transients_lsst`) had zero messages -- confirmed via
Kafka watermark offsets (not just a poll timeout), including an 8-sample
watch over 100 continuous seconds showing zero movement. This is the LSST
broker's actual current state, not a connectivity problem: the ZTF broker
(kafka-ztf.fink-broker.org, this session's `avro_ztf.py`) returned a real
alert within seconds on the equivalent request. No live LSST packet of any
kind could be obtained to verify field names against, so none of the
field-name choices below carry that verification.

This reader therefore fails LOUDLY and by name, never silently, on any
missing expected key (see `_require`) -- a real packet with different field
names should surface as a clear `KeyError`-style message naming exactly what
was expected and absent, not a wrong value silently computed from `None` or
a misread field. Treat the first `ValueError` this raises against real
traffic as a signal this module needs a golden-fixture rewrite against that
real packet (CLAUDE.md §12), exactly the same way `avro_ztf.py` was built
this session -- not a bug to patch around inline.

-----------------------------------------------------------------------------
Assumed schema (DPDD convention)
-----------------------------------------------------------------------------
Top level: `diaSource` (the triggering detection), `diaObject` (host-level
aggregate, may be `None` for a brand-new object), `prvDiaSources` (prior
detections), `prvDiaForcedSources` (forced photometry -- LSST's forced-
photometry service means non-detections here carry an actual measured flux,
unlike ZTF's limiting-magnitude-only upper limits; see avro_ztf.py's
docstring for that contrast).

Per-source fields assumed: `diaSourceId`, `ra`, `dec` (the live Fink LSST
schema; older DPDD drafts used `decl` -- verified against a real archived
packet 2026-11-06), `band` (single character, one of ugrizy -- NOT a
numeric `fid` the way ZTF encodes it), `midpointMjdTai` (already MJD, NOT JD
-- no `-2400000.5` conversion needed, unlike ZTF/BTSbot), `psfFlux`,
`psfFluxErr` (already **calibrated flux**, nanojansky, AB zeropoint 31.4 --
CLAUDE.md §3's own worked example for exactly this convention; NOT a
magnitude+zeropoint pair the way ZTF alerts are, so no `mag_to_flux` step is
needed here, only `convert_zeropoint(..., from_zeropoint=31.4)`).

`diaObject` fields assumed for `StaticContext`: no native photo-z field is
assumed present (Rubin's own DIA object catalog does not carry a host
redshift; a future Fink cross-match layer analogous to ZTF's `mangrove`
might add one, but none is assumed here) -- `host_photoz`/`host_photoz_err`
always `None`, matching both existing readers' treatment of surveys with no
native redshift field. `host_offset_arcsec`/`host_mag_r`/`host_ellipticity`/
`real_bogus_score`: no DPDD-standard field name is confidently known for
these without a live packet to check against, so all default to `None`
rather than guessing a specific key name and silently mismapping it.

Cutouts: assumed named `cutoutScience`/`cutoutTemplate`/`cutoutDifference`
with a `stampData` sub-field, mirroring ZTF's Fink packaging (both surveys'
alerts are relayed through the same Fink cutout-attachment convention per
Fink's own architecture) -- but the stamp SIZE is not assumed to be 63x63
(that is ZTF-specific, confirmed empirically on the ZTF golden alert);
`CanonicalAlert.validate()` (schema.py, FROZEN) hard-enforces `(3, 63, 63)`
whenever cutouts are not `None`, so if Rubin's actual stamp size differs,
`decode_cutouts` here deliberately drops to `cutouts=None` (logged) rather
than let `validate()` reject the whole alert over a cutout-shape mismatch --
classification does not depend on cutouts existing.
"""

import gzip
import io
import logging
import math
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from src.readers.zeropoint import convert_zeropoint
from src.schema import CanonicalAlert, Observation, StaticContext, Survey

logger = logging.getLogger(__name__)

# Rubin/LSST alert-packet flux convention (DPDD): calibrated flux in
# nanojansky, AB zeropoint 31.4 -- CLAUDE.md §3's own named example for a
# real (non-SNANA) LSST reader. NOT verified against a live packet this
# session (see module docstring) -- if real LSST flux turns out to be on a
# different zeropoint, this constant is the one place to correct it.
LSST_FLUX_ZEROPOINT = 31.4

_VALID_BANDS = {"u", "g", "r", "i", "z", "y"}

_dust_query = None
_DUSTMAPS_DATA_DIR = "data/raw/dustmaps"


class SkippedRow(Exception):
    """A single prvDiaSources/prvDiaForcedSources row could not be
    converted -- caught and counted by the caller (CLAUDE.md §2.5), never
    allowed to fail the whole alert."""


def _require(d: dict, key: str, context: str) -> object:
    if key not in d or d[key] is None:
        raise SkippedRow(f"{context}: missing required field {key!r}")
    return d[key]


def _band_to_lam_eff(band: str) -> float:
    from src.schema import LAM_EFF

    if band not in _VALID_BANDS:
        raise SkippedRow(f"unrecognized LSST band {band!r}; expected one of {sorted(_VALID_BANDS)}")
    return LAM_EFF[band]  # schema.py's LAM_EFF keys "u","g","r","i","z","y" directly for LSST


def _source_to_observation(row: dict, is_detection: bool) -> Observation:
    mjd = _require(row, "midpointMjdTai", "diaSource row")
    band = _require(row, "band", "diaSource row")
    flux_raw = _require(row, "psfFlux", "diaSource row")
    flux_err_raw = _require(row, "psfFluxErr", "diaSource row")
    if not flux_err_raw > 0:
        raise SkippedRow(f"psfFluxErr={flux_err_raw!r} not > 0")
    flux, flux_err = convert_zeropoint(
        float(flux_raw), float(flux_err_raw), from_zeropoint=LSST_FLUX_ZEROPOINT
    )
    return Observation(
        mjd=float(mjd),
        lam_eff=_band_to_lam_eff(band),
        flux=flux,
        flux_err=flux_err,
        is_detection=is_detection,
    )


def _mw_ebv_and_gal_lat(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Same lazy-singleton SFD-map pattern as avro_ztf.py / btsbot.py."""
    global _dust_query
    if _dust_query is None:
        from dustmaps.config import config
        from dustmaps.sfd import SFDQuery

        config["data_dir"] = _DUSTMAPS_DATA_DIR
        _dust_query = SFDQuery()
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return float(_dust_query(coord)), float(coord.galactic.b.deg)


def decode_cutout_stamp(stamp_data: bytes) -> np.ndarray:
    """Identical decode path to avro_ztf.py's own -- see that module for a
    tested synthetic round-trip; not re-tested here since the decode logic
    itself is survey-agnostic (gzip + single-HDU FITS), only the assumed
    dict key names and stamp size differ."""
    with gzip.open(io.BytesIO(stamp_data)) as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)


def _decode_cutouts(alert: dict) -> Optional[np.ndarray]:
    try:
        sci = decode_cutout_stamp(alert["cutoutScience"]["stampData"])
        ref = decode_cutout_stamp(alert["cutoutTemplate"]["stampData"])
        diff = decode_cutout_stamp(alert["cutoutDifference"]["stampData"])
    except Exception as exc:  # noqa: BLE001 -- see module docstring: degrade, never propagate
        logger.warning(
            "avro_lsst: object_id=%s cutout decode failed (%s: %s), storing cutouts=None",
            alert.get("diaObjectId") or alert.get("objectId"), type(exc).__name__, exc,
        )
        return None
    stacked = np.stack([sci, ref, diff], axis=0)
    if stacked.shape != (3, 63, 63):
        logger.warning(
            "avro_lsst: cutout stamp shape %s != the (63,63)-per-plane shape "
            "CanonicalAlert.validate() requires -- storing cutouts=None rather "
            "than let a cutout-shape mismatch reject an otherwise-good alert "
            "(see module docstring: LSST stamp size was never confirmed live)",
            sci.shape,
        )
        return None
    return stacked


def lsst_alert_to_canonical(alert: dict) -> CanonicalAlert:
    """Convert one decoded Fink LSST alert `dict` into a `CanonicalAlert`.

    Raises:
        ValueError: if the triggering `diaSource` itself cannot be converted,
            or if `diaObjectId`/`ra`/`dec` (needed for object_id and static
            context) are absent -- CLAUDE.md §5.1: every real alert
            originates from a detection, this is not a normal edge case.
    """
    dia_source = alert.get("diaSource")
    if not dia_source:
        raise ValueError("alert has no diaSource -- not a source-triggered LSST alert")

    object_id = dia_source.get("diaObjectId") or dia_source.get("diaSourceId")
    if object_id is None:
        raise ValueError("diaSource has neither diaObjectId nor diaSourceId")
    object_id = str(object_id)

    try:
        observations = [_source_to_observation(dia_source, is_detection=True)]
    except SkippedRow as exc:
        raise ValueError(f"object_id={object_id!r}: triggering diaSource unusable ({exc})") from exc

    n_skipped = 0
    for row in alert.get("prvDiaSources") or []:
        try:
            observations.append(_source_to_observation(row, is_detection=True))
        except SkippedRow as exc:
            n_skipped += 1
            logger.debug("avro_lsst: object_id=%s skipped prvDiaSources row (%s)", object_id, exc)
    for row in alert.get("prvDiaForcedSources") or []:
        try:
            observations.append(_source_to_observation(row, is_detection=False))
        except SkippedRow as exc:
            n_skipped += 1
            logger.debug("avro_lsst: object_id=%s skipped prvDiaForcedSources row (%s)", object_id, exc)
    if n_skipped:
        logger.warning("avro_lsst: object_id=%s skipped %d prior-source row(s)", object_id, n_skipped)

    observations.sort(key=lambda o: o.mjd)

    ra = dia_source.get("ra")
    dec = dia_source.get("dec")
    if ra is None or dec is None:
        raise ValueError(f"object_id={object_id!r}: diaSource missing ra/dec")
    mw_ebv, gal_lat = _mw_ebv_and_gal_lat(float(ra), float(dec))

    # diaObject-level fields: no DPDD-standard host offset/mag/ellipticity/
    # real-bogus field name is confidently known without a live packet to
    # check (see module docstring) -- always None rather than guess.
    static = StaticContext(
        host_photoz=None,
        host_photoz_err=None,
        host_offset_arcsec=None,
        host_ellipticity=None,
        host_mag_r=None,
        mw_ebv=mw_ebv,
        gal_lat=gal_lat,
        real_bogus_score=None,
    )

    canonical = CanonicalAlert(
        object_id=object_id,
        survey=Survey.LSST,
        observations=observations,
        static=static,
        cutouts=_decode_cutouts(alert),
        class_id=None,
    )
    canonical.validate()
    return canonical
