"""Fink ZTF live-alert reader -> CanonicalAlert (CLAUDE.md §6 step 13, Day 13).

Consumes one already-decoded Fink alert `dict` (`fink_client.consumer.
AlertConsumer.poll()`'s second return value — the client handles Avro
decoding using the schema embedded in the Kafka message key, so this module
never touches Avro/Kafka itself, same separation `src.serve.ingest` already
keeps from `src.serve.replay`'s Kafka producer).

-----------------------------------------------------------------------------
Verified against a live alert, not guessed — see tests/fixtures/golden_avro_ztf_alert.py
-----------------------------------------------------------------------------
Field names and the flux conversion below were checked against one real alert
polled live from Fink's `fink_sn_candidates_ztf` topic during this session
(2026-09-07), not inferred from documentation alone. This is the SAME raw ZTF
alert-packet schema `src.readers.btsbot` already established and tested
(`magpsf`/`sigmapsf`/`magzpsci`/`isdiffpos`) — see that module's docstring for
the derivation, not re-derived here — with one real difference: a live Fink
packet's `isdiffpos` is the literal string `"t"`/`"f"` (confirmed on the live
alert), not the `bool` btsbot.py's `polars.read_csv` already produced from the
same underlying CSV export.

    flux_raw     = sign * 10**(-0.4*(magpsf - magzpsci))   # sign: "t"->+1, else -1
    flux_err_raw = |flux_raw| * (ln(10)/2.5) * sigmapsf
    (flux, flux_err) = src.readers.zeropoint.convert_zeropoint(
        flux_raw, flux_err_raw, from_zeropoint=magzpsci)   # -> SNANA FLUXCAL zp=27.5

-----------------------------------------------------------------------------
prv_candidates -- detections AND non-detection upper limits, both real here
-----------------------------------------------------------------------------
Unlike BTSbot's metadata_v10 (every row a real detection, CLAUDE.md §5.1
Day-8 finding), a live ZTF alert's `prv_candidates` list genuinely contains
non-detection rows (`magpsf=None`, only a 5-sigma limiting magnitude
`diffmaglim` known) alongside real prior detections (`magpsf` populated) --
confirmed on the golden fixture alert: 19 of 25 `prv_candidates` rows are
non-detections. A non-detection is NOT the same kind of information as
ELAsTiCC2's forced photometry (CLAUDE.md §2.2: a real, possibly-negative
flux measured at every epoch) -- it is only a bound. Represented here as
`flux=0.0` (no measured excess, not a claim the true flux is exactly zero)
with `flux_err` set from the limiting depth read as a 5-sigma bound:

    limiting_flux_raw = 10**(-0.4*(diffmaglim - magzpsci)) / 5.0

then the same `convert_zeropoint` rescaling, `is_detection=False`. This is a
stated modeling assumption (CLAUDE.md §12: "state assumptions out loud"), not
something verified against an external convention -- flag for review if a
future session finds ZTF non-detections need different treatment.

Rows missing `magzpsci` entirely, or with `sigmapsf<=0` on a detection row,
are skipped (never silently -- CLAUDE.md §2.5) with a logged reason and count.

-----------------------------------------------------------------------------
Static context -- identical field mapping to src.readers.btsbot
-----------------------------------------------------------------------------
Taken from the alert's own triggering `candidate` (the most recent row, same
"latest row" convention btsbot.py uses across an object's full history --
here there is only ever one row to take it from, since each Kafka message is
one alert). `host_photoz`/`host_photoz_err`/`host_ellipticity` are always
`None` -- the ZTF alert-packet schema has no redshift or ellipticity field
(same finding as btsbot.py; `mangrove`, Fink's own host cross-match, was
checked on the golden alert and carries no populated distance/redshift field
either -- string `"None"` sentinels throughout for this object). Fink's own
science-module outputs (`rf_snia_vs_nonia`, `snn_snia_vs_nonia`, `cdsxmatch`,
`anomaly_score`, etc.) are deliberately never read here: `StaticContext`
(schema.py, FROZEN) has no slot for them, and feeding another classifier's
output into this model's inputs would be exactly the kind of shortcut
CLAUDE.md §2.6 bans for SNTYPE/REDSHIFT_HELIO -- these fields are available
for pre-filtering/reporting only (see src/serve/live_fink.py), never as model
input.

-----------------------------------------------------------------------------
Cutouts
-----------------------------------------------------------------------------
`cutoutScience`/`cutoutTemplate`/`cutoutDifference` each carry a gzip-
compressed single-HDU FITS stamp (`stampData` bytes) -- confirmed 63x63
pixels on the golden alert, matching CanonicalAlert.cutouts' documented
`(3, 63, 63)` shape exactly (schema.py's comment was written with this
convention in mind). Decoded via `decode_cutout_stamp` and stacked
[science, template, difference]. A decode failure (malformed/missing stamp)
logs a warning and falls back to `cutouts=None` rather than dropping the
whole alert -- classification does not depend on cutouts existing.
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
from src.schema import LAM_EFF, CanonicalAlert, Observation, StaticContext, Survey

logger = logging.getLogger(__name__)

# Same convention as src.readers.btsbot._FID_TO_LAM_EFF_KEY -- duplicated
# rather than imported (CLAUDE.md §12: "three similar lines is better than a
# premature abstraction"); both readers parse the same ZTF alert-packet
# schema but are otherwise independent (batch CSV vs. live Kafka).
_FID_TO_LAM_EFF_KEY = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}

_SRMAG1_SENTINEL = -999.0
_DRB_SENTINEL = -999.0

# Lazy singleton, same rationale as btsbot.py's own `_dust_query`: loading
# the SFD FITS maps from disk is the expensive part, not the per-point query.
_dust_query = None
_DUSTMAPS_DATA_DIR = "data/raw/dustmaps"


class SkippedRow(Exception):
    """A single prv_candidates row could not be converted -- caught and
    counted by the caller, never allowed to fail the whole alert (CLAUDE.md
    §2.5: log and continue, don't silently drop, but one bad historical row
    must not discard an otherwise-good live detection)."""


def _fid_to_lam_eff(fid: int) -> float:
    key = _FID_TO_LAM_EFF_KEY.get(int(fid))
    if key is None:
        raise SkippedRow(f"unrecognized fid {fid!r}; expected one of {sorted(_FID_TO_LAM_EFF_KEY)}")
    return LAM_EFF[key]


def _isdiffpos_sign(value) -> float:
    """Live Fink packets encode this as the string "t"/"f" (verified on the
    golden alert) -- checked explicitly, not via truthiness (which would
    silently treat the string "f" as truthy)."""
    return 1.0 if value == "t" else -1.0


def _detection_observation(row: dict) -> Observation:
    sign = _isdiffpos_sign(row["isdiffpos"])
    magzpsci = row["magzpsci"]
    flux_raw = sign * 10 ** (-0.4 * (row["magpsf"] - magzpsci))
    flux_err_raw = abs(flux_raw) * (math.log(10) / 2.5) * row["sigmapsf"]
    flux, flux_err = convert_zeropoint(flux_raw, flux_err_raw, from_zeropoint=magzpsci)
    return Observation(
        mjd=float(row["jd"]) - 2400000.5,
        lam_eff=_fid_to_lam_eff(row["fid"]),
        flux=float(flux),
        flux_err=float(flux_err),
        is_detection=True,
    )


def _nondetection_observation(row: dict) -> Observation:
    """See module docstring, "prv_candidates -- detections AND non-detection
    upper limits". Raises `SkippedRow` if `diffmaglim`/`magzpsci` is missing
    -- a non-detection row with neither is not even a usable bound."""
    magzpsci = row.get("magzpsci")
    diffmaglim = row.get("diffmaglim")
    if magzpsci is None or diffmaglim is None:
        raise SkippedRow("non-detection row missing magzpsci and/or diffmaglim")
    limiting_flux_raw = 10 ** (-0.4 * (diffmaglim - magzpsci)) / 5.0
    flux, flux_err = convert_zeropoint(0.0, limiting_flux_raw, from_zeropoint=magzpsci)
    return Observation(
        mjd=float(row["jd"]) - 2400000.5,
        lam_eff=_fid_to_lam_eff(row["fid"]),
        flux=float(flux),
        flux_err=float(flux_err),
        is_detection=False,
    )


def _row_to_observation(row: dict) -> Observation:
    if row.get("magzpsci") is None:
        raise SkippedRow("missing magzpsci")
    if row.get("magpsf") is not None:
        if not (row.get("sigmapsf") or 0) > 0:
            raise SkippedRow(f"sigmapsf={row.get('sigmapsf')!r} not > 0 on a detection row")
        return _detection_observation(row)
    return _nondetection_observation(row)


def _srmag1_to_host_mag_r(value: Optional[float]) -> Optional[float]:
    if value is None or value == _SRMAG1_SENTINEL:
        return None
    return float(value)


def _drb_to_real_bogus_score(value: Optional[float]) -> Optional[float]:
    if value is None or value == _DRB_SENTINEL:
        return None
    return float(value)


def _mw_ebv_and_gal_lat(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Single-point query -- live ingestion processes one alert at a time,
    so there is no batch to amortize the ~400x speedup btsbot.py's batched
    version relies on (that reader's own module docstring). The SFD FITS
    maps themselves are still loaded once per process via `_dust_query`."""
    global _dust_query
    if _dust_query is None:
        from dustmaps.config import config
        from dustmaps.sfd import SFDQuery

        config["data_dir"] = _DUSTMAPS_DATA_DIR
        _dust_query = SFDQuery()
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return float(_dust_query(coord)), float(coord.galactic.b.deg)


def decode_cutout_stamp(stamp_data: bytes) -> np.ndarray:
    """One gzip-compressed single-HDU FITS postage stamp -> `(H, W)` float32
    array. See tests/test_readers_avro_ztf.py for a synthetic round-trip
    test (real ZTF stamp bytes are not committed to this repo)."""
    with gzip.open(io.BytesIO(stamp_data)) as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)


def _decode_cutouts(alert: dict) -> Optional[np.ndarray]:
    """`(3, H, W)` [science, template, difference], or `None` if any stamp is
    missing/undecodable/mis-sized -- a cutout failure must never drop the
    alert itself (classification runs on the light curve, not on cutouts).

    Verified live (Day 13 session): a ZTF object near a CCD-quadrant edge
    gets a stamp SMALLER than 63x63 on that side (observed shapes down to
    (63, 42)) -- still perfectly decodable FITS, just not the schema's
    assumed size. `CanonicalAlert.validate()` (schema.py, FROZEN)
    hard-rejects any non-`None` cutouts array that isn't exactly
    `(3, 63, 63)`, so an early version of this function let that raise all
    the way up to `ztf_alert_to_canonical`, discarding the alert's entire
    LIGHT CURVE over an undersized cutout -- 9/1015 alerts in the first live
    run. Checking the shape here and degrading to `cutouts=None` (same
    pattern `avro_lsst.py` already uses) fixes that: an off-size cutout now
    costs only the cutout, never the classification.
    """
    try:
        sci = decode_cutout_stamp(alert["cutoutScience"]["stampData"])
        ref = decode_cutout_stamp(alert["cutoutTemplate"]["stampData"])
        diff = decode_cutout_stamp(alert["cutoutDifference"]["stampData"])
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any decode
        # failure degrades to no-cutouts, never propagates (see docstring).
        logger.warning(
            "avro_ztf: object_id=%s cutout decode failed (%s: %s), storing cutouts=None",
            alert.get("objectId"), type(exc).__name__, exc,
        )
        return None
    stacked = np.stack([sci, ref, diff], axis=0)
    if stacked.shape != (3, 63, 63):
        logger.warning(
            "avro_ztf: object_id=%s cutout stamp shape %s != the (63,63)-per-plane "
            "shape CanonicalAlert.validate() requires (likely a CCD-edge object) -- "
            "storing cutouts=None rather than discard the whole alert over it",
            alert.get("objectId"), sci.shape,
        )
        return None
    return stacked


def ztf_alert_to_canonical(alert: dict) -> CanonicalAlert:
    """Convert one decoded Fink ZTF alert `dict` into a `CanonicalAlert`
    carrying this alert's full available history (`candidate` +
    `prv_candidates`) -- consistent with how `src.serve.state.merge_history`
    already dedups overlapping historical rows across consecutive alerts
    for the same object (CLAUDE.md §5.1), so re-sending the same
    `prv_candidates` window alert-to-alert is handled correctly downstream,
    not something this function needs to worry about.

    Raises:
        ValueError: if the triggering `candidate` itself cannot be converted
            (missing magzpsci, or sigmapsf<=0) -- this is not a normal edge
            case (CLAUDE.md §5.1: every real alert originates from a
            detection), unlike a bad `prv_candidates` row, which is skipped
            and counted instead (see module docstring).
    """
    object_id = str(alert["objectId"])
    cand = alert["candidate"]
    prv = alert.get("prv_candidates") or []

    try:
        observations = [_row_to_observation(cand)]
    except SkippedRow as exc:
        raise ValueError(f"object_id={object_id!r}: triggering candidate unusable ({exc})") from exc

    n_skipped = 0
    for row in prv:
        try:
            observations.append(_row_to_observation(row))
        except SkippedRow as exc:
            n_skipped += 1
            logger.debug("avro_ztf: object_id=%s skipped prv_candidates row (%s)", object_id, exc)
    if n_skipped:
        logger.warning(
            "avro_ztf: object_id=%s skipped %d/%d prv_candidates row(s)",
            object_id, n_skipped, len(prv),
        )

    observations.sort(key=lambda o: o.mjd)

    mw_ebv, gal_lat = _mw_ebv_and_gal_lat(cand["ra"], cand["dec"])
    static = StaticContext(
        host_photoz=None,
        host_photoz_err=None,
        host_offset_arcsec=float(cand["distpsnr1"]) if cand.get("distpsnr1") is not None else None,
        host_ellipticity=None,
        host_mag_r=_srmag1_to_host_mag_r(cand.get("srmag1")),
        mw_ebv=mw_ebv,
        gal_lat=gal_lat,
        real_bogus_score=_drb_to_real_bogus_score(cand.get("drb")),
    )

    canonical = CanonicalAlert(
        object_id=object_id,
        survey=Survey.ZTF,
        observations=observations,
        static=static,
        cutouts=_decode_cutouts(alert),
        class_id=None,  # live stream: no spectroscopic label exists at ingest time
    )
    canonical.validate()
    return canonical
