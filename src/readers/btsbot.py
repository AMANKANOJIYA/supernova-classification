"""BTSbot v10 metadata (Zenodo 10.5281/zenodo.10839691) -> CanonicalAlert
reader (CLAUDE.md §6 step 4, Day 8 session). Metadata only — the 19.6 GB
`images_v10.zip` cutout archive is out of scope this session (Day 10).

Source: `data/raw/btsbot/metadata_v10/{train,val,test}_cand_v10_N100_programid1.csv`,
one row per ZTF alert (not per object) — 87 columns of raw ZTF alert-packet
fields plus BTS's own bright-transient triage columns. Column definitions
verified against the ZTF alert-packet schema
(https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html), not
inferred, since A_FORMAT.TXT (ELAsTiCC2's format doc) has no ZTF equivalent
in this repo.

-----------------------------------------------------------------------------
Flux zeropoint (CLAUDE.md §3) — the highest-risk item this session
-----------------------------------------------------------------------------
BTSbot metadata is NOT on any flux scale — it carries `magpsf`/`sigmapsf`
(PSF-fit magnitude/uncertainty), `magzpsci` (that alert's own photometric
zeropoint, verified to vary row-to-row: 21.4-27.3 mag across the full
dataset — NOT a fixed 31.4 constant the way CLAUDE.md §3 describes for a
hypothetical nJy-flux LSST/ZTF alert-stream reader), and `isdiffpos`
(t/1 = sci-minus-ref = positive; f/0 = ref-minus-sci = negative).

    flux_raw     = sign * 10**(-0.4 * (magpsf - magzpsci))   # on magzpsci's own zp
    flux_err_raw = |flux_raw| * (ln(10)/2.5) * sigmapsf       # standard mag-err propagation

`flux_raw` is then rescaled onto FLUXCAL zp=27.5 via
`src.readers.zeropoint.convert_zeropoint(flux_raw, flux_err_raw,
from_zeropoint=magzpsci)` — `from_zeropoint` is passed PER ROW (not a
module-level constant), since `magzpsci` genuinely varies alert to alert.
Verified end to end against two hand-picked real objects, independently
computed with plain Python math (not by calling this reader or even
`convert_zeropoint`) — see tests/fixtures/golden_btsbot_object.py and
tests/test_readers_btsbot.py.

`magzpsciunc` (the zeropoint's own uncertainty, itself a few-micromag
systematic) is not folded into `flux_err` — only `sigmapsf` is propagated,
matching standard practice for this field.

-----------------------------------------------------------------------------
class_id — always None (Day 8 chat decision)
-----------------------------------------------------------------------------
metadata_v10 carries NO spectroscopic subtype string anywhere in its 87
columns — verified empirically. `label`/`is_SN`/`source_set` are BTSbot's
own binary bright-transient-vs-reject triage output, not a 3-class SN
subtype; using them as a stand-in for `class_id` would silently misrepresent
binary triage as spectroscopic typing (see chat report). This reader never
touches those columns and always emits `class_id=None`. The real
spectroscopic type (Ia/Ib/Ic/II/etc, keyed by the same `objectId` this
reader uses for `object_id`) is fetched and mapped separately, from the
external BTS Sample Explorer, by `src/readers/bts_labels.py` — kept out of
this reader deliberately, so the (offline, fast) metadata parse never
depends on a live network call. Joining that module's output onto this
reader's `CanonicalAlert.object_id` is a Day 9 fine-tuning-time step.

-----------------------------------------------------------------------------
is_detection — always True
-----------------------------------------------------------------------------
Verified empirically across all 409,107 rows (train+val+test): `isdiffpos`
is `True` for every single row, and there is no forced-photometry/upper-
limit row type in this metadata table at all (contrast ELAsTiCC2, whose
PHOT files carry ~30 days of forced photometry per object — CLAUDE.md
§5.1). Every BTSbot alert reaching this table already passed the ZTF
alert-stream's S/N-based detection trigger, so every row is a real
detection — this is also why every reconstructed flux here comes out
positive (see chat report re: Gate A check #2).

-----------------------------------------------------------------------------
Static host context vs. Tier A's seven fields
-----------------------------------------------------------------------------
Per-object StaticContext is built from that object's chronologically LAST
row (max `jd`) — `sgscore1`/`srmag1` are exactly constant across an
object's rows (looked up once from the PS1 catalog, not per-epoch); `ra`/
`dec`/`distpsnr1` show only ~arcsec-level astrometric jitter; `drb`
genuinely varies per alert (each has its own cutout), so "latest row's drb"
is a documented choice, not something forced by the data.

  host_photoz          -> None, always (BTSbot provides no redshift field)
  host_photoz_err      -> None, always
  host_offset_arcsec   -> distpsnr1 (arcsec; no sentinel found empirically —
                          0/409,107 rows hit -999 or any other implausible
                          value for distpsnr1 or sgscore1)
  host_ellipticity     -> None, always (BTSbot provides no ellipticity field)
  host_mag_r           -> srmag1 (PS1 r-band mag of nearest source within 30
                          arcsec); sentinel -999.0 -> None (hit on ~11% of rows)
  mw_ebv               -> dustmaps.sfd.SFDQuery(ra, dec) (schema's own
                          docstring: "from dustmaps") — not a CSV column
  gal_lat              -> astropy SkyCoord(ra, dec).galactic.b.deg — not a
                          CSV column, same derivation as the ELAsTiCC2 reader
  real_bogus_score     -> drb (RealBogus deep-learning score); null on
                          ~6.6% of rows (26,948/409,107) AND a rare -999.0
                          sentinel (1 row) -> both become None
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import polars as pl

from src.readers.elasticc2 import ReadStats, alerts_to_dataframe, write_parquet  # noqa: F401 (re-exported)
from src.readers.zeropoint import convert_zeropoint
from src.schema import LAM_EFF, CanonicalAlert, Observation, StaticContext, Survey

logger = logging.getLogger(__name__)

_FID_TO_LAM_EFF_KEY = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}

_SRMAG1_SENTINEL = -999.0
_DRB_SENTINEL = -999.0

_REQUIRED_COLUMNS = [
    "objectId",
    "jd",
    "fid",
    "magpsf",
    "sigmapsf",
    "magzpsci",
    "isdiffpos",
    "ra",
    "dec",
    "distpsnr1",
    "sgscore1",
    "srmag1",
    "drb",
]

# SFD dust-map FITS files were placed here this session (Harvard Dataverse,
# dustmaps' default fetch source, returned 403 in this environment — fetched
# from a mirror instead; see chat report). Relative to repo root, matching
# every other `data/raw/...` path in this codebase (readers are always run
# from repo root, per the Makefile).
_DUSTMAPS_DATA_DIR = "data/raw/dustmaps"

_dust_query = None  # lazily constructed — see _mw_ebv()


def _fid_to_lam_eff(fid: int) -> float:
    key = _FID_TO_LAM_EFF_KEY.get(int(fid))
    if key is None:
        raise ValueError(f"unrecognized fid {fid!r}; expected one of {sorted(_FID_TO_LAM_EFF_KEY)}")
    return LAM_EFF[key]


def _srmag1_to_host_mag_r(value: Optional[float]) -> Optional[float]:
    if value is None or value == _SRMAG1_SENTINEL:
        return None
    return float(value)


def _drb_to_real_bogus_score(value: Optional[float]) -> Optional[float]:
    if value is None or value == _DRB_SENTINEL:
        return None
    return float(value)


def _mw_ebv_and_gal_lat_batch(ra_deg, dec_deg):
    """Milky Way E(B-V) (from dustmaps, per the schema docstring) and
    galactic latitude for a whole array of RA/Dec at once. Batched rather
    than one `SkyCoord`/`SFDQuery` call per object — same rationale as
    `elasticc2._gal_lat_deg_batch` (~400x faster there; this reader's first,
    unbatched version took 141s over the full train+val+test set, almost
    entirely in ~19,500 one-point `SkyCoord`/`SFDQuery` constructions).
    """
    global _dust_query
    if _dust_query is None:
        from dustmaps.config import config
        from dustmaps.sfd import SFDQuery

        config["data_dir"] = _DUSTMAPS_DATA_DIR
        _dust_query = SFDQuery()
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return _dust_query(coord).astype(float), coord.galactic.b.deg


def read_candidates_csv(
    csv_path: str, survey: Survey = Survey.ZTF
) -> tuple[list[CanonicalAlert], ReadStats]:
    """Read one BTSbot metadata_v10 split CSV into `CanonicalAlert` objects,
    one per unique `objectId`, `observations` sorted ascending by `mjd`.

    Args:
        csv_path: path to a `{train,val,test}_cand_v10_N100_programid1.csv`.
        survey: defaults to `Survey.ZTF` (CLAUDE.md §5, Tier B).

    Returns:
        `(alerts, stats)`. `class_id` is always `None` (see module
        docstring). `stats` reports rows/objects discarded and why
        (CLAUDE.md §2.5).
    """
    # Two distinct granularities of discard, per CLAUDE.md §2.5 ("log every
    # discarded row with count and reason") and to keep `ReadStats`'s
    # existing object-level semantics honest (elsewhere in this codebase,
    # e.g. elasticc2.py, `n_objects_discarded == sum(discard_reasons.values())`
    # by construction — one discard entry == one object never reaching
    # `alerts`). Row-level discards below are logged with real row counts
    # but only counted into `discard_reasons`/`n_objects_discarded` for the
    # (rare) objects they leave with zero remaining rows — most bad rows
    # belong to objects that still have other good rows and are not lost.
    discard_reasons: dict[str, int] = {}

    def _discard(reason: str, n: int = 1) -> None:
        discard_reasons[reason] = discard_reasons.get(reason, 0) + n

    df = pl.read_csv(csv_path, columns=_REQUIRED_COLUMNS)
    object_ids_before = set(df["objectId"].unique().to_list())

    n_rows_total = df.height
    missing_zp_mask = df["magzpsci"].is_null()
    n_missing_zp = int(missing_zp_mask.sum())
    if n_missing_zp:
        logger.warning(
            "btsbot reader: discarding %d/%d row(s) with null magzpsci in %s",
            n_missing_zp,
            n_rows_total,
            csv_path,
        )
    df = df.filter(~missing_zp_mask)

    non_positive_sigma_mask = df["sigmapsf"] <= 0
    n_non_positive_sigma = int(non_positive_sigma_mask.sum())
    if n_non_positive_sigma:
        logger.warning(
            "btsbot reader: discarding %d row(s) with sigmapsf<=0 in %s",
            n_non_positive_sigma,
            csv_path,
        )
    df = df.filter(~non_positive_sigma_mask)

    object_ids_after = set(df["objectId"].unique().to_list())
    n_objects_fully_lost = len(object_ids_before - object_ids_after)
    if n_objects_fully_lost:
        _discard(
            "every row discarded (missing magzpsci and/or non-positive sigmapsf)",
            n_objects_fully_lost,
        )
        logger.warning(
            "btsbot reader: %d object(s) in %s lost ALL rows to the row-level "
            "discards above and will not appear in the output",
            n_objects_fully_lost,
            csv_path,
        )

    # First pass: per-object observations + the chronologically-last row
    # (everything except mw_ebv/gal_lat, which are computed in a second,
    # batched pass below — see _mw_ebv_and_gal_lat_batch).
    object_ids: list[str] = []
    observations_by_object: list[list[Observation]] = []
    last_rows: list[dict] = []
    for object_id, group in df.group_by("objectId", maintain_order=False):
        object_id = object_id[0] if isinstance(object_id, tuple) else object_id
        group = group.sort("jd")

        observations = []
        for row in group.iter_rows(named=True):
            sign = 1.0 if row["isdiffpos"] else -1.0
            flux_raw = sign * 10 ** (-0.4 * (row["magpsf"] - row["magzpsci"]))
            flux_err_raw = abs(flux_raw) * (math.log(10) / 2.5) * row["sigmapsf"]
            flux, flux_err = convert_zeropoint(
                flux_raw, flux_err_raw, from_zeropoint=row["magzpsci"]
            )
            observations.append(
                Observation(
                    mjd=float(row["jd"]) - 2400000.5,
                    lam_eff=_fid_to_lam_eff(row["fid"]),
                    flux=float(flux),
                    flux_err=float(flux_err),
                    is_detection=True,  # see module docstring
                )
            )

        object_ids.append(str(object_id))
        observations_by_object.append(observations)
        last_rows.append(group.row(-1, named=True))

    if last_rows:
        last_ra = np.array([r["ra"] for r in last_rows])
        last_dec = np.array([r["dec"] for r in last_rows])
        mw_ebv_arr, gal_lat_arr = _mw_ebv_and_gal_lat_batch(last_ra, last_dec)
    else:
        mw_ebv_arr, gal_lat_arr = [], []

    alerts: list[CanonicalAlert] = []
    for object_id, observations, last, mw_ebv, gal_lat in zip(
        object_ids, observations_by_object, last_rows, mw_ebv_arr, gal_lat_arr
    ):
        static = StaticContext(
            host_photoz=None,
            host_photoz_err=None,
            host_offset_arcsec=float(last["distpsnr1"]),
            host_ellipticity=None,
            host_mag_r=_srmag1_to_host_mag_r(last["srmag1"]),
            mw_ebv=float(mw_ebv),
            gal_lat=float(gal_lat),
            real_bogus_score=_drb_to_real_bogus_score(last["drb"]),
        )

        alert = CanonicalAlert(
            object_id=object_id,
            survey=survey,
            observations=observations,
            static=static,
            cutouts=None,  # metadata only this session — images_v10 is Day 10
            class_id=None,  # see module docstring
        )
        alert.validate()
        alerts.append(alert)

    stats = ReadStats(
        n_objects_read=len(alerts),
        n_objects_discarded=sum(discard_reasons.values()),
        discard_reasons=discard_reasons,
    )
    logger.info(
        "btsbot reader: %s -> %d objects read, %d row(s) discarded (%s)",
        csv_path,
        stats.n_objects_read,
        stats.n_objects_discarded,
        stats.discard_reasons or "no discards",
    )
    return alerts, stats


# Filenames as shipped by the Zenodo release — CLAUDE.md: "Train/val/test
# splits ship with the release — use them rather than re-splitting."
SPLIT_FILES: dict[str, str] = {
    "train": "train_cand_v10_N100_programid1.csv",
    "val": "val_cand_v10_N100_programid1.csv",
    "test": "test_cand_v10_N100_programid1.csv",
}


def read_all_splits(
    metadata_dir: str, survey: Survey = Survey.ZTF
) -> dict[str, tuple[list[CanonicalAlert], ReadStats]]:
    """Read every split named in `SPLIT_FILES` from `metadata_dir`
    (`data/raw/btsbot/metadata_v10/`). Splits are read and returned
    separately, never merged or re-split, per CLAUDE.md's instruction to
    use the release's own train/val/test assignment."""
    return {
        split: read_candidates_csv(os.path.join(metadata_dir, filename), survey=survey)
        for split, filename in SPLIT_FILES.items()
    }
