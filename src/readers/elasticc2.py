"""ELAsTiCC2 SNANA FITS -> CanonicalAlert reader (CLAUDE.md §6 step 4).

Format reference: `data/raw/elasticc2/A_FORMAT.TXT`, general SNANA FITS
format in SNANA manual §4.36.1. Every convention below was verified against
the real downloaded files (SNIa-SALT3, SNIIb+HostXT_V19), not inferred from
documentation alone — see this session's chat report for the empirical
checks (PTROBS indexing, sentinel values, PHOTFLAG bits).

-----------------------------------------------------------------------------
PTROBS_MIN / PTROBS_MAX
-----------------------------------------------------------------------------
1-indexed, inclusive on both ends. `PTROBS_MAX - PTROBS_MIN + 1 == NOBS`,
verified against real data. The Python slice for one object's photometry is
`phot_data[PTROBS_MIN - 1 : PTROBS_MAX]` — this slice is confirmed to
contain exactly NOBS rows and never the `MJD == -777` end-of-lightcurve
terminator (that terminator sits immediately after, at 0-indexed position
`PTROBS_MAX`, i.e. one past the last row this slice includes; the next
object's PTROBS_MIN begins right after the terminator). A_FORMAT.TXT
documents the pointers' existence and the -777 sentinel but not the
indexing convention itself — this was verified empirically here, and cross-
checked against LSSTDESC/elasticc's own lib_elasticc2/read_snana.py (read
for reference this session, not installed — see chat report), which uses
the identical `PTROBS_MIN - 1` / `[ptrmin:ptrmax]` slicing.

-----------------------------------------------------------------------------
Flux zeropoint (CLAUDE.md §3)
-----------------------------------------------------------------------------
`FLUXCAL`/`FLUXCALERR` in ELAsTiCC2 PHOT files are already on the SNANA
FLUXCAL convention §3 requires (AB zeropoint 27.5) — A_FORMAT.TXT states
`mag(SNANA sim) = 27.5 - 2.5*log10(FLUXCAL)` directly. No conversion is
applied here; `flux`/`flux_err` are FLUXCAL/FLUXCALERR passed through
unchanged. (Contrast: A_FORMAT.TXT separately notes the LSST *alert*
stream's own psFlux uses zeropoint 31.4 — a real future ZTF/LSST-alert
reader will need to convert; this ELAsTiCC2 SNANA reader does not.)

-----------------------------------------------------------------------------
is_detection
-----------------------------------------------------------------------------
`PHOTFLAG & 4096 != 0` (A_FORMAT.TXT: "+= 4096 for detection, roughly
SNR>5"). The first detection additionally carries `+= 2048` (trigger),
giving PHOTFLAG=6144 for that one row — both cases have the 4096 bit set,
so a single bitwise check covers both.

-----------------------------------------------------------------------------
Sentinel values (CLAUDE.md: "-9 is a missing-value sentinel... check
A_FORMAT.TXT for other sentinels of the same kind")
-----------------------------------------------------------------------------
Four distinct sentinel conventions were found empirically, not one:
  -9      HOSTGAL_PHOTOZ[_ERR], HOSTGAL_SPECZ[_ERR], HOSTGAL_SNSEP,
          REDSHIFT_HELIO[_ERR] (per A_FORMAT.TXT; not hit in the data
          checked here, since this sample always had a redshift estimate)
  -9999   HOSTGAL_ELLIPTICITY — undocumented in A_FORMAT.TXT
  999     HOSTGAL_MAG_[band] (positive, opposite sign convention) —
          undocumented in A_FORMAT.TXT
A_FORMAT.TXT's claim that "-999 indicates no 2nd host" for every
HOSTGAL2_xxx field does NOT hold uniformly: empirically HOSTGAL2_RA uses
-999, but HOSTGAL2_PHOTOZ/HOSTGAL2_SNSEP use -9 and HOSTGAL2_OBJID uses 0.
This reader does not currently populate any HOSTGAL2_* field into
StaticContext (the schema has no slot for a second host), so this
discrepancy doesn't affect output today, but is recorded here since it
directly contradicts the format doc and would matter if a second-host
field is ever added.

Each sentinel-eligible HEAD field is converted to `None` by comparing
against its OWN specific sentinel value — never inferred from
HOSTGAL_NMATCH. HOSTGAL_NMATCH counts matches passing a DDLR<4 quality
threshold; it is not a "host record exists" flag — an object with
HOSTGAL_NMATCH=0 was found (empirically) to still carry real, non-sentinel
HOSTGAL_PHOTOZ/HOSTGAL_MAG_r/etc. values for its nearest candidate host
(A_FORMAT.TXT: "First listed host has smallest DDLR and may not be the
true host"). Gating on NMATCH instead of the field's own sentinel value
would have silently thrown away real host data for these objects.

-----------------------------------------------------------------------------
Label derivation
-----------------------------------------------------------------------------
class_id is derived from `SIM_TYPE_INDEX` (confirmed empirically, across
both downloaded models, to be the same integer as the "gentype" column in
elasticc2_gentypeofclassid.csv — e.g. SNIa-SALT3 -> 10 -> classid 2222,
SNIIb+HostXT_V19 -> 37 -> classid 2224) via `taxonomy.load_gentype_of_classid`,
never from the directory or file name. The directory-to-model table in this
session's instructions is for locating files only.
"""

import glob
import logging
import os
import re
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

from src.schema import LAM_EFF, CanonicalAlert, Observation, StaticContext, Survey

logger = logging.getLogger(__name__)

_DETECTION_BIT = 4096
_END_OF_LIGHTCURVE_MJD = -777.0

_BAND_TO_LAM_EFF_KEY = {"u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "Y": "y"}

# (head_field, sentinel_value) for every StaticContext field sourced from a
# sentinel-eligible HEAD column. See module docstring for how each value was
# determined empirically.
_SENTINEL = {
    "HOSTGAL_PHOTOZ": -9.0,
    "HOSTGAL_PHOTOZ_ERR": -9.0,
    "HOSTGAL_SNSEP": -9.0,
    "HOSTGAL_ELLIPTICITY": -9999.0,
    "HOSTGAL_MAG_r": 999.0,
}


def _sentinel_to_none(value: float, head_field: str) -> "float | None":
    sentinel = _SENTINEL[head_field]
    return None if value == sentinel else float(value)


def _band_to_lam_eff(band: str) -> float:
    key = band.strip()
    if key not in _BAND_TO_LAM_EFF_KEY:
        raise ValueError(f"unrecognized BAND value {band!r}; expected one of ugrizY")
    return LAM_EFF[_BAND_TO_LAM_EFF_KEY[key]]


def _gal_lat_deg_batch(ra_deg, dec_deg):
    """Galactic latitude, degrees, for a whole array of RA/Dec at once.

    Not a HEAD field — HEAD only has RA/DEC; this is a derived quantity,
    computed the same way for every reader. Batched rather than one
    `SkyCoord` per object: measured ~400x faster in this session (a single
    vectorized `SkyCoord` call vs. one object construction per row), which
    matters at 80 files x ~4000 objects each.
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return coord.galactic.b.deg


@dataclass(frozen=True)
class ReadStats:
    """Per-file or per-directory read summary. CLAUDE.md §2.5: every
    discarded row/object must be logged with a count and a reason — this is
    that count, returned to the caller rather than only logged, so a
    pipeline script can assert on it (e.g. in CI) without scraping logs.

    `n_head_files_found`/`n_phot_files_found`/`n_pairs_read_ok` are
    directory-level file counts (default 0; only `read_model_directory`
    populates them) — object-level `n_objects_discarded` is row-level and
    cannot by itself distinguish "one row was bad" from "one whole file
    silently failed to read", which is exactly the gap that let a truncated
    download pass as a handful of discarded objects instead of ~1/40 of a
    class going missing. See `assert_directory_complete`.
    """

    n_objects_read: int
    n_objects_discarded: int
    discard_reasons: dict[str, int]
    n_head_files_found: int = 0
    n_phot_files_found: int = 0
    n_pairs_read_ok: int = 0

    def __add__(self, other: "ReadStats") -> "ReadStats":
        merged_reasons = dict(self.discard_reasons)
        for reason, count in other.discard_reasons.items():
            merged_reasons[reason] = merged_reasons.get(reason, 0) + count
        return ReadStats(
            n_objects_read=self.n_objects_read + other.n_objects_read,
            n_objects_discarded=self.n_objects_discarded + other.n_objects_discarded,
            discard_reasons=merged_reasons,
            n_head_files_found=self.n_head_files_found + other.n_head_files_found,
            n_phot_files_found=self.n_phot_files_found + other.n_phot_files_found,
            n_pairs_read_ok=self.n_pairs_read_ok + other.n_pairs_read_ok,
        )


def assert_directory_complete(
    model_dir: str, stats: ReadStats, expected_files: int = 40
) -> None:
    """Fail loudly if `model_dir` did not yield exactly `expected_files`
    HEAD.FITS and `expected_files` PHOT.FITS files, all successfully read.

    Added per Day-3 CLAUDE.md amendment: `read_model_directory`'s existing
    per-file log-and-continue (needed so one corrupt download doesn't crash
    the whole batch — see TestReadModelDirectoryResilience) reports discards
    at the object level, via `n_objects_discarded`/`discard_reasons`. That is
    exactly the wrong granularity for a whole missing/corrupt *file*: a
    truncated download that fails to open contributes only "1 discarded
    object" (one exception, logged once) to those counters, not "~1/40th of
    this class's objects are gone." This function checks the file-level
    counts `read_model_directory` also now tracks and raises instead of
    warning, since a warning is exactly what would keep letting this pass
    silently.

    Raises:
        RuntimeError: if fewer (or more) than `expected_files` HEAD files,
            PHOT files, or successfully-read pairs were found.
    """
    if (
        stats.n_head_files_found != expected_files
        or stats.n_phot_files_found != expected_files
        or stats.n_pairs_read_ok != expected_files
    ):
        raise RuntimeError(
            f"{model_dir}: incomplete directory — expected exactly "
            f"{expected_files} HEAD.FITS and {expected_files} PHOT.FITS files, all "
            f"successfully read. Found {stats.n_head_files_found} HEAD file(s), "
            f"{stats.n_phot_files_found} PHOT file(s) on disk; {stats.n_pairs_read_ok} "
            f"pair(s) successfully read. discard_reasons={stats.discard_reasons or 'none'}"
        )


def read_head_phot_file(
    head_path: str,
    phot_path: str,
    gentype_to_classid: dict[int, int],
    survey: Survey = Survey.LSST,
) -> tuple[list[CanonicalAlert], ReadStats]:
    """Read one HEAD/PHOT FITS(.gz) pair into `CanonicalAlert` objects.

    Args:
        head_path: path to a `*_HEAD.FITS` or `*_HEAD.FITS.gz` file.
        phot_path: path to the matching `*_PHOT.FITS[.gz]` file.
        gentype_to_classid: from `taxonomy.load_gentype_of_classid()`. Maps
            SIM_TYPE_INDEX (== SNANA gentype, verified empirically — see
            module docstring) to an ELAsTiCC2 taxonomy `class_id`.
        survey: defaults to `Survey.LSST` — ELAsTiCC2 simulates the LSST
            survey (CLAUDE.md §5, Tier A).

    Returns:
        `(alerts, stats)`. `alerts` is one `CanonicalAlert` per HEAD row
        that was not discarded; `stats` reports how many were read/
        discarded and why (CLAUDE.md §2.5).
    """
    discard_reasons: dict[str, int] = {}

    def _discard(reason: str) -> None:
        discard_reasons[reason] = discard_reasons.get(reason, 0) + 1

    with fits.open(head_path) as head_hdul, fits.open(phot_path) as phot_hdul:
        head = head_hdul[1].data
        phot = phot_hdul[1].data
        gal_lat_col = _gal_lat_deg_batch(head["RA"], head["DEC"])

        alerts: list[CanonicalAlert] = []

        for head_idx, row in enumerate(head):
            snid = int(row["SNID"])
            ptrobs_min = int(row["PTROBS_MIN"])
            ptrobs_max = int(row["PTROBS_MAX"])
            nobs = int(row["NOBS"])

            if nobs <= 0 or ptrobs_max < ptrobs_min:
                _discard("nobs<=0 or empty PTROBS range")
                logger.warning(
                    "elasticc2 reader: discarding SNID=%d (NOBS=%d, PTROBS=[%d,%d])",
                    snid,
                    nobs,
                    ptrobs_min,
                    ptrobs_max,
                )
                continue

            obs_rows = phot[ptrobs_min - 1 : ptrobs_max]

            if len(obs_rows) != nobs:
                _discard("PTROBS slice length != NOBS")
                logger.warning(
                    "elasticc2 reader: discarding SNID=%d, PTROBS slice gave %d rows, "
                    "NOBS says %d",
                    snid,
                    len(obs_rows),
                    nobs,
                )
                continue

            if (obs_rows["MJD"] == _END_OF_LIGHTCURVE_MJD).any():
                _discard("end-of-lightcurve sentinel (-777) found inside PTROBS range")
                logger.warning(
                    "elasticc2 reader: discarding SNID=%d, found MJD=-777 terminator "
                    "inside its own PTROBS range (unexpected)",
                    snid,
                )
                continue

            sim_type_index = int(row["SIM_TYPE_INDEX"])
            class_id = gentype_to_classid.get(sim_type_index)
            if class_id is None:
                _discard(f"gentype {sim_type_index} not in gentype_to_classid mapping")
                logger.warning(
                    "elasticc2 reader: discarding SNID=%d, SIM_TYPE_INDEX=%d has no "
                    "entry in gentype_to_classid",
                    snid,
                    sim_type_index,
                )
                continue

            # Columnar access (obs_rows['MJD'] etc.), not per-row-per-field FITS
            # record access — the latter measured ~4x slower in this session
            # (astropy FITS record row iteration has real per-field overhead)
            # and this reader needs to process 80 files x ~4000 objects each.
            mjd_col = obs_rows["MJD"]
            band_col = obs_rows["BAND"]
            fluxcal_col = obs_rows["FLUXCAL"]
            fluxcalerr_col = obs_rows["FLUXCALERR"]
            photflag_col = obs_rows["PHOTFLAG"]
            observations = [
                Observation(
                    mjd=float(mjd_col[i]),
                    lam_eff=_band_to_lam_eff(str(band_col[i])),
                    flux=float(fluxcal_col[i]),
                    flux_err=float(fluxcalerr_col[i]),
                    is_detection=bool(int(photflag_col[i]) & _DETECTION_BIT),
                )
                for i in range(nobs)
            ]
            observations.sort(key=lambda o: o.mjd)  # defensive; verified already-sorted in practice

            static = StaticContext(
                host_photoz=_sentinel_to_none(float(row["HOSTGAL_PHOTOZ"]), "HOSTGAL_PHOTOZ"),
                host_photoz_err=_sentinel_to_none(
                    float(row["HOSTGAL_PHOTOZ_ERR"]), "HOSTGAL_PHOTOZ_ERR"
                ),
                host_offset_arcsec=_sentinel_to_none(
                    float(row["HOSTGAL_SNSEP"]), "HOSTGAL_SNSEP"
                ),
                host_ellipticity=_sentinel_to_none(
                    float(row["HOSTGAL_ELLIPTICITY"]), "HOSTGAL_ELLIPTICITY"
                ),
                host_mag_r=_sentinel_to_none(float(row["HOSTGAL_MAG_r"]), "HOSTGAL_MAG_r"),
                mw_ebv=float(row["MWEBV"]),
                gal_lat=float(gal_lat_col[head_idx]),
                real_bogus_score=None,  # no image-based classifier exists for pure simulation (§5.1)
            )

            alert = CanonicalAlert(
                object_id=str(snid),
                survey=survey,
                observations=observations,
                static=static,
                cutouts=None,  # ELAsTiCC2 is photometry-level; always None (§5.1)
                class_id=class_id,
            )
            alert.validate()
            alerts.append(alert)

        stats = ReadStats(
            n_objects_read=len(alerts),
            n_objects_discarded=sum(discard_reasons.values()),
            discard_reasons=discard_reasons,
        )
        return alerts, stats


_HEAD_FILE_RE = re.compile(r"^(.*)_HEAD\.FITS(\.gz)?$")


def _find_head_phot_pairs(model_dir: str) -> list[tuple[str, str]]:
    """Locate matching `*_HEAD.FITS[.gz]` / `*_PHOT.FITS[.gz]` pairs in a
    model directory, sorted for deterministic processing order."""
    pairs = []
    for head_path in sorted(glob.glob(os.path.join(model_dir, "*_HEAD.FITS*"))):
        match = _HEAD_FILE_RE.match(os.path.basename(head_path))
        if not match:
            continue
        base, gz_suffix = match.group(1), match.group(2) or ""
        phot_path = os.path.join(model_dir, f"{base}_PHOT.FITS{gz_suffix}")
        if os.path.exists(phot_path):
            pairs.append((head_path, phot_path))
        else:
            logger.warning(
                "elasticc2 reader: found %s but no matching PHOT file %s", head_path, phot_path
            )
    return pairs


def read_model_directory(
    model_dir: str,
    gentype_to_classid: dict[int, int],
    survey: Survey = Survey.LSST,
) -> tuple[list[CanonicalAlert], ReadStats]:
    """Read every HEAD/PHOT pair in a model directory (CLAUDE.md: "Each
    model directory holds 40 gzipped _HEAD.FITS and 40 _PHOT.FITS files").

    Args:
        model_dir: path to one `ELASTICC2_TRAIN_02_<model>/` directory.
        gentype_to_classid: see `read_head_phot_file`.
        survey: see `read_head_phot_file`.

    Returns:
        `(alerts, stats)`, combined across every HEAD/PHOT pair found.
    """
    all_alerts: list[CanonicalAlert] = []
    total_stats = ReadStats(n_objects_read=0, n_objects_discarded=0, discard_reasons={})
    n_pairs_read_ok = 0

    for head_path, phot_path in _find_head_phot_pairs(model_dir):
        try:
            alerts, stats = read_head_phot_file(head_path, phot_path, gentype_to_classid, survey)
        except Exception as exc:
            # A single corrupt/truncated download (encountered for real this
            # session — a truncated gzip mid-download) must not take down the
            # whole directory's read. Logged individually with the file path
            # and full error, counted in aggregate under the exception type
            # per CLAUDE.md §2.5. This is deliberately still a per-*object*
            # discard count (1, not NOBS-many) — file-level loss is tracked
            # separately below via n_head_files_found/n_phot_files_found/
            # n_pairs_read_ok, and must be checked with
            # `assert_directory_complete`, since this counter alone cannot
            # distinguish "a handful of bad rows" from "an entire file
            # silently vanished."
            reason = f"file read error ({type(exc).__name__})"
            total_stats = total_stats + ReadStats(
                n_objects_read=0, n_objects_discarded=1, discard_reasons={reason: 1}
            )
            logger.error(
                "elasticc2 reader: failed to read %s / %s: %s: %s",
                head_path,
                phot_path,
                type(exc).__name__,
                exc,
            )
            continue
        n_pairs_read_ok += 1
        all_alerts.extend(alerts)
        total_stats = total_stats + stats

    total_stats = ReadStats(
        n_objects_read=total_stats.n_objects_read,
        n_objects_discarded=total_stats.n_objects_discarded,
        discard_reasons=total_stats.discard_reasons,
        n_head_files_found=len(glob.glob(os.path.join(model_dir, "*_HEAD.FITS*"))),
        n_phot_files_found=len(glob.glob(os.path.join(model_dir, "*_PHOT.FITS*"))),
        n_pairs_read_ok=n_pairs_read_ok,
    )

    logger.info(
        "elasticc2 reader: %s -> %d objects read, %d discarded (%s); "
        "%d HEAD files found, %d PHOT files found, %d pairs read ok",
        model_dir,
        total_stats.n_objects_read,
        total_stats.n_objects_discarded,
        total_stats.discard_reasons or "no discards",
        total_stats.n_head_files_found,
        total_stats.n_phot_files_found,
        total_stats.n_pairs_read_ok,
    )
    return all_alerts, total_stats


def alerts_to_dataframe(alerts: list[CanonicalAlert]):
    """Flatten `CanonicalAlert` objects into a Polars DataFrame, one row per
    object, ready for `write_parquet`. Observation-level fields become
    list columns; `StaticContext` fields are flattened to top-level columns.
    """
    import polars as pl

    records = []
    for alert in alerts:
        records.append(
            {
                "object_id": alert.object_id,
                "survey": alert.survey.value,
                "class_id": alert.class_id,
                "mjd": [o.mjd for o in alert.observations],
                "lam_eff": [o.lam_eff for o in alert.observations],
                "flux": [o.flux for o in alert.observations],
                "flux_err": [o.flux_err for o in alert.observations],
                "is_detection": [o.is_detection for o in alert.observations],
                "host_photoz": alert.static.host_photoz,
                "host_photoz_err": alert.static.host_photoz_err,
                "host_offset_arcsec": alert.static.host_offset_arcsec,
                "host_ellipticity": alert.static.host_ellipticity,
                "host_mag_r": alert.static.host_mag_r,
                "mw_ebv": alert.static.mw_ebv,
                "gal_lat": alert.static.gal_lat,
                "real_bogus_score": alert.static.real_bogus_score,
            }
        )
    return pl.DataFrame(records)


def write_parquet(alerts: list[CanonicalAlert], output_path: str) -> None:
    """Write `alerts` to `output_path` as Parquet (CLAUDE.md §6 step 4:
    "Write to data/interim/ as Parquet")."""
    df = alerts_to_dataframe(alerts)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.write_parquet(output_path)
