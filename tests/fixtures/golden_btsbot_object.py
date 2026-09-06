"""Golden fixture: two hand-verified BTSbot v10 objects, traced by hand from
the raw metadata_v10 CSV rows through the zeropoint conversion this session
added (CLAUDE.md §3, §12: hand-verify one object end to end, commit it as a
fixture, write the reader against the test).

Source file (not committed — data/ is gitignored; re-derive from the real
CSV if this ever needs updating):
    data/raw/btsbot/metadata_v10/val_cand_v10_N100_programid1.csv

-----------------------------------------------------------------------------
Flux convention (Day 8 finding, the highest-risk item this session)
-----------------------------------------------------------------------------
BTSbot metadata_v10 is NOT on any flux scale at all — it carries the raw ZTF
alert-packet fields (`magpsf`, `sigmapsf`, `magzpsci`, `isdiffpos`), i.e.
*magnitudes* on a per-alert photometric zeropoint (`magzpsci` varies row to
row, unlike ELAsTiCC2's fixed FLUXCAL scale or the fixed-31.4 LSST-alert
convention CLAUDE.md §3 describes for a hypothetical future reader). Verified
against the ZTF alert-packet schema
(https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html):
`magzpsci` = "Magnitude zero point for photometry estimates" and
`isdiffpos` = "t/1 => sci-minus-ref (positive); f/0 => ref-minus-sci
(negative)".

Conversion applied here, independent of `src.readers.zeropoint`, straight
from the definitions:
    flux_raw     = sign * 10**(-0.4*(magpsf - magzpsci))   # on magzpsci's own zp
    flux_err_raw = |flux_raw| * (ln(10)/2.5) * sigmapsf     # standard mag-err propagation
    sign         = +1 if isdiffpos else -1
Then `src.readers.zeropoint.convert_zeropoint(flux_raw, flux_err_raw,
from_zeropoint=magzpsci)` rescales onto the SNANA FLUXCAL 27.5 convention —
`from_zeropoint` varies PER ROW here, unlike a single global 31.4 constant.
`magzpsciunc` (the zeropoint's own uncertainty) is not folded into
flux_err — only `sigmapsf` is propagated, matching standard practice for
this field (magzpsciunc is a few-micro-mag systematic, negligible next to
sigmapsf).

`mjd = jd - 2400000.5` (standard JD -> MJD).

-----------------------------------------------------------------------------
Object 1: ZTF19abiszoe — 8 alerts (both ztf_g, ztf_r), clean host fields
-----------------------------------------------------------------------------
`source_set=trues`, `label=1` in the raw CSV (BTS-confirmed bright
transient) — irrelevant to `class_id` this session (see chat report: this
session's reader always emits `class_id=None`; spectroscopic-type join is a
separate module, `src/readers/bts_labels.py`).

Every row for this object, sorted ascending by `jd` (as read from the CSV;
verified this object's rows are not already sorted in the source file, so
the reader's own sort is exercised, not just passed through):
(jd, fid, magpsf, sigmapsf, magzpsci, isdiffpos)
"""

OBJECT_ID = "ZTF19abiszoe"

RAW_ROWS = [
    (2458694.7118981, 1, 18.649986267089844, 0.0719746425747871, 26.15098571777344, True),
    (2458698.6779861, 2, 18.350414276123047, 0.0954082012176513, 25.95841407775879, True),
    (2458705.6638426, 2, 18.51445770263672, 0.0772358477115631, 26.151456832885746, True),
    (2458708.6540394, 1, 18.75904083251953, 0.111432708799839, 26.111040115356445, True),
    (2458712.7388079, 2, 19.02228736877441, 0.1420579105615615, 26.10928726196289, True),
    (2458715.6550694, 1, 19.35818672180176, 0.1028162986040115, 26.118186950683597, True),
    (2458718.6443056, 2, 19.21329116821289, 0.1418308764696121, 26.12529182434082, True),
    (2458718.6573148, 1, 19.70516967773437, 0.1505766659975052, 26.098169326782227, True),
]
NOBS = 8

# Expected mjd (rows already in jd-ascending order above) and flux/flux_err
# on the FLUXCAL zp=27.5 convention, computed independently (plain Python
# math, not via the reader or even via src.readers.zeropoint.convert_zeropoint)
# per the formula in the module docstring above.
EXPECTED_MJD = [row[0] - 2400000.5 for row in RAW_ROWS]

EXPECTED_FLUX = [
    288.9309512293282,
    267.033379289093,
    327.6290369617668,
    242.78159444269326,
    189.8955825458824,
    141.66901417975282,
    164.02762330547722,
    99.18993149805867,
]

EXPECTED_FLUX_ERR = [
    19.153549317402394,
    23.465344778402006,
    23.306495899989127,
    24.917480504870934,
    24.845970469731263,
    13.41567463717972,
    21.42710308147621,
    13.756278108848582,
]

# fid 1 -> ztf_g, fid 2 -> ztf_r (LAM_EFF keys) — never a raw band index.
EXPECTED_FID = [row[1] for row in RAW_ROWS]

# Every alert in this metadata release is a real ZTF alert-stream detection
# (S/N above the survey's alert-triggering threshold) — there is no
# forced-photometry/upper-limit row type in this CSV at all (contrast
# ELAsTiCC2, whose PHOT files carry ~30 days of forced photometry per
# object). Verified empirically this session: `isdiffpos` is `True` for
# ALL 409,107 rows across train+val+test, and there is no analogous
# "is this row a real detection" column that ever reads False. See chat
# report §"Anything about real data that contradicts an assumption built on
# simulations" for the implication for Gate A check #2 (negative fluxes).
EXPECTED_IS_DETECTION = [True] * NOBS

# Static context fields, taken from this object's chronologically LAST row
# (jd=2458718.6573148, fid=1, candid=964157313615015020). Justified
# empirically this session: `sgscore1`/`srmag1` are exactly constant across
# all 8 rows for this object (PS1-catalog crossmatch values, looked up once,
# independent of epoch); `distpsnr1`/ra/dec show only ~0.0001-0.2 arcsec-
# level astrometric jitter row to row (negligible); `drb` genuinely varies
# per alert (each alert has its own difference-image cutout) — using the
# last row's drb is a documented judgment call, not something forced by the
# data being constant.
EXPECTED_HOST_OFFSET_ARCSEC = 10.08835506439209  # distpsnr1, last row
EXPECTED_HOST_MAG_R = 16.849899291992188  # srmag1, last row (not the -999 sentinel)
EXPECTED_REAL_BOGUS_SCORE = 0.999688982963562  # drb, last row

# host_photoz / host_photoz_err / host_ellipticity: BTSbot provides none of
# these at all (verified against the ZTF alert-packet schema — no redshift
# or ellipticity field exists anywhere in the 87-column metadata_v10 export).
# Always None for every BTSbot object, not a per-object sentinel case.
EXPECTED_HOST_PHOTOZ = None
EXPECTED_HOST_PHOTOZ_ERR = None
EXPECTED_HOST_ELLIPTICITY = None

# mw_ebv / gal_lat: not HEAD/CSV fields — derived, same as the ELAsTiCC2
# reader, from this object's last-row RA/Dec (224.7381897, 20.0530004).
# mw_ebv computed independently via dustmaps.sfd.SFDQuery (SFD98 map, fetched
# this session — see chat report on the Harvard Dataverse 403 and the mirror
# used instead); gal_lat via astropy.coordinates.SkyCoord(...).galactic.b.deg.
EXPECTED_MW_EBV = 0.04362025
EXPECTED_GAL_LAT_DEG = 60.02971902995888


# -----------------------------------------------------------------------------
# Object 2: ZTF19aamfupk — 3 alerts, exercises BOTH sentinels found in this
# data (srmag1=-999 "no PS1 r-band match", drb=<null> "acai/drb not computed
# for this alert" — a real, non-rare case: drb is null on 26,948/409,107
# rows, ~6.6%, not a one-off). Neither must survive into StaticContext as
# the raw sentinel/null value re-emitted uninterpreted as "-999" — both must
# become Python None.
# -----------------------------------------------------------------------------

SENTINEL_OBJECT_ID = "ZTF19aamfupk"

SENTINEL_RAW_ROWS = [
    (2458573.6395949, 2, 18.17814445495605, 0.0707112699747085, 25.94914436340332, True),
    (2458573.6586227, 1, 19.93438720703125, 0.2017474323511123, 25.997386932373047, True),
    (2458582.677963, 2, 18.749984741210938, 0.1074327379465103, 25.891984939575195, True),
]
SENTINEL_NOBS = 3

SENTINEL_EXPECTED_MJD = [row[0] - 2400000.5 for row in SENTINEL_RAW_ROWS]

SENTINEL_EXPECTED_FLUX = [
    307.6505591444426,
    66.70442780072639,
    163.52872670211042,
]
SENTINEL_EXPECTED_FLUX_ERR = [
    20.036507625145415,
    12.394766773246392,
    16.181038050976706,
]

# Raw srmag1 == -999.0 and raw drb == null on all 3 rows for this object —
# both must convert to None, never pass through as -999.0.
SENTINEL_EXPECTED_HOST_MAG_R = None
SENTINEL_EXPECTED_REAL_BOGUS_SCORE = None

# Last row (jd=2458582.677963): distpsnr1=1.154861330986023, ra=70.2723313,
# dec=73.6730673 — distpsnr1 has no sentinel convention found in this data
# (verified empirically: 0/409,107 rows hit -999 or any other implausible
# value for distpsnr1 or sgscore1), so this passes through unconverted.
SENTINEL_EXPECTED_HOST_OFFSET_ARCSEC = 1.154861330986023
SENTINEL_EXPECTED_MW_EBV = 0.16284922
SENTINEL_EXPECTED_GAL_LAT_DEG = 17.62867327318394
