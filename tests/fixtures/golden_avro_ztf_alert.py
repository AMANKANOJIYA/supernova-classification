"""Golden fixture: one real, live Fink ZTF alert (topic `fink_sn_candidates_ztf`,
polled 2026-09-07 during Day 13's live-Fink session — CLAUDE.md §12: hand-verify
one object end to end against the source docs, commit it as a fixture, write the
reader against the test), used to verify `src/readers/avro_ztf.py`.

Not a replay of BTSbot metadata (`golden_btsbot_object.py`) — this is the raw
Fink Kafka packet shape (`candidate` + `prv_candidates`, both real ZTF
alert-packet sub-records, per `fink_client.consumer.AlertConsumer.poll`),
which is what `avro_ztf.py` actually receives on the wire, and it is the first
fixture in this repo that exercises real ZTF **non-detection** rows (19 of
this object's 25 `prv_candidates` entries have `magpsf=None`, i.e. real
5-sigma-limiting-magnitude upper limits, not detections) — BTSbot's
metadata_v10 has no such row type at all (CLAUDE.md §5.1 Day-8 finding).

Only the fields `avro_ztf.py` actually reads are kept (dropped: the ~40
Fink-computed science-module fields like `cdsxmatch`/`rf_snia_vs_nonia`/
`mangrove`/`anomaly_score` — CLAUDE.md §2.6's spirit: none of these are model
inputs, `StaticContext` has no slot for them, so they are irrelevant to this
reader and dropped from the fixture to keep it readable). Cutout stamps
(`cutoutScience`/`cutoutTemplate`/`cutoutDifference`) are also dropped here —
`decode_cutout_stamp` is tested separately in
`tests/test_readers_avro_ztf.py` against a synthetic gzipped FITS stamp, not
against real (large, binary) ZTF postage-stamp bytes.

-----------------------------------------------------------------------------
Flux convention — identical to `src.readers.btsbot` (same ZTF alert-packet
schema; see that reader's module docstring for the derivation), NOT re-derived
here:
    flux_raw     = sign * 10**(-0.4*(magpsf - magzpsci))   # sign: 't'->+1, 'f'->-1
    flux_err_raw = |flux_raw| * (ln(10)/2.5) * sigmapsf
then rescaled onto SNANA FLUXCAL zp=27.5 via
`src.readers.zeropoint.convert_zeropoint(..., from_zeropoint=magzpsci)`,
`magzpsci` taken per-row (verified varying row-to-row below: 25.44-26.44).

One difference from btsbot.py: live Fink packets encode `isdiffpos` as the
literal string `"t"`/`"f"` (verified directly on this alert — every
detection row here is `"t"`), not the CSV-parsed Python `bool` btsbot.py's
`polars.read_csv` already produced. `avro_ztf.py` must check for `"t"`
explicitly, not truthiness of the raw value.

-----------------------------------------------------------------------------
Non-detection (upper-limit) rows — genuinely new handling, absent from
btsbot.py and from any existing reader in this repo:
`prv_candidates` rows with `magpsf=None` carry only a 5-sigma limiting
magnitude (`diffmaglim`), never an actual flux measurement — unlike
ELAsTiCC2's forced photometry (CLAUDE.md §2.2), which gives a real,
possibly-negative flux at every epoch regardless of detection. Represented
here as `flux=0.0` (no measured excess, not a claim the true flux is exactly
zero) with `flux_err` set from the limiting depth treated as a 5-sigma bound:
    limiting_flux_raw = 10**(-0.4*(diffmaglim - magzpsci)) / 5.0
then the same `convert_zeropoint` rescaling. `is_detection=False`.

-----------------------------------------------------------------------------
Object: ZTF26ablxxjw — 26 total rows (1 triggering `candidate` + 25
`prv_candidates`: 6 real detections across fid 1/2 (ztf_g/ztf_r), 19
non-detection upper limits across fid 1/2/3 (ztf_g/ztf_r/ztf_i) — the only
fixture in this repo exercising all three ZTF bands on one object.
"""

OBJECT_ID = "ZTF26ablxxjw"

# The triggering candidate (always a real detection — this alert exists
# because its own S/N crossed the ZTF alert-stream's trigger threshold).
CANDIDATE = {
    "jd": 2461281.6727662,
    "fid": 1,
    "magpsf": 19.020832061767578,
    "sigmapsf": 0.20117300748825073,
    "magzpsci": 26.219831466674805,
    "isdiffpos": "t",
    "diffmaglim": 19.483070373535156,
    "ra": 268.7154116,
    "dec": -16.3315096,
    "distpsnr1": 0.11315658688545227,
    "srmag1": 17.546600341796875,
    "drb": 0.9995948076248169,
}

# prv_candidates, exactly as received (jd-ascending is NOT guaranteed on the
# wire — this list is already ascending here, but the reader's own sort is
# exercised regardless since CANDIDATE's own jd is the largest of all 26).
PRV_CANDIDATES = [
    {"jd": 2461251.6925463, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.302200317382812, "isdiffpos": None, "diffmaglim": 19.47130012512207},
    {"jd": 2461251.7522338, "fid": 2, "magpsf": None, "sigmapsf": None, "magzpsci": 26.440200805664062, "isdiffpos": None, "diffmaglim": 19.57159996032715},
    {"jd": 2461254.7949537, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.829999923706055, "isdiffpos": None, "diffmaglim": 19.33370018005371},
    {"jd": 2461254.8320255, "fid": 2, "magpsf": None, "sigmapsf": None, "magzpsci": 26.30150032043457, "isdiffpos": None, "diffmaglim": 19.359899520874023},
    {"jd": 2461256.6891667, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.229900360107422, "isdiffpos": None, "diffmaglim": 19.992399215698242},
    {"jd": 2461256.8121759, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.835800170898438, "isdiffpos": None, "diffmaglim": 19.53179931640625},
    {"jd": 2461257.7125463, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.228500366210938, "isdiffpos": None, "diffmaglim": 20.407699584960938},
    {"jd": 2461257.8062616, "fid": 2, "magpsf": 18.78219985961914, "sigmapsf": 0.13838399946689606, "magzpsci": 26.318199157714844, "isdiffpos": "t", "diffmaglim": 19.65239906311035},
    {"jd": 2461258.6863426, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.854000091552734, "isdiffpos": None, "diffmaglim": 19.533599853515625},
    {"jd": 2461258.6935069, "fid": 2, "magpsf": 18.819499969482422, "sigmapsf": 0.11961500346660614, "magzpsci": 26.375499725341797, "isdiffpos": "t", "diffmaglim": 20.24970054626465},
    {"jd": 2461259.6709144, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.671100616455078, "isdiffpos": None, "diffmaglim": 19.242599487304688},
    {"jd": 2461260.6885995, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.23900032043457, "isdiffpos": None, "diffmaglim": 20.319900512695312},
    {"jd": 2461261.6892593, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.0310001373291, "isdiffpos": None, "diffmaglim": 20.026599884033203},
    {"jd": 2461262.7793403, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.436899185180664, "isdiffpos": None, "diffmaglim": 19.4237003326416},
    {"jd": 2461262.8119676, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 25.68869972229004, "isdiffpos": None, "diffmaglim": 19.245100021362305},
    {"jd": 2461263.6879282, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.464599609375, "isdiffpos": None, "diffmaglim": 19.17729949951172},
    {"jd": 2461263.7082407, "fid": 2, "magpsf": None, "sigmapsf": None, "magzpsci": 26.246299743652344, "isdiffpos": None, "diffmaglim": 20.059900283813477},
    {"jd": 2461263.7681134, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.026599884033203, "isdiffpos": None, "diffmaglim": 19.980100631713867},
    {"jd": 2461264.6911458, "fid": 1, "magpsf": None, "sigmapsf": None, "magzpsci": 26.193899154663086, "isdiffpos": None, "diffmaglim": 20.32740020751953},
    {"jd": 2461267.6680093, "fid": 1, "magpsf": 19.83139991760254, "sigmapsf": 0.16678500175476074, "magzpsci": 26.33139991760254, "isdiffpos": "t", "diffmaglim": 20.495800018310547},
    {"jd": 2461270.7103588, "fid": 1, "magpsf": 19.898799896240234, "sigmapsf": 0.2054930031299591, "magzpsci": 26.2898006439209, "isdiffpos": "t", "diffmaglim": 20.435300827026367},
    {"jd": 2461278.6672338, "fid": 2, "magpsf": None, "sigmapsf": None, "magzpsci": 26.349199295043945, "isdiffpos": None, "diffmaglim": 19.64240074157715},
    {"jd": 2461278.6895139, "fid": 3, "magpsf": None, "sigmapsf": None, "magzpsci": 25.77549934387207, "isdiffpos": None, "diffmaglim": 19.069700241088867},
    {"jd": 2461278.7298148, "fid": 1, "magpsf": 19.139999389648438, "sigmapsf": 0.27247700095176697, "magzpsci": 26.14699935913086, "isdiffpos": "t", "diffmaglim": 19.088499069213867},
    {"jd": 2461279.7314236, "fid": 1, "magpsf": 18.99720001220703, "sigmapsf": 0.2295749932527542, "magzpsci": 25.978200912475586, "isdiffpos": "t", "diffmaglim": 18.862199783325195},
]

NOBS = 26  # 1 candidate + 25 prv_candidates

# Expected (mjd, fid, flux, flux_err, is_detection) for all 26 rows, mjd-ascending
# -- computed independently (plain Python math against the formulas in the
# module docstring above, not via the reader or even via
# src.readers.zeropoint.convert_zeropoint) from CANDIDATE + PRV_CANDIDATES.
EXPECTED_ROWS = [
    (61251.192546300124, 1, 0.0, 325.4692368617074, False),
    (61251.25223380001, 2, 0.0, 296.7495106993483, False),
    (61254.294953700155, 3, 0.0, 369.44511090194203, False),
    (61254.332025500014, 2, 0.0, 360.6369216615601, False),
    (61256.1891667, 1, 0.0, 201.4050284799926, False),
    (61256.31217590021, 3, 0.0, 307.8295228130304, False),
    (61257.21254630014, 1, 0.0, 137.38843031610116, False),
    (61257.306261599995, 2, 3069.870504592006, 391.27456228240806, True),
    (61258.1863425998, 3, 0.0, 307.3194549422391, False),
    (61258.19350690022, 2, 2966.197142642501, 326.78442511628276, True),
    (61259.1709143999, 3, 0.0, 401.78174509397456, False),
    (61260.188599499874, 1, 0.0, 148.96004354249325, False),
    (61261.18925930001, 1, 0.0, 195.15965515654858, False),
    (61262.27934029978, 3, 0.0, 340.05554632504754, False),
    (61262.31196760014, 1, 0.0, 400.8574755470696, False),
    (61263.18792819977, 3, 0.0, 426.6877708351809, False),
    (61263.208240699954, 2, 0.0, 189.26481393435415, False),
    (61263.268113399856, 1, 0.0, 203.69939680965484, False),
    (61264.191145800054, 1, 0.0, 147.93465139659514, False),
    (61267.168009299785, 1, 1167.9924432470614, 179.4207661518527, True),
    (61270.2103587999, 1, 1097.690843485321, 207.75561038448237, True),
    (61278.16723380005, 2, 0.0, 278.0159414055775, False),
    (61278.18951389985, 3, 0.0, 471.1399149768903, False),
    (61278.22981479997, 1, 2208.005974259188, 554.1224869476963, True),
    (61279.23142360011, 1, 2518.372655405915, 532.5007887173938, True),
    (61281.1727661998, 1, 2464.1501941822544, 456.57545847136345, True),
]

EXPECTED_MJD = [r[0] for r in EXPECTED_ROWS]
EXPECTED_FID = [r[1] for r in EXPECTED_ROWS]
EXPECTED_FLUX = [r[2] for r in EXPECTED_ROWS]
EXPECTED_FLUX_ERR = [r[3] for r in EXPECTED_ROWS]
EXPECTED_IS_DETECTION = [r[4] for r in EXPECTED_ROWS]

# Static context, from CANDIDATE (the alert's own, i.e. most recent, row —
# same "latest row" convention as btsbot.py). host_photoz/host_photoz_err/
# host_ellipticity: None always -- the ZTF alert-packet schema has no
# redshift or ellipticity field (identical to btsbot.py's own finding).
EXPECTED_HOST_OFFSET_ARCSEC = 0.11315658688545227  # distpsnr1
EXPECTED_HOST_MAG_R = 17.546600341796875  # srmag1
EXPECTED_REAL_BOGUS_SCORE = 0.9995948076248169  # drb
EXPECTED_HOST_PHOTOZ = None
EXPECTED_HOST_PHOTOZ_ERR = None
EXPECTED_HOST_ELLIPTICITY = None

# mw_ebv / gal_lat: derived from CANDIDATE's ra/dec (268.7154116, -16.3315096)
# via dustmaps.sfd.SFDQuery / astropy SkyCoord, same as every other reader.
# This object sits at low galactic latitude (b=4.6 deg) -- correspondingly
# high reddening is expected and is what was actually computed, not a
# suspicious outlier.
EXPECTED_MW_EBV = 0.7245166301727295
EXPECTED_GAL_LAT_DEG = 4.61633141541887
