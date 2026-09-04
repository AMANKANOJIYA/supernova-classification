"""Golden fixture: one hand-verified ELAsTiCC2 object, traced by hand from
HEAD.FITS through PTROBS_MIN/PTROBS_MAX to its PHOT.FITS rows.

Source files (not committed — data/ is gitignored; re-derive by re-running
the extraction below against the real files if this ever needs updating):
    data/raw/elasticc2/ELASTICC2_TRAIN_02_SNIa-SALT3/
        ELASTICC2_TRAIN_02_NONIaMODEL0-0001_HEAD.FITS.gz  (object index 0)
        ELASTICC2_TRAIN_02_NONIaMODEL0-0001_PHOT.FITS.gz

HEAD row 0: SNID=62642131, PTROBS_MIN=1, PTROBS_MAX=33, NOBS=33.
PTROBS convention verified empirically (not inferred from A_FORMAT.TXT,
which documents the fields but not the indexing): 1-indexed, inclusive on
both ends. Python slice = phot[PTROBS_MIN-1 : PTROBS_MAX]. Confirmed:
this slice has exactly NOBS=33 rows, none with the MJD=-777 end-of-
lightcurve sentinel; the row at 0-indexed position PTROBS_MAX (34th row,
1-indexed) is MJD=-777, immediately followed by the next object's
PTROBS_MIN. This matches LSSTDESC/elasticc's own lib_elasticc2/read_snana.py
(`ptrmin = PTROBS_MIN - 1`, then `data[ptrmin:ptrmax]`), read for reference
during this session (not installed as a dependency — see chat report).

Cross-checked independently against
ELASTICC2_TRAIN_02_SNIa-SALT3.DUMP (truth summary, one row per object):
    SN: 62642131 12500 1 10 0 4.64818e-01 4.66167e-01 ... PEAKMJD=61328.104
    MJD_DETECT_FIRST=61331.109 MJD_DETECT_LAST=61331.109 NOBS=33 ...
GENTYPE=10 there matches SIM_TYPE_INDEX=10 in HEAD. A single detection
(MJD_DETECT_FIRST == MJD_DETECT_LAST) matches the single PHOTFLAG=6144 row
found below (all 32 others are PHOTFLAG=0, non-detections/forced
photometry) — this object is a real-world instance of CLAUDE.md §5.1's
"first alert carries no forced photometry, every later one reaches back":
32 of 33 rows here are pre/around-detection forced photometry.
"""

SNID = 62642131
HEAD_FILE = "ELASTICC2_TRAIN_02_NONIaMODEL0-0001_HEAD.FITS.gz"
MODEL_DIR = "ELASTICC2_TRAIN_02_SNIa-SALT3"

# Raw HEAD fields, read directly off the FITS row (float() applied, no
# further processing).
RAW_HEAD = {
    "SNID": 62642131,
    "RA": 324.38663571811895,
    "DEC": -5.109444237142185,
    "MWEBV": 0.039476774632930756,
    "MWEBV_ERR": 0.001973838545382023,
    "HOSTGAL_PHOTOZ": 0.43985652923583984,
    "HOSTGAL_PHOTOZ_ERR": 0.42368999123573303,
    "HOSTGAL_SNSEP": 0.5939101576805115,
    "HOSTGAL_ELLIPTICITY": 0.6189000010490417,
    "HOSTGAL_MAG_r": 22.607332229614258,
    "SIM_TYPE_INDEX": 10,
}

# Expected class_id derived via SIM_TYPE_INDEX=10 -> gentype 10 ->
# elasticc2_gentypeofclassid.csv -> classid 2222 (Ia). NOT derived from the
# directory name (which happens to also say SNIa-SALT3, but that is
# coincidental for this fixture, not the mechanism under test).
EXPECTED_CLASS_ID = 2222

# Every raw PTROBS-sliced PHOT row, in file order (already ascending by MJD
# in the source — verified by inspection, not assumed): (mjd, band, fluxcal,
# fluxcalerr, photflag).
RAW_PHOT_ROWS = [
    (61127.4072, "z", -20.374576568603516, 41.474788665771484, 0),
    (61133.4143, "z", -58.37596130371094, 51.58866882324219, 0),
    (61146.4158, "r", 1.244686245918274, 7.040173053741455, 0),
    (61155.427, "z", 2.8769421577453613, 22.68814468383789, 0),
    (61160.4259, "z", -18.98322105407715, 17.896535873413086, 0),
    (61161.3462, "Y", -14.234416961669922, 40.62339782714844, 0),
    (61161.3687, "Y", -25.751277923583984, 34.35419845581055, 0),
    (61178.3701, "r", -1.178865671157837, 6.3242106437683105, 0),
    (61178.3956, "i", 6.9859490394592285, 8.236458778381348, 0),
    (61227.2941, "z", -20.900569915771484, 21.999765396118164, 0),
    (61227.3153, "Y", 31.035499572753906, 33.637001037597656, 0),
    (61227.3453, "r", -25.434646606445312, 14.045605659484863, 0),
    (61246.3052, "Y", 33.85614013671875, 25.293832778930664, 0),
    (61246.3279, "Y", 11.215193748474121, 29.2027587890625, 0),
    (61260.3081, "g", 0.18639704585075378, 4.287863731384277, 0),
    (61263.297, "r", 2.00000262260437, 6.750197887420654, 0),
    (61263.321, "i", 3.2207729816436768, 8.884330749511719, 0),
    (61268.1322, "g", -2.5636098384857178, 3.509051561355591, 0),
    (61268.1569, "r", 0.18965913355350494, 5.916012287139893, 0),
    (61291.0094, "u", -19.82541275024414, 9.47125244140625, 0),
    (61291.0334, "g", 8.548070907592773, 4.955066680908203, 0),
    (61294.1337, "u", -3.3308517932891846, 5.639345169067383, 0),
    (61294.1565, "g", -1.9126195907592773, 2.498701333999634, 0),
    (61296.0988, "u", -0.1243290901184082, 6.1486921310424805, 0),
    (61296.1222, "r", -3.6667940616607666, 6.354576587677002, 0),
    (61296.1304, "r", 9.053269386291504, 6.201106548309326, 0),
    (61296.1546, "i", 7.009267330169678, 8.607889175415039, 0),
    (61298.1769, "u", 0.8232421278953552, 7.303699016571045, 0),
    (61298.1999, "g", 0.5456579327583313, 4.159746170043945, 0),
    (61303.1383, "Y", 10.170055389404297, 29.468324661254883, 0),
    (61303.1614, "Y", 22.23649024963379, 28.856708526611328, 0),
    (61331.1106, "z", 133.82135009765625, 11.183006286621094, 6144),
    (61331.1346, "Y", 95.36376953125, 36.91653060913086, 0),
]

NOBS = 33
assert len(RAW_PHOT_ROWS) == NOBS

# PHOTFLAG & 4096 (CLAUDE.md A_FORMAT.TXT: "+= 4096 for detection"). Exactly
# one row here has it set (index 31, the same one PHOTFLAG=6144 (4096+2048,
# the 1st-detection trigger bit) singles out) -- matches
# MJD_DETECT_FIRST==MJD_DETECT_LAST in the DUMP truth row above.
EXPECTED_IS_DETECTION = [bool(row[4] & 4096) for row in RAW_PHOT_ROWS]
assert sum(EXPECTED_IS_DETECTION) == 1
assert EXPECTED_IS_DETECTION[31] is True

FIRST_DETECTION_MJD = RAW_PHOT_ROWS[31][0]  # 61331.1106

# Expected galactic latitude (degrees), computed independently via
# astropy.coordinates.SkyCoord(ra, dec, frame="icrs").galactic.b.deg from
# the golden object's RA/DEC above (324.38663571811895, -5.109444237142185).
# Not a HEAD field -- HEAD only has RA/DEC; gal_lat is a derived quantity
# the reader must compute.
EXPECTED_GAL_LAT_DEG = -38.925138950141594


# -----------------------------------------------------------------------------
# A second object, same file, hand-picked because it exercises every sentinel
# value found in this session (see chat report): -9 (HOSTGAL_PHOTOZ/_ERR,
# HOSTGAL_SNSEP), -9999 (HOSTGAL_ELLIPTICITY), and 999 (HOSTGAL_MAG_r) all at
# once, none of which should survive into StaticContext -- each must become
# None, not the raw sentinel number.
# -----------------------------------------------------------------------------

SENTINEL_SNID = 11903027
SENTINEL_HEAD_INDEX = 335  # 0-indexed position within HEAD file 0001

RAW_SENTINEL_HEAD = {
    "SNID": 11903027,
    "RA": 42.08456463728632,
    "DEC": -23.45644257133382,
    "MWEBV": 0.020194901153445244,
    "HOSTGAL_PHOTOZ": -9.0,
    "HOSTGAL_PHOTOZ_ERR": -9.0,
    "HOSTGAL_SNSEP": -9.0,
    "HOSTGAL_ELLIPTICITY": -9999.0,
    "HOSTGAL_MAG_r": 999.0,
    "SIM_TYPE_INDEX": 10,
}

# Every one of these must map to None in StaticContext -- CLAUDE.md: "None
# for genuinely absent fields, never a sentinel."
EXPECTED_SENTINEL_STATIC = {
    "host_photoz": None,
    "host_photoz_err": None,
    "host_offset_arcsec": None,
    "host_ellipticity": None,
    "host_mag_r": None,
}

EXPECTED_SENTINEL_GAL_LAT_DEG = -63.38295828197109
