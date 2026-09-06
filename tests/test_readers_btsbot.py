"""Golden test for src/readers/btsbot.py, per CLAUDE.md §12: hand-verify one
object end to end against the source docs, commit it as a fixture, confirm
the test fails before the reader exists, then write the reader against it.

Uses a small in-memory CSV built from real values pulled off the actual
downloaded metadata_v10 file (data/ is gitignored, so the real file itself
isn't committed) — see tests/fixtures/golden_btsbot_object.py for the full
derivation and the independent flux/mw_ebv/gal_lat computation.
"""

import csv
import io

import pytest

from src.schema import LAM_EFF, Survey
from tests.fixtures import golden_btsbot_object as golden

_COLUMNS = [
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

# Real per-row ra/dec/distpsnr1/drb for ZTF19abiszoe (pulled off the real
# file this session; sgscore1/srmag1 are exactly constant for this object).
_OBJ1_RA = [224.738204, 224.7381992, 224.7381919, 224.7381896, 224.7382353, 224.7381963, 224.7381972, 224.7381897]
_OBJ1_DEC = [20.0529498, 20.052942, 20.0529237, 20.0529309, 20.0528961, 20.0529224, 20.0529343, 20.0530004]
_OBJ1_DISTPSNR1 = [10.025057792663574, 10.039207458496094, 10.059367179870604, 10.068832397460938, 9.906900405883787, 10.044217109680176, 10.04401969909668, 10.08835506439209]
_OBJ1_SGSCORE1 = 0.1495240032672882
_OBJ1_SRMAG1 = 16.849899291992188
_OBJ1_DRB = [1.0, 0.9993834495544434, 0.999997854232788, 0.9999843835830688, 0.9999847412109376, 0.999996781349182, 0.9998766183853148, 0.999688982963562]

_OBJ2_RA = [70.2723528, 70.2723654, 70.2723313]
_OBJ2_DEC = [73.6730848, 73.673137, 73.6730673]
_OBJ2_DISTPSNR1 = [1.1532946825027466, 1.1198748350143433, 1.154861330986023]
_OBJ2_SGSCORE1 = 0.5


def _write_csv(path):
    rows = []
    # Object 1 rows deliberately written in a NON-jd-sorted order, to
    # exercise the reader's own sort — reversed relative to golden.RAW_ROWS.
    obj1_indices = list(reversed(range(len(golden.RAW_ROWS))))
    for i in obj1_indices:
        jd, fid, magpsf, sigmapsf, magzpsci, isdiffpos = golden.RAW_ROWS[i]
        rows.append(
            {
                "objectId": golden.OBJECT_ID,
                "jd": jd,
                "fid": fid,
                "magpsf": magpsf,
                "sigmapsf": sigmapsf,
                "magzpsci": magzpsci,
                "isdiffpos": isdiffpos,
                "ra": _OBJ1_RA[i],
                "dec": _OBJ1_DEC[i],
                "distpsnr1": _OBJ1_DISTPSNR1[i],
                "sgscore1": _OBJ1_SGSCORE1,
                "srmag1": _OBJ1_SRMAG1,
                "drb": _OBJ1_DRB[i],
            }
        )

    for i, (jd, fid, magpsf, sigmapsf, magzpsci, isdiffpos) in enumerate(golden.SENTINEL_RAW_ROWS):
        rows.append(
            {
                "objectId": golden.SENTINEL_OBJECT_ID,
                "jd": jd,
                "fid": fid,
                "magpsf": magpsf,
                "sigmapsf": sigmapsf,
                "magzpsci": magzpsci,
                "isdiffpos": isdiffpos,
                "ra": _OBJ2_RA[i],
                "dec": _OBJ2_DEC[i],
                "distpsnr1": _OBJ2_DISTPSNR1[i],
                "sgscore1": _OBJ2_SGSCORE1,
                "srmag1": -999.0,
                "drb": "",  # empty -> null, matches the real drb=null rows
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def alerts(tmp_path_factory):
    from src.readers import btsbot

    tmp_dir = tmp_path_factory.mktemp("btsbot_golden")
    csv_path = tmp_dir / "golden.csv"
    _write_csv(csv_path)
    alerts, stats = btsbot.read_candidates_csv(str(csv_path))
    return {a.object_id: a for a in alerts}, stats


class TestGoldenObjectObservations:
    def test_object_present(self, alerts):
        by_id, _ = alerts
        assert golden.OBJECT_ID in by_id

    def test_observation_count(self, alerts):
        by_id, _ = alerts
        assert len(by_id[golden.OBJECT_ID].observations) == golden.NOBS

    def test_mjd_sorted_ascending_matches_expected(self, alerts):
        """CSV rows were written in reverse-jd order — this also proves the
        reader sorts, not just passes rows through."""
        by_id, _ = alerts
        obs = by_id[golden.OBJECT_ID].observations
        assert [o.mjd for o in obs] == pytest.approx(golden.EXPECTED_MJD)
        mjds = [o.mjd for o in obs]
        assert mjds == sorted(mjds)

    def test_flux_and_flux_err_converted_to_fluxcal_27_5(self, alerts):
        """The highest-risk item this session: BTSbot's magpsf/magzpsci
        must be converted to SNANA FLUXCAL zp=27.5, not passed through as
        magnitudes and not left on the per-row magzpsci scale."""
        by_id, _ = alerts
        obs = by_id[golden.OBJECT_ID].observations
        assert [o.flux for o in obs] == pytest.approx(golden.EXPECTED_FLUX, rel=1e-6)
        assert [o.flux_err for o in obs] == pytest.approx(golden.EXPECTED_FLUX_ERR, rel=1e-6)

    def test_band_mapped_to_lam_eff_never_fid_index(self, alerts):
        by_id, _ = alerts
        obs = by_id[golden.OBJECT_ID].observations
        fid_to_key = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}
        expected = [LAM_EFF[fid_to_key[fid]] for fid in golden.EXPECTED_FID]
        # obs is mjd-sorted; golden.EXPECTED_FID is in golden.RAW_ROWS (jd-asc) order already
        assert [o.lam_eff for o in obs] == pytest.approx(expected)

    def test_is_detection_always_true(self, alerts):
        """Every BTSbot metadata_v10 row is a real ZTF alert-stream
        detection -- there is no forced-photometry/upper-limit row type in
        this CSV (see golden fixture docstring)."""
        by_id, _ = alerts
        obs = by_id[golden.OBJECT_ID].observations
        assert [o.is_detection for o in obs] == golden.EXPECTED_IS_DETECTION

    def test_alert_passes_schema_validate(self, alerts):
        by_id, _ = alerts
        by_id[golden.OBJECT_ID].validate()


class TestGoldenObjectMetadata:
    def test_class_id_always_none(self, alerts):
        """Day 8 chat decision: metadata_v10 has no spectroscopic subtype
        column at all (only a binary bright-transient/reject label) --
        the reader must never fabricate a class_id. Spectroscopic typing is
        a separate, later join (src/readers/bts_labels.py)."""
        by_id, _ = alerts
        assert by_id[golden.OBJECT_ID].class_id is None

    def test_survey_is_ztf(self, alerts):
        by_id, _ = alerts
        assert by_id[golden.OBJECT_ID].survey == Survey.ZTF

    def test_cutouts_is_none(self, alerts):
        by_id, _ = alerts
        assert by_id[golden.OBJECT_ID].cutouts is None

    def test_static_context_matches_expected(self, alerts):
        by_id, _ = alerts
        static = by_id[golden.OBJECT_ID].static
        assert static.host_offset_arcsec == pytest.approx(golden.EXPECTED_HOST_OFFSET_ARCSEC)
        assert static.host_mag_r == pytest.approx(golden.EXPECTED_HOST_MAG_R)
        assert static.real_bogus_score == pytest.approx(golden.EXPECTED_REAL_BOGUS_SCORE)
        assert static.host_photoz is golden.EXPECTED_HOST_PHOTOZ
        assert static.host_photoz_err is golden.EXPECTED_HOST_PHOTOZ_ERR
        assert static.host_ellipticity is golden.EXPECTED_HOST_ELLIPTICITY
        assert static.mw_ebv == pytest.approx(golden.EXPECTED_MW_EBV, rel=1e-3)
        assert static.gal_lat == pytest.approx(golden.EXPECTED_GAL_LAT_DEG)


class TestSentinelObject:
    """ZTF19aamfupk: srmag1=-999 (no PS1 r-band match) and drb=null
    (acai/drb not computed for this alert) on every row -- both must become
    None, never pass through as the raw sentinel."""

    def test_flux_and_flux_err(self, alerts):
        by_id, _ = alerts
        obs = by_id[golden.SENTINEL_OBJECT_ID].observations
        assert len(obs) == golden.SENTINEL_NOBS
        assert [o.mjd for o in obs] == pytest.approx(golden.SENTINEL_EXPECTED_MJD)
        assert [o.flux for o in obs] == pytest.approx(golden.SENTINEL_EXPECTED_FLUX, rel=1e-6)
        assert [o.flux_err for o in obs] == pytest.approx(
            golden.SENTINEL_EXPECTED_FLUX_ERR, rel=1e-6
        )

    def test_srmag1_sentinel_becomes_none(self, alerts):
        by_id, _ = alerts
        static = by_id[golden.SENTINEL_OBJECT_ID].static
        assert static.host_mag_r is golden.SENTINEL_EXPECTED_HOST_MAG_R

    def test_drb_null_becomes_none(self, alerts):
        by_id, _ = alerts
        static = by_id[golden.SENTINEL_OBJECT_ID].static
        assert static.real_bogus_score is golden.SENTINEL_EXPECTED_REAL_BOGUS_SCORE

    def test_distpsnr1_has_no_sentinel_passes_through(self, alerts):
        by_id, _ = alerts
        static = by_id[golden.SENTINEL_OBJECT_ID].static
        assert static.host_offset_arcsec == pytest.approx(
            golden.SENTINEL_EXPECTED_HOST_OFFSET_ARCSEC
        )

    def test_mw_ebv_and_gal_lat(self, alerts):
        by_id, _ = alerts
        static = by_id[golden.SENTINEL_OBJECT_ID].static
        assert static.mw_ebv == pytest.approx(golden.SENTINEL_EXPECTED_MW_EBV, rel=1e-3)
        assert static.gal_lat == pytest.approx(golden.SENTINEL_EXPECTED_GAL_LAT_DEG)
