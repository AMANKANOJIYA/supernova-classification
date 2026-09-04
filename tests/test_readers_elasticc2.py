"""Golden test for src/readers/elasticc2.py, per CLAUDE.md §12: hand-verify
one object end to end against the source docs, commit it as a fixture, then
write the reader against the test — not the other way around.

All tests here require the real ELAsTiCC2 SNIa-SALT3 files on disk
(data/ is gitignored) and are skipped otherwise.
"""

from pathlib import Path

import pytest

from src import taxonomy
from src.schema import Survey
from tests.fixtures import golden_elasticc2_object as golden

_MODEL_DIR = Path("data/raw/elasticc2") / golden.MODEL_DIR
_HEAD_PATH = _MODEL_DIR / golden.HEAD_FILE
_PHOT_PATH = _MODEL_DIR / golden.HEAD_FILE.replace("_HEAD", "_PHOT")
_GENTYPE_CSV = Path("data/raw/elasticc2/elasticc2_gentypeofclassid.csv")

pytestmark = pytest.mark.skipif(
    not (_HEAD_PATH.exists() and _PHOT_PATH.exists() and _GENTYPE_CSV.exists()),
    reason="real ELAsTiCC2 SNIa-SALT3 HEAD/PHOT files or gentype CSV not present on disk",
)


@pytest.fixture(scope="module")
def gentype_to_classid():
    return taxonomy.load_gentype_of_classid(str(_GENTYPE_CSV))


@pytest.fixture(scope="module")
def alerts(gentype_to_classid):
    from src.readers import elasticc2

    alerts, stats = elasticc2.read_head_phot_file(
        str(_HEAD_PATH), str(_PHOT_PATH), gentype_to_classid
    )
    return {a.object_id: a for a in alerts}, stats


class TestGoldenObjectObservations:
    def test_object_present(self, alerts):
        by_id, _ = alerts
        assert str(golden.SNID) in by_id

    def test_observation_count_matches_nobs(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        assert len(alert.observations) == golden.NOBS

    def test_mjd_matches_raw_phot_rows_in_order(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        expected_mjd = [row[0] for row in golden.RAW_PHOT_ROWS]
        assert [obs.mjd for obs in alert.observations] == pytest.approx(expected_mjd)

    def test_flux_and_flux_err_are_fluxcal_unconverted(self, alerts):
        """CLAUDE.md §3: flux must be on the FLUXCAL/zeropoint-27.5
        convention. A_FORMAT.TXT confirms ELAsTiCC2 FLUXCAL already IS on
        that convention (mag = 27.5 - 2.5*log10(FLUXCAL)) -- so the reader
        must pass FLUXCAL/FLUXCALERR through unconverted, not rescale them."""
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        expected_flux = [row[2] for row in golden.RAW_PHOT_ROWS]
        expected_flux_err = [row[3] for row in golden.RAW_PHOT_ROWS]
        assert [obs.flux for obs in alert.observations] == pytest.approx(expected_flux)
        assert [obs.flux_err for obs in alert.observations] == pytest.approx(expected_flux_err)

    def test_is_detection_matches_photflag_bit_4096(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        assert [obs.is_detection for obs in alert.observations] == golden.EXPECTED_IS_DETECTION

    def test_band_mapped_to_correct_lam_eff(self, alerts):
        from src.schema import LAM_EFF

        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        band_to_key = {"u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "Y": "y"}
        expected_lam_eff = [LAM_EFF[band_to_key[row[1]]] for row in golden.RAW_PHOT_ROWS]
        assert [obs.lam_eff for obs in alert.observations] == pytest.approx(expected_lam_eff)

    def test_observations_sorted_ascending_by_mjd(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        mjds = [obs.mjd for obs in alert.observations]
        assert mjds == sorted(mjds)

    def test_alert_passes_schema_validate(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        alert.validate()  # must not raise


class TestGoldenObjectLabelAndMetadata:
    def test_class_id_derived_from_gentype_not_directory_name(self, alerts):
        """SIM_TYPE_INDEX=10 -> gentype 10 -> classid 2222 via the CSV
        mapping. The directory is named SNIa-SALT3, but that name plays no
        role in this assertion's derivation."""
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        assert alert.class_id == golden.EXPECTED_CLASS_ID

    def test_survey_is_lsst(self, alerts):
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        assert alert.survey == Survey.LSST

    def test_cutouts_is_none(self, alerts):
        """CLAUDE.md §5.1: ELAsTiCC2 has no pixel data. Always None, not a
        bug."""
        by_id, _ = alerts
        alert = by_id[str(golden.SNID)]
        assert alert.cutouts is None

    def test_static_context_matches_head_fields(self, alerts):
        by_id, _ = alerts
        static = by_id[str(golden.SNID)].static
        assert static.host_photoz == pytest.approx(golden.RAW_HEAD["HOSTGAL_PHOTOZ"])
        assert static.host_photoz_err == pytest.approx(golden.RAW_HEAD["HOSTGAL_PHOTOZ_ERR"])
        assert static.host_offset_arcsec == pytest.approx(golden.RAW_HEAD["HOSTGAL_SNSEP"])
        assert static.host_ellipticity == pytest.approx(golden.RAW_HEAD["HOSTGAL_ELLIPTICITY"])
        assert static.host_mag_r == pytest.approx(golden.RAW_HEAD["HOSTGAL_MAG_r"])
        assert static.mw_ebv == pytest.approx(golden.RAW_HEAD["MWEBV"])

    def test_gal_lat_computed_from_ra_dec(self, alerts):
        by_id, _ = alerts
        static = by_id[str(golden.SNID)].static
        assert static.gal_lat == pytest.approx(golden.EXPECTED_GAL_LAT_DEG, abs=1e-6)

    def test_real_bogus_score_is_none(self, alerts):
        """No image-based real/bogus classifier exists for pure simulation
        -- CLAUDE.md §5.1."""
        by_id, _ = alerts
        static = by_id[str(golden.SNID)].static
        assert static.real_bogus_score is None


class TestSentinelHandling:
    """The second golden object: every sentinel value found this session
    (-9, -9999, 999) must become None, never pass through as a number."""

    def test_sentinel_object_present(self, alerts):
        by_id, _ = alerts
        assert str(golden.SENTINEL_SNID) in by_id

    def test_all_sentinel_fields_become_none(self, alerts):
        by_id, _ = alerts
        static = by_id[str(golden.SENTINEL_SNID)].static
        for field_name, expected in golden.EXPECTED_SENTINEL_STATIC.items():
            assert getattr(static, field_name) is expected, (
                f"{field_name} should be None (sentinel), got {getattr(static, field_name)!r}"
            )

    def test_non_sentinel_fields_still_populated(self, alerts):
        """mw_ebv and gal_lat are not sentinel-coded fields for this object
        -- they must survive even though the host fields don't."""
        by_id, _ = alerts
        static = by_id[str(golden.SENTINEL_SNID)].static
        assert static.mw_ebv == pytest.approx(golden.RAW_SENTINEL_HEAD["MWEBV"])
        assert static.gal_lat == pytest.approx(golden.EXPECTED_SENTINEL_GAL_LAT_DEG, abs=1e-6)

    def test_sentinel_object_class_id(self, alerts):
        by_id, _ = alerts
        # SIM_TYPE_INDEX=10 for this object too -> same classid 2222.
        assert by_id[str(golden.SENTINEL_SNID)].class_id == 2222


class TestReadStats:
    def test_reads_all_objects_in_file(self, alerts):
        _, stats = alerts
        assert stats.n_objects_read == 4247  # HEAD file 0001 object count, verified by hand

    def test_no_unexpected_discards_on_clean_file(self, alerts):
        _, stats = alerts
        assert stats.n_objects_discarded == 0
