"""Golden test for src/readers/avro_ztf.py, per CLAUDE.md §12: hand-verify one
object end to end against a real, live-polled Fink alert, commit it as a
fixture, write the reader against the test. See
tests/fixtures/golden_avro_ztf_alert.py for the full derivation.
"""

import gzip
import io

import numpy as np
import pytest
from astropy.io import fits

from src.readers import avro_ztf
from src.schema import Survey
from tests.fixtures import golden_avro_ztf_alert as golden


def _make_alert(candidate=None, prv_candidates=None, object_id=None, with_cutouts=False):
    alert = {
        "objectId": object_id or golden.OBJECT_ID,
        "candidate": dict(candidate if candidate is not None else golden.CANDIDATE),
        "prv_candidates": prv_candidates if prv_candidates is not None else [dict(r) for r in golden.PRV_CANDIDATES],
    }
    if with_cutouts:
        # CanonicalAlert.validate() (schema.py, FROZEN) hard-enforces cutouts
        # shaped exactly (3, 63, 63) -- the real ZTF stamp size (verified on
        # the golden alert) -- so the synthetic stamp must match that size.
        stamp = _make_gzipped_fits_stamp(np.arange(63 * 63, dtype=np.float32).reshape(63, 63))
        alert["cutoutScience"] = {"fileName": "sci.fits", "stampData": stamp}
        alert["cutoutTemplate"] = {"fileName": "ref.fits", "stampData": stamp}
        alert["cutoutDifference"] = {"fileName": "diff.fits", "stampData": stamp}
    return alert


def _make_gzipped_fits_stamp(data: np.ndarray) -> bytes:
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    buf = io.BytesIO()
    hdu.writeto(buf)
    return gzip.compress(buf.getvalue())


@pytest.fixture(scope="module")
def canonical():
    return avro_ztf.ztf_alert_to_canonical(_make_alert())


class TestGoldenObjectObservations:
    def test_object_id(self, canonical):
        assert canonical.object_id == golden.OBJECT_ID

    def test_observation_count(self, canonical):
        assert len(canonical.observations) == golden.NOBS

    def test_mjd_sorted_ascending_matches_expected(self, canonical):
        mjds = [o.mjd for o in canonical.observations]
        assert mjds == pytest.approx(golden.EXPECTED_MJD)
        assert mjds == sorted(mjds)

    def test_flux_and_flux_err_converted_to_fluxcal_27_5(self, canonical):
        """The highest-risk item: magpsf/magzpsci -> FLUXCAL zp=27.5, both
        for real detections and for non-detection upper limits."""
        flux = [o.flux for o in canonical.observations]
        flux_err = [o.flux_err for o in canonical.observations]
        assert flux == pytest.approx(golden.EXPECTED_FLUX, abs=1e-6)
        assert flux_err == pytest.approx(golden.EXPECTED_FLUX_ERR, rel=1e-6)

    def test_is_detection_matches_expected(self, canonical):
        assert [o.is_detection for o in canonical.observations] == golden.EXPECTED_IS_DETECTION

    def test_non_detection_rows_have_zero_flux(self, canonical):
        """Every non-detection row's flux is exactly 0.0 (no measured
        excess) -- never a nonzero placeholder, and never dropped."""
        non_det = [o for o in canonical.observations if not o.is_detection]
        assert len(non_det) == golden.EXPECTED_IS_DETECTION.count(False)
        assert all(o.flux == 0.0 for o in non_det)
        assert all(o.flux_err > 0 for o in non_det)

    def test_band_mapped_to_lam_eff_never_fid_index(self, canonical):
        from src.schema import LAM_EFF

        fid_to_key = {1: "ztf_g", 2: "ztf_r", 3: "ztf_i"}
        by_mjd = sorted(zip(golden.EXPECTED_MJD, golden.EXPECTED_FID))
        expected_lam = [LAM_EFF[fid_to_key[fid]] for _, fid in by_mjd]
        assert [o.lam_eff for o in canonical.observations] == pytest.approx(expected_lam)

    def test_alert_passes_schema_validate(self, canonical):
        canonical.validate()


class TestGoldenObjectMetadata:
    def test_survey_is_ztf(self, canonical):
        assert canonical.survey == Survey.ZTF

    def test_class_id_always_none(self, canonical):
        assert canonical.class_id is None

    def test_static_context_matches_expected(self, canonical):
        static = canonical.static
        assert static.host_offset_arcsec == pytest.approx(golden.EXPECTED_HOST_OFFSET_ARCSEC)
        assert static.host_mag_r == pytest.approx(golden.EXPECTED_HOST_MAG_R)
        assert static.real_bogus_score == pytest.approx(golden.EXPECTED_REAL_BOGUS_SCORE)
        assert static.host_photoz is golden.EXPECTED_HOST_PHOTOZ
        assert static.host_photoz_err is golden.EXPECTED_HOST_PHOTOZ_ERR
        assert static.host_ellipticity is golden.EXPECTED_HOST_ELLIPTICITY
        assert static.mw_ebv == pytest.approx(golden.EXPECTED_MW_EBV, rel=1e-3)
        assert static.gal_lat == pytest.approx(golden.EXPECTED_GAL_LAT_DEG, rel=1e-3)

    def test_cutouts_none_when_absent(self, canonical):
        assert canonical.cutouts is None


class TestCutoutDecoding:
    def test_decode_round_trip(self):
        data = np.arange(9, dtype=np.float32).reshape(3, 3) * 1.5
        stamp = _make_gzipped_fits_stamp(data)
        decoded = avro_ztf.decode_cutout_stamp(stamp)
        assert decoded.shape == (3, 3)
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, data)

    def test_alert_with_cutouts_stacks_sci_ref_diff(self):
        canonical = avro_ztf.ztf_alert_to_canonical(_make_alert(with_cutouts=True))
        assert canonical.cutouts is not None
        assert canonical.cutouts.shape == (3, 63, 63)

    def test_malformed_cutout_falls_back_to_none(self):
        alert = _make_alert(with_cutouts=True)
        alert["cutoutScience"]["stampData"] = b"not a real gzip stream"
        canonical = avro_ztf.ztf_alert_to_canonical(alert)
        assert canonical.cutouts is None

    def test_undersized_ccd_edge_cutout_degrades_to_none_not_fatal(self):
        """Verified live (Day 13): a real ZTF object near a CCD-quadrant
        edge gets a stamp smaller than 63x63 (observed down to (63, 42)) --
        this must degrade the cutout to None, never discard the whole
        alert's light curve over it (see avro_ztf.py's _decode_cutouts
        docstring for the live bug this test pins down)."""
        undersized = _make_gzipped_fits_stamp(np.zeros((63, 42), dtype=np.float32))
        alert = _make_alert(with_cutouts=True)
        alert["cutoutScience"]["stampData"] = undersized
        alert["cutoutTemplate"]["stampData"] = undersized
        alert["cutoutDifference"]["stampData"] = undersized
        canonical = avro_ztf.ztf_alert_to_canonical(alert)
        assert canonical.cutouts is None
        assert len(canonical.observations) == golden.NOBS  # light curve intact


class TestNonDetectionAndSkipping:
    def test_missing_magzpsci_row_is_skipped_not_fatal(self):
        prv = [dict(r) for r in golden.PRV_CANDIDATES]
        prv[0]["magzpsci"] = None
        canonical = avro_ztf.ztf_alert_to_canonical(_make_alert(prv_candidates=prv))
        assert len(canonical.observations) == golden.NOBS - 1

    def test_zero_sigmapsf_detection_row_is_skipped_not_fatal(self):
        prv = [dict(r) for r in golden.PRV_CANDIDATES]
        assert prv[7]["magpsf"] is not None  # the row this test mutates is a real detection
        prv[7]["sigmapsf"] = 0.0
        canonical = avro_ztf.ztf_alert_to_canonical(_make_alert(prv_candidates=prv))
        assert len(canonical.observations) == golden.NOBS - 1

    def test_unusable_triggering_candidate_raises(self):
        bad_candidate = dict(golden.CANDIDATE)
        bad_candidate["magzpsci"] = None
        with pytest.raises(ValueError, match="triggering candidate unusable"):
            avro_ztf.ztf_alert_to_canonical(_make_alert(candidate=bad_candidate))

    def test_isdiffpos_f_gives_negative_flux(self):
        candidate = dict(golden.CANDIDATE)
        candidate["isdiffpos"] = "f"
        canonical = avro_ztf.ztf_alert_to_canonical(_make_alert(candidate=candidate, prv_candidates=[]))
        triggering = max(canonical.observations, key=lambda o: o.mjd)
        assert triggering.flux < 0
