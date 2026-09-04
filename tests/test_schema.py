"""Golden tests for the frozen canonical schema (CLAUDE.md §2.1, §3)."""

import numpy as np
import pytest

from src.schema import CanonicalAlert, LAM_EFF, Observation, StaticContext, Survey


def _static(**overrides) -> StaticContext:
    defaults = dict(
        host_photoz=0.05,
        host_photoz_err=0.01,
        host_offset_arcsec=0.3,
        host_ellipticity=0.2,
        host_mag_r=18.5,
        mw_ebv=0.02,
        gal_lat=45.0,
        real_bogus_score=0.9,
    )
    defaults.update(overrides)
    return StaticContext(**defaults)


def _obs(mjd: float, flux: float = 100.0, flux_err: float = 5.0) -> Observation:
    return Observation(
        mjd=mjd, lam_eff=LAM_EFF["ztf_r"], flux=flux, flux_err=flux_err, is_detection=True
    )


def _alert(observations: list[Observation], cutouts=None) -> CanonicalAlert:
    return CanonicalAlert(
        object_id="ZTF21abc",
        survey=Survey.ZTF,
        observations=observations,
        static=_static(),
        cutouts=cutouts,
        class_id=None,
    )


class TestValidateSortedMjd:
    def test_ascending_mjd_passes(self):
        alert = _alert([_obs(1.0), _obs(2.0), _obs(3.0)])
        alert.validate()  # must not raise

    def test_single_observation_passes(self):
        alert = _alert([_obs(1.0)])
        alert.validate()

    def test_empty_observations_passes(self):
        alert = _alert([])
        alert.validate()

    def test_out_of_order_mjd_raises(self):
        alert = _alert([_obs(2.0), _obs(1.0)])
        with pytest.raises(ValueError, match="mjd"):
            alert.validate()

    def test_out_of_order_later_in_list_raises(self):
        alert = _alert([_obs(1.0), _obs(2.0), _obs(3.0), _obs(2.5)])
        with pytest.raises(ValueError, match="mjd"):
            alert.validate()


class TestValidateFluxErr:
    def test_positive_flux_err_passes(self):
        alert = _alert([_obs(1.0, flux_err=0.001)])
        alert.validate()

    def test_zero_flux_err_raises(self):
        alert = _alert([_obs(1.0, flux_err=0.0)])
        with pytest.raises(ValueError, match="flux_err"):
            alert.validate()

    def test_negative_flux_err_raises(self):
        alert = _alert([_obs(1.0, flux_err=-1.0)])
        with pytest.raises(ValueError, match="flux_err"):
            alert.validate()


class TestNegativeFluxIsValid:
    """CLAUDE.md §2.2: negative flux is real, early-time signal — must never raise."""

    def test_negative_flux_does_not_raise(self):
        alert = _alert([_obs(1.0, flux=-42.0), _obs(2.0, flux=-3.5), _obs(3.0, flux=10.0)])
        alert.validate()  # must not raise

    def test_all_negative_flux_does_not_raise(self):
        alert = _alert([_obs(1.0, flux=-100.0)])
        alert.validate()


class TestValidateCutoutsShape:
    def test_none_cutouts_passes(self):
        alert = _alert([_obs(1.0)], cutouts=None)
        alert.validate()

    def test_correct_shape_passes(self):
        cutouts = np.zeros((3, 63, 63), dtype=np.float32)
        alert = _alert([_obs(1.0)], cutouts=cutouts)
        alert.validate()

    def test_wrong_spatial_shape_raises(self):
        cutouts = np.zeros((3, 32, 32), dtype=np.float32)
        alert = _alert([_obs(1.0)], cutouts=cutouts)
        with pytest.raises(ValueError, match="cutouts"):
            alert.validate()

    def test_wrong_channel_count_raises(self):
        cutouts = np.zeros((1, 63, 63), dtype=np.float32)
        alert = _alert([_obs(1.0)], cutouts=cutouts)
        with pytest.raises(ValueError, match="cutouts"):
            alert.validate()

    def test_extra_dimension_raises(self):
        cutouts = np.zeros((3, 63, 63, 1), dtype=np.float32)
        alert = _alert([_obs(1.0)], cutouts=cutouts)
        with pytest.raises(ValueError, match="cutouts"):
            alert.validate()


class TestImmutability:
    """Frozen dataclasses: mutation should raise, not silently succeed."""

    def test_observation_is_frozen(self):
        obs = _obs(1.0)
        with pytest.raises(Exception):
            obs.mjd = 2.0

    def test_canonical_alert_is_frozen(self):
        alert = _alert([_obs(1.0)])
        with pytest.raises(Exception):
            alert.object_id = "other"


class TestLamEff:
    def test_all_lsst_bands_present(self):
        for band in ("u", "g", "r", "i", "z", "y"):
            assert band in LAM_EFF

    def test_all_ztf_bands_present(self):
        for band in ("ztf_g", "ztf_r", "ztf_i"):
            assert band in LAM_EFF

    def test_wavelengths_are_nanometres_and_increasing_within_survey(self):
        lsst_order = [LAM_EFF[b] for b in ("u", "g", "r", "i", "z", "y")]
        assert lsst_order == sorted(lsst_order)
        ztf_order = [LAM_EFF[b] for b in ("ztf_g", "ztf_r", "ztf_i")]
        assert ztf_order == sorted(ztf_order)
