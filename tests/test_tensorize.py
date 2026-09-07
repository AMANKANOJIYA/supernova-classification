"""Tests for src/features/tensorize.py. Hand-constructed CanonicalAlert
fixtures only — no real data, no network, per CLAUDE.md §12 (golden tests)
and this session's explicit scope.
"""

import math

import pytest
import torch

from src.features import tensorize
from src.schema import LAM_EFF, CanonicalAlert, Observation, StaticContext, Survey


def _static() -> StaticContext:
    return StaticContext(
        host_photoz=0.05,
        host_photoz_err=0.01,
        host_offset_arcsec=0.3,
        host_ellipticity=0.2,
        host_mag_r=18.5,
        mw_ebv=0.02,
        gal_lat=45.0,
        real_bogus_score=0.9,
    )


def _obs(mjd, flux=100.0, flux_err=5.0, lam_eff=LAM_EFF["ztf_r"], is_detection=True) -> Observation:
    return Observation(
        mjd=mjd, lam_eff=lam_eff, flux=flux, flux_err=flux_err, is_detection=is_detection
    )


def _alert(observations, object_id="ZTF21abc", class_id=2222) -> CanonicalAlert:
    alert = CanonicalAlert(
        object_id=object_id,
        survey=Survey.ZTF,
        observations=observations,
        static=_static(),
        cutouts=None,
        class_id=class_id,
    )
    alert.validate()  # every fixture must itself be schema-valid
    return alert


# ---------------------------------------------------------------------------
# _signed_log1p itself: the exact definition matters (sign extraction, not a
# plain log1p that would return NaN on a negative argument).
# ---------------------------------------------------------------------------


class TestSignedLog1p:
    def test_negative_argument_does_not_produce_nan(self):
        x = torch.tensor([-10.0])
        result = tensorize._signed_log1p(x)
        assert torch.isfinite(result).all()

    def test_matches_sign_times_log1p_of_absolute_value(self):
        for value in (-1000.0, -1.0, -0.001, 0.0, 0.001, 1.0, 1000.0):
            x = torch.tensor([value])
            expected = math.copysign(1.0, value) * math.log1p(abs(value)) if value != 0 else 0.0
            assert tensorize._signed_log1p(x).item() == pytest.approx(expected, abs=1e-5)

    def test_is_odd_sign_preserving(self):
        x = torch.tensor([3.7, 250.0, 0.02])
        assert torch.allclose(tensorize._signed_log1p(-x), -tensorize._signed_log1p(x))

    def test_a_plain_log1p_would_have_produced_nan_here(self):
        """Documents exactly the failure this function exists to avoid."""
        x = torch.tensor([-10.0])
        assert torch.isnan(torch.log1p(x)).all()
        assert not torch.isnan(tensorize._signed_log1p(x)).any()


# ---------------------------------------------------------------------------
# Required fixtures
# ---------------------------------------------------------------------------


class TestSingleObservation:
    def test_shape_and_no_crash(self):
        alert = _alert([_obs(60000.0)])
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape == (1, tensorize.NUM_FEATURES)

    def test_delta_t_is_zero(self):
        alert = _alert([_obs(60000.0)])
        result = tensorize.tensorize_alert(alert)
        assert result.features[0, 0].item() == pytest.approx(0.0)

    def test_class_id_and_object_id_passthrough(self):
        alert = _alert([_obs(60000.0)], object_id="ZTF99xyz", class_id=2223)
        result = tensorize.tensorize_alert(alert)
        assert result.object_id == "ZTF99xyz"
        assert result.class_id == 2223

    def test_unlabelled_alert_has_none_class_id(self):
        alert = _alert([_obs(60000.0)], class_id=None)
        result = tensorize.tensorize_alert(alert)
        assert result.class_id is None


class TestSeasonalGap:
    def test_forty_day_gap_reflected_in_delta_t(self):
        alert = _alert(
            [
                _obs(60000.0),
                _obs(60005.0),
                _obs(60045.0),  # 40-day gap after the previous point
                _obs(60050.0),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        delta_t = result.features[:, 0]
        assert delta_t.tolist() == pytest.approx([0.0, 5.0, 45.0, 50.0])
        gap = (delta_t[2] - delta_t[1]).item()
        assert gap == pytest.approx(40.0)

    def test_no_nan_or_inf_across_the_gap(self):
        alert = _alert([_obs(60000.0), _obs(60045.0)])
        result = tensorize.tensorize_alert(alert)
        assert torch.isfinite(result.features).all()


class TestAllNegativeFlux:
    def test_no_nan_or_inf(self):
        alert = _alert(
            [
                _obs(60000.0, flux=-50.0, flux_err=5.0),
                _obs(60001.0, flux=-3.0, flux_err=1.0),
                _obs(60002.0, flux=-0.5, flux_err=2.0),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert torch.isfinite(result.features).all()

    def test_flux_norm_is_negative(self):
        alert = _alert([_obs(60000.0, flux=-50.0, flux_err=5.0)])
        result = tensorize.tensorize_alert(alert)
        flux_norm = result.features[0, 1].item()
        assert flux_norm < 0.0
        # signed_log1p(-10) = -log1p(10)
        assert flux_norm == pytest.approx(-math.log1p(10.0), abs=1e-5)

    def test_flux_abs_norm_is_negative_and_matches_formula(self):
        alert = _alert([_obs(60000.0, flux=-50.0, flux_err=5.0)])
        result = tensorize.tensorize_alert(alert)
        flux_abs_norm = result.features[0, 2].item()
        assert flux_abs_norm < 0.0
        assert flux_abs_norm == pytest.approx(-math.log1p(50.0), abs=1e-5)


class TestOneBandEntirelyAbsent:
    def test_only_g_band_present_no_crash(self):
        alert = _alert(
            [
                _obs(60000.0, lam_eff=LAM_EFF["ztf_g"]),
                _obs(60001.0, lam_eff=LAM_EFF["ztf_g"]),
                _obs(60002.0, lam_eff=LAM_EFF["ztf_g"]),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape == (3, tensorize.NUM_FEATURES)
        wavelength_col = result.features[:, 4]
        expected = tensorize._wavelength_norm(LAM_EFF["ztf_g"])
        assert wavelength_col.tolist() == pytest.approx([expected] * 3, abs=1e-5)


class TestNearZeroAmplitude:
    def test_no_nan_or_inf_tiny_flux(self):
        alert = _alert(
            [
                _obs(60000.0, flux=1e-8, flux_err=5.0),
                _obs(60001.0, flux=-1e-8, flux_err=5.0),
                _obs(60002.0, flux=0.0, flux_err=5.0),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert torch.isfinite(result.features).all()

    def test_no_nan_or_inf_tiny_flux_err(self):
        """The dangerous direction: a very small (but schema-valid, > 0)
        flux_err could blow up flux/flux_err — signed_log1p must still keep
        this finite."""
        alert = _alert(
            [
                _obs(60000.0, flux=1.0, flux_err=1e-6),
                _obs(60001.0, flux=-1.0, flux_err=1e-6),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert torch.isfinite(result.features).all()

    def test_zero_flux_gives_zero_flux_norm(self):
        alert = _alert([_obs(60000.0, flux=0.0, flux_err=5.0)])
        result = tensorize.tensorize_alert(alert)
        assert result.features[0, 1].item() == pytest.approx(0.0)
        assert result.features[0, 2].item() == pytest.approx(0.0)  # flux_abs_norm too


class TestEmptyOrUndetectedRaise:
    def test_empty_observations_raises(self):
        alert = CanonicalAlert(
            object_id="empty",
            survey=Survey.ZTF,
            observations=[],
            static=_static(),
            cutouts=None,
            class_id=None,
        )
        with pytest.raises(ValueError, match="no observations"):
            tensorize.tensorize_alert(alert)

    def test_no_detections_raises(self):
        alert = _alert([_obs(60000.0, is_detection=False), _obs(60001.0, is_detection=False)])
        with pytest.raises(ValueError, match="no detections"):
            tensorize.tensorize_alert(alert)


# ---------------------------------------------------------------------------
# Order invariance for equal-mjd observations (same bug class as Day 1's
# colour_at_peak: picking/aggregating in a way that depends on array order).
# ---------------------------------------------------------------------------


class TestSameMjdOrderInvariance:
    def test_swapping_two_equal_mjd_rows_swaps_only_their_own_output_rows(self):
        obs_a = _obs(60000.0, flux=10.0, lam_eff=LAM_EFF["ztf_g"])
        obs_b = _obs(60000.0, flux=20.0, lam_eff=LAM_EFF["ztf_r"])  # same mjd, different band
        obs_c = _obs(60001.0, flux=30.0)

        forward = tensorize.tensorize_alert(_alert([obs_a, obs_b, obs_c], object_id="fwd"))
        swapped = tensorize.tensorize_alert(_alert([obs_b, obs_a, obs_c], object_id="swp"))

        assert torch.allclose(forward.features[0], swapped.features[1])
        assert torch.allclose(forward.features[1], swapped.features[0])
        assert torch.allclose(forward.features[2], swapped.features[2])

    def test_first_detection_mjd_is_order_independent(self):
        """Two same-mjd detections at the very start: Δt=0 for both,
        regardless of which one appears first in the input list."""
        obs_a = _obs(60000.0, lam_eff=LAM_EFF["ztf_g"])
        obs_b = _obs(60000.0, lam_eff=LAM_EFF["ztf_r"])

        forward = tensorize.tensorize_alert(_alert([obs_a, obs_b], object_id="fwd"))
        swapped = tensorize.tensorize_alert(_alert([obs_b, obs_a], object_id="swp"))

        assert forward.features[:, 0].tolist() == pytest.approx([0.0, 0.0])
        assert swapped.features[:, 0].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# The test that matters: truncation invariance. If this fails, normalization
# leaks — see the module docstring in src/features/tensorize.py.
# ---------------------------------------------------------------------------


class TestWindowing:
    """Day-4 addendum: train/serve distribution mismatch fix. Default
    windowing truncates to [first_detection_mjd - 30d, last_detection_mjd]."""

    def test_forced_photometry_before_window_is_dropped_by_default(self):
        alert = _alert(
            [
                _obs(59900.0, is_detection=False),  # 100 days before first detection
                _obs(60000.0, is_detection=True),
                _obs(60005.0, is_detection=True),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape[0] == 2

    def test_forced_photometry_after_last_detection_is_dropped_by_default(self):
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60005.0, is_detection=True),
                _obs(60500.0, is_detection=False),  # far post-fade baseline
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape[0] == 2

    def test_forced_photometry_inside_window_is_kept(self):
        alert = _alert(
            [
                _obs(59980.0, is_detection=False),  # 20 days before first detection
                _obs(60000.0, is_detection=True),
                _obs(60005.0, is_detection=True),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape[0] == 3

    def test_window_boundaries_are_inclusive(self):
        alert = _alert(
            [
                _obs(59970.0, is_detection=False),  # exactly 30 days before first detection
                _obs(60000.0, is_detection=True),
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape[0] == 2

    def test_windowing_never_drops_a_detection(self):
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60100.0, is_detection=True),  # 100 days after first detection
            ]
        )
        result = tensorize.tensorize_alert(alert)
        assert result.features.shape[0] == 2

    def test_pre_detection_window_days_none_disables_windowing(self):
        alert = _alert(
            [
                _obs(59500.0, is_detection=False),  # 500 days before first detection
                _obs(60000.0, is_detection=True),
                _obs(60600.0, is_detection=False),  # 600 days after last detection
            ]
        )
        result = tensorize.tensorize_alert(alert, pre_detection_window_days=None)
        assert result.features.shape[0] == 3

    def test_custom_window_width_is_respected(self):
        alert = _alert(
            [
                _obs(59950.0, is_detection=False),  # 50 days before first detection
                _obs(60000.0, is_detection=True),
            ]
        )
        default_windowed = tensorize.tensorize_alert(alert)
        assert default_windowed.features.shape[0] == 1  # dropped at the default 30d

        wide_windowed = tensorize.tensorize_alert(alert, pre_detection_window_days=60.0)
        assert wide_windowed.features.shape[0] == 2  # kept at 60d


class TestMaxSpanDays:
    """Day-12 addendum: right-edge cap, independent of `pre_detection_window_days`.
    Window becomes `[first_detection - pre_detection_window_days,
    min(last_detection, first_detection + max_span_days)]`."""

    def test_default_none_preserves_prior_behaviour(self):
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60800.0, is_detection=True),  # 800 days after first detection
            ]
        )
        with_default = tensorize.tensorize_alert(alert)
        explicit_none = tensorize.tensorize_alert(alert, max_span_days=None)
        assert torch.allclose(with_default.features, explicit_none.features)
        assert with_default.features.shape[0] == 2  # nothing capped

    def test_late_detection_beyond_span_is_dropped(self):
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60300.0, is_detection=True),  # 300 days after first detection
            ]
        )
        result = tensorize.tensorize_alert(alert, max_span_days=200.0)
        assert result.features.shape[0] == 1  # the 300-day-later detection is capped away

    def test_span_boundary_is_inclusive(self):
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60200.0, is_detection=True),  # exactly at the 200-day cap
            ]
        )
        result = tensorize.tensorize_alert(alert, max_span_days=200.0)
        assert result.features.shape[0] == 2

    def test_span_never_widens_past_last_detection(self):
        """A generous max_span_days must not pull in forced photometry past
        the object's own last detection -- min(last_detection, ...) in the
        semantics, not a plain additive cap."""
        alert = _alert(
            [
                _obs(60000.0, is_detection=True),
                _obs(60010.0, is_detection=True),
                _obs(60500.0, is_detection=False),  # far post-fade baseline
            ]
        )
        result = tensorize.tensorize_alert(alert, max_span_days=10_000.0)
        assert result.features.shape[0] == 2  # forced photometry at 60500 still excluded

    def test_composes_with_pre_detection_window_days_none(self):
        """max_span_days must still cap the right edge even when the left
        edge is fully unbounded."""
        alert = _alert(
            [
                _obs(59000.0, is_detection=False),  # 1000 days before first detection
                _obs(60000.0, is_detection=True),
                _obs(60300.0, is_detection=True),  # 300 days after first detection
            ]
        )
        result = tensorize.tensorize_alert(
            alert, pre_detection_window_days=None, max_span_days=200.0
        )
        assert result.features.shape[0] == 2  # early point kept, late detection capped away

    def test_both_none_is_fully_unwindowed(self):
        alert = _alert(
            [
                _obs(59000.0, is_detection=False),
                _obs(60000.0, is_detection=True),
                _obs(60900.0, is_detection=False),
            ]
        )
        result = tensorize.tensorize_alert(
            alert, pre_detection_window_days=None, max_span_days=None
        )
        assert result.features.shape[0] == 3


class TestTruncationInvariance:
    def test_truncated_tensorization_matches_prefix_of_full_tensorization(self):
        observations = [
            _obs(60000.0, flux=5.0, flux_err=2.0, lam_eff=LAM_EFF["ztf_g"]),
            _obs(60003.0, flux=80.0, flux_err=4.0, lam_eff=LAM_EFF["ztf_r"]),
            _obs(60010.0, flux=250.0, flux_err=6.0, lam_eff=LAM_EFF["ztf_g"]),  # the eventual peak
            _obs(60020.0, flux=120.0, flux_err=5.0, lam_eff=LAM_EFF["ztf_r"]),
            _obs(60035.0, flux=15.0, flux_err=3.0, lam_eff=LAM_EFF["ztf_g"]),
        ]
        full = tensorize.tensorize_alert(_alert(observations, object_id="full"))

        for t in range(1, len(observations) + 1):
            truncated = tensorize.tensorize_alert(
                _alert(observations[:t], object_id=f"truncated_{t}")
            )
            assert torch.allclose(truncated.features, full.features[:t]), (
                f"tensorization diverged at truncation t={t}: normalization is "
                "leaking future information"
            )

    def test_truncation_invariance_holds_even_when_peak_is_excluded(self):
        """Extra emphasis on the case the docstring calls out: an early
        truncation that excludes the eventual peak entirely must produce
        features identical to what a live, peak-blind stream would compute
        — the whole reason peak-relative normalization is rejected."""
        observations = [
            _obs(60000.0, flux=5.0, flux_err=2.0),
            _obs(60003.0, flux=8.0, flux_err=2.0),
            _obs(60010.0, flux=500.0, flux_err=10.0),  # peak, excluded below
        ]
        full = tensorize.tensorize_alert(_alert(observations, object_id="full"))
        early = tensorize.tensorize_alert(_alert(observations[:2], object_id="early"))
        assert torch.allclose(early.features, full.features[:2])
