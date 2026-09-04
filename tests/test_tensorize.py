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
        wavelength_col = result.features[:, 3]
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
# Batching: padding, masking, collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_mask_matches_original_lengths(self):
        short = tensorize.tensorize_alert(_alert([_obs(60000.0), _obs(60001.0)], object_id="short"))
        long_ = tensorize.tensorize_alert(
            _alert([_obs(60000.0 + i) for i in range(5)], object_id="long")
        )
        batch = tensorize.collate_fn([short, long_])

        assert batch.features.shape == (2, 5, tensorize.NUM_FEATURES)
        assert batch.mask.shape == (2, 5)
        assert batch.mask[0].tolist() == [True, True, False, False, False]
        assert batch.mask[1].tolist() == [True, True, True, True, True]

    def test_object_id_and_class_id_order_preserved(self):
        a = tensorize.tensorize_alert(_alert([_obs(60000.0)], object_id="a", class_id=2222))
        b = tensorize.tensorize_alert(_alert([_obs(60000.0), _obs(60001.0)], object_id="b", class_id=None))
        batch = tensorize.collate_fn([a, b])
        assert batch.object_id == ["a", "b"]
        assert batch.class_id == [2222, None]

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            tensorize.collate_fn([])

    def test_padded_positions_do_not_influence_masked_computation(self):
        """Padded rows must be safely ignorable by any consumer that
        actually respects the mask — prove it by replacing padded feature
        values with garbage and showing a masked-mean is unaffected."""
        short = tensorize.tensorize_alert(_alert([_obs(60000.0), _obs(60001.0)], object_id="short"))
        long_ = tensorize.tensorize_alert(
            _alert([_obs(60000.0 + i) for i in range(5)], object_id="long")
        )
        batch = tensorize.collate_fn([short, long_])

        def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_f = mask.unsqueeze(-1).to(features.dtype)
            summed = (features * mask_f).sum(dim=1)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            return summed / counts

        baseline = masked_mean(batch.features, batch.mask)

        garbage = batch.features.clone()
        garbage_fill = torch.full_like(garbage, 1e9)
        garbage = torch.where(batch.mask.unsqueeze(-1), garbage, garbage_fill)
        # confirm the padded region really did change
        assert not torch.equal(garbage, batch.features)

        sabotaged = masked_mean(garbage, batch.mask)
        assert torch.allclose(baseline, sabotaged, atol=1e-5)

    def test_padded_positions_can_even_be_nan_without_affecting_masked_output(self):
        short = tensorize.tensorize_alert(_alert([_obs(60000.0), _obs(60001.0)], object_id="short"))
        long_ = tensorize.tensorize_alert(
            _alert([_obs(60000.0 + i) for i in range(5)], object_id="long")
        )
        batch = tensorize.collate_fn([short, long_])

        nan_features = torch.where(
            batch.mask.unsqueeze(-1), batch.features, torch.full_like(batch.features, float("nan"))
        )

        def masked_sum(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_f = mask.unsqueeze(-1).to(features.dtype)
            safe = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
            return (safe * mask_f).sum(dim=1)

        assert torch.isfinite(masked_sum(nan_features, batch.mask)).all()


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
