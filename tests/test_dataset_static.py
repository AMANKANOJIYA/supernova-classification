"""Tests for src/train/dataset_static.py -- synthetic CanonicalAlert
fixtures only, no real data required (mirrors tests/test_dataset.py)."""

import pytest

from src.models.static_encoder import NUM_STATIC_FIELDS, StaticFeatureNormalizer
from src.schema import CanonicalAlert, Observation, StaticContext, Survey
from src.train import dataset_static as ds


def _static(**overrides) -> StaticContext:
    base = dict(
        host_photoz=0.1, host_photoz_err=0.01, host_offset_arcsec=0.5,
        host_ellipticity=0.2, host_mag_r=20.0, mw_ebv=0.05, gal_lat=30.0,
        real_bogus_score=None,
    )
    base.update(overrides)
    return StaticContext(**base)


def _alert(object_id, class_id, n_obs=3, static=None) -> CanonicalAlert:
    observations = [
        Observation(mjd=float(i), lam_eff=622.2, flux=10.0, flux_err=1.0, is_detection=True)
        for i in range(n_obs)
    ]
    return CanonicalAlert(
        object_id=object_id, survey=Survey.LSST, observations=observations,
        static=static or _static(), cutouts=None, class_id=class_id,
    )


class TestLightCurveStaticDataset:
    def test_carries_static_features_through(self):
        alerts = [_alert("a", 2222, static=_static(host_photoz=0.5))]
        norm = StaticFeatureNormalizer.fit(alerts)
        lcds = ds.LightCurveStaticDataset(alerts, frozenset(["a"]), pre_detection_window_days=30.0, normalizer=norm)
        assert len(lcds) == 1
        assert lcds[0].static_features.shape == (2 * NUM_STATIC_FIELDS,)

    def test_filters_to_given_object_ids(self):
        alerts = [_alert("a", 2222), _alert("b", 2223)]
        norm = StaticFeatureNormalizer.fit(alerts)
        lcds = ds.LightCurveStaticDataset(alerts, frozenset(["a"]), pre_detection_window_days=30.0, normalizer=norm)
        assert len(lcds) == 1
        assert lcds[0].object_id == "a"


class TestCollateStaticExamples:
    def test_pads_and_stacks_static_features(self):
        alerts = [_alert("a", 2222, n_obs=2), _alert("b", 2222, n_obs=5)]
        norm = StaticFeatureNormalizer.fit(alerts)
        ids = frozenset(["a", "b"])
        lcds = ds.LightCurveStaticDataset(alerts, ids, pre_detection_window_days=30.0, normalizer=norm)
        batch = ds.collate_static_examples([lcds[i] for i in range(len(lcds))])
        assert batch.features.shape[1] == 5
        assert batch.static_features.shape == (2, 2 * NUM_STATIC_FIELDS)
        lengths_sorted = sorted(batch.lengths.tolist())
        assert lengths_sorted == [2, 5]

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            ds.collate_static_examples([])
