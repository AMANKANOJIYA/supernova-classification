"""Unit tests for src/train/dataset.py -- synthetic CanonicalAlert fixtures
only, no real data required."""

import pytest
import torch

from src.schema import CanonicalAlert, Observation, StaticContext, Survey
from src.train import dataset as ds


def _static():
    return StaticContext(
        host_photoz=0.1,
        host_photoz_err=0.01,
        host_offset_arcsec=0.5,
        host_ellipticity=0.2,
        host_mag_r=20.0,
        mw_ebv=0.05,
        gal_lat=30.0,
        real_bogus_score=None,
    )


def _obs(mjd, flux=10.0, is_detection=True):
    return Observation(mjd=mjd, lam_eff=622.2, flux=flux, flux_err=1.0, is_detection=is_detection)


def _alert(object_id, class_id, n_obs=5):
    observations = [_obs(60000.0 + i) for i in range(n_obs)]
    return CanonicalAlert(
        object_id=object_id,
        survey=Survey.LSST,
        observations=observations,
        static=_static(),
        cutouts=None,
        class_id=class_id,
    )


def _make_alerts(n_per_class=30):
    alerts = []
    for class_id in (2222, 2223, 2224):
        for i in range(n_per_class):
            alerts.append(_alert(f"{class_id}_{i}", class_id))
    return alerts


class TestSplitByObjectId:
    def test_splits_are_disjoint_and_cover_everything(self):
        alerts = _make_alerts()
        split = ds.split_by_object_id(alerts, seed=42)
        all_ids = {a.object_id for a in alerts}
        assert split.train | split.val | split.test == all_ids
        assert split.train & split.val == frozenset()
        assert split.train & split.test == frozenset()
        assert split.val & split.test == frozenset()

    def test_split_is_deterministic_given_same_seed(self):
        alerts = _make_alerts()
        split_a = ds.split_by_object_id(alerts, seed=42)
        split_b = ds.split_by_object_id(alerts, seed=42)
        assert split_a == split_b

    def test_different_seeds_can_give_different_splits(self):
        alerts = _make_alerts()
        split_a = ds.split_by_object_id(alerts, seed=1)
        split_b = ds.split_by_object_id(alerts, seed=2)
        assert split_a != split_b

    def test_stratifies_per_class(self):
        """Each class contributes proportionally to train, not just the
        aggregate -- otherwise random assignment could drift class balance
        away from the natural composition (CLAUDE.md §11 Day-4)."""
        alerts = _make_alerts(n_per_class=100)
        split = ds.split_by_object_id(alerts, seed=42, train_frac=0.7, val_frac=0.15)
        train_dist = ds.class_distribution(alerts, split.train)
        for class_id in (2222, 2223, 2224):
            assert train_dist[class_id] == 70

    def test_duplicate_object_id_raises(self):
        alerts = [_alert("dup", 2222), _alert("dup", 2223)]
        with pytest.raises(ValueError, match="duplicate object_id"):
            ds.split_by_object_id(alerts, seed=42)


class TestClassDistribution:
    def test_counts_only_within_given_ids(self):
        alerts = _make_alerts(n_per_class=5)
        ids = frozenset(a.object_id for a in alerts if a.class_id == 2222)
        dist = ds.class_distribution(alerts, ids)
        assert dist == {2222: 5}


class TestLightCurveDataset:
    def test_filters_to_given_object_ids(self):
        alerts = _make_alerts(n_per_class=3)
        ids = frozenset(a.object_id for a in alerts if a.class_id == 2222)
        lcds = ds.LightCurveDataset(alerts, ids, pre_detection_window_days=30.0)
        assert len(lcds) == 3
        for i in range(len(lcds)):
            assert lcds[i].class_index == 0  # 2222 -> index 0

    def test_class_index_matches_target_class_ids_order(self):
        alerts = _make_alerts(n_per_class=1)
        all_ids = frozenset(a.object_id for a in alerts)
        lcds = ds.LightCurveDataset(alerts, all_ids, pre_detection_window_days=30.0)
        by_class_id = {}
        for i in range(len(lcds)):
            ex = lcds[i]
            source_class_id = int(ex.object_id.split("_")[0])
            by_class_id[source_class_id] = ex.class_index
        assert by_class_id == {2222: 0, 2223: 1, 2224: 2}


class TestCollateExamples:
    def test_pads_and_masks_correctly(self):
        alerts = [_alert("a", 2222, n_obs=2), _alert("b", 2222, n_obs=5)]
        ids = frozenset(a.object_id for a in alerts)
        lcds = ds.LightCurveDataset(alerts, ids, pre_detection_window_days=30.0)
        items = [lcds[i] for i in range(len(lcds))]
        batch = ds.collate_examples(items)
        assert batch.features.shape[1] == 5
        lengths_sorted = sorted(batch.lengths.tolist())
        assert lengths_sorted == [2, 5]
        for i, length in enumerate(batch.lengths.tolist()):
            assert batch.mask[i, :length].all()
            assert not batch.mask[i, length:].any()

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            ds.collate_examples([])

    def test_padded_positions_do_not_influence_masked_computation(self):
        """Day 6: ported from the deleted tensorize.collate_fn's test suite
        (that implementation was removed as an untested-in-production
        duplicate; this property is real and worth keeping on the
        implementation actually used). Padded rows must be safely ignorable
        by any consumer that respects the mask."""
        alerts = [_alert("a", 2222, n_obs=2), _alert("b", 2222, n_obs=5)]
        ids = frozenset(a.object_id for a in alerts)
        lcds = ds.LightCurveDataset(alerts, ids, pre_detection_window_days=30.0)
        batch = ds.collate_examples([lcds[i] for i in range(len(lcds))])

        def masked_mean(features, mask):
            mask_f = mask.unsqueeze(-1).to(features.dtype)
            summed = (features * mask_f).sum(dim=1)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            return summed / counts

        baseline = masked_mean(batch.features, batch.mask)

        garbage = torch.where(
            batch.mask.unsqueeze(-1), batch.features, torch.full_like(batch.features, 1e9)
        )
        assert not torch.equal(garbage, batch.features)  # padded region really did change
        assert torch.allclose(baseline, masked_mean(garbage, batch.mask), atol=1e-5)
