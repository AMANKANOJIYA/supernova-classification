"""Tests for src/serve/state.py -- the Redis rolling-history merge. Only
`merge_history` (pure, no Redis) is exercised here; `get_and_merge`/
`get_alert`'s thin Redis I/O wrapper is covered by the end-to-end replay
run against a real Redis container (infra/docker-compose.yml), not unit
tests, per CLAUDE.md §12 ("golden tests... commit it as a fixture").
"""

import pytest

from src.schema import Observation, StaticContext, Survey
from src.serve.ingest import NormalizedAlert
from src.serve.state import merge_history


def _static() -> StaticContext:
    return StaticContext(
        host_photoz=0.05, host_photoz_err=0.01, host_offset_arcsec=0.3,
        host_ellipticity=0.2, host_mag_r=18.5, mw_ebv=0.02, gal_lat=45.0,
        real_bogus_score=0.9,
    )


def _obs(mjd, flux=100.0, flux_err=5.0, is_detection=True) -> Observation:
    return Observation(mjd=mjd, lam_eff=472.3, flux=flux, flux_err=flux_err, is_detection=is_detection)


def _normalized(object_id, new_observations, class_id=2222) -> NormalizedAlert:
    return NormalizedAlert(
        object_id=object_id, survey=Survey.ZTF, new_observations=new_observations,
        static=_static(), class_id=class_id,
    )


class TestNonMonotonicMerge:
    """CLAUDE.md §5.1: "as new detections arrive, an object's sequence can
    gain points before its current endpoint, because forced photometry
    between two detections enters the window only once the later detection
    exists." A later alert's bundled forced photometry can predate
    everything currently stored -- the merge must insert it in sorted
    position, not append it at the end."""

    def test_early_forced_photometry_arriving_late_is_inserted_before_existing_points(self):
        # Alert 1 (this object's first-ever alert): a detection at day 10.
        state_1 = merge_history(None, _normalized("ZTFX", [_obs(60010.0, is_detection=True)]))
        assert [o["mjd"] for o in state_1["observations"]] == [60010.0]

        # Alert 2 (a LATER trigger, day 20): bundles forced photometry from
        # day 3 -- before anything previously stored -- plus its own new
        # detection at day 20.
        state_2 = merge_history(
            state_1,
            _normalized(
                "ZTFX",
                [_obs(60003.0, is_detection=False), _obs(60020.0, is_detection=True)],
            ),
        )
        mjds = [o["mjd"] for o in state_2["observations"]]
        assert mjds == [60003.0, 60010.0, 60020.0]  # sorted, not append order [60010, 60003, 60020]

    def test_result_is_always_sorted_regardless_of_arrival_order(self):
        state = merge_history(None, _normalized("ZTFY", [_obs(60050.0)]))
        state = merge_history(state, _normalized("ZTFY", [_obs(60010.0)]))
        state = merge_history(state, _normalized("ZTFY", [_obs(60030.0)]))
        mjds = [o["mjd"] for o in state["observations"]]
        assert mjds == sorted(mjds)
        assert mjds == [60010.0, 60030.0, 60050.0]


class TestDedup:
    def test_resent_forced_photometry_point_is_not_duplicated(self):
        shared = _obs(60005.0, is_detection=False)
        state = merge_history(None, _normalized("ZTFZ", [shared, _obs(60010.0)]))
        state = merge_history(state, _normalized("ZTFZ", [shared, _obs(60015.0)]))
        mjds = [o["mjd"] for o in state["observations"]]
        assert mjds == [60005.0, 60010.0, 60015.0]  # the shared 60005 point appears once


class TestFieldPersistence:
    def test_object_id_survey_static_and_class_id_persist_across_merges(self):
        state = merge_history(None, _normalized("ZTFW", [_obs(60000.0)], class_id=2223))
        state = merge_history(state, _normalized("ZTFW", [_obs(60010.0)], class_id=2223))
        assert state["object_id"] == "ZTFW"
        assert state["survey"] == "ztf"
        assert state["class_id"] == 2223
        assert state["static"]["host_photoz"] == pytest.approx(0.05)

    def test_first_seen_class_id_is_not_overwritten_by_a_later_none(self):
        state = merge_history(None, _normalized("ZTFV", [_obs(60000.0)], class_id=2224))
        state = merge_history(state, _normalized("ZTFV", [_obs(60010.0)], class_id=None))
        assert state["class_id"] == 2224
