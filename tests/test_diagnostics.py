"""Unit tests for src/diagnostics.py — synthetic CanonicalAlert fixtures
only, no real data required (contrast tests/test_readers_elasticc2.py, the
golden test, which needs the real downloaded files)."""

import ast
import math
from pathlib import Path

from src import diagnostics
from src.schema import CanonicalAlert, Observation, StaticContext, Survey

_EVAL_DIR = Path(__file__).resolve().parent.parent / "src" / "eval"


def _alert(object_id, class_id, observations, static=None):
    return CanonicalAlert(
        object_id=object_id,
        survey=Survey.LSST,
        observations=observations,
        static=static
        or StaticContext(
            host_photoz=0.1,
            host_photoz_err=0.01,
            host_offset_arcsec=0.5,
            host_ellipticity=0.2,
            host_mag_r=20.0,
            mw_ebv=0.05,
            gal_lat=30.0,
            real_bogus_score=None,
        ),
        cutouts=None,
        class_id=class_id,
    )


def _obs(mjd, flux, flux_err=1.0, lam_eff=622.2, is_detection=True):
    return Observation(
        mjd=mjd, lam_eff=lam_eff, flux=flux, flux_err=flux_err, is_detection=is_detection
    )


class TestImportDirection:
    """This session's CLAUDE.md ruling: src/diagnostics.py may import from
    src/eval (compute-only), but nothing under src/eval may import it back
    -- src/eval judges models (frozen, §2.1); src/diagnostics judges the
    parse. Mirrors the existing plot-free regression test style in
    tests/test_eval.py::TestPhaseCurvesHasNoMatplotlib."""

    def test_no_eval_module_imports_diagnostics(self):
        offenders = []
        for path in sorted(_EVAL_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(a.name.split(".")[-1] == "diagnostics" for a in node.names):
                        offenders.append(str(path))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.split(".")[-1] == "diagnostics" or any(
                        a.name == "diagnostics" for a in node.names
                    ):
                        offenders.append(str(path))
        assert offenders == [], f"src/eval/ modules importing src.diagnostics: {offenders}"


class TestNegativeFluxFraction:
    def test_mixed_signs(self):
        alerts = [
            _alert("a", 2222, [_obs(0, 10.0), _obs(1, -5.0), _obs(2, -3.0), _obs(3, 8.0)]),
        ]
        assert diagnostics.negative_flux_fraction(alerts) == 0.5

    def test_all_positive_gives_zero(self):
        alerts = [_alert("a", 2222, [_obs(0, 10.0), _obs(1, 5.0)])]
        assert diagnostics.negative_flux_fraction(alerts) == 0.0

    def test_no_observations_gives_nan(self):
        assert math.isnan(diagnostics.negative_flux_fraction([]))


class TestClassCounts:
    def test_counts_by_class_id(self):
        alerts = [
            _alert("a", 2222, [_obs(0, 1.0)]),
            _alert("b", 2222, [_obs(0, 1.0)]),
            _alert("c", 2223, [_obs(0, 1.0)]),
        ]
        assert diagnostics.class_counts(alerts) == {2222: 2, 2223: 1}

    def test_unlabelled_counted_under_none(self):
        alerts = [_alert("a", None, [_obs(0, 1.0)])]
        assert diagnostics.class_counts(alerts) == {None: 1}


class TestDetectionsAndForcedPhotometry:
    def test_split_by_is_detection(self):
        alerts = [
            _alert(
                "a",
                2222,
                [
                    _obs(0, 1.0, is_detection=True),
                    _obs(1, 1.0, is_detection=False),
                    _obs(2, 1.0, is_detection=False),
                ],
            )
        ]
        assert diagnostics.detections_per_object(alerts) == [1]
        assert diagnostics.forced_photometry_per_object(alerts) == [2]


class TestObservationsPerBand:
    def test_groups_by_band_name(self):
        from src.schema import LAM_EFF

        alerts = [
            _alert(
                "a",
                2222,
                [
                    _obs(0, 1.0, lam_eff=LAM_EFF["r"]),
                    _obs(1, 1.0, lam_eff=LAM_EFF["r"]),
                    _obs(2, 1.0, lam_eff=LAM_EFF["g"]),
                ],
            )
        ]
        assert diagnostics.observations_per_band(alerts) == {"r": 2, "g": 1}

    def test_unrecognized_lam_eff_bucketed_not_dropped(self):
        alerts = [_alert("a", 2222, [_obs(0, 1.0, lam_eff=123.456)])]
        counts = diagnostics.observations_per_band(alerts)
        assert sum(counts.values()) == 1
        assert list(counts.keys()) == ["unknown(lam_eff=123.456)"]


class TestTimeSpanDays:
    def test_span_is_max_minus_min_mjd(self):
        alerts = [_alert("a", 2222, [_obs(10, 1.0), _obs(25, 1.0), _obs(15, 1.0)])]
        assert diagnostics.time_span_days(alerts) == [15.0]

    def test_object_with_no_observations_skipped(self):
        alerts = [_alert("a", 2222, [])]
        assert diagnostics.time_span_days(alerts) == []


class TestStaticFieldNoneFraction:
    def test_fraction_none_per_field(self):
        alerts = [
            _alert(
                "a",
                2222,
                [_obs(0, 1.0)],
                static=StaticContext(
                    host_photoz=None,
                    host_photoz_err=None,
                    host_offset_arcsec=0.5,
                    host_ellipticity=0.2,
                    host_mag_r=20.0,
                    mw_ebv=0.05,
                    gal_lat=30.0,
                    real_bogus_score=None,
                ),
            ),
            _alert(
                "b",
                2222,
                [_obs(0, 1.0)],
                static=StaticContext(
                    host_photoz=0.2,
                    host_photoz_err=0.02,
                    host_offset_arcsec=0.6,
                    host_ellipticity=0.3,
                    host_mag_r=21.0,
                    mw_ebv=0.06,
                    gal_lat=31.0,
                    real_bogus_score=None,
                ),
            ),
        ]
        fractions = diagnostics.static_field_none_fraction(alerts)
        assert fractions["host_photoz"] == 0.5
        assert fractions["host_offset_arcsec"] == 0.0
        assert fractions["real_bogus_score"] == 1.0

    def test_empty_alerts_gives_empty_dict(self):
        assert diagnostics.static_field_none_fraction([]) == {}


class TestSampleObjectsPerClass:
    def test_caps_at_n_per_class_deterministically(self):
        alerts = [_alert(f"obj{i}", 2222, [_obs(0, 1.0)]) for i in (3, 1, 2)]
        sampled = diagnostics.sample_objects_per_class(alerts, n_per_class=2)
        assert [a.object_id for a in sampled[2222]] == ["obj1", "obj2"]

    def test_groups_separately_per_class(self):
        alerts = [
            _alert("a", 2222, [_obs(0, 1.0)]),
            _alert("b", 2223, [_obs(0, 1.0)]),
        ]
        sampled = diagnostics.sample_objects_per_class(alerts, n_per_class=20)
        assert set(sampled.keys()) == {2222, 2223}


class TestPlotLightCurveRuns:
    """Just confirm a figure is producible without raising -- visual
    correctness isn't something a unit test can check (same rationale as
    tests/test_eval.py::TestPlotting)."""

    def test_plot_with_mixed_sign_flux_and_forced_photometry(self):
        import matplotlib.pyplot as plt

        alert = _alert(
            "a",
            2222,
            [
                _obs(0, 10.0, is_detection=True),
                _obs(1, -3.0, is_detection=False),
                _obs(2, 15.0, is_detection=True),
            ],
        )
        fig = diagnostics.plot_light_curve(alert, "Ia")
        assert fig is not None
        plt.close(fig)


class TestBuildGateAReport:
    def test_bundles_all_fields(self):
        alerts = [_alert("a", 2222, [_obs(0, 10.0), _obs(1, -5.0)])]
        report = diagnostics.build_gate_a_report(alerts)
        assert report.negative_flux_fraction == 0.5
        assert report.class_counts_by_object == {2222: 1}
        assert report.detections_per_object == [2]
        assert report.time_span_days == [1.0]
