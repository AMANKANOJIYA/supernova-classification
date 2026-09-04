"""Tests for src/eval/ using synthetic inputs with analytically known answers."""

import math

import matplotlib

matplotlib.use("Agg")  # headless — no display in CI or this session

import numpy as np
import pytest

from src.eval import baselines, metrics, phase_curves, plots, report


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_perfect_predictions_is_diagonal(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])
        cm = metrics.confusion_matrix(y_true, y_pred, num_classes=3)
        assert np.array_equal(cm.raw, np.diag([2, 2, 1]))
        assert np.allclose(cm.normalized, np.eye(3))

    def test_known_small_matrix(self):
        # true=0: predicted [0,0,1] -> row [2,1,0]
        # true=1: predicted [1,2]   -> row [0,1,1]
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 2])
        cm = metrics.confusion_matrix(y_true, y_pred, num_classes=3)
        expected_raw = np.array([[2, 1, 0], [0, 1, 1], [0, 0, 0]])
        assert np.array_equal(cm.raw, expected_raw)
        expected_norm = np.array([[2 / 3, 1 / 3, 0], [0, 0.5, 0.5], [0, 0, 0]])
        assert np.allclose(cm.normalized, expected_norm)

    def test_zero_support_row_is_zero_not_nan(self):
        y_true = np.array([0, 0])
        y_pred = np.array([0, 1])
        cm = metrics.confusion_matrix(y_true, y_pred, num_classes=3)
        assert not np.isnan(cm.normalized).any()
        assert np.array_equal(cm.normalized[2], [0, 0, 0])


class TestPerClassPRF1:
    def test_known_values(self):
        # class 0: tp=2 fp=1 fn=0 -> P=2/3 R=1   F1=0.8
        # class 1: tp=1 fp=0 fn=1 -> P=1   R=0.5 F1=2/3
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        result = metrics.per_class_precision_recall_f1(y_true, y_pred, num_classes=2)
        assert result.precision[0] == pytest.approx(2 / 3)
        assert result.recall[0] == pytest.approx(1.0)
        assert result.f1[0] == pytest.approx(0.8)
        assert result.precision[1] == pytest.approx(1.0)
        assert result.recall[1] == pytest.approx(0.5)
        assert result.f1[1] == pytest.approx(2 / 3)
        assert list(result.support) == [2, 2]

    def test_zero_support_class_is_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 1])
        result = metrics.per_class_precision_recall_f1(y_true, y_pred, num_classes=2)
        assert result.support[1] == 0
        assert result.precision[1] == 0.0
        assert result.recall[1] == 0.0
        assert result.f1[1] == 0.0


class TestMacroF1:
    def test_matches_hand_computed_mean(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        # from TestPerClassPRF1.test_known_values: F1 = [0.8, 2/3] -> mean 11/15
        assert metrics.macro_f1(y_true, y_pred, num_classes=2) == pytest.approx((0.8 + 2 / 3) / 2)

    def test_perfect_predictions_gives_one(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        assert metrics.macro_f1(y_true, y_true, num_classes=3) == pytest.approx(1.0)

    def test_no_bare_accuracy_function_exists(self):
        assert not hasattr(metrics, "accuracy")

    def test_majority_class_imbalance_asymmetry(self):
        """The reason this harness exists (CLAUDE.md §7 Gate A, §11): a
        14:1-imbalanced majority-class predictor scores high accuracy but
        poor macro-F1."""
        rng = np.random.RandomState(0)
        n_majority, n_minority = 140, 10
        y_true = np.array([0] * n_majority + [1] * n_minority)
        y_pred = np.zeros_like(y_true)  # always predicts class 0

        accuracy = float(np.mean(y_pred == y_true))  # computed directly, not via a library accuracy()
        assert accuracy == pytest.approx(140 / 150)
        assert accuracy > 0.9

        macro = metrics.macro_f1(y_true, y_pred, num_classes=2)
        assert macro < 0.6  # class 1: precision=recall=F1=0; class 0 F1 < 1.0 doesn't matter, mean is pulled down hard


class TestPerClassPrAuc:
    def test_perfect_separation_gives_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        )
        ap = metrics.per_class_pr_auc(y_true, y_prob, num_classes=2)
        assert ap[0] == pytest.approx(1.0)
        assert ap[1] == pytest.approx(1.0)

    def test_worst_case_ranking_gives_low_ap(self):
        # class 1's positives are ranked last -> low average precision
        y_true = np.array([0, 0, 0, 1])
        y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]])
        ap = metrics.per_class_pr_auc(y_true, y_prob, num_classes=2)
        # single positive for class 1 ranked last of 4 -> precision at that point = 1/4
        assert ap[1] == pytest.approx(0.25)

    def test_zero_positives_gives_nan(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([[0.5, 0.5]] * 3)
        ap = metrics.per_class_pr_auc(y_true, y_prob, num_classes=2)
        assert math.isnan(ap[1])


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_gives_zero_ece(self):
        # 10 predictions at confidence 0.7, exactly 7 correct -> accuracy == confidence in the one populated bin.
        y_prob = np.array([[0.7, 0.3]] * 10)
        y_true = np.array([0] * 7 + [1] * 3)
        result = metrics.expected_calibration_error(y_true, y_prob, num_bins=10)
        assert result.ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_gives_positive_ece(self):
        y_prob = np.array([[0.99, 0.01]] * 10)
        y_true = np.array([0] * 5 + [1] * 5)  # only 50% accurate despite 99% confidence
        result = metrics.expected_calibration_error(y_true, y_prob, num_bins=10)
        assert result.ece == pytest.approx(0.49, abs=1e-6)

    def test_bin_count_sums_to_n(self):
        rng = np.random.RandomState(1)
        y_prob = rng.dirichlet([1, 1, 1], size=50)
        y_true = rng.randint(0, 3, size=50)
        result = metrics.expected_calibration_error(y_true, y_prob, num_bins=15)
        assert result.bin_count.sum() == 50

    def test_confidence_one_lands_in_last_bin(self):
        y_prob = np.array([[1.0, 0.0]])
        y_true = np.array([0])
        result = metrics.expected_calibration_error(y_true, y_prob, num_bins=5)
        assert result.bin_count[-1] == 1
        assert result.bin_count.sum() == 1


# ---------------------------------------------------------------------------
# baselines.py
# ---------------------------------------------------------------------------


class TestMajorityClassBaseline:
    def test_fits_the_majority_class(self):
        y_train = np.array([0] * 14 + [1] * 1)
        clf = baselines.MajorityClassBaseline().fit(y_train, num_classes=2)
        assert clf.majority_class_ == 0

    def test_predict_shape_and_value(self):
        y_train = np.array([1, 1, 1, 0])
        clf = baselines.MajorityClassBaseline().fit(y_train, num_classes=2)
        preds = clf.predict(5)
        assert preds.shape == (5,)
        assert np.all(preds == 1)

    def test_predict_proba_is_one_hot(self):
        y_train = np.array([2, 2, 0])
        clf = baselines.MajorityClassBaseline().fit(y_train, num_classes=3)
        proba = clf.predict_proba(4)
        assert proba.shape == (4, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba[:, 2] == 1.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            baselines.MajorityClassBaseline().predict(3)


class TestMulticlassLogisticRegression:
    def test_fits_linearly_separable_data(self):
        rng = np.random.RandomState(0)
        n = 60
        X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(n, 2))
        X1 = rng.normal(loc=[3, 3], scale=0.5, size=(n, 2))
        X2 = rng.normal(loc=[-3, 3], scale=0.5, size=(n, 2))
        X = np.concatenate([X0, X1, X2])
        y = np.array([0] * n + [1] * n + [2] * n)

        clf = baselines.MulticlassLogisticRegression(num_classes=3, n_iter=1500, seed=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        accuracy = float(np.mean(preds == y))
        assert accuracy > 0.95

    def test_predict_proba_rows_sum_to_one(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 5))
        y = rng.randint(0, 3, size=20)
        clf = baselines.MulticlassLogisticRegression(num_classes=3, n_iter=50, seed=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_before_fit_raises(self):
        clf = baselines.MulticlassLogisticRegression(num_classes=3)
        with pytest.raises(RuntimeError):
            clf.predict_proba(np.zeros((2, 5)))


class TestExtractFiveFeatures:
    def test_hand_computed_values(self):
        # Two bands, clean synthetic rise-peak-decline light curve.
        mjd = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 10.0])
        flux = np.array([10.0, 50.0, 100.0, 60.0, 20.0, 25.0])  # last point: red band near peak
        lam_eff = np.array([480.0, 480.0, 480.0, 480.0, 480.0, 620.0])  # g,g,g,g,g,r
        is_detection = np.array([True] * 6)

        features = baselines.extract_five_features(mjd, flux, lam_eff, is_detection)
        peak_flux, rise_time, decline_time, colour, duration = features

        assert peak_flux == pytest.approx(100.0)
        assert rise_time == pytest.approx(10.0)  # peak at mjd=10, first detection mjd=0
        assert decline_time == pytest.approx(10.0)  # last detection mjd=20, peak mjd=10
        assert duration == pytest.approx(20.0)
        # colour: blue band (g) brightest point in window is the peak itself (flux=100);
        # red band (r) has one point in window (flux=25).
        expected_colour = -2.5 * math.log10(100.0) - (-2.5 * math.log10(25.0))
        assert colour == pytest.approx(expected_colour)

    def test_no_detections_raises(self):
        mjd = np.array([1.0, 2.0])
        flux = np.array([10.0, 20.0])
        lam_eff = np.array([480.0, 480.0])
        is_detection = np.array([False, False])
        with pytest.raises(ValueError):
            baselines.extract_five_features(mjd, flux, lam_eff, is_detection)

    def test_colour_independent_of_input_order(self):
        """Regression: colour must pick each band's brightest in-window point,
        not whichever point happens to come first for that wavelength."""
        mjd = np.array([5.0, 10.0, 15.0, 10.0])
        flux = np.array([50.0, 100.0, 60.0, 25.0])
        lam_eff = np.array([480.0, 480.0, 480.0, 620.0])
        is_detection = np.array([True] * 4)
        forward = baselines.extract_five_features(mjd, flux, lam_eff, is_detection)

        # Same data, g-band points reordered so the brightest one comes first.
        mjd2 = np.array([10.0, 5.0, 15.0, 10.0])
        flux2 = np.array([100.0, 50.0, 60.0, 25.0])
        lam_eff2 = np.array([480.0, 480.0, 480.0, 620.0])
        reordered = baselines.extract_five_features(mjd2, flux2, lam_eff2, is_detection)

        assert forward[3] == pytest.approx(reordered[3])

    def test_single_band_falls_back_colour_to_zero(self):
        mjd = np.array([0.0, 5.0, 10.0])
        flux = np.array([10.0, 50.0, 20.0])
        lam_eff = np.array([480.0, 480.0, 480.0])  # only one wavelength present
        is_detection = np.array([True, True, True])
        features = baselines.extract_five_features(mjd, flux, lam_eff, is_detection)
        assert features[3] == pytest.approx(0.0)

    def test_negative_flux_does_not_crash_feature_extraction(self):
        """CLAUDE.md §2.2: negative flux is real and must be tolerated everywhere."""
        mjd = np.array([0.0, 5.0, 10.0])
        flux = np.array([-5.0, 50.0, 10.0])
        lam_eff = np.array([480.0, 480.0, 480.0])
        is_detection = np.array([True, True, True])
        features = baselines.extract_five_features(mjd, flux, lam_eff, is_detection)
        assert features[0] == pytest.approx(50.0)  # peak flux unaffected by the negative point


# ---------------------------------------------------------------------------
# phase_curves.py
# ---------------------------------------------------------------------------


class TestSelectAtCutoff:
    def _pred(self, object_id, day, probs=(0.5, 0.5)):
        return phase_curves.PhasePrediction(
            object_id=object_id, days_since_first_detection=day, true_class=0, probs=np.array(probs)
        )

    def test_picks_latest_at_or_before_cutoff(self):
        preds = [self._pred("a", -3), self._pred("a", 0), self._pred("a", 10)]
        selected, n_excluded = phase_curves.select_at_cutoff(preds, cutoff=5)
        assert n_excluded == 0
        assert len(selected) == 1
        assert selected[0].days_since_first_detection == 0

    def test_excludes_object_with_no_eligible_prediction(self):
        preds = [self._pred("a", 10), self._pred("b", 0)]
        selected, n_excluded = phase_curves.select_at_cutoff(preds, cutoff=5)
        assert n_excluded == 1
        assert len(selected) == 1
        assert selected[0].object_id == "b"

    def test_full_cutoff_picks_latest_overall(self):
        preds = [self._pred("a", -3), self._pred("a", 0), self._pred("a", 40)]
        selected, n_excluded = phase_curves.select_at_cutoff(preds, cutoff=None)
        assert n_excluded == 0
        assert selected[0].days_since_first_detection == 40

    def test_negative_cutoff(self):
        preds = [self._pred("a", -10), self._pred("a", -2), self._pred("a", 5)]
        selected, n_excluded = phase_curves.select_at_cutoff(preds, cutoff=-5)
        assert n_excluded == 0
        assert selected[0].days_since_first_detection == -10


class TestCutoffLabel:
    def test_full_is_none(self):
        assert phase_curves.cutoff_label(None) == "full"

    def test_integer_cutoff(self):
        assert phase_curves.cutoff_label(-5.0) == "-5"
        assert phase_curves.cutoff_label(30.0) == "30"


class TestEvaluateAtCutoffs:
    def test_produces_one_result_per_cutoff(self):
        preds = [
            phase_curves.PhasePrediction("a", 0, true_class=0, probs=np.array([0.9, 0.1])),
            phase_curves.PhasePrediction("b", 0, true_class=1, probs=np.array([0.2, 0.8])),
        ]
        results = phase_curves.evaluate_at_cutoffs(
            preds, num_classes=2, cutoffs=(0.0, None)
        )
        assert set(results.keys()) == {"0", "full"}
        assert results["0"].n_objects == 2
        assert results["0"].macro_f1 == pytest.approx(1.0)

    def test_empty_selection_reports_nan_macro_f1_not_crash(self):
        preds = [phase_curves.PhasePrediction("a", 10, true_class=0, probs=np.array([1.0, 0.0]))]
        results = phase_curves.evaluate_at_cutoffs(preds, num_classes=2, cutoffs=(-5.0,))
        assert results["-5"].n_objects == 0
        assert results["-5"].n_excluded == 1
        assert math.isnan(results["-5"].macro_f1)


class TestPhaseCurvesHasNoMatplotlib:
    """Regression for the metrics.py/phase_curves.py <-> plots.py split:
    compute stays plot-free so it's usable anywhere a figure isn't wanted."""

    def test_phase_curves_module_does_not_import_matplotlib(self):
        assert "matplotlib" not in vars(phase_curves)

    def test_metrics_module_does_not_import_matplotlib(self):
        assert "matplotlib" not in vars(metrics)


class TestPlotting:
    """Just confirm figures are producible without raising — visual
    correctness isn't something a unit test can check."""

    def test_plot_macro_f1_vs_phase_runs(self):
        preds = [
            phase_curves.PhasePrediction("a", 0, true_class=0, probs=np.array([0.9, 0.1])),
            phase_curves.PhasePrediction("b", 0, true_class=1, probs=np.array([0.2, 0.8])),
        ]
        results = phase_curves.evaluate_at_cutoffs(preds, num_classes=2, cutoffs=(0.0, None))
        fig = plots.plot_macro_f1_vs_phase(
            results, class_names=["Ia", "II"], cutoff_order=(0.0, None)
        )
        assert fig is not None

    def test_plot_confusion_matrices_runs(self):
        preds = [
            phase_curves.PhasePrediction("a", 0, true_class=0, probs=np.array([0.9, 0.1])),
            phase_curves.PhasePrediction("b", 0, true_class=1, probs=np.array([0.2, 0.8])),
        ]
        results = phase_curves.evaluate_at_cutoffs(preds, num_classes=2, cutoffs=(0.0, None))
        fig = plots.plot_confusion_matrices(
            results, class_names=["Ia", "II"], cutoff_order=(0.0, None)
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------


class TestRunEvalReport:
    def test_writes_expected_files_and_returns_matching_dict(self, tmp_path):
        preds = [
            phase_curves.PhasePrediction("a", 0, true_class=0, probs=np.array([0.9, 0.1])),
            phase_curves.PhasePrediction("b", 0, true_class=1, probs=np.array([0.2, 0.8])),
            phase_curves.PhasePrediction("a", 20, true_class=0, probs=np.array([0.7, 0.3])),
        ]
        out_dict = report.run_eval_report(
            preds,
            num_classes=2,
            class_names=["Ia", "II"],
            output_dir=str(tmp_path),
            cutoffs=(0.0, None),
        )

        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "macro_f1_vs_phase.png").exists()
        assert (tmp_path / "confusion_matrices.png").exists()

        import json

        on_disk = json.loads((tmp_path / "metrics.json").read_text())
        assert on_disk == out_dict
        assert set(on_disk.keys()) == {"0", "full"}
        assert on_disk["0"]["n_objects"] == 2

    def test_mismatched_class_names_raises(self, tmp_path):
        with pytest.raises(ValueError):
            report.run_eval_report(
                [], num_classes=3, class_names=["Ia", "II"], output_dir=str(tmp_path)
            )

    def test_nan_is_json_null_not_bare_nan(self, tmp_path):
        preds = [phase_curves.PhasePrediction("a", 10, true_class=0, probs=np.array([1.0, 0.0]))]
        out_dict = report.run_eval_report(
            preds,
            num_classes=2,
            class_names=["Ia", "II"],
            output_dir=str(tmp_path),
            cutoffs=(-5.0,),
        )
        assert out_dict["-5"]["macro_f1"] is None
        raw_text = (tmp_path / "metrics.json").read_text()
        assert "NaN" not in raw_text
