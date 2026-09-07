"""Tests for src/serve/model.py::AnomalyScorer -- specifically the Day 13
full-3-component-combination path added on top of Day 12's
reconstruction-error-only baseline. Builds tiny synthetic checkpoints/
artifacts rather than depending on the real (large, slow-to-produce)
Day 10/13 outputs, so this test suite runs fast and doesn't depend on
`src/train/persist_anomaly_scorers.py` having been run.
"""

import json
import pickle

import numpy as np
import pytest
import torch

from src.anomaly.decoder import DecoderTrainConfig, ReconstructionDecoder
from src.anomaly.pineforest import NUM_ENGINEERED_FEATURES
from src.features.tensorize import NUM_FEATURES
from src.models.lc_encoder import LCEncoder
from src.schema import CanonicalAlert, Observation, StaticContext, Survey
from src.serve.model import AnomalyScorer

# AnomalyScorer.__init__ hardcodes the encoder architecture (hidden_size=128,
# num_layers=2) to match the real checkpoint -- the synthetic checkpoint here
# must match exactly for load_state_dict to succeed.
_ENCODER_HIDDEN_SIZE = 128


class _ConstantForestScorer:
    """A trivial, picklable stand-in for PineForestScorer/IsolationForestScorer
    -- only `.score(X) -> np.ndarray` is required by AnomalyScorer."""

    def score(self, X: np.ndarray) -> np.ndarray:
        return X.sum(axis=1)


def _make_alert(object_id: str, n_obs: int = 5) -> CanonicalAlert:
    observations = [
        Observation(mjd=60000.0 + i, lam_eff=472.3, flux=100.0 + i, flux_err=5.0, is_detection=True)
        for i in range(n_obs)
    ]
    static = StaticContext(
        host_photoz=None, host_photoz_err=None, host_offset_arcsec=0.2, host_ellipticity=None,
        host_mag_r=18.0, mw_ebv=0.05, gal_lat=30.0, real_bogus_score=0.9,
    )
    return CanonicalAlert(
        object_id=object_id, survey=Survey.ZTF, observations=observations,
        static=static, cutouts=None, class_id=None,
    )


@pytest.fixture()
def artifact_dir(tmp_path):
    torch.manual_seed(0)

    encoder = LCEncoder(input_size=NUM_FEATURES, hidden_size=_ENCODER_HIDDEN_SIZE, num_layers=2, dropout=0.2, num_classes=3)
    encoder_ckpt = tmp_path / "encoder.pt"
    torch.save({"model_state_dict": encoder.state_dict()}, encoder_ckpt)

    decoder_cfg = DecoderTrainConfig(encoder_hidden_size=_ENCODER_HIDDEN_SIZE, decoder_hidden_size=8, dropout=0.0)
    decoder = ReconstructionDecoder(
        encoder_hidden_size=decoder_cfg.encoder_hidden_size,
        hidden_size=decoder_cfg.decoder_hidden_size,
        dropout=decoder_cfg.dropout,
    )
    decoder_ckpt = tmp_path / "decoder.pt"
    torch.save({"decoder_state_dict": decoder.state_dict(), "config": decoder_cfg.__dict__}, decoder_ckpt)

    results_json = tmp_path / "anomaly_results.json"
    with open(results_json, "w") as f:
        json.dump(
            {
                "reference_zscore_stats": {
                    "reconstruction_error": [1.0, 0.5],
                    "mahalanobis": [10.0, 3.0],
                    "isoforest": [0.0, 1.0],
                }
            },
            f,
        )

    return {
        "encoder_checkpoint": encoder_ckpt,
        "decoder_checkpoint": decoder_ckpt,
        "results_json": results_json,
        "tmp_path": tmp_path,
    }


def _batch_for(alerts: list[CanonicalAlert]):
    lengths = torch.tensor([len(a.observations) for a in alerts])
    t_max = int(lengths.max())
    features = torch.zeros(len(alerts), t_max, NUM_FEATURES)
    for i, a in enumerate(alerts):
        for t, obs in enumerate(a.observations):
            features[i, t] = torch.tensor([float(t), 1.0, 1.0, 1.0, 0.0, 1.0])
    mask = torch.arange(t_max).unsqueeze(0) < lengths.unsqueeze(1)
    return features, lengths, mask


class TestReconstructionErrorOnlyFallback:
    def test_missing_persisted_artifacts_falls_back(self, artifact_dir):
        scorer = AnomalyScorer(
            encoder_checkpoint=artifact_dir["encoder_checkpoint"],
            decoder_checkpoint=artifact_dir["decoder_checkpoint"],
            results_json=artifact_dir["results_json"],
            mahalanobis_npz=artifact_dir["tmp_path"] / "does_not_exist.npz",
            engineered_normalizer_npz=artifact_dir["tmp_path"] / "does_not_exist2.npz",
            forest_pkl=artifact_dir["tmp_path"] / "does_not_exist.pkl",
        )
        assert scorer.full_combination is False

        alerts = [_make_alert("a"), _make_alert("b")]
        features, lengths, mask = _batch_for(alerts)
        scores_with_alerts = scorer.score(features, lengths, mask, alerts=alerts)
        scores_without_alerts = scorer.score(features, lengths, mask)
        # Falling back must give the SAME reconstruction-error-only values
        # whether or not `alerts` is passed -- alerts is ignored, not required.
        np.testing.assert_allclose(scores_with_alerts, scores_without_alerts)
        assert scores_with_alerts.shape == (2,)


class TestFullCombination:
    @pytest.fixture()
    def full_artifact_dir(self, artifact_dir):
        tmp_path = artifact_dir["tmp_path"]
        rng = np.random.default_rng(0)

        maha_npz = tmp_path / "mahalanobis.npz"
        np.savez(maha_npz, mean=np.zeros(_ENCODER_HIDDEN_SIZE), precision=np.eye(_ENCODER_HIDDEN_SIZE))

        eng_npz = tmp_path / "engineered_normalizer.npz"
        np.savez(eng_npz, mean=np.zeros(NUM_ENGINEERED_FEATURES), std=np.ones(NUM_ENGINEERED_FEATURES))

        forest_pkl = tmp_path / "forest.pkl"
        with open(forest_pkl, "wb") as f:
            pickle.dump(_ConstantForestScorer(), f)

        artifact_dir.update({"mahalanobis_npz": maha_npz, "engineered_normalizer_npz": eng_npz, "forest_pkl": forest_pkl})
        return artifact_dir

    def test_full_combination_detected_and_used(self, full_artifact_dir):
        scorer = AnomalyScorer(
            encoder_checkpoint=full_artifact_dir["encoder_checkpoint"],
            decoder_checkpoint=full_artifact_dir["decoder_checkpoint"],
            results_json=full_artifact_dir["results_json"],
            mahalanobis_npz=full_artifact_dir["mahalanobis_npz"],
            engineered_normalizer_npz=full_artifact_dir["engineered_normalizer_npz"],
            forest_pkl=full_artifact_dir["forest_pkl"],
        )
        assert scorer.full_combination is True

        alerts = [_make_alert("a", n_obs=5), _make_alert("b", n_obs=3)]
        features, lengths, mask = _batch_for(alerts)
        combined = scorer.score(features, lengths, mask, alerts=alerts)
        assert combined.shape == (2,)
        assert np.all(np.isfinite(combined))

    def test_full_combination_differs_from_reconstruction_only(self, full_artifact_dir):
        """The whole point of persisting the scorers: the served score
        should actually change (not silently equal reconstruction-error
        alone) once mahalanobis/isoforest are folded in."""
        alerts = [_make_alert("a", n_obs=5), _make_alert("b", n_obs=3)]
        features, lengths, mask = _batch_for(alerts)

        full_scorer = AnomalyScorer(
            encoder_checkpoint=full_artifact_dir["encoder_checkpoint"],
            decoder_checkpoint=full_artifact_dir["decoder_checkpoint"],
            results_json=full_artifact_dir["results_json"],
            mahalanobis_npz=full_artifact_dir["mahalanobis_npz"],
            engineered_normalizer_npz=full_artifact_dir["engineered_normalizer_npz"],
            forest_pkl=full_artifact_dir["forest_pkl"],
        )
        recon_only_scorer = AnomalyScorer(
            encoder_checkpoint=full_artifact_dir["encoder_checkpoint"],
            decoder_checkpoint=full_artifact_dir["decoder_checkpoint"],
            results_json=full_artifact_dir["results_json"],
            mahalanobis_npz=full_artifact_dir["tmp_path"] / "missing.npz",
            engineered_normalizer_npz=full_artifact_dir["tmp_path"] / "missing2.npz",
            forest_pkl=full_artifact_dir["tmp_path"] / "missing.pkl",
        )

        combined = full_scorer.score(features, lengths, mask, alerts=alerts)
        recon_only = recon_only_scorer.score(features, lengths, mask, alerts=alerts)
        assert not np.allclose(combined, recon_only)

    def test_alerts_required_for_full_combination(self, full_artifact_dir):
        """Omitting `alerts` must degrade to reconstruction-error-only
        rather than raise -- a caller that forgets to pass alerts gets a
        safe (if incomplete) answer, not a crash."""
        scorer = AnomalyScorer(
            encoder_checkpoint=full_artifact_dir["encoder_checkpoint"],
            decoder_checkpoint=full_artifact_dir["decoder_checkpoint"],
            results_json=full_artifact_dir["results_json"],
            mahalanobis_npz=full_artifact_dir["mahalanobis_npz"],
            engineered_normalizer_npz=full_artifact_dir["engineered_normalizer_npz"],
            forest_pkl=full_artifact_dir["forest_pkl"],
        )
        alerts = [_make_alert("a")]
        features, lengths, mask = _batch_for(alerts)
        result = scorer.score(features, lengths, mask)  # no alerts=
        assert result.shape == (1,)
        assert np.all(np.isfinite(result))
