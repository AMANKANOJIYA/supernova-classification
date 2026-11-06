"""Serving classifier: ONNX export of the LC+static fusion model (deployed
2026-11-05, CLAUDE.md §11). Uses a randomly initialised model written to a
temp checkpoint, so no trained artifact or data is needed."""

import numpy as np
import pytest
import torch

from src.features.tensorize import NUM_FEATURES
from src.models.fusion import LCStaticFusionModel
from src.models.static_encoder import NUM_STATIC_FIELDS, StaticFeatureNormalizer, build_static_vector
from src.schema import StaticContext
from src.serve.model import ONNXClassifier, export_tier_b_classifier_to_onnx, load_lc_static_checkpoint

STATIC_FULL = StaticContext(
    host_photoz=None, host_photoz_err=None, host_offset_arcsec=0.8, host_ellipticity=None,
    host_mag_r=18.2, mw_ebv=0.04, gal_lat=35.0, real_bogus_score=0.97,
)
STATIC_EMPTY = StaticContext(*([None] * 8))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch.manual_seed(0)
    model = LCStaticFusionModel(lc_input_size=NUM_FEATURES, static_hidden_size=32, static_output_size=16)
    d = tmp_path_factory.mktemp("clf")
    ckpt = d / "with_static_checkpoint.pt"
    rng = np.random.default_rng(0)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "static_normalizer_mean": rng.normal(size=NUM_STATIC_FIELDS).tolist(),
         "static_normalizer_std": (rng.uniform(0.5, 2, size=NUM_STATIC_FIELDS)).tolist()},
        ckpt,
    )
    onnx_path = d / "with_static_model.onnx"
    export_tier_b_classifier_to_onnx(checkpoint_path=ckpt, onnx_path=onnx_path)
    return ckpt, onnx_path


def _batch(lengths, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros((len(lengths), max(lengths), NUM_FEATURES), dtype=np.float32)
    for i, n in enumerate(lengths):
        x[i, :n] = rng.normal(size=(n, NUM_FEATURES))
    return x, np.array(lengths, dtype=np.int64)


def test_onnx_matches_torch_fusion_model_at_every_real_timestep(artifacts):
    ckpt, onnx_path = artifacts
    model, normalizer = load_lc_static_checkpoint(ckpt)
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    x, lengths = _batch([7, 3, 5])
    statics = [STATIC_FULL, STATIC_EMPTY, STATIC_FULL]
    s = np.stack([build_static_vector(st, normalizer) for st in statics])
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(s)).numpy()
    (got,) = clf.session.run(["logits"], {"features": x, "static_features": clf.static_matrix(statics)})
    for i, n in enumerate(lengths):
        np.testing.assert_allclose(got[i, :n], ref[i, :n], rtol=1e-5, atol=1e-5)


def test_padding_and_batching_do_not_change_predictions(artifacts):
    ckpt, onnx_path = artifacts
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    x, lengths = _batch([7, 3, 5])
    statics = [STATIC_FULL, STATIC_EMPTY, STATIC_FULL]
    batched = clf.predict_probs(x, lengths, statics)
    for i, n in enumerate(lengths):
        single = clf.predict_probs(x[i : i + 1, :n], lengths[i : i + 1], [statics[i]])
        np.testing.assert_allclose(batched[i], single[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, rtol=1e-5)


def test_future_timesteps_do_not_change_past_predictions(artifacts):
    """CLAUDE.md §2.3, for the exported graph: the prediction as of step t
    is identical whether or not later steps are present."""
    ckpt, onnx_path = artifacts
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    x, _ = _batch([9])
    full = clf.session.run(["logits"], {"features": x, "static_features": clf.static_matrix([STATIC_FULL])})[0]
    for t in range(1, 9):
        trunc = clf.session.run(["logits"], {"features": x[:, :t], "static_features": clf.static_matrix([STATIC_FULL])})[0]
        np.testing.assert_allclose(trunc[0, t - 1], full[0, t - 1], rtol=1e-5, atol=1e-6)


def test_all_missing_static_context_gives_finite_probabilities(artifacts):
    ckpt, onnx_path = artifacts
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    v = clf.static_matrix([STATIC_EMPTY])[0]
    assert np.all(v[:NUM_STATIC_FIELDS] == 0.0) and np.all(v[NUM_STATIC_FIELDS:] == 1.0)
    x, lengths = _batch([4])
    p = clf.predict_probs(x, lengths, [STATIC_EMPTY])
    assert np.all(np.isfinite(p))


def test_static_count_must_match_batch(artifacts):
    ckpt, onnx_path = artifacts
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    x, lengths = _batch([4, 2])
    with pytest.raises(ValueError):
        clf.predict_probs(x, lengths, [STATIC_FULL])


def test_normalizer_is_the_one_stored_in_the_checkpoint(artifacts):
    ckpt, onnx_path = artifacts
    clf = ONNXClassifier(onnx_path=onnx_path, checkpoint_path=ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert isinstance(clf.normalizer, StaticFeatureNormalizer)
    np.testing.assert_allclose(clf.normalizer.mean, saved["static_normalizer_mean"])
    np.testing.assert_allclose(clf.normalizer.std, saved["static_normalizer_std"])
