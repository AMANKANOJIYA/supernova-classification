"""Tests for src/models/fusion.py."""

import pytest
import torch

from src.models.fusion import LCStaticDRBFusionModel, LCStaticFusionModel, load_shared_submodules
from src.models.static_encoder import NUM_STATIC_FIELDS


class TestLCStaticFusionModel:
    def test_output_shape(self):
        model = LCStaticFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, num_classes=3,
        )
        features = torch.randn(5, 10, 6)
        lengths = torch.tensor([10, 8, 5, 3, 1])
        static_features = torch.randn(5, 2 * NUM_STATIC_FIELDS)
        logits = model(features, lengths, static_features)
        assert logits.shape == (5, 10, 3)

    def test_gradients_flow_to_both_branches(self):
        model = LCStaticFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, num_classes=3,
        )
        features = torch.randn(3, 4, 6)
        lengths = torch.tensor([4, 3, 2])
        static_features = torch.randn(3, 2 * NUM_STATIC_FIELDS)
        logits = model(features, lengths, static_features)
        logits.sum().backward()

        assert model.gru.weight_ih_l0.grad is not None
        assert torch.isfinite(model.gru.weight_ih_l0.grad).all()
        assert model.static_encoder.mlp[0].weight.grad is not None
        assert torch.isfinite(model.static_encoder.mlp[0].weight.grad).all()

    def test_static_features_affect_output(self):
        """Confirms the static branch is actually wired into the head, not
        a dead end -- two objects with identical light curves but different
        static context must get different logits."""
        torch.manual_seed(0)
        model = LCStaticFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, num_classes=3,
        )
        model.eval()
        features = torch.randn(1, 4, 6).expand(2, 4, 6)
        lengths = torch.tensor([4, 4])
        static_a = torch.zeros(1, 2 * NUM_STATIC_FIELDS)
        static_b = torch.ones(1, 2 * NUM_STATIC_FIELDS)
        static_features = torch.cat([static_a, static_b], dim=0)

        with torch.no_grad():
            logits = model(features, lengths, static_features)
        assert not torch.allclose(logits[0], logits[1])

    def test_padded_timesteps_do_not_crash_or_produce_nan(self):
        model = LCStaticFusionModel(lc_input_size=6, lc_hidden_size=8, static_hidden_size=4, static_output_size=4)
        features = torch.zeros(2, 5, 6)
        features[0, :2] = torch.randn(2, 6)
        features[1, :5] = torch.randn(5, 6)
        lengths = torch.tensor([2, 5])
        static_features = torch.randn(2, 2 * NUM_STATIC_FIELDS)
        logits = model(features, lengths, static_features)
        assert torch.isfinite(logits).all()


class TestLCStaticDRBFusionModel:
    """Day 9 real_bogus_score ablation model (CLAUDE.md §6 step 9)."""

    def test_output_shape(self):
        model = LCStaticDRBFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, drb_hidden_size=4, drb_output_size=2,
            num_classes=3,
        )
        features = torch.randn(5, 10, 6)
        lengths = torch.tensor([10, 8, 5, 3, 1])
        static_features = torch.randn(5, 2 * NUM_STATIC_FIELDS)
        drb_features = torch.randn(5, 2)
        logits = model(features, lengths, static_features, drb_features)
        assert logits.shape == (5, 10, 3)

    def test_drb_branch_affects_output(self):
        """The drb branch must actually be wired into the head -- two
        objects identical except for drb_features must get different
        logits."""
        torch.manual_seed(0)
        model = LCStaticDRBFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, drb_hidden_size=4, drb_output_size=2,
        )
        model.eval()
        features = torch.randn(1, 4, 6).expand(2, 4, 6)
        lengths = torch.tensor([4, 4])
        static_features = torch.zeros(1, 2 * NUM_STATIC_FIELDS).expand(2, 2 * NUM_STATIC_FIELDS)
        drb_features = torch.tensor([[0.0, 1.0], [3.0, 0.0]])

        with torch.no_grad():
            logits = model(features, lengths, static_features, drb_features)
        assert not torch.allclose(logits[0], logits[1])


class TestLoadSharedSubmodules:
    """`load_shared_submodules` warm-starts gru/layer_norm/static_encoder
    from a Tier A `LCStaticFusionModel` checkpoint into a fresh
    `LCStaticDRBFusionModel`, leaving head/drb_branch untouched."""

    def test_copies_matching_submodules_and_leaves_head_random(self):
        source = LCStaticFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, num_classes=3,
        )
        target = LCStaticDRBFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, drb_hidden_size=4, drb_output_size=2,
            num_classes=3,
        )
        original_head_weight = target.head.weight.clone()

        load_shared_submodules(target, source.state_dict())

        assert torch.equal(target.gru.weight_ih_l0, source.gru.weight_ih_l0)
        assert torch.equal(target.layer_norm.weight, source.layer_norm.weight)
        assert torch.equal(target.static_encoder.mlp[0].weight, source.static_encoder.mlp[0].weight)
        # head has a different input width (lc_hidden_size + static_output_size
        # + drb_output_size vs lc_hidden_size + static_output_size) -- must be
        # untouched, not partially/incorrectly copied.
        assert torch.equal(target.head.weight, original_head_weight)

    def test_raises_on_shape_mismatch(self):
        source = LCStaticFusionModel(
            lc_input_size=6, lc_hidden_size=32, lc_num_layers=2,  # mismatched hidden size
            static_hidden_size=8, static_output_size=4, num_classes=3,
        )
        target = LCStaticDRBFusionModel(
            lc_input_size=6, lc_hidden_size=16, lc_num_layers=2,
            static_hidden_size=8, static_output_size=4, drb_hidden_size=4, drb_output_size=2,
            num_classes=3,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            load_shared_submodules(target, source.state_dict())
