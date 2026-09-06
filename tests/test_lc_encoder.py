"""Unit tests for src/models/lc_encoder.py -- synthetic tensors only. The
end-to-end causality property (tensorize windowing + model composed) is
Gate B's job (tests/test_gate_b.py); these tests cover the model in
isolation: shapes, padding/masking correctness, and that the model itself
does not leak across batch items or future timesteps.
"""

import torch

from src.models.lc_encoder import LCEncoder


def _make_model(input_size=6, hidden_size=16, num_layers=2, dropout=0.0, num_classes=3):
    torch.manual_seed(0)
    return LCEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    )


class TestShapes:
    def test_output_shape_matches_batch_and_num_classes(self):
        model = _make_model()
        features = torch.randn(4, 10, 6)
        lengths = torch.tensor([10, 7, 3, 1])
        logits = model(features, lengths)
        assert logits.shape == (4, 10, 3)

    def test_single_item_single_timestep(self):
        model = _make_model()
        features = torch.randn(1, 1, 6)
        lengths = torch.tensor([1])
        logits = model(features, lengths)
        assert logits.shape == (1, 1, 3)


class TestBidirectionalIsFalse:
    def test_gru_is_not_bidirectional(self):
        """CLAUDE.md §2.3 is absolute -- this must never become a config
        option. Checking the actual constructed nn.GRU, not just trusting
        the constructor call."""
        model = _make_model()
        assert model.gru.bidirectional is False


class TestNoLeakageBetweenBatchItems:
    def test_short_sequence_output_unaffected_by_longer_batchmate(self):
        """A packed GRU must never let one sequence's padding influence
        another sequence's real timesteps. Verify by comparing a short
        sequence's output when run alone vs. batched alongside a much
        longer, differently-valued sequence."""
        model = _make_model(dropout=0.0)
        model.eval()

        torch.manual_seed(1)
        short = torch.randn(1, 3, 6)
        long_ = torch.randn(1, 8, 6)

        alone_logits = model(short, torch.tensor([3]))

        padded = torch.zeros(2, 8, 6)
        padded[0, :3] = short[0]
        padded[1, :8] = long_[0]
        # garbage in the padded region of the short sequence -- must not
        # affect its own real-timestep output either.
        padded[0, 3:] = 1e9
        batched_logits = model(padded, torch.tensor([3, 8]))

        assert torch.allclose(alone_logits[0], batched_logits[0, :3], atol=1e-5)


class TestCausalAtModelLevel:
    def test_truncated_prefix_matches_full_run_up_to_truncation(self):
        """The model itself (given identical, already-windowed input up to
        t) must not change its prediction at t when more of the future is
        appended. This is the same property Gate B checks end-to-end
        (tensorize + model); here it is isolated to the model."""
        model = _make_model(dropout=0.0)
        model.eval()

        torch.manual_seed(2)
        full = torch.randn(1, 6, 6)

        full_logits = model(full, torch.tensor([6]))

        for t in range(1, 7):
            truncated_logits = model(full[:, :t], torch.tensor([t]))
            assert torch.allclose(truncated_logits[0], full_logits[0, :t], atol=1e-5), (
                f"causality violated at t={t}"
            )


class TestEvalModeIsDeterministic:
    def test_two_forward_passes_in_eval_mode_match(self):
        model = _make_model(dropout=0.5)  # dropout on, but eval() must disable it
        model.eval()
        features = torch.randn(2, 5, 6)
        lengths = torch.tensor([5, 3])
        out1 = model(features, lengths)
        out2 = model(features, lengths)
        assert torch.equal(out1, out2)
