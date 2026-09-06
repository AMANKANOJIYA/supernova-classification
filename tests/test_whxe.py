"""Tests for src/models/whxe.py using hand-computable synthetic cases --
same philosophy as tests/test_eval.py (CLAUDE.md §12: golden tests before
generation), since this is new, non-trivial math that must be verified
before any real (hours-long) training run uses it.
"""

import math

import pytest
import torch

from src import taxonomy
from src.models import whxe


class TestNodeIndexing:
    def test_covers_every_taxonomy_node_exactly_once(self):
        assert whxe.NUM_TAXONOMY_NODES == len(taxonomy.ordered_class_ids())
        indices = {whxe.node_index(cid) for cid in taxonomy.ordered_class_ids()}
        assert indices == set(range(whxe.NUM_TAXONOMY_NODES))


class TestDepthFromRoot:
    def test_root_is_zero(self):
        assert whxe.depth_from_root(taxonomy.ROOT_ID) == 0

    def test_known_depths(self):
        assert whxe.depth_from_root(1000) == 1
        assert whxe.depth_from_root(2000) == 1
        assert whxe.depth_from_root(2200) == 2
        assert whxe.depth_from_root(2220) == 3
        assert whxe.depth_from_root(2222) == 4

    def test_target_leaves_share_one_depth(self):
        depths = {whxe.depth_from_root(c) for c in taxonomy.TARGET_CLASS_IDS}
        assert depths == {4}


class TestAncestorPath:
    def test_excludes_root(self):
        path = whxe.ancestor_path(2222)
        assert taxonomy.ROOT_ID not in path

    def test_known_path(self):
        assert whxe.ancestor_path(2222) == (2222, 2220, 2200, 2000)
        assert whxe.ancestor_path(2223) == (2223, 2220, 2200, 2000)
        assert whxe.ancestor_path(2224) == (2224, 2220, 2200, 2000)


class TestComputeNodeClassWeights:
    def test_hand_computed_values(self):
        # N_all=60, N_labels=3
        leaf_counts = {2222: 10, 2223: 20, 2224: 30}
        weights = whxe.compute_node_class_weights(leaf_counts)

        assert weights[2222] == pytest.approx(60 / (3 * 10))
        assert weights[2223] == pytest.approx(60 / (3 * 20))
        assert weights[2224] == pytest.approx(60 / (3 * 30))
        # internal ancestors: subtree total = 10+20+30=60 for all three
        # (every example in this toy set falls under 2220/2200/2000)
        assert weights[2220] == pytest.approx(60 / (3 * 60))
        assert weights[2200] == pytest.approx(60 / (3 * 60))
        assert weights[2000] == pytest.approx(60 / (3 * 60))

    def test_parent_weight_is_lower_than_every_leaf_weight(self):
        """Paper, Sec 2: "parent classes have a lower weighting than leaf
        nodes (as parent classes include all sibling leaves)"."""
        leaf_counts = {2222: 172484, 2223: 240359, 2224: 430774}
        weights = whxe.compute_node_class_weights(leaf_counts)
        for leaf in (2222, 2223, 2224):
            assert weights[2220] < weights[leaf]
            assert weights[2200] < weights[leaf]
            assert weights[2000] < weights[leaf]


class TestWhxeLoss:
    _NODE_WEIGHTS_UNIT = {c: 1.0 for c in (2222, 2223, 2224, 2220, 2200, 2000)}

    def test_hand_computed_single_example_uniform_logits(self):
        """All logits zero -> every level's softmax is uniform over its
        sibling group. Loss reduces to a closed-form sum of
        lambda(node)*log(|siblings|), hand-verifiable term by term."""
        logits = torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES)
        class_index = torch.tensor([0])  # leaf_class_ids[0] = 2222
        mask = torch.ones(1, 1, dtype=torch.bool)

        loss = whxe.whxe_loss(
            logits, class_index, mask, leaf_class_ids=(2222, 2223, 2224),
            node_weights=self._NODE_WEIGHTS_UNIT,
        )

        # h=0: 2222 among children(2220)=[2222,2223,2224,2225,2226] (5-way)
        # h=1: 2220 among children(2200)=[2220,2230,2240]           (3-way)
        # h=2: 2200 among children(2000)=[2200,2300]                (2-way)
        # h=3: 2000 among children(ROOT)=[1000,2000]                (2-way)
        expected = (
            math.exp(-0.5 * 4) * math.log(5)
            + math.exp(-0.5 * 3) * math.log(3)
            + math.exp(-0.5 * 2) * math.log(2)
            + math.exp(-0.5 * 1) * math.log(2)
        )
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_node_weights_scale_the_loss_linearly(self):
        logits = torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES)
        class_index = torch.tensor([0])
        mask = torch.ones(1, 1, dtype=torch.bool)

        base = whxe.whxe_loss(
            logits, class_index, mask, (2222, 2223, 2224), self._NODE_WEIGHTS_UNIT
        )
        doubled_weights = {k: 2.0 * v for k, v in self._NODE_WEIGHTS_UNIT.items()}
        doubled = whxe.whxe_loss(logits, class_index, mask, (2222, 2223, 2224), doubled_weights)
        assert doubled.item() == pytest.approx(2 * base.item())

    def test_padded_timesteps_are_excluded(self):
        """Second timestep is garbage + unmasked -- must not affect the loss
        (same invariant CLAUDE.md's masking convention requires elsewhere)."""
        logits = torch.zeros(1, 2, whxe.NUM_TAXONOMY_NODES)
        logits[0, 1] = 1e6  # garbage in the padded slot
        class_index = torch.tensor([0])
        mask = torch.tensor([[True, False]])

        loss = whxe.whxe_loss(
            logits, class_index, mask, (2222, 2223, 2224), self._NODE_WEIGHTS_UNIT
        )
        single_step = whxe.whxe_loss(
            torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES),
            class_index,
            torch.ones(1, 1, dtype=torch.bool),
            (2222, 2223, 2224),
            self._NODE_WEIGHTS_UNIT,
        )
        assert loss.item() == pytest.approx(single_step.item())

    def test_per_example_leaf_weight_is_applied_correctly(self):
        """Batch of the three different true classes with DIFFERENT leaf
        weights -- confirms per-example gather picks the right weight, not
        a broadcast average or a fixed index."""
        logits = torch.zeros(3, 1, whxe.NUM_TAXONOMY_NODES)
        class_index = torch.tensor([0, 1, 2])  # one of each: 2222, 2223, 2224
        mask = torch.ones(3, 1, dtype=torch.bool)
        weights = dict(self._NODE_WEIGHTS_UNIT)
        weights[2222], weights[2223], weights[2224] = 1.0, 10.0, 100.0

        loss = whxe.whxe_loss(logits, class_index, mask, (2222, 2223, 2224), weights)

        # per-example loss = leaf-level term (weight varies: 1/10/100 * lambda(2222)*log5)
        #                     + fixed shared terms (2220/2200/2000, weight=1 each)
        leaf_term_unit = math.exp(-0.5 * 4) * math.log(5)
        shared_terms = (
            math.exp(-0.5 * 3) * math.log(3)
            + math.exp(-0.5 * 2) * math.log(2)
            + math.exp(-0.5 * 1) * math.log(2)
        )
        expected_mean = (
            (1.0 * leaf_term_unit + shared_terms)
            + (10.0 * leaf_term_unit + shared_terms)
            + (100.0 * leaf_term_unit + shared_terms)
        ) / 3
        assert loss.item() == pytest.approx(expected_mean, rel=1e-4)

    def test_raises_on_leaves_at_different_depths(self):
        with pytest.raises(ValueError, match="depth"):
            whxe.whxe_loss(
                torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES),
                torch.tensor([0]),
                torch.ones(1, 1, dtype=torch.bool),
                (1000, 2222),  # depths 1 and 4
                {1000: 1.0, 2222: 1.0},
            )

    def test_raises_on_leaves_with_different_parents_at_some_level(self):
        # 2222 (parent 2220) and 2232 KN (parent 2230) are both depth 4 but
        # diverge at the leaf level -- no single shared sibling group there.
        with pytest.raises(ValueError, match="parent"):
            whxe.whxe_loss(
                torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES),
                torch.tensor([0]),
                torch.ones(1, 1, dtype=torch.bool),
                (2222, 2232),
                {2222: 1.0, 2232: 1.0, 2220: 1.0, 2230: 1.0, 2200: 1.0, 2000: 1.0},
            )


class TestLeafProbabilities:
    def test_renormalizes_over_just_the_target_leaves(self):
        logits = torch.zeros(1, 1, whxe.NUM_TAXONOMY_NODES)
        logits[0, 0, whxe.node_index(2222)] = 1.0
        logits[0, 0, whxe.node_index(2223)] = 2.0
        logits[0, 0, whxe.node_index(2224)] = 3.0
        logits[0, 0, whxe.node_index(2225)] = 1000.0  # must be ignored entirely

        probs = whxe.leaf_probabilities(logits, (2222, 2223, 2224))
        expected = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1)
        assert torch.allclose(probs[0, 0], expected, atol=1e-6)
        assert probs[0, 0].sum().item() == pytest.approx(1.0)


class TestPredictedNodeMatches:
    def _logits_favoring(self, winners: dict[int, list[int]]) -> torch.Tensor:
        """winners: {sibling_group_representative_parent_children_list: [...]}
        not used directly -- helper builds logits favoring a specific node at
        each level by setting its logit high and its siblings' logits low."""
        logits = torch.zeros(whxe.NUM_TAXONOMY_NODES)
        for node_id, siblings in winners.items():
            logits[whxe.node_index(node_id)] = 10.0
            for s in siblings:
                if s != node_id:
                    logits[whxe.node_index(s)] = -10.0
        return logits

    def test_cascade_reaches_2220_when_every_level_favors_it(self):
        logits = self._logits_favoring(
            {
                2000: taxonomy.children(taxonomy.ROOT_ID),  # 2000 beats 1000
                2200: taxonomy.children(2000),  # 2200 beats 2300
                2220: taxonomy.children(2200),  # 2220 beats 2230/2240
            }
        )
        assert whxe.predicted_node_matches(logits, 2220).item() is True

    def test_cascade_misses_2220_when_top_level_favors_static(self):
        logits = self._logits_favoring({1000: taxonomy.children(taxonomy.ROOT_ID)})
        assert whxe.predicted_node_matches(logits, 2220).item() is False

    def test_cascade_misses_2220_when_only_the_last_level_flips(self):
        logits = self._logits_favoring(
            {
                2000: taxonomy.children(taxonomy.ROOT_ID),
                2200: taxonomy.children(2000),
                2230: taxonomy.children(2200),  # Fast beats SN-like here
            }
        )
        assert whxe.predicted_node_matches(logits, 2220).item() is False

    def test_batched_input(self):
        good = self._logits_favoring(
            {
                2000: taxonomy.children(taxonomy.ROOT_ID),
                2200: taxonomy.children(2000),
                2220: taxonomy.children(2200),
            }
        )
        bad = self._logits_favoring({1000: taxonomy.children(taxonomy.ROOT_ID)})
        batch = torch.stack([good, bad])
        matches = whxe.predicted_node_matches(batch, 2220)
        assert matches.tolist() == [True, False]
