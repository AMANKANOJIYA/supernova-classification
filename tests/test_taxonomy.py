"""Tests for the ELAsTiCC2 taxonomy tree (CLAUDE.md §4)."""

from pathlib import Path

import pytest

from src import taxonomy
from src.taxonomy import ROOT_ID, TargetClass


class TestParent:
    def test_leaf_parent(self):
        assert taxonomy.parent(2222) == 2220

    def test_mid_level_parent(self):
        assert taxonomy.parent(2220) == 2200
        assert taxonomy.parent(2200) == 2000

    def test_top_level_parent_is_root(self):
        assert taxonomy.parent(1000) == ROOT_ID
        assert taxonomy.parent(2000) == ROOT_ID

    def test_root_has_no_parent(self):
        assert taxonomy.parent(ROOT_ID) is None

    def test_recurring_branch(self):
        assert taxonomy.parent(2332) == 2330
        assert taxonomy.parent(2330) == 2300
        assert taxonomy.parent(2300) == 2000

    def test_invalid_id_raises(self):
        with pytest.raises(ValueError):
            taxonomy.parent(9999)


class TestParents:
    def test_leaf_full_chain(self):
        assert taxonomy.parents(2222) == [2220, 2200, 2000, ROOT_ID]

    def test_root_chain_is_empty(self):
        assert taxonomy.parents(ROOT_ID) == []

    def test_top_level_chain(self):
        assert taxonomy.parents(1000) == [ROOT_ID]


class TestChildren:
    def test_root_children(self):
        assert taxonomy.children(ROOT_ID) == [1000, 2000]

    def test_sn_like_children(self):
        assert taxonomy.children(2220) == [2222, 2223, 2224, 2225, 2226]

    def test_leaf_has_no_children(self):
        assert taxonomy.children(2222) == []

    def test_static_has_no_children(self):
        assert taxonomy.children(1000) == []

    def test_variable_children(self):
        assert taxonomy.children(2000) == [2200, 2300]


class TestIsLeaf:
    def test_target_classes_are_leaves(self):
        for cid in (2222, 2223, 2224):
            assert taxonomy.is_leaf(cid)

    def test_static_is_a_leaf_despite_being_top_level(self):
        assert taxonomy.is_leaf(1000)

    def test_internal_nodes_are_not_leaves(self):
        for cid in (ROOT_ID, 2000, 2200, 2220, 2300, 2320):
            assert not taxonomy.is_leaf(cid)


class TestToTargetClass:
    def test_ia_leaf(self):
        assert taxonomy.to_target_class(2222) == TargetClass.IA

    def test_ibc_leaf(self):
        assert taxonomy.to_target_class(2223) == TargetClass.IBC

    def test_ii_leaf(self):
        assert taxonomy.to_target_class(2224) == TargetClass.II

    @pytest.mark.parametrize(
        "cid",
        [2225, 2226, 2232, 2233, 2234, 2235, 2242, 2243, 2244, 2245, 2246, 2322, 2332, 1000],
    )
    def test_non_target_leaves_map_to_other(self, cid):
        assert taxonomy.to_target_class(cid) == TargetClass.OTHER

    def test_internal_nodes_map_to_other(self):
        assert taxonomy.to_target_class(2220) == TargetClass.OTHER
        assert taxonomy.to_target_class(2200) == TargetClass.OTHER

    def test_root_maps_to_other(self):
        assert taxonomy.to_target_class(ROOT_ID) == TargetClass.OTHER

    def test_invalid_id_raises(self):
        with pytest.raises(ValueError):
            taxonomy.to_target_class(1234)


class TestOrderedClassIds:
    def test_contains_every_node_exactly_once(self):
        order = taxonomy.ordered_class_ids()
        assert len(order) == len(set(order)) == 30  # 29 tree nodes + root

    def test_deterministic_across_calls(self):
        assert taxonomy.ordered_class_ids() == taxonomy.ordered_class_ids()

    def test_root_is_first(self):
        assert taxonomy.ordered_class_ids()[0] == ROOT_ID

    def test_parent_precedes_child(self):
        order = taxonomy.ordered_class_ids()
        index = {cid: i for i, cid in enumerate(order)}
        for cid in order:
            if cid != ROOT_ID:
                assert index[taxonomy.parent(cid)] < index[cid]


class TestClassIndex:
    def test_matches_ordered_class_ids_position(self):
        order = taxonomy.ordered_class_ids()
        for i, cid in enumerate(order):
            assert taxonomy.class_index(cid) == i


class TestLoadGentypeMapping:
    """CLAUDE.md: if the CSVs aren't on disk, write against documented columns
    and skip rather than invent data. As of Day 1, data/raw/elasticc2/ is
    empty, so this is skipped — see the ASSUMPTION note in
    src/taxonomy.py::load_gentype_mapping about unconfirmed column names.
    """

    _CLASSID_PATH = Path("data/raw/elasticc2/elasticc2_classidofgentype.csv")
    _GENTYPE_PATH = Path("data/raw/elasticc2/elasticc2_gentypeofclassid.csv")

    @pytest.mark.skipif(
        not _CLASSID_PATH.exists(),
        reason=(
            "elasticc2_classidofgentype.csv not present on disk yet (expected — "
            "Day 0 downloads incomplete); column names are also unconfirmed "
            "against real data. Re-enable once the file lands, per CLAUDE.md."
        ),
    )
    def test_loads_mapping_from_real_file(self):
        mapping = taxonomy.load_gentype_mapping(
            str(self._GENTYPE_PATH), str(self._CLASSID_PATH)
        )
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for gentype, class_id in mapping.items():
            assert isinstance(gentype, int)
            assert class_id in taxonomy._VALID_IDS  # noqa: SLF001 (internal check)
