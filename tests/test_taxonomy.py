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


class TestLoadGentypeOfClassidRealFile:
    """CLAUDE.md: if the CSVs aren't on disk, write against documented columns
    and skip rather than invent data. As of Day 1, data/raw/elasticc2/ is
    empty, so this is skipped. Column names ('gentype', 'classid') are no
    longer a guess — confirmed directly by the user on Day 2 — but the real
    file's actual content is still unverified until it lands.
    """

    _PATH = Path("data/raw/elasticc2/elasticc2_gentypeofclassid.csv")

    @pytest.mark.skipif(
        not _PATH.exists(),
        reason=(
            "elasticc2_gentypeofclassid.csv not present on disk yet (expected — "
            "Day 0 downloads incomplete). Re-enable once the file lands."
        ),
    )
    def test_loads_mapping_from_real_file(self):
        mapping = taxonomy.load_gentype_of_classid(str(self._PATH))
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for gentype, class_id in mapping.items():
            assert isinstance(gentype, int)
            assert class_id in taxonomy._VALID_IDS  # noqa: SLF001 (internal check)


class TestLoadClassidOfGentypeRealFile:
    _PATH = Path("data/raw/elasticc2/elasticc2_classidofgentype.csv")

    @pytest.mark.skipif(
        not _PATH.exists(),
        reason=(
            "elasticc2_classidofgentype.csv not present on disk yet (expected — "
            "Day 0 downloads incomplete). Re-enable once the file lands."
        ),
    )
    def test_loads_mapping_from_real_file(self):
        mapping = taxonomy.load_classid_of_gentype(str(self._PATH))
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for gentype, class_ids in mapping.items():
            assert isinstance(gentype, int)
            assert isinstance(class_ids, list)
            for class_id in class_ids:
                assert class_id in taxonomy._VALID_IDS  # noqa: SLF001 (internal check)


# ---------------------------------------------------------------------------
# The functions above are unverified against the *real* CSVs (not on disk
# yet), but the column-name contract the user confirmed ('gentype', 'classid')
# and the documented blank-handling behavior can be tested now against
# synthetic CSVs, independent of Day 3's data drop.
# ---------------------------------------------------------------------------


class TestLoadGentypeOfClassidSynthetic:
    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("gentype,not_classid\n10,2222\n")
        with pytest.raises(ValueError, match="classid"):
            taxonomy.load_gentype_of_classid(str(path))

    def test_blank_gentype_rows_are_skipped_and_logged_not_raised(self, tmp_path, caplog):
        """Blank gentype is legitimate here (category-level classIds like
        2220 SN-like have no single specific gentype) -- must not raise."""
        path = tmp_path / "with_blanks.csv"
        path.write_text("gentype,classid\n10,2222\n,2220\n,2230\n20,2223\n")
        with caplog.at_level("INFO", logger="src.taxonomy"):
            mapping = taxonomy.load_gentype_of_classid(str(path))
        assert mapping == {10: 2222, 20: 2223}
        assert any("skipped 2" in record.message for record in caplog.records)

    def test_no_blanks_logs_nothing(self, tmp_path, caplog):
        path = tmp_path / "no_blanks.csv"
        path.write_text("gentype,classid\n10,2222\n20,2223\n")
        with caplog.at_level("INFO", logger="src.taxonomy"):
            taxonomy.load_gentype_of_classid(str(path))
        assert caplog.records == []

    def test_blank_rows_do_not_count_toward_duplicate_check(self, tmp_path):
        """Two category-level rows, both blank, must not look like a
        duplicate-gentype collision -- blanks are excluded before the
        uniqueness check runs."""
        path = tmp_path / "blanks_not_dupes.csv"
        path.write_text("gentype,classid\n,2220\n,2230\n10,2222\n")
        mapping = taxonomy.load_gentype_of_classid(str(path))
        assert mapping == {10: 2222}

    def test_duplicate_nonblank_gentype_raises(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_text("gentype,classid\n10,2222\n10,2223\n")
        with pytest.raises(ValueError, match="unique"):
            taxonomy.load_gentype_of_classid(str(path))

    def test_clean_file_loads_correctly(self, tmp_path):
        path = tmp_path / "clean.csv"
        path.write_text("gentype,classid\n10,2222\n20,2223\n30,2224\n")
        mapping = taxonomy.load_gentype_of_classid(str(path))
        assert mapping == {10: 2222, 20: 2223, 30: 2224}


class TestLoadClassidOfGentypeSynthetic:
    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("not_gentype,classid\n10,2222\n")
        with pytest.raises(ValueError, match="gentype"):
            taxonomy.load_classid_of_gentype(str(path))

    def test_repeated_gentype_is_intentional_not_an_error(self, tmp_path):
        """Every SN gentype has a row for its category (2220) as well as its
        specific classid — repetition is expected, not a data error."""
        path = tmp_path / "repeated.csv"
        path.write_text("gentype,classid\n10,2220\n10,2222\n")
        mapping = taxonomy.load_classid_of_gentype(str(path))
        assert mapping == {10: [2220, 2222]}

    def test_blank_gentype_rows_are_skipped_and_logged(self, tmp_path, caplog):
        # classid=22 is a stand-in category-level (two-digit) classId row with no gentype.
        path = tmp_path / "with_blanks.csv"
        path.write_text("gentype,classid\n10,2220\n,22\n,22\n20,2223\n")
        with caplog.at_level("INFO", logger="src.taxonomy"):
            mapping = taxonomy.load_classid_of_gentype(str(path))
        assert mapping == {10: [2220], 20: [2223]}
        assert any("skipped 2" in record.message for record in caplog.records)

    def test_no_blanks_logs_nothing(self, tmp_path, caplog):
        path = tmp_path / "no_blanks.csv"
        path.write_text("gentype,classid\n10,2220\n20,2223\n")
        with caplog.at_level("INFO", logger="src.taxonomy"):
            taxonomy.load_classid_of_gentype(str(path))
        assert caplog.records == []


class TestGentypeFileAsymmetry:
    """The real distinction between the two files (both have legitimate
    blank-gentype category rows -- that part is symmetric): non-blank
    gentype is unique in gentypeofclassid, but repeats across the hierarchy
    in classidofgentype. Same underlying data, read through each file."""

    def test_gentype_of_classid_is_keyed_uniquely(self, tmp_path):
        path = tmp_path / "gentypeofclassid.csv"
        path.write_text("gentype,classid\n10,2222\n,2220\n20,2223\n")
        mapping = taxonomy.load_gentype_of_classid(str(path))
        assert mapping == {10: 2222, 20: 2223}

    def test_classid_of_gentype_repeats_the_same_gentype_across_the_hierarchy(self, tmp_path):
        path = tmp_path / "classidofgentype.csv"
        path.write_text("gentype,classid\n10,2220\n10,2222\n,2220\n20,2223\n")
        mapping = taxonomy.load_classid_of_gentype(str(path))
        assert mapping == {10: [2220, 2222], 20: [2223]}
