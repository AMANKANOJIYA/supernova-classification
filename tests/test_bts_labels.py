"""Unit tests for src/readers/bts_labels.py -- pure classification logic and
the join, no network required (fetch_bts_sample itself is not exercised
here; see chat report for the real fetch's join-rate verification)."""

import polars as pl
import pytest

from src.readers import bts_labels
from src.schema import CanonicalAlert, Observation, StaticContext, Survey


def _static():
    return StaticContext(
        host_photoz=None,
        host_photoz_err=None,
        host_offset_arcsec=0.5,
        host_ellipticity=None,
        host_mag_r=20.0,
        mw_ebv=0.05,
        gal_lat=30.0,
        real_bogus_score=0.9,
    )


def _alert(object_id, class_id=None):
    return CanonicalAlert(
        object_id=object_id,
        survey=Survey.ZTF,
        observations=[Observation(mjd=60000.0, lam_eff=633.9, flux=10.0, flux_err=1.0, is_detection=True)],
        static=_static(),
        cutouts=None,
        class_id=class_id,
    )


class TestClassifyBtsType:
    @pytest.mark.parametrize(
        "type_str,expected_class_id",
        [
            ("SN Ia", 2222),
            ("SN Ia-91T", 2222),
            ("SN Ia-91bg", 2226),  # must NOT fold into 2222
            ("SN Iax", 2225),  # must NOT fold into 2222
            ("SN Ib", 2223),
            ("SN Ic", 2223),
            ("SN Ic-BL", 2223),
            ("SN Ib/c", 2223),
            ("SN II", 2224),
            ("SN IIn", 2224),
            ("SN IIP", 2224),
            ("SN IIL", 2224),
            ("SN IIb", 2224),  # §4: IIb is II, not Ib/c
            ("AGN", 2332),
            ("SLSN-I", 2242),
            ("SLSN-II", 2242),
            ("TDE", 2243),
            ("TDE-H-He", 2243),
            ("TDE-He", 2243),
            ("TDE-featureless", 2243),
            ("ILRT", 2244),
            ("Ca-rich", 2245),
            ("SN Ca-rich-Ca", 2245),
        ],
    )
    def test_mapped_types(self, type_str, expected_class_id):
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id == expected_class_id
        assert result.category == "mapped"

    @pytest.mark.parametrize("type_str", ["-", "Other", "other"])
    def test_unclassified_types_are_none_not_a_negative(self, type_str):
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id is None
        assert result.category == "unclassified"

    @pytest.mark.parametrize("type_str", ["CV?", "AGN?"])
    def test_uncertain_types_are_none(self, type_str):
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id is None
        assert result.category == "uncertain"

    @pytest.mark.parametrize(
        "type_str",
        [
            "SN Ia-pec",
            "SN II-pec",
            "SN Ib-pec",
            "SN Ic-pec",
            "SN IIn-pec",
            "SN Ia-CSM",
            "SN Ia-SC",
        ],
    )
    def test_ambiguous_peculiar_types_are_reported_not_bucketed(self, type_str):
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id is None
        assert result.category == "ambiguous"

    @pytest.mark.parametrize("type_str", ["CV", "nova", "LBV", "LRN"])
    def test_confirmed_nonsn_with_no_tree_node(self, type_str):
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id is None
        assert result.category == "no_tree_node"

    @pytest.mark.parametrize("type_str", ["SN Ibn", "SN Icn"])
    def test_csm_interacting_excluded_not_ibc(self, type_str):
        """Day 8 overturn: Ibn/Icn must NOT map to 2223 (Ib/c) despite the
        superficially matching name -- excluded from the 3-class set
        entirely, distinct from no_tree_node (no matching node exists) and
        from ambiguous (family itself unclear)."""
        result = bts_labels.classify_bts_type(type_str)
        assert result.class_id is None
        assert result.category == "excluded_csm_interacting"

    def test_unrecognized_string_is_safety_net_not_silent(self, caplog):
        result = bts_labels.classify_bts_type("SN Totally-New-Type")
        assert result.class_id is None
        assert result.category == "unrecognized"
        assert "unrecognized BTS type string" in caplog.text

    def test_every_known_type_is_disjoint_across_buckets(self):
        """No string should be double-counted across the four None-yielding
        buckets or MAPPED -- a duplicate would mean two different rules
        disagree about the same string."""
        all_none_buckets = [
            bts_labels.UNCLASSIFIED,
            bts_labels.UNCERTAIN,
            bts_labels.AMBIGUOUS,
            bts_labels.NO_TREE_NODE,
            bts_labels.EXCLUDED_CSM_INTERACTING,
        ]
        seen = set()
        for bucket in all_none_buckets:
            assert not (bucket & seen), f"overlap found: {bucket & seen}"
            seen |= bucket
        assert not (seen & set(bts_labels.MAPPED)), "a string is in both MAPPED and a None bucket"


class TestJoinObjectIds:
    def test_join_matches_by_ztfid(self):
        sample = pl.DataFrame(
            {
                "ZTFID": ["ZTF20aaa", "ZTF20bbb"],
                "type": ["SN Ia", "-"],
            }
        )
        result = bts_labels.join_object_ids(["ZTF20aaa", "ZTF20bbb", "ZTF20ccc"], sample)
        assert result["ZTF20aaa"].class_id == 2222
        assert result["ZTF20aaa"].category == "mapped"
        assert result["ZTF20bbb"].class_id is None
        assert result["ZTF20bbb"].category == "unclassified"

    def test_object_absent_from_bts_sample_is_none_not_unclassified(self):
        """Distinguish 'BTS has no row for this object at all' from 'BTS
        has a row but its type is unclassified (-)' -- both are ultimately
        unusable, but for different reasons worth telling apart."""
        sample = pl.DataFrame({"ZTFID": ["ZTF20aaa"], "type": ["SN Ia"]})
        result = bts_labels.join_object_ids(["ZTF20aaa", "ZTF20ccc"], sample)
        assert result["ZTF20ccc"] is None


class TestApplyClassificationsToAlerts:
    """Day 9: joining BTS types directly onto `CanonicalAlert`s (the actual
    fine-tuning-time step described in this module's docstring)."""

    def test_mapped_object_gets_class_id_filled_in(self):
        alerts = [_alert("ZTF20aaa"), _alert("ZTF20bbb")]
        sample = pl.DataFrame({"ZTFID": ["ZTF20aaa", "ZTF20bbb"], "type": ["SN Ia", "SN Ib"]})
        new_alerts, stats = bts_labels.apply_classifications_to_alerts(alerts, sample)

        by_id = {a.object_id: a for a in new_alerts}
        assert by_id["ZTF20aaa"].class_id == 2222
        assert by_id["ZTF20bbb"].class_id == 2223
        assert stats == {"mapped_ia": 1, "mapped_ibc": 1}

    def test_non_mapped_and_absent_objects_stay_none(self):
        alerts = [_alert("ZTF20aaa"), _alert("ZTF20bbb"), _alert("ZTF20ccc")]
        sample = pl.DataFrame({"ZTFID": ["ZTF20aaa", "ZTF20bbb"], "type": ["-", "AGN"]})
        new_alerts, stats = bts_labels.apply_classifications_to_alerts(alerts, sample)

        by_id = {a.object_id: a for a in new_alerts}
        assert by_id["ZTF20aaa"].class_id is None  # unclassified
        assert by_id["ZTF20bbb"].class_id == 2332  # mapped, but non-3-class leaf
        assert by_id["ZTF20ccc"].class_id is None  # not in BTS sample at all
        assert stats == {"unclassified": 1, "mapped_other_leaf": 1, "not_in_bts_sample": 1}

    def test_excluded_csm_interacting_stays_none_not_ibc(self):
        """Day 8 overturn, exercised through the real join path: an Ibn/Icn
        object must never end up with class_id=2223."""
        alerts = [_alert("ZTF20aaa")]
        sample = pl.DataFrame({"ZTFID": ["ZTF20aaa"], "type": ["SN Ibn"]})
        new_alerts, stats = bts_labels.apply_classifications_to_alerts(alerts, sample)

        assert new_alerts[0].class_id is None
        assert stats == {"excluded_csm_interacting": 1}

    def test_every_input_alert_accounted_for_exactly_once(self):
        alerts = [_alert(f"ZTF{i}") for i in range(5)]
        sample = pl.DataFrame({"ZTFID": ["ZTF0", "ZTF1"], "type": ["SN Ia", "-"]})
        _, stats = bts_labels.apply_classifications_to_alerts(alerts, sample)
        assert sum(stats.values()) == len(alerts)

    def test_only_class_id_changes_other_fields_untouched(self):
        alert = _alert("ZTF20aaa")
        sample = pl.DataFrame({"ZTFID": ["ZTF20aaa"], "type": ["SN Ia"]})
        new_alerts, _ = bts_labels.apply_classifications_to_alerts([alert], sample)

        new_alert = new_alerts[0]
        assert new_alert.class_id == 2222
        assert new_alert.observations == alert.observations
        assert new_alert.static == alert.static
        assert new_alert.survey == alert.survey
