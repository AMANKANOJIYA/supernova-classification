"""Tests for src/models/static_encoder.py -- hand-constructed fixtures plus
one real-data check (CLAUDE.md Day-7: "verify no sentinel values survive
into the encoder")."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import static_encoder as se
from src.schema import CanonicalAlert, Observation, StaticContext, Survey


def _static(**overrides) -> StaticContext:
    base = dict(
        host_photoz=0.1,
        host_photoz_err=0.01,
        host_offset_arcsec=0.5,
        host_ellipticity=0.2,
        host_mag_r=20.0,
        mw_ebv=0.05,
        gal_lat=30.0,
        real_bogus_score=None,
    )
    base.update(overrides)
    return StaticContext(**base)


def _alert(static, object_id="obj") -> CanonicalAlert:
    return CanonicalAlert(
        object_id=object_id,
        survey=Survey.LSST,
        observations=[Observation(mjd=1.0, lam_eff=622.2, flux=1.0, flux_err=1.0, is_detection=True)],
        static=static,
        cutouts=None,
        class_id=2222,
    )


class TestRealBogusScoreOmitted:
    def test_static_fields_excludes_real_bogus_score(self):
        assert "real_bogus_score" not in se.STATIC_FIELDS
        assert se.NUM_STATIC_FIELDS == 7


class TestStaticFeatureNormalizer:
    def test_fit_hand_computed_mean_and_std(self):
        alerts = [
            _alert(_static(host_photoz=0.1)),
            _alert(_static(host_photoz=0.2)),
            _alert(_static(host_photoz=0.3)),
        ]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        idx = se.STATIC_FIELDS.index("host_photoz")
        assert norm.mean[idx] == pytest.approx(0.2)
        assert norm.std[idx] == pytest.approx(np.std([0.1, 0.2, 0.3]))

    def test_missing_values_excluded_from_fit_statistics(self):
        alerts = [
            _alert(_static(host_photoz=0.1)),
            _alert(_static(host_photoz=None)),
            _alert(_static(host_photoz=0.3)),
        ]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        idx = se.STATIC_FIELDS.index("host_photoz")
        # nanmean/nanstd over [0.1, 0.3] only, not treating the missing entry as 0
        assert norm.mean[idx] == pytest.approx(0.2)

    def test_degenerate_zero_variance_field_does_not_divide_by_zero(self):
        alerts = [_alert(_static(gal_lat=30.0)) for _ in range(5)]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        idx = se.STATIC_FIELDS.index("gal_lat")
        assert norm.std[idx] == pytest.approx(1.0)  # guarded, not 0.0
        transformed = norm.transform(se._raw_static_row(alerts[0].static))
        assert np.isfinite(transformed).all()


class TestBuildStaticVector:
    def test_shape_and_no_missing_case(self):
        alerts = [_alert(_static())]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        vec = se.build_static_vector(alerts[0].static, norm)
        assert vec.shape == (2 * se.NUM_STATIC_FIELDS,)
        # no missing fields -> all indicators (second half) are 0.0
        assert np.allclose(vec[se.NUM_STATIC_FIELDS :], 0.0)

    def test_missing_field_gets_zero_value_and_indicator_one(self):
        alerts = [_alert(_static(host_offset_arcsec=None)), _alert(_static(host_offset_arcsec=0.7))]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        vec = se.build_static_vector(alerts[0].static, norm)
        idx = se.STATIC_FIELDS.index("host_offset_arcsec")
        assert vec[idx] == pytest.approx(0.0)  # zero-filled
        assert vec[se.NUM_STATIC_FIELDS + idx] == pytest.approx(1.0)  # indicator set

    def test_present_field_indicator_is_zero(self):
        alerts = [_alert(_static(host_offset_arcsec=None)), _alert(_static(host_offset_arcsec=0.7))]
        norm = se.StaticFeatureNormalizer.fit(alerts)
        vec = se.build_static_vector(alerts[1].static, norm)
        idx = se.STATIC_FIELDS.index("host_offset_arcsec")
        assert vec[se.NUM_STATIC_FIELDS + idx] == pytest.approx(0.0)


class TestAssertNoSentinelValues:
    def test_passes_on_clean_data(self):
        se.assert_no_sentinel_values([_alert(_static())])

    def test_raises_on_a_surviving_sentinel(self):
        # host_mag_r=999.0 is one of the raw, unconverted SNANA sentinels --
        # this should never happen post-reader, but the check must catch it.
        alerts = [_alert(_static(host_mag_r=999.0))]
        with pytest.raises(ValueError, match="sentinel"):
            se.assert_no_sentinel_values(alerts)

    def test_none_is_not_flagged_as_a_sentinel(self):
        """A field the reader already converted to None must NOT trip this
        check -- only a raw sentinel VALUE surviving is a bug."""
        se.assert_no_sentinel_values([_alert(_static(host_mag_r=None))])


class TestStaticEncoderModule:
    def test_output_shape(self):
        encoder = se.StaticEncoder(output_size=16, hidden_size=32)
        batch = torch.zeros(4, 2 * se.NUM_STATIC_FIELDS)
        out = encoder(batch)
        assert out.shape == (4, 16)

    def test_gradients_flow(self):
        encoder = se.StaticEncoder(output_size=8, hidden_size=16)
        batch = torch.randn(3, 2 * se.NUM_STATIC_FIELDS, requires_grad=True)
        out = encoder(batch)
        out.sum().backward()
        assert batch.grad is not None
        assert torch.isfinite(batch.grad).all()


# ---------------------------------------------------------------------------
# Real-data check: CLAUDE.md Day-7 instruction, "Sentinel conversion already
# happened in the reader. Verify no sentinel values survive into the
# encoder." Golden-test pattern (CLAUDE.md §12): needs the real parquet,
# skipped if absent, like tests/test_gate_b.py.
# ---------------------------------------------------------------------------

PARQUET = Path("data/interim/elasticc2/train_02_full_3class.parquet")


@pytest.mark.skipif(not PARQUET.exists(), reason="cached ELAsTiCC2 parquet not present on disk")
def test_no_sentinel_values_in_real_interim_parquet():
    import polars as pl

    df = pl.read_parquet(str(PARQUET), columns=list(se.STATIC_FIELDS))
    offending = 0
    for field in se.STATIC_FIELDS:
        col = df[field]
        non_null = col.drop_nulls()
        for sentinel in se._KNOWN_SENTINELS:
            offending += int((non_null == sentinel).sum())
    assert offending == 0, (
        f"found {offending} raw-sentinel value(s) surviving into the real interim "
        "parquet's static fields -- reader-level sentinel conversion regression"
    )
