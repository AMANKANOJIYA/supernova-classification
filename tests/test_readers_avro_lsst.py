"""Tests for src/readers/avro_lsst.py.

Not a golden test against a real Fink LSST packet -- CLAUDE.md §12's "hand-
verify one object against the source docs" could not be done this session:
every Fink LSST livestream topic was confirmed empty (see that module's own
docstring, and the Day 13 session report). These tests instead verify (a)
the arithmetic against a synthetic packet built from the module's OWN
documented field-name assumptions, and (b) that a genuinely malformed or
unexpected packet fails loudly and by name rather than silently -- the
property that matters most given the schema itself is unverified.
"""

import gzip
import io

import numpy as np
import pytest
from astropy.io import fits

from src.readers import avro_lsst
from src.schema import LAM_EFF, Survey


def _dia_source(**overrides):
    row = {
        "diaSourceId": 111,
        "diaObjectId": 42,
        "ra": 150.0,
        "decl": 2.0,
        "band": "r",
        "midpointMjdTai": 60123.5,
        "psfFlux": 500.0,
        "psfFluxErr": 25.0,
    }
    row.update(overrides)
    return row


def _make_alert(prv_sources=None, forced_sources=None, with_cutouts=False):
    alert = {
        "diaSource": _dia_source(),
        "prvDiaSources": prv_sources if prv_sources is not None else [],
        "prvDiaForcedSources": forced_sources if forced_sources is not None else [],
    }
    if with_cutouts:
        data = np.arange(63 * 63, dtype=np.float32).reshape(63, 63)
        hdu = fits.PrimaryHDU(data)
        buf = io.BytesIO()
        hdu.writeto(buf)
        stamp = gzip.compress(buf.getvalue())
        for name in ("cutoutScience", "cutoutTemplate", "cutoutDifference"):
            alert[name] = {"stampData": stamp}
    return alert


class TestBasicConversion:
    def test_object_id_from_dia_object_id(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        assert canonical.object_id == "42"

    def test_survey_is_lsst(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        assert canonical.survey == Survey.LSST

    def test_flux_converted_from_nanojansky_zp_31_4(self):
        """psfFlux=500.0, psfFluxErr=25.0 at zp=31.4 -> FLUXCAL zp=27.5,
        computed independently here (not via convert_zeropoint) as the
        golden check."""
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        scale = 10 ** ((27.5 - 31.4) / 2.5)  # nJy -> FLUXCAL: divide by ~36.3
        expected_flux = 500.0 * scale
        expected_flux_err = 25.0 * scale
        obs = canonical.observations[0]
        assert obs.flux == pytest.approx(expected_flux)
        assert obs.flux_err == pytest.approx(expected_flux_err)

    def test_mjd_passthrough_no_jd_conversion(self):
        """midpointMjdTai is already MJD -- unlike ZTF/BTSbot's jd, there is
        no -2400000.5 offset to apply."""
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        assert canonical.observations[0].mjd == pytest.approx(60123.5)

    def test_band_mapped_to_lam_eff(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        assert canonical.observations[0].lam_eff == pytest.approx(LAM_EFF["r"])

    def test_triggering_source_is_a_detection(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        assert canonical.observations[0].is_detection is True

    def test_forced_sources_are_non_detections(self):
        forced = [_dia_source(midpointMjdTai=60120.0, psfFlux=10.0, psfFluxErr=15.0)]
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(forced_sources=forced))
        forced_obs = [o for o in canonical.observations if o.mjd == pytest.approx(60120.0)]
        assert len(forced_obs) == 1
        assert forced_obs[0].is_detection is False

    def test_prv_sources_are_detections(self):
        prv = [_dia_source(midpointMjdTai=60110.0, psfFlux=200.0, psfFluxErr=20.0)]
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(prv_sources=prv))
        prv_obs = [o for o in canonical.observations if o.mjd == pytest.approx(60110.0)]
        assert len(prv_obs) == 1
        assert prv_obs[0].is_detection is True

    def test_observations_sorted_by_mjd(self):
        prv = [_dia_source(midpointMjdTai=60130.0), _dia_source(midpointMjdTai=60100.0)]
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(prv_sources=prv))
        mjds = [o.mjd for o in canonical.observations]
        assert mjds == sorted(mjds)

    def test_static_context_host_fields_none(self):
        """No DPDD-standard host offset/mag/ellipticity/real-bogus field
        name is assumed known (module docstring) -- must be None, not a
        guessed value."""
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        static = canonical.static
        assert static.host_photoz is None
        assert static.host_offset_arcsec is None
        assert static.host_ellipticity is None
        assert static.host_mag_r is None
        assert static.real_bogus_score is None

    def test_alert_passes_schema_validate(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert())
        canonical.validate()


class TestFailsLoudlyOnMalformedInput:
    """The property that matters most given the schema itself is unverified
    (module docstring): a packet that doesn't match these assumptions must
    raise with a clear message, never silently mis-map a field."""

    def test_missing_dia_source_raises(self):
        with pytest.raises(ValueError, match="no diaSource"):
            avro_lsst.lsst_alert_to_canonical({"diaSource": None})

    def test_missing_dia_object_id_and_source_id_raises(self):
        alert = _make_alert()
        del alert["diaSource"]["diaObjectId"]
        del alert["diaSource"]["diaSourceId"]
        with pytest.raises(ValueError, match="diaObjectId nor diaSourceId"):
            avro_lsst.lsst_alert_to_canonical(alert)

    def test_missing_psf_flux_raises(self):
        alert = _make_alert()
        del alert["diaSource"]["psfFlux"]
        with pytest.raises(ValueError, match="triggering diaSource unusable"):
            avro_lsst.lsst_alert_to_canonical(alert)

    def test_unrecognized_band_skips_row_not_fatal_for_prv(self):
        prv = [_dia_source(midpointMjdTai=60110.0, band="unknown_band")]
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(prv_sources=prv))
        # only the triggering source survives; the bad prv row was skipped
        assert len(canonical.observations) == 1

    def test_missing_ra_decl_raises(self):
        alert = _make_alert()
        alert["diaSource"]["ra"] = None
        with pytest.raises(ValueError, match="missing ra/decl"):
            avro_lsst.lsst_alert_to_canonical(alert)


class TestCutouts:
    def test_correctly_shaped_cutouts_decode(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(with_cutouts=True))
        assert canonical.cutouts is not None
        assert canonical.cutouts.shape == (3, 63, 63)

    def test_missing_cutouts_gives_none(self):
        canonical = avro_lsst.lsst_alert_to_canonical(_make_alert(with_cutouts=False))
        assert canonical.cutouts is None
