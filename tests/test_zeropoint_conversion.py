"""CLAUDE.md §3, Day-6 audit finding: every reader must convert to the SNANA
FLUXCAL AB-27.5 convention before constructing an `Observation`. This guards
two things: (1) the conversion arithmetic itself, and (2) that any *future*
reader module constructing `Observation` from a non-SNANA source actually
calls it -- so a later LSST/ZTF alert reader can't silently skip the
conversion the way this project's audit found nothing currently does.
"""

import glob
import math
import os

import pytest

from src.readers.zeropoint import SNANA_FLUXCAL_ZEROPOINT, convert_zeropoint

# elasticc2.py's own docstring confirms ELAsTiCC2 SNANA FLUXCAL is already on
# the 27.5 convention -- it is the one reader exempt from calling
# convert_zeropoint, by construction, not by oversight.
_EXEMPT_READERS = {"elasticc2.py", "zeropoint.py", "__init__.py"}


class TestConvertZeropoint:
    def test_lsst_nanojansky_is_divided_by_approximately_36(self):
        # LSST alert-packet flux is nJy, AB zeropoint 31.4. Going DOWN to
        # zp 27.5 makes the number SMALLER: 1 FLUXCAL unit is ~36.3 nJy.
        # (Audit 2026-11-05: this test previously asserted a x36.3
        # multiplication, copied from CLAUDE.md §3's wording, and so
        # encoded the inverted conversion it was meant to guard against.)
        flux, flux_err = convert_zeropoint(1.0, 0.1, from_zeropoint=31.4)
        expected_scale = 10 ** ((SNANA_FLUXCAL_ZEROPOINT - 31.4) / 2.5)
        assert 1.0 / expected_scale == pytest.approx(36.307805, abs=1e-3)
        assert flux == pytest.approx(1.0 * expected_scale)
        assert flux_err == pytest.approx(0.1 * expected_scale)

    @pytest.mark.parametrize("from_zp", [21.4, 25.9, 26.2, 26.44, 27.5, 31.4])
    @pytest.mark.parametrize("flux_in", [0.37, 1.0, 631.0, 8000.0, 2.5e6])
    def test_magnitude_is_invariant_under_conversion(self, from_zp, flux_in):
        """The physical definition, independent of any factor written down
        anywhere: the AB magnitude a flux implies must be the same before
        and after rescaling to zp 27.5. This is the test that would have
        caught the inverted conversion."""
        flux_out, _ = convert_zeropoint(flux_in, 1.0, from_zeropoint=from_zp)
        mag_in = from_zp - 2.5 * math.log10(flux_in)
        mag_out = SNANA_FLUXCAL_ZEROPOINT - 2.5 * math.log10(flux_out)
        assert mag_out == pytest.approx(mag_in, abs=1e-9)

    def test_ztf_magnitude_converts_to_fluxcal_independent_of_magzpsci(self):
        """ZTF: flux_raw = 10**(-0.4*(magpsf - magzpsci)) on the alert's own
        zeropoint. After a correct conversion the zeropoint cancels and the
        result is exactly 10**(-0.4*(magpsf - 27.5)) whatever magzpsci was."""
        magpsf = 19.0
        expected = 10 ** (-0.4 * (magpsf - SNANA_FLUXCAL_ZEROPOINT))
        for magzpsci in (25.5, 26.0, 26.44):
            raw = 10 ** (-0.4 * (magpsf - magzpsci))
            flux, _ = convert_zeropoint(raw, 1.0, from_zeropoint=magzpsci)
            assert flux == pytest.approx(expected, rel=1e-12)

    def test_already_on_snana_scale_is_identity(self):
        flux, flux_err = convert_zeropoint(123.4, 5.6, from_zeropoint=SNANA_FLUXCAL_ZEROPOINT)
        assert flux == pytest.approx(123.4)
        assert flux_err == pytest.approx(5.6)

    def test_flux_norm_ratio_is_invariant_to_zeropoint_scale(self):
        """The whole reason flux_norm=flux/flux_err needs no correction:
        scaling both flux and flux_err by the same factor leaves their ratio
        unchanged."""
        raw_flux, raw_flux_err = 50.0, 3.0
        scaled_flux, scaled_flux_err = convert_zeropoint(
            raw_flux, raw_flux_err, from_zeropoint=31.4
        )
        assert (raw_flux / raw_flux_err) == pytest.approx(scaled_flux / scaled_flux_err)

    def test_flux_abs_norm_input_is_not_invariant_to_zeropoint_scale(self):
        """The opposite of the above: flux alone (feeding flux_abs_norm) IS
        distorted by an unconverted zeropoint -- this is the actual bug
        this whole module exists to prevent."""
        raw_flux = 50.0
        scaled_flux, _ = convert_zeropoint(raw_flux, 3.0, from_zeropoint=31.4)
        assert scaled_flux != pytest.approx(raw_flux)


class TestEveryNonSnanaReaderUsesConvertZeropoint:
    def test_no_reader_module_skips_the_conversion(self):
        reader_dir = os.path.join(os.path.dirname(__file__), "..", "src", "readers")
        offending = []
        for path in sorted(glob.glob(os.path.join(reader_dir, "*.py"))):
            name = os.path.basename(path)
            if name in _EXEMPT_READERS:
                continue
            source = open(path).read()
            if "Observation(" not in source:
                continue  # doesn't construct Observations -- not a reader in this sense
            if "convert_zeropoint" not in source:
                offending.append(name)
        assert not offending, (
            f"reader module(s) {offending} construct Observation but never call "
            "src.readers.zeropoint.convert_zeropoint -- CLAUDE.md §3 requires "
            "every non-SNANA reader to convert to the FLUXCAL AB-27.5 zeropoint "
            "before constructing an Observation"
        )
