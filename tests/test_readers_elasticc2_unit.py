"""Unit tests for src/readers/elasticc2.py that don't need real ELAsTiCC2
data — pure logic (sentinel handling, band mapping, file-pair discovery,
per-file error resilience). Always runs, unlike tests/test_readers_elasticc2.py
(the golden test), which needs the real downloaded files.
"""

from pathlib import Path

import pytest

from src.readers import elasticc2


class TestSentinelToNone:
    def test_matching_sentinel_becomes_none(self):
        assert elasticc2._sentinel_to_none(-9.0, "HOSTGAL_PHOTOZ") is None

    def test_non_sentinel_value_passes_through(self):
        assert elasticc2._sentinel_to_none(0.3, "HOSTGAL_PHOTOZ") == pytest.approx(0.3)

    def test_each_field_uses_its_own_sentinel(self):
        assert elasticc2._sentinel_to_none(-9999.0, "HOSTGAL_ELLIPTICITY") is None
        assert elasticc2._sentinel_to_none(999.0, "HOSTGAL_MAG_r") is None
        # -9999 is not HOSTGAL_MAG_r's sentinel (999 is) -- must NOT become None.
        assert elasticc2._sentinel_to_none(-9999.0, "HOSTGAL_MAG_r") == pytest.approx(-9999.0)


class TestBandToLamEff:
    def test_all_lsst_bands_map(self):
        from src.schema import LAM_EFF

        assert elasticc2._band_to_lam_eff("u") == LAM_EFF["u"]
        assert elasticc2._band_to_lam_eff("g") == LAM_EFF["g"]
        assert elasticc2._band_to_lam_eff("r") == LAM_EFF["r"]
        assert elasticc2._band_to_lam_eff("i") == LAM_EFF["i"]
        assert elasticc2._band_to_lam_eff("z") == LAM_EFF["z"]

    def test_uppercase_y_maps_to_lowercase_lam_eff_key(self):
        """A_FORMAT.TXT documents the PHOT BAND value as capital 'Y', but
        LAM_EFF's key is lowercase 'y' -- verified empirically against real
        data (see chat report). This is the exact mapping that bug would
        hit."""
        from src.schema import LAM_EFF

        assert elasticc2._band_to_lam_eff("Y") == LAM_EFF["y"]

    def test_unrecognized_band_raises(self):
        with pytest.raises(ValueError, match="unrecognized BAND"):
            elasticc2._band_to_lam_eff("X")

    def test_padded_band_is_stripped(self):
        from src.schema import LAM_EFF

        assert elasticc2._band_to_lam_eff(" g ") == LAM_EFF["g"]


class TestFindHeadPhotPairs:
    def test_finds_matching_gz_pairs(self, tmp_path):
        (tmp_path / "M0-0001_HEAD.FITS.gz").write_bytes(b"x")
        (tmp_path / "M0-0001_PHOT.FITS.gz").write_bytes(b"x")
        (tmp_path / "M0-0002_HEAD.FITS.gz").write_bytes(b"x")
        (tmp_path / "M0-0002_PHOT.FITS.gz").write_bytes(b"x")
        pairs = elasticc2._find_head_phot_pairs(str(tmp_path))
        assert len(pairs) == 2

    def test_finds_uncompressed_pairs(self, tmp_path):
        (tmp_path / "M0-0001_HEAD.FITS").write_bytes(b"x")
        (tmp_path / "M0-0001_PHOT.FITS").write_bytes(b"x")
        pairs = elasticc2._find_head_phot_pairs(str(tmp_path))
        assert len(pairs) == 1
        assert pairs[0] == (
            str(tmp_path / "M0-0001_HEAD.FITS"),
            str(tmp_path / "M0-0001_PHOT.FITS"),
        )

    def test_head_without_matching_phot_is_skipped(self, tmp_path):
        (tmp_path / "M0-0001_HEAD.FITS.gz").write_bytes(b"x")
        # no matching PHOT file
        pairs = elasticc2._find_head_phot_pairs(str(tmp_path))
        assert pairs == []

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "M0-0001_HEAD.FITS.gz").write_bytes(b"x")
        (tmp_path / "M0-0001_PHOT.FITS.gz").write_bytes(b"x")
        (tmp_path / "README.txt").write_bytes(b"x")
        (tmp_path / "some.DUMP").write_bytes(b"x")
        pairs = elasticc2._find_head_phot_pairs(str(tmp_path))
        assert len(pairs) == 1


class TestAssertDirectoryComplete:
    """CLAUDE.md §5.1: "File-count completeness must be asserted, not
    assumed" -- the exact guard for a truncated download silently removing
    ~1/40 of a class. Synthetic `ReadStats`, no real FITS files needed: this
    function only ever inspects the counts on the object it's handed."""

    def _stats(self, **overrides):
        base = dict(
            n_objects_read=100,
            n_objects_discarded=0,
            discard_reasons={},
            n_head_files_found=40,
            n_phot_files_found=40,
            n_pairs_read_ok=40,
        )
        base.update(overrides)
        return elasticc2.ReadStats(**base)

    def test_passes_when_all_counts_match_expected(self):
        elasticc2.assert_directory_complete("fake_dir", self._stats(), expected_files=40)

    def test_raises_on_missing_head_file(self):
        stats = self._stats(n_head_files_found=39)
        with pytest.raises(RuntimeError, match="incomplete directory"):
            elasticc2.assert_directory_complete("fake_dir", stats, expected_files=40)

    def test_raises_on_missing_phot_file(self):
        stats = self._stats(n_phot_files_found=39)
        with pytest.raises(RuntimeError, match="incomplete directory"):
            elasticc2.assert_directory_complete("fake_dir", stats, expected_files=40)

    def test_raises_when_a_pair_failed_to_read_despite_files_present_on_disk(self):
        """The exact truncated-download scenario CLAUDE.md §5.1 names: both
        files are on disk (found=40/40) but one raised during read, so only
        39 pairs were actually read -- object-level discard counts alone
        (n_objects_discarded=1) would hugely understate this as "one bad
        row," not "one entire missing file"."""
        stats = self._stats(
            n_pairs_read_ok=39,
            n_objects_discarded=1,
            discard_reasons={"file read error (EOFError)": 1},
        )
        with pytest.raises(RuntimeError, match="incomplete directory"):
            elasticc2.assert_directory_complete("fake_dir", stats, expected_files=40)

    def test_raises_on_more_files_than_expected_too(self):
        """Not just a floor -- an unexpected extra file (e.g. a stray
        duplicate download) should also be caught, not silently accepted."""
        stats = self._stats(n_head_files_found=41, n_phot_files_found=41, n_pairs_read_ok=41)
        with pytest.raises(RuntimeError, match="incomplete directory"):
            elasticc2.assert_directory_complete("fake_dir", stats, expected_files=40)

    def test_error_message_includes_discard_reasons(self):
        stats = self._stats(
            n_pairs_read_ok=39,
            n_objects_discarded=1,
            discard_reasons={"file read error (OSError)": 1},
        )
        with pytest.raises(RuntimeError, match="file read error"):
            elasticc2.assert_directory_complete("fake_dir", stats, expected_files=40)


class TestReadModelDirectoryResilience:
    """One corrupt/unreadable file pair must not take down the whole
    directory read -- encountered for real this session (a truncated gzip
    from an interrupted download crashed the entire batch before this was
    added)."""

    def test_one_bad_pair_does_not_crash_the_others(self, tmp_path, monkeypatch):
        for i in (1, 2, 3):
            (tmp_path / f"M0-000{i}_HEAD.FITS.gz").write_bytes(b"x")
            (tmp_path / f"M0-000{i}_PHOT.FITS.gz").write_bytes(b"x")

        from src.schema import CanonicalAlert, Observation, StaticContext, Survey

        def fake_read_head_phot_file(head_path, phot_path, gentype_to_classid, survey):
            if "0002" in head_path:
                raise OSError("simulated corrupt gzip")
            alert = CanonicalAlert(
                object_id=head_path,
                survey=survey,
                observations=[
                    Observation(mjd=1.0, lam_eff=622.2, flux=1.0, flux_err=1.0, is_detection=True)
                ],
                static=StaticContext(None, None, None, None, None, 0.1, 0.0, None),
                cutouts=None,
                class_id=2222,
            )
            return [alert], elasticc2.ReadStats(
                n_objects_read=1, n_objects_discarded=0, discard_reasons={}
            )

        monkeypatch.setattr(elasticc2, "read_head_phot_file", fake_read_head_phot_file)

        alerts, stats = elasticc2.read_model_directory(str(tmp_path), gentype_to_classid={})

        assert len(alerts) == 2  # files 0001 and 0003 succeeded
        assert stats.n_objects_read == 2
        assert stats.n_objects_discarded == 1
        assert any("file read error" in reason for reason in stats.discard_reasons)
        assert stats.discard_reasons["file read error (OSError)"] == 1
