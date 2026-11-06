"""Shared flux-zeropoint conversion for readers of non-SNANA-native sources
(CLAUDE.md §3, Day-6 audit finding).

`CanonicalAlert.Observation.flux`/`flux_err` must be on the SNANA `FLUXCAL`
convention, AB zeropoint 27.5. ELAsTiCC2's SNANA `FLUXCAL` is already on this
scale (`src/readers/elasticc2.py` does no conversion) -- that reader is the
one exception, not a template. A real LSST/ZTF alert-stream reader receives
flux in nJy, AB zeropoint 31.4, not 27.5. `convert_zeropoint` exists so that
conversion is one shared, tested function every non-SNANA reader calls,
rather than something each reader could reinvent -- or, as this project did
for a while, simply omit.
"""

SNANA_FLUXCAL_ZEROPOINT = 27.5


def convert_zeropoint(
    flux: float, flux_err: float, from_zeropoint: float
) -> tuple[float, float]:
    """Rescale `(flux, flux_err)` from `from_zeropoint` to
    `SNANA_FLUXCAL_ZEROPOINT`.

    `mag = zp - 2.5*log10(flux)` is invariant under a zeropoint change iff
    `flux` is rescaled by `10**((SNANA_FLUXCAL_ZEROPOINT - from_zeropoint) /
    2.5)`. For LSST nJy (zp 31.4) that DIVIDES by ~36.3; for a ZTF
    `magzpsci` near 26 it multiplies by ~3.6.

    Audit 2026-11-05: until then this used the reciprocal factor
    (`from_zeropoint - 27.5`), which does NOT preserve magnitude -- every
    Tier B / live ZTF flux came out a median ~10x too faint (row-dependent,
    via magzpsci) and LSST ~1318x too bright. The tests encoded the same
    inversion; `tests/test_zeropoint_conversion.py` now checks magnitude
    invariance directly. `flux_err` scales by the identical factor (it sits on the same
    additive flux scale as `flux`), which is exactly why `flux/flux_err`
    (this project's `flux_norm` feature, `src/features/tensorize.py`) is
    scale-invariant and needs no correction of its own — only `flux`/
    `flux_err` individually (and therefore `flux_abs_norm`/`flux_err_norm`)
    do.

    Args:
        flux: flux on the `from_zeropoint` scale.
        flux_err: 1-sigma flux uncertainty on the `from_zeropoint` scale.
        from_zeropoint: the AB zeropoint `flux`/`flux_err` are currently on
            (e.g. 31.4 for LSST/ZTF alert-packet nJy flux).

    Returns:
        `(flux, flux_err)` rescaled onto `SNANA_FLUXCAL_ZEROPOINT`.
    """
    scale = 10 ** ((SNANA_FLUXCAL_ZEROPOINT - from_zeropoint) / 2.5)
    return flux * scale, flux_err * scale
