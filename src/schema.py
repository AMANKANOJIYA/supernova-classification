"""Canonical cross-survey alert schema. FROZEN — see CLAUDE.md §2.1 and §3.

Every data source (ELAsTiCC2, ZTF/BTSbot, Fink) converts into `CanonicalAlert`.
Do not modify this file after Day 1-2 without stopping to ask; a change here
invalidates every downstream comparison across surveys.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Survey(str, Enum):
    ZTF = "ztf"
    LSST = "lsst"
    ELASTICC = "elasticc"


@dataclass(frozen=True)
class Observation:
    """A single photometric measurement.

    Attributes:
        mjd: Modified Julian Date of the observation, days.
        lam_eff: Filter effective wavelength, nanometres. Use `LAM_EFF` below —
            never hardcode a band index (see CLAUDE.md §3 for why).
        flux: Calibrated flux. MAY BE NEGATIVE — difference-image photometry on
            faint, early-time sources routinely goes negative; that is real
            information, not noise to be clipped. Never clip at zero.
        flux_err: 1-sigma flux uncertainty. Must be > 0.
        is_detection: False for forced photometry / upper limits, True for a
            real detection above the survey's significance threshold.
    """

    mjd: float
    lam_eff: float
    flux: float
    flux_err: float
    is_detection: bool


@dataclass(frozen=True)
class StaticContext:
    """Per-object metadata that does not vary with time.

    Units: `host_photoz` is dimensionless redshift; `host_offset_arcsec` is
    arcseconds; `host_ellipticity` is dimensionless (0-1); `host_mag_r` is AB
    magnitude; `mw_ebv` is Milky Way E(B-V) reddening in magnitudes; `gal_lat`
    is galactic latitude in degrees; `real_bogus_score` is dimensionless in
    [0, 1]. All fields are Optional because not every survey/pipeline provides
    every value (e.g. ELAsTiCC2 simulates zero to two candidate hosts).
    """

    host_photoz: Optional[float]
    host_photoz_err: Optional[float]
    host_offset_arcsec: Optional[float]
    host_ellipticity: Optional[float]
    host_mag_r: Optional[float]
    mw_ebv: Optional[float]
    gal_lat: Optional[float]
    real_bogus_score: Optional[float]


@dataclass(frozen=True)
class CanonicalAlert:
    """One transient object's full canonical record.

    Attributes:
        object_id: Survey-native identifier, as a string.
        survey: Which survey this alert originated from.
        observations: Photometry, sorted ascending by `mjd`.
        static: Time-independent per-object metadata.
        cutouts: `(3, 63, 63)` array of (science, reference, difference)
            postage stamps, or None. Always None for simulated data (ELAsTiCC2
            has no pixel-level simulation) — see CLAUDE.md §5.1. This is not a
            bug.
        class_id: ELAsTiCC2 taxonomy id (CLAUDE.md §4), or None if unlabelled.
            Never a local string label.
    """

    object_id: str
    survey: Survey
    observations: list[Observation]
    static: StaticContext
    cutouts: Optional[np.ndarray]
    class_id: Optional[int]

    def validate(self) -> None:
        """Assert the invariants this schema promises.

        Raises:
            ValueError: if `observations` is not sorted ascending by `mjd`,
                if any `flux_err` is not strictly positive, or if `cutouts` is
                present but is not shaped `(3, 63, 63)`.

        A negative `flux` is explicitly valid and must never raise here — see
        CLAUDE.md §2.2.
        """
        mjds = [obs.mjd for obs in self.observations]
        if mjds != sorted(mjds):
            raise ValueError(
                f"observations for object_id={self.object_id!r} are not sorted "
                f"ascending by mjd: {mjds}"
            )

        for obs in self.observations:
            if not obs.flux_err > 0:
                raise ValueError(
                    f"observations for object_id={self.object_id!r} contains "
                    f"flux_err={obs.flux_err} at mjd={obs.mjd}, but flux_err "
                    "must be > 0"
                )

        if self.cutouts is not None and self.cutouts.shape != (3, 63, 63):
            raise ValueError(
                f"cutouts for object_id={self.object_id!r} has shape "
                f"{self.cutouts.shape}, expected (3, 63, 63)"
            )


# Filter effective wavelengths, nanometres. Conditioning models on this
# continuous value (rather than a categorical band index) is what makes ZTF
# and LSST bands land in a shared input space — see CLAUDE.md §3. Do not
# "simplify" this into band indices.
LAM_EFF: dict[str, float] = {
    # LSST
    "u": 368.6,
    "g": 480.3,
    "r": 622.2,
    "i": 754.5,
    "z": 869.1,
    "y": 971.0,
    # ZTF
    "ztf_g": 472.3,
    "ztf_r": 633.9,
    "ztf_i": 788.6,
}
