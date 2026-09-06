"""Gate-A / data-integrity diagnostics on parsed `CanonicalAlert` data.

Deliberately NOT under `src/eval/`. `src/eval/` (frozen, CLAUDE.md §2.1)
judges *model* performance — predictions against truth, at various phase
cutoffs. This module runs before any model exists: it judges whether the
*parse itself* looks physically sane (CLAUDE.md §7 Gate A: "Plot 20 light
curves per tier ... If they look like noise, the reader is wrong regardless
of whether it ran"). Nothing here is scored against a metric; nothing here
is comparable across model checkpoints. That's a different kind of judgment
than `src/eval/` exists to make, so it gets its own module rather than an
exemption carved into a file §2.1 calls frozen.

Import direction is one-way: this module may import from `src.eval`'s
compute modules (metrics/phase_curves), but nothing under `src/eval/` may
import this back — enforced by
`tests/test_diagnostics.py::test_eval_modules_do_not_import_diagnostics`,
mirroring the existing `TestPhaseCurvesHasNoMatplotlib` regression test that
keeps `phase_curves.py`/`metrics.py` plot-free.

Magnitude conversion (CLAUDE.md §2.2: "Convert to magnitude only for
plotting"): `mag = ZP - 2.5*log10(flux)` with `ZP = 27.5` (the SNANA FLUXCAL
zeropoint, CLAUDE.md §3) is only defined for `flux > 0`. Points with
`flux <= 0` are real information (§2.2) — they are shown in the flux panel
of every light-curve plot, and silently omitted from the magnitude panel
only (which cannot represent them), never dropped from any count/statistic
in this module.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from src.schema import LAM_EFF, CanonicalAlert

_FLUXCAL_ZEROPOINT = 27.5

# lam_eff (nm) -> band display name, built by inverting the schema's
# canonical LAM_EFF table. Exact float equality is safe here: every
# Observation.lam_eff in this codebase is assigned directly from a LAM_EFF
# value (see src/readers/elasticc2.py:_band_to_lam_eff), never derived
# arithmetically, so no floating-point drift is possible between the two.
_BAND_NAME_OF_LAM_EFF: dict[float, str] = {v: k for k, v in LAM_EFF.items()}


def _band_name(lam_eff: float) -> str:
    return _BAND_NAME_OF_LAM_EFF.get(lam_eff, f"unknown(lam_eff={lam_eff})")


# ---------------------------------------------------------------------------
# Pure compute — no plotting, no I/O. Kept separate so these are usable (and
# unit-testable) without a matplotlib backend, same rationale as
# src/eval/phase_curves.py vs src/eval/plots.py.
# ---------------------------------------------------------------------------


def negative_flux_fraction(alerts: Sequence[CanonicalAlert]) -> float:
    """Fraction of all observations (across all `alerts`) with `flux < 0`.

    CLAUDE.md §7 Gate A: "Assert negative fluxes exist... Zero means upper
    limits were dropped or something went through magnitudes." Returns NaN
    (not ZeroDivisionError) if `alerts` contains no observations at all —
    that is itself a red flag the caller should check for, not a crash.
    """
    n_total = 0
    n_negative = 0
    for alert in alerts:
        for obs in alert.observations:
            n_total += 1
            if obs.flux < 0:
                n_negative += 1
    return float("nan") if n_total == 0 else n_negative / n_total


def class_counts(alerts: Sequence[CanonicalAlert]) -> dict[Optional[int], int]:
    """Object counts grouped by `class_id` (taxonomy id, or `None` if
    unlabelled)."""
    counts: dict[Optional[int], int] = {}
    for alert in alerts:
        counts[alert.class_id] = counts.get(alert.class_id, 0) + 1
    return counts


def detections_per_object(alerts: Sequence[CanonicalAlert]) -> list[int]:
    """Count of `is_detection=True` observations, one entry per object."""
    return [sum(1 for obs in a.observations if obs.is_detection) for a in alerts]


def forced_photometry_per_object(alerts: Sequence[CanonicalAlert]) -> list[int]:
    """Count of `is_detection=False` (forced photometry / upper limit)
    observations, one entry per object."""
    return [sum(1 for obs in a.observations if not obs.is_detection) for a in alerts]


def observations_per_band(alerts: Sequence[CanonicalAlert]) -> dict[str, int]:
    """Total observation count grouped by band display name (e.g. 'r')."""
    counts: dict[str, int] = {}
    for alert in alerts:
        for obs in alert.observations:
            band = _band_name(obs.lam_eff)
            counts[band] = counts.get(band, 0) + 1
    return counts


def time_span_days(alerts: Sequence[CanonicalAlert]) -> list[float]:
    """`max(mjd) - min(mjd)` per object. Empty-observation objects (should
    not occur past `CanonicalAlert.validate()`, but this module does not
    assume it was called) are skipped, not zero-filled."""
    spans = []
    for alert in alerts:
        if not alert.observations:
            continue
        mjds = [obs.mjd for obs in alert.observations]
        spans.append(max(mjds) - min(mjds))
    return spans


_STATIC_FIELDS: tuple[str, ...] = (
    "host_photoz",
    "host_photoz_err",
    "host_offset_arcsec",
    "host_ellipticity",
    "host_mag_r",
    "mw_ebv",
    "gal_lat",
    "real_bogus_score",
)


def static_field_none_fraction(alerts: Sequence[CanonicalAlert]) -> dict[str, float]:
    """Fraction of objects where each `StaticContext` field is `None`
    (post-sentinel-conversion). CLAUDE.md §7 Gate A / this session's Task 3d:
    "A field that's None for nearly everything isn't a usable feature."
    Returns an empty dict (not a divide-by-zero) if `alerts` is empty.
    """
    if not alerts:
        return {}
    n = len(alerts)
    return {
        field: sum(1 for a in alerts if getattr(a.static, field) is None) / n
        for field in _STATIC_FIELDS
    }


def sample_objects_per_class(
    alerts: Sequence[CanonicalAlert], n_per_class: int = 20
) -> dict[Optional[int], list[CanonicalAlert]]:
    """Up to `n_per_class` objects per `class_id`, chosen deterministically
    (sorted ascending by `object_id`, then take the first `n_per_class`) so
    repeated runs over the same data select the same objects — no `random`,
    no dependence on file/read order."""
    by_class: dict[Optional[int], list[CanonicalAlert]] = {}
    for alert in alerts:
        by_class.setdefault(alert.class_id, []).append(alert)
    return {
        class_id: sorted(group, key=lambda a: a.object_id)[:n_per_class]
        for class_id, group in by_class.items()
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------

_DETECTION_MARKER = "o"
_FORCED_PHOT_MARKER = "^"


def plot_light_curve(alert: CanonicalAlert, class_name: str) -> "matplotlib.figure.Figure":
    """One object's light curve: flux panel (top, all points, includes
    negative flux) and magnitude panel (bottom, `flux > 0` points only,
    y-axis inverted so brighter is up — the conventional SN light-curve
    view, which is where an Ia's rise/decline or a II's plateau is visually
    recognizable per CLAUDE.md §7 Gate A).

    Detections (`is_detection=True`) are plotted as filled circles; forced
    photometry / upper limits (`is_detection=False`) as open triangles —
    both panels use the same marker convention. Points are grouped and
    colored by band.
    """
    import matplotlib.pyplot as plt

    fig, (ax_flux, ax_mag) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    by_band: dict[str, list] = {}
    for obs in alert.observations:
        by_band.setdefault(_band_name(obs.lam_eff), []).append(obs)

    colors = plt.cm.viridis_r(
        [i / max(len(by_band) - 1, 1) for i in range(len(by_band))]
    )

    for color, (band, obs_list) in zip(colors, sorted(by_band.items())):
        for marker, is_det in ((_DETECTION_MARKER, True), (_FORCED_PHOT_MARKER, False)):
            pts = [o for o in obs_list if o.is_detection == is_det]
            if not pts:
                continue
            ax_flux.errorbar(
                [o.mjd for o in pts],
                [o.flux for o in pts],
                yerr=[o.flux_err for o in pts],
                fmt=marker,
                color=color,
                markersize=4,
                alpha=0.8,
                linestyle="none",
                label=f"{band} ({'det' if is_det else 'forced'})",
            )
            mag_pts = [o for o in pts if o.flux > 0]
            if mag_pts:
                mags = [_FLUXCAL_ZEROPOINT - 2.5 * math.log10(o.flux) for o in mag_pts]
                ax_mag.plot(
                    [o.mjd for o in mag_pts],
                    mags,
                    marker,
                    color=color,
                    markersize=4,
                    alpha=0.8,
                    linestyle="none",
                )

    ax_flux.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax_flux.set_ylabel("FLUXCAL (ZP=27.5)")
    ax_flux.legend(fontsize=7, ncol=2, loc="best")

    ax_mag.invert_yaxis()
    ax_mag.set_ylabel("magnitude (flux>0 only)")
    ax_mag.set_xlabel("MJD")

    n_det = sum(1 for o in alert.observations if o.is_detection)
    n_forced = sum(1 for o in alert.observations if not o.is_detection)
    fig.suptitle(
        f"SNID={alert.object_id}  class={class_name} ({alert.class_id})  "
        f"detections={n_det}  forced-phot={n_forced}"
    )
    fig.tight_layout()
    return fig


def save_gate_a_light_curves(
    samples: dict[Optional[int], list[CanonicalAlert]],
    class_name_of: dict[Optional[int], str],
    output_dir: str,
) -> list[str]:
    """Render and save every sampled light curve to
    `output_dir/<class_name>/<object_id>.png`. Returns the list of paths
    written."""
    import matplotlib.pyplot as plt

    written: list[str] = []
    for class_id, alerts in samples.items():
        class_name = class_name_of.get(class_id, f"unlabelled({class_id})")
        class_dir = os.path.join(output_dir, class_name.replace("/", "-"))
        os.makedirs(class_dir, exist_ok=True)
        for alert in alerts:
            fig = plot_light_curve(alert, class_name)
            path = os.path.join(class_dir, f"{alert.object_id}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(path)
    return written


@dataclass(frozen=True)
class GateAReport:
    """Everything Task 3 (parts b/c/d) asks for, in one bundle so a driver
    script can compute it once and both print and persist it."""

    negative_flux_fraction: float
    class_counts_by_object: dict[Optional[int], int]
    detections_per_object: list[int]
    forced_photometry_per_object: list[int]
    observations_per_band: dict[str, int]
    time_span_days: list[float]
    static_field_none_fraction: dict[str, float]


def build_gate_a_report(alerts: Sequence[CanonicalAlert]) -> GateAReport:
    return GateAReport(
        negative_flux_fraction=negative_flux_fraction(alerts),
        class_counts_by_object=class_counts(alerts),
        detections_per_object=detections_per_object(alerts),
        forced_photometry_per_object=forced_photometry_per_object(alerts),
        observations_per_band=observations_per_band(alerts),
        time_span_days=time_span_days(alerts),
        static_field_none_fraction=static_field_none_fraction(alerts),
    )
