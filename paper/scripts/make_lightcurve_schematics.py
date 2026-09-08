"""Figure 2: the three supernova types, drawn from REAL light curves in the
project's own Tier A (ELAsTiCC2) interim corpus -- not idealised curves.

Object selection (documented here, not hidden): candidates were drawn from
data/interim/elasticc2/train_02_full_3class.parquet by a cheap polars
group-by/list-length scan (no full-corpus CanonicalAlert reconstruction) for
objects with 40-80 detections, then visually screened (a handful per class,
rendered to a scratch PNG, inspected) for one showing its class's textbook
morphology cleanly:
  - Ia  35264248: smooth, asymmetric rise (~25 d) then slower decline.
  - Ibc 46704854: fast rise, sharp/steep decline -- Type Ib/c light curves
                  are powered by radioactive decay in a much smaller
                  ejecta mass than Ia, so they rise and fall faster.
  - II  36031694: a clear, ~85-day plateau (Type IIP) -- hydrogen
                  recombination in the still-attached envelope temporarily
                  holds the photosphere's temperature (and so its
                  brightness) nearly constant before the envelope is fully
                  recombined and the light curve drops onto the radioactive
                  tail.
This is a real, on-disk selection (object ids are given, reproducible by
anyone re-running the same query against the same parquet) rather than a
fabricated schematic.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train.dataset import load_alerts

PARQUET = "data/interim/elasticc2/train_02_full_3class.parquet"
OUT_PATH = "paper/figures/fig2_lightcurve_schematics.pdf"

# class -> (object_id, display color, mechanism annotation)
CLASS_COLOR = {"Ia": "#D98324", "Ibc": "#2E5C8A", "II": "#3E8E5A"}
SELECTION = {
    "Ia": {
        "object_id": "35264248",
        "title": "Type Ia",
        "mechanism": "Thermonuclear disruption of a\nwhite dwarf -- radioactive $^{56}$Ni\ndecay powers a smooth rise and\na slower, near-exponential decline.",
    },
    "Ibc": {
        "object_id": "46704854",
        "title": "Type Ib/c",
        "mechanism": "Core collapse of a massive star\nstripped of its H (and, for Ic, He)\nenvelope -- a smaller ejecta mass\nrises and declines faster than Ia.",
    },
    "II": {
        "object_id": "36031694",
        "title": "Type II",
        "mechanism": "Core collapse with the H envelope\nstill attached -- hydrogen\nrecombination holds the\nphotosphere's brightness on a\nplateau before the radioactive tail.",
    },
}


def main() -> None:
    ids = frozenset(v["object_id"] for v in SELECTION.values())
    alerts = load_alerts(PARQUET, object_ids=ids)
    by_id = {a.object_id: a for a in alerts}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)
    for ax, (cls, meta) in zip(axes, SELECTION.items()):
        alert = by_id[meta["object_id"]]
        color = CLASS_COLOR[cls]
        detections = [o for o in alert.observations if o.is_detection]
        t0 = min(o.mjd for o in detections)

        forced = [o for o in alert.observations if not o.is_detection]
        if forced:
            ax.errorbar(
                [o.mjd - t0 for o in forced],
                [o.flux for o in forced],
                yerr=[o.flux_err for o in forced],
                fmt=".",
                color="0.75",
                ms=3,
                elinewidth=0.6,
                zorder=1,
                label="forced photometry",
            )
        ax.errorbar(
            [o.mjd - t0 for o in detections],
            [o.flux for o in detections],
            yerr=[o.flux_err for o in detections],
            fmt="o",
            color=color,
            ms=4,
            elinewidth=0.8,
            zorder=2,
            label="detection",
        )
        ax.axhline(0.0, color="0.85", lw=0.7, zorder=0)
        ax.set_xlim(-15, 155)
        ax.set_title(f"{meta['title']}  (SNID {meta['object_id']})", fontsize=10, color=color)
        ax.set_xlabel("Days since first detection")
        ax.text(
            0.97,
            0.04,
            meta["mechanism"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.8, alpha=0.95),
        )
    axes[0].set_ylabel("FLUXCAL (AB zeropoint 27.5)")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
