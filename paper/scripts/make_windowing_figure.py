"""Figure 3: paired histograms of per-object sequence length, unwindowed vs.
windowed (`pre_detection_window_days=30`, tensorize.py's default) -- read
from paper/data_derived/windowing_stats.json, itself produced by
compute_windowing_stats.py from a stratified 3,000-object sample of the
Tier A interim parquet via the existing, unmodified reader/tensorize code.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = "paper/data_derived/windowing_stats.json"
OUT_PATH = "paper/figures/fig3_windowing_histograms.pdf"


def main() -> None:
    d = json.load(open(IN_PATH))
    unwindowed = d["seq_len_unwindowed_raw"]
    windowed = d["seq_len_windowed_raw"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = list(range(0, 420, 15))
    ax.hist(unwindowed, bins=bins, alpha=0.55, color="#8A2E5C", label="unwindowed (full archived history)")
    ax.hist(windowed, bins=bins, alpha=0.65, color="#2E5C8A", label="windowed (30-day pre-detection lookback)")
    ax.axvline(d["mean_seq_len_unwindowed"], color="#8A2E5C", ls="--", lw=1.2)
    ax.axvline(d["mean_seq_len_windowed"], color="#2E5C8A", ls="--", lw=1.2)
    ax.set_xlabel("Sequence length (observations per object)")
    ax.set_ylabel(f"Objects ($n$={d['n_sample_objects']} sampled from Tier A)")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
