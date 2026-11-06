"""Figure 6: four real Tier B (BTSbot) test-split prediction panels -- three
correct classifications (one per class) and one confident misclassification
-- each showing the light curve as the deployed model saw it, its real
science/template/difference cutout triplet, and the model's own class
probabilities.

This runs the ALREADY-TRAINED, deployed checkpoint
(outputs/finetune_bts_zpfix/with_static_checkpoint.pt, the LC+static
fine-tune on zeropoint-corrected Tier B, seed 42 -- the model serving runs)
forward, in eval mode, no gradient, on real
held-out test objects -- this is INFERENCE with an existing frozen
checkpoint, not training or metric recomputation. It is the only way to put
real probabilities on this figure, since no per-object prediction file was
ever persisted anywhere in outputs/ or reports/ (only aggregate metrics.json
files exist) -- see paper/NUMBERS_GAPS.md.

Object selection (documented, not hidden -- see this script's `main` and the
printed selection log, and paper/data_derived/prediction_panels.json, which
collect_numbers.py reads for the caption/prose numbers):
  - Candidate pool: Tier B test-split objects with a 3-class label (Ia/Ibc/II
    from the BTS Sample Explorer join) AND a precomputed cutout triplet
    under data/interim/btsbot_cutouts/test/{object_id}.json.
  - Correct panels: for each class, the highest-confidence correctly-
    classified object with at least MIN_OBS_WELL_SAMPLED observations.
  - Misclassification panel: among true-Ibc-predicted-Ia objects (the
    project's own recorded dominant confusion, CLAUDE.md Sec 11), the
    most confident wrong call with the fewest observations (a sparse light
    curve) -- falls back to the next dominant confusion pattern actually
    present in the test split if no true-Ibc-as-Ia case exists, and prints
    which pattern was actually used.
"""

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src import taxonomy
from src.features import tensorize
from src.models.static_encoder import build_static_vector
from src.readers import bts_labels
from src.serve.model import load_lc_static_checkpoint
from src.train.dataset import load_alerts

PARQUET = "data/interim/btsbot_test.parquet"
CHECKPOINT = Path("outputs/finetune_bts_zpfix/with_static_checkpoint.pt")
CUTOUT_DIR = Path("data/interim/btsbot_cutouts/test")
OUT_PATH = "paper/figures/fig6_prediction_panels.pdf"
SELECTION_JSON = Path("paper/data_derived/prediction_panels.json")

PRE_DETECTION_WINDOW_DAYS = 30.0  # matches src.train.finetune_bts.PRE_DETECTION_WINDOW_DAYS
MIN_OBS_WELL_SAMPLED = 25

CLASS_NAMES = ["Ia", "Ib/c", "II"]
CLASS_COLORS = {"Ia": "#D98324", "Ib/c": "#2E5C8A", "II": "#3E8E5A"}
BAND_COLORS = {"ztf_g": "#2E8B57", "ztf_r": "#C0392B", "ztf_i": "#B8860B"}
BAND_NAMES = {472.3: "ztf_g", 633.9: "ztf_r", 788.6: "ztf_i"}


def band_name(lam_eff: float) -> str:
    return min(BAND_NAMES.items(), key=lambda kv: abs(kv[0] - lam_eff))[1]


def load_model():
    return load_lc_static_checkpoint(CHECKPOINT)


@torch.no_grad()
def predict_full(model, normalizer, alert) -> np.ndarray:
    """Softmax probabilities at the object's own last real timestep ("full"
    cutoff, matching src.eval.phase_curves' cutoff=None convention)."""
    t = tensorize.tensorize_alert(alert, pre_detection_window_days=PRE_DETECTION_WINDOW_DAYS)
    features = t.features.unsqueeze(0)  # (1, T, F)
    lengths = torch.tensor([features.shape[1]])
    static = torch.from_numpy(build_static_vector(alert.static, normalizer)).unsqueeze(0)
    logits = model(features, lengths, static)  # (1, T, 3)
    probs = torch.softmax(logits[0, -1], dim=-1).numpy()
    return probs


def load_cutout_triplet(object_id: str) -> dict[str, np.ndarray]:
    with open(CUTOUT_DIR / f"{object_id}.json") as f:
        payload = json.load(f)["images"]
    out = {}
    for name, b64 in payload.items():
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        out[name] = np.asarray(img)
    return out


def main() -> None:
    alerts = load_alerts(PARQUET)
    bts_sample = bts_labels.fetch_bts_sample()
    labeled, join_stats = bts_labels.apply_classifications_to_alerts(alerts, bts_sample)
    print("join_stats:", join_stats)

    cutout_ids = {p.stem for p in CUTOUT_DIR.glob("*.json")}
    class_index_of = {cid: i for i, cid in enumerate(taxonomy.TARGET_CLASS_IDS)}

    candidates = [
        a for a in labeled if a.class_id in class_index_of and a.object_id in cutout_ids
    ]
    print(f"{len(candidates)} candidate test objects (3-class labeled + cutout available)")

    model, normalizer = load_model()

    records = []
    for a in candidates:
        probs = predict_full(model, normalizer, a)
        true_idx = class_index_of[a.class_id]
        pred_idx = int(np.argmax(probs))
        n_obs = len(a.observations)
        records.append(
            {
                "alert": a,
                "probs": probs,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "correct": true_idx == pred_idx,
                "n_obs": n_obs,
                "confidence": float(probs[pred_idx]),
            }
        )

    selected: dict[str, dict] = {}

    # Three correct panels, one per class: highest-confidence correct call
    # among well-sampled objects (falls back to any correct call if no
    # well-sampled one exists for a class).
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        pool = [r for r in records if r["correct"] and r["true_idx"] == cls_idx]
        well_sampled = [r for r in pool if r["n_obs"] >= MIN_OBS_WELL_SAMPLED]
        use_pool = well_sampled if well_sampled else pool
        if not use_pool:
            print(f"WARNING: no correct test-split example for class {cls_name}")
            continue
        best = max(use_pool, key=lambda r: r["confidence"])
        selected[f"correct_{cls_name}"] = best
        print(
            f"correct[{cls_name}]: {best['alert'].object_id} conf={best['confidence']:.3f} "
            f"n_obs={best['n_obs']} well_sampled_pool={len(well_sampled)}"
        )

    # Misclassification panel: prefer true Ib/c predicted Ia (the project's
    # recorded dominant confusion); fall back to whatever wrong, confident
    # confusion actually exists in this candidate pool otherwise.
    ibc_as_ia = [r for r in records if not r["correct"] and r["true_idx"] == 1 and r["pred_idx"] == 0]
    if ibc_as_ia:
        pool = ibc_as_ia
        pattern = "true Ib/c -> predicted Ia"
    else:
        wrong = [r for r in records if not r["correct"]]
        if not wrong:
            raise RuntimeError("no misclassified candidate exists in this pool at all")
        from collections import Counter

        pattern_counts = Counter((r["true_idx"], r["pred_idx"]) for r in wrong)
        dominant_true, dominant_pred = pattern_counts.most_common(1)[0][0]
        pool = [r for r in wrong if r["true_idx"] == dominant_true and r["pred_idx"] == dominant_pred]
        pattern = f"true {CLASS_NAMES[dominant_true]} -> predicted {CLASS_NAMES[dominant_pred]} (fallback: no true-Ibc-as-Ia case in this pool)"
    # most confident wrong call, tie-broken toward sparser light curves
    misclass = sorted(pool, key=lambda r: (-r["confidence"], r["n_obs"]))[0]
    selected["misclassified"] = misclass
    print(
        f"misclassified[{pattern}]: {misclass['alert'].object_id} conf={misclass['confidence']:.3f} "
        f"n_obs={misclass['n_obs']}"
    )

    # -----------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------
    panel_order = ["correct_Ia", "correct_Ib/c", "correct_II", "misclassified"]
    panel_titles = {
        "correct_Ia": "Correct: Ia",
        "correct_Ib/c": "Correct: Ib/c",
        "correct_II": "Correct: II",
        "misclassified": f"Misclassified ({pattern.split(' (')[0]})",
    }

    fig = plt.figure(figsize=(11.5, 10.6))
    outer = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.42, wspace=0.28, top=0.94, bottom=0.03, left=0.06, right=0.98
    )

    for panel_i, key in enumerate(panel_order):
        rec = selected[key]
        alert = rec["alert"]
        probs = rec["probs"]
        true_name = CLASS_NAMES[rec["true_idx"]]
        pred_name = CLASS_NAMES[rec["pred_idx"]]

        inner = gridspec.GridSpecFromSubplotSpec(
            3, 3, subplot_spec=outer[panel_i // 2, panel_i % 2],
            height_ratios=[3.2, 1.3, 0.9], hspace=0.55, wspace=0.15,
        )

        ax_lc = fig.add_subplot(inner[0, :])
        detections = [o for o in alert.observations if o.is_detection]
        t0 = min(o.mjd for o in detections) if detections else min(o.mjd for o in alert.observations)
        for band, color in BAND_COLORS.items():
            band_obs = [o for o in alert.observations if band_name(o.lam_eff) == band]
            det = [o for o in band_obs if o.is_detection]
            forced = [o for o in band_obs if not o.is_detection]
            if forced:
                ax_lc.errorbar(
                    [o.mjd - t0 for o in forced], [o.flux for o in forced],
                    yerr=[o.flux_err for o in forced], fmt="x", color=color, ms=4,
                    elinewidth=0.6, alpha=0.5, label=f"{band} (forced)",
                )
            if det:
                ax_lc.errorbar(
                    [o.mjd - t0 for o in det], [o.flux for o in det],
                    yerr=[o.flux_err for o in det], fmt="o", color=color, ms=4.5,
                    elinewidth=0.8, label=f"{band} (detection)",
                )
        ax_lc.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        ax_lc.set_xlabel("Days since first detection", fontsize=8)
        ax_lc.set_ylabel("FLUXCAL", fontsize=8)
        ax_lc.tick_params(labelsize=7)
        ax_lc.legend(fontsize=6, frameon=False, ncol=2, loc="best")
        border_color = "0.3" if rec["correct"] else "#B03A2E"
        title = (
            f"{panel_titles[key]} -- {alert.object_id}\n"
            f"true={true_name}, predicted={pred_name} (p={rec['confidence']:.2f}), n_obs={rec['n_obs']}"
        )
        ax_lc.set_title(title, fontsize=8.3, color=border_color)
        for spine in ax_lc.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.4)

        cutouts = load_cutout_triplet(alert.object_id)
        for j, plane in enumerate(("science", "template", "difference")):
            ax_img = fig.add_subplot(inner[1, j])
            ax_img.imshow(cutouts[plane], cmap="gray", origin="lower")
            ax_img.set_title(plane, fontsize=7)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

        ax_bar = fig.add_subplot(inner[2, :])
        y_pos = np.arange(3)
        colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
        ax_bar.barh(y_pos, probs, color=colors, height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax_bar.set_xlim(0, 1)
        ax_bar.tick_params(axis="x", labelsize=6)
        for yi, p in zip(y_pos, probs):
            if p > 0.82:
                ax_bar.text(p - 0.03, yi, f"{p:.2f}", va="center", ha="right", fontsize=6.5, color="white", fontweight="bold")
            else:
                ax_bar.text(p + 0.03, yi, f"{p:.2f}", va="center", ha="left", fontsize=6.5, color="black")
        ax_bar.invert_yaxis()

    fig.suptitle(
        "Deployed model (LC-only fine-tune) on real Tier B test objects: "
        "light curve as tensorized, science/template/difference cutouts, class probabilities",
        fontsize=10, y=0.985,
    )
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")

    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps({
        "checkpoint": str(CHECKPOINT),
        "misclassification_pattern": pattern,
        "min_obs_well_sampled": MIN_OBS_WELL_SAMPLED,
        "panels": {
            key: {
                "object_id": selected[key]["alert"].object_id,
                "true": CLASS_NAMES[selected[key]["true_idx"]],
                "predicted": CLASS_NAMES[selected[key]["pred_idx"]],
                "confidence": selected[key]["confidence"],
                "n_obs": selected[key]["n_obs"],
            }
            for key in panel_order
        },
    }, indent=2))
    print(f"Wrote {SELECTION_JSON}")


if __name__ == "__main__":
    main()
