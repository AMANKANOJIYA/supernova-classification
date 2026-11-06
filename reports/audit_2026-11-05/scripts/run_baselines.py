"""Part B: fit + persist the two Table-4.1 baselines (majority-class, 5-feature
multinomial logistic regression), on BOTH tiers, using the frozen
src/eval/baselines.py implementations and the frozen src/eval/report.py
harness -- no new metric code.

Splits are the model's own:
  Tier A: split_by_object_id(seed=42, 0.7/0.15) over
          data/interim/elasticc2/train_02_full_3class.parquet (configs/train/*.yaml)
  Tier B: src.train.finetune_bts.load_tier_b (shipped BTSbot splits + BTS
          Sample Explorer join), 3-class-labelled objects only.

Protocol (stated, not hidden):
  * Both baselines are FIT on full-light-curve training features (all
    detections). Each is EVALUATED at the same phase cutoffs as the models,
    with features recomputed from detections with
    mjd <= first_detection_mjd + cutoff only.
  * A cutoff with zero detections available (always the case at -5) yields no
    prediction for that object -> counted in n_excluded by the frozen harness.
  * extract_five_features raises on an object with zero detections; such
    objects are counted and logged, never silently dropped (CLAUDE.md 2.5).

Usage: python run_baselines.py <repo_root_for_src> <output_dir>
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

from src import taxonomy  # noqa: E402
from src.eval.baselines import (  # noqa: E402
    FEATURE_NAMES, MajorityClassBaseline, MulticlassLogisticRegression, extract_five_features,
)
from src.eval.phase_curves import PhasePrediction  # noqa: E402
from src.eval.report import run_eval_report  # noqa: E402
from src.train.dataset import load_object_class_ids, split_by_object_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("baselines")

CLASS_IDS = list(taxonomy.TARGET_CLASS_IDS)
CLS_IDX = {c: i for i, c in enumerate(CLASS_IDS)}
CLASS_NAMES = ["Ia", "Ib/c", "II"]
TIER_A_PARQUET = "data/interim/elasticc2/train_02_full_3class.parquet"
TIER_A_CUTOFFS = (-5.0, 0.0, 5.0, 15.0, 30.0, None)
TIER_B_CUTOFFS = (0.0, 5.0, 15.0, 30.0, None)
SEED = 42


def features_at_cutoffs(mjd, lam, flux, det, cutoffs):
    """Returns {cutoff: (days_since_first_det, feature_vec) or None}."""
    mjd = np.asarray(mjd, float); lam = np.asarray(lam, float)
    flux = np.asarray(flux, float); det = np.asarray(det, bool)
    if not det.any():
        return None
    first = mjd[det].min()
    last_delta = mjd[det].max() - first
    out = {}
    for c in cutoffs:
        if c is None:
            m = np.ones_like(det)
            d = last_delta
        else:
            m = mjd <= first + c
            d = min(c, last_delta)
        if not (det & m).any():
            out[c] = None
            continue
        out[c] = (float(d), extract_five_features(mjd[m], flux[m], lam[m], det[m]))
    return out


def featurize(rows_iter, cutoffs):
    ids, ys, feats = [], [], []
    n_no_det = 0
    for oid, cid, mjd, lam, flux, det in rows_iter:
        r = features_at_cutoffs(mjd, lam, flux, det, cutoffs)
        if r is None:
            n_no_det += 1
            continue
        ids.append(oid); ys.append(CLS_IDX[cid]); feats.append(r)
    return ids, np.array(ys), feats, n_no_det


def iter_parquet(path, object_ids, chunk=50_000):
    lf = pl.scan_parquet(path).select(["object_id", "class_id", "mjd", "lam_eff", "flux", "is_detection"])
    lf = lf.filter(pl.col("object_id").is_in(list(object_ids)))
    df = lf.collect()
    log.info("loaded %d rows from %s", df.height, path)
    for off in range(0, df.height, chunk):
        sl = df.slice(off, chunk)
        yield from zip(*(sl[c].to_list() for c in sl.columns))


def evaluate(name, ids, ys, feats, cutoffs, fitted_lr, majority, outdir):
    results = {}
    for model_name in ("majority", "logreg"):
        preds = []
        for oid, y, f in zip(ids, ys, feats):
            for c in cutoffs:
                if f.get(c) is None:
                    continue
                d, x = f[c]
                if model_name == "majority":
                    p = majority.predict_proba(1)[0]
                else:
                    p = fitted_lr.predict_proba(x[None, :])[0]
                preds.append(PhasePrediction(object_id=oid, days_since_first_detection=d, true_class=int(y), probs=p))
        m = run_eval_report(preds, num_classes=3, class_names=CLASS_NAMES,
                            output_dir=str(outdir / model_name / "eval"), cutoffs=cutoffs)
        results[model_name] = {
            k: {"n_objects": v["n_objects"], "n_excluded": v["n_excluded"], "macro_f1": v["macro_f1"],
                "ia_vs_rest_roc_auc": v["roc_auc"][0] if v.get("roc_auc") else None,
                "per_class_f1": v["per_class"]["f1"], "support": v["per_class"]["support"],
                "confusion_raw": v["confusion"]["raw"]}
            for k, v in m.items() if isinstance(v, dict) and "macro_f1" in v
        }
        log.info("%s %s full: macro_f1=%.4f", name, model_name, results[model_name]["full"]["macro_f1"])
    return results


def fit(ys_train, feats_train, name, outdir):
    X = np.stack([f[None][1] for f in feats_train])
    maj = MajorityClassBaseline().fit(ys_train, 3)
    lr = MulticlassLogisticRegression(num_classes=3, seed=0).fit(X, ys_train)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "logreg_model.npz", weights=lr.weights_, feature_mean=lr._feature_mean,
             feature_std=lr._feature_std, feature_names=np.array(FEATURE_NAMES),
             majority_class=maj.majority_class_)
    log.info("%s: fit on %d objects, majority class=%s", name, len(ys_train), CLASS_NAMES[maj.majority_class_])
    return maj, lr


def main():
    t0 = time.time()
    report = {"protocol": __doc__, "class_order": CLASS_NAMES,
              "lr_hyperparams": {"learning_rate": 0.1, "n_iter": 2000, "l2": 1e-3, "seed": 0}}

    # ---------------- Tier A
    ocids = load_object_class_ids(TIER_A_PARQUET)
    split = split_by_object_id(ocids, seed=SEED, train_frac=0.7, val_frac=0.15)
    log.info("Tier A split sizes: train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test))
    tr_ids, tr_y, tr_f, tr_nodet = featurize(iter_parquet(TIER_A_PARQUET, split.train), [None])
    te_ids, te_y, te_f, te_nodet = featurize(iter_parquet(TIER_A_PARQUET, split.test), TIER_A_CUTOFFS)
    log.info("Tier A: zero-detection objects skipped: train=%d test=%d", tr_nodet, te_nodet)
    outA = OUT / "tier_a"
    maj, lr = fit(tr_y, tr_f, "tier_a", outA)
    report["tier_a"] = {
        "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "n_train_fit": len(tr_ids), "n_test_scored": len(te_ids),
        "n_zero_detection_skipped": {"train": tr_nodet, "test": te_nodet},
        "majority_class": CLASS_NAMES[maj.majority_class_],
        "results": evaluate("tier_a", te_ids, te_y, te_f, TIER_A_CUTOFFS, lr, maj, outA),
    }

    # ---------------- Tier B
    from src.readers import bts_labels
    from src.train.finetune_bts import load_tier_b
    tb = load_tier_b(bts_labels.fetch_bts_sample())

    def rows(alerts):
        for a in alerts:
            if a.class_id not in CLS_IDX:
                continue
            o = a.observations
            yield (a.object_id, a.class_id, [x.mjd for x in o], [x.lam_eff for x in o],
                   [x.flux for x in o], [x.is_detection for x in o])

    btr_ids, btr_y, btr_f, btr_nodet = featurize(rows(tb.train), [None])
    bte_ids, bte_y, bte_f, bte_nodet = featurize(rows(tb.test), TIER_B_CUTOFFS)
    outB = OUT / "tier_b"
    majB, lrB = fit(btr_y, btr_f, "tier_b", outB)
    report["tier_b"] = {
        "n_train_fit": len(btr_ids), "n_test_scored": len(bte_ids),
        "n_zero_detection_skipped": {"train": btr_nodet, "test": bte_nodet},
        "majority_class": CLASS_NAMES[majB.majority_class_],
        "results": evaluate("tier_b", bte_ids, bte_y, bte_f, TIER_B_CUTOFFS, lrB, majB, outB),
    }
    report["elapsed_seconds"] = time.time() - t0
    (OUT / "baseline_report.json").write_text(json.dumps(report, indent=1, default=float))
    log.info("wrote %s (%.0fs)", OUT / "baseline_report.json", time.time() - t0)


if __name__ == "__main__":
    main()
