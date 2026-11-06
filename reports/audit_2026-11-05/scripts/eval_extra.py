"""Inference-only evaluations on the Tier B test split (no training):
  (a) zero-shot LC-only: Tier A LC-only checkpoint (outputs/train/checkpoint.pt),
      unmodified -- the missing like-for-like baseline for the LC-only transfer gain.
  (b) deployed ONNX: outputs/finetune_bts/lc_only_model.onnx (the file serving
      uses; mtime Sep 7 06:58, i.e. exported BEFORE the Sep 7 11:43 accidental
      re-run) -- does it still carry the original 0.6716 weights?
Same dataset construction, cutoffs, harness and ROC-AUC helper as finetune_bts.py.
Usage: python eval_extra.py <repo_root> <onnx_path> <out_dir>
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(sys.argv[1]); ONNX = Path(sys.argv[2]); OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.models.lc_encoder import LCEncoder
from src.readers import bts_labels
from src.train import finetune_bts as fb
from src.train.dataset import LightCurveDataset, collate_examples
from src.train.pretrain import collect_phase_predictions
from src.eval.phase_curves import PhasePrediction
from src.eval.report import run_eval_report
import onnxruntime as ort
from torch.utils.data import DataLoader

tb = fb.load_tier_b(bts_labels.fetch_bts_sample())
ds = LightCurveDataset(tb.test, frozenset(a.object_id for a in tb.test), fb.PRE_DETECTION_WINDOW_DAYS)
res = {}

def summarize(name, preds):
    m = run_eval_report(preds, num_classes=3, class_names=fb.CLASS_NAMES,
                        output_dir=str(OUT / name / "eval"), cutoffs=fb.TIER_B_CUTOFFS)
    res[name] = {"macro_f1_full": m["full"]["macro_f1"],
                 "ia_vs_rest_roc_auc": fb._ia_vs_rest_roc_auc(preds),
                 "per_class_f1_full": m["full"]["per_class"]["f1"],
                 "phase_curve": {k: v["macro_f1"] for k, v in m.items() if isinstance(v, dict) and "macro_f1" in v}}
    print(name, json.dumps(res[name]))

# (a) zero-shot LC-only
ck = torch.load("outputs/train/checkpoint.pt", map_location="cpu", weights_only=False)
c = ck["config"]["model"]
model = LCEncoder(input_size=c["input_size"], hidden_size=c["hidden_size"], num_layers=c["num_layers"],
                  dropout=c["dropout"], num_classes=c["num_classes"])
model.load_state_dict(ck["model_state_dict"])
with torch.no_grad():
    summarize("zero_shot_lc_only", collect_phase_predictions(model, ds, torch.device("cpu")))

# (b) deployed ONNX
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
preds = []
for batch in DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples):
    (logits,) = sess.run(["logits"], {"features": batch.features.numpy().astype(np.float32)})
    z = logits - logits.max(-1, keepdims=True); p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    for i, oid in enumerate(batch.object_id):
        for t in range(int(batch.lengths[i])):
            preds.append(PhasePrediction(object_id=oid, days_since_first_detection=float(batch.delta_t_days[i, t]),
                                         true_class=int(batch.class_index[i]), probs=p[i, t]))
summarize("deployed_onnx_lc_only", preds)
(OUT / "eval_extra.json").write_text(json.dumps(res, indent=1))
