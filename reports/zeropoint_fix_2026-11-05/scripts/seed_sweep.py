"""Fine-tune seed sweep, two data conditions, identical code otherwise.
Usage: python seed_sweep.py <repo> <out_dir>"""
import json, sys, os, logging
from pathlib import Path
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1]); OUT = Path(sys.argv[2])
logging.basicConfig(level=logging.WARNING)
import src.train.finetune_bts as fb
res = {}
for cond, d in [("fixed_flux", "data/interim"), ("old_flux_same_order", "data/interim_oldflux")]:
    for seed in (42, 1, 2, 3, 4):
        tag = f"{cond}_seed{seed}"
        fb.SEED = seed; fb.BTSBOT_INTERIM_DIR = Path(d)
        fb.OUTPUT_ROOT = OUT / tag / "outputs"; fb.REPORT_DIR = OUT / tag / "report"
        fb.main()
        r = json.load(open(fb.REPORT_DIR / "gate_c_report.json"))
        res[tag] = {v: {"macro_f1": r[v]["macro_f1_full"], "roc": r[v]["ia_vs_rest_roc_auc"], "f1": r[v]["per_class_f1_full"]}
                    for v in ("zero_shot", "with_static", "lc_only", "with_static_drb")}
        print(tag, {v: round(x["macro_f1"], 4) for v, x in res[tag].items()}, flush=True)
        json.dump(res, open(OUT / "seed_sweep.json", "w"), indent=1)
