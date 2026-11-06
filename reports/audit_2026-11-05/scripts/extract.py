import json, sys, os
R = "/Users/akaza/Desktop/Test/Supernova detection/"
files = {
 "TierA LC-only (step6)": "outputs/train/eval/metrics.json",
 "TierA LC+static (step8)": "outputs/train_static/eval/metrics.json",
 "TierA LC+static no-photoz": "outputs/train_static_no_photoz/eval/metrics.json",
 "TierA WHXE": "outputs/train_whxe/eval/metrics.json",
 "TierB zero-shot": "outputs/finetune_bts/zero_shot/eval/metrics.json",
 "TierB LC+static FT": "outputs/finetune_bts/with_static/eval/metrics.json",
 "TierB LC-only FT": "outputs/finetune_bts/lc_only/eval/metrics.json",
 "TierB LC+static+DRB FT": "outputs/finetune_bts/with_static_drb/eval/metrics.json",
 "span100": "reports/span_ablation/span_100/metrics.json",
 "span200": "reports/span_ablation/span_200/metrics.json",
 "spanfull": "reports/span_ablation/span_full/metrics.json",
}
out = {}
for name, p in files.items():
    d = json.load(open(R+p)); rows = {}
    for k, v in d.items():
        if not isinstance(v, dict) or "macro_f1" not in v: continue
        rows[k] = dict(n=v["n_objects"], macro_f1=v["macro_f1"], f1=v["per_class"]["f1"],
                       support=v["per_class"]["support"], roc_auc=v.get("roc_auc"),
                       cm=v["confusion"]["raw"])
    out[name] = dict(path=p, cutoffs=rows)
    print(f"\n== {name}  [{p}]")
    for k, r in rows.items():
        ra = r['roc_auc'][0] if r['roc_auc'] else None
        print(f"  cut={k:>5} n={r['n']:>6} mF1={r['macro_f1']:.4f} IaROC={ra:.4f} F1(Ia,Ibc,II)={[round(x,4) for x in r['f1']]} sup={r['support']} cm={r['cm']}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
