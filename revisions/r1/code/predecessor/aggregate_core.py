#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    for p in args.results_dir.glob("*/result.json"):
        j=json.loads(p.read_text())
        if not j.get("test_evaluated"): continue
        m=j["metrics"]
        rows.append({
            "run_id":p.parent.name,"model":j["model"],"modal":j["modal"],"seed":j["seed"],
            "roc_auc":m["roc_auc"],"ap":m["ap"],"f1_val_threshold":m["f1_val_threshold"],
            "p_at_5pct":m["p_at_5pct"],"r_at_10fpr":m["r_at_10fpr"],
            "best_epoch":j["best_epoch"],"best_val_auc":j["best_val_auc"],
            "pos_weight":j["pos_weight"],
        })
    df=pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out.with_suffix(".per_seed.csv"),index=False)
    summ=[]
    for (model,modal),g in df.groupby(["model","modal"]):
        z={"model":model,"modal":modal,"n_seeds":len(g)}
        for c in ["roc_auc","ap","f1_val_threshold","p_at_5pct","r_at_10fpr"]:
            z[c+"_mean"]=g[c].mean()
            z[c+"_sd"]=g[c].std(ddof=1) if len(g)>1 else 0.0
        summ.append(z)
    pd.DataFrame(summ).sort_values(["modal","roc_auc_mean"],ascending=[True,False]).to_csv(args.out,index=False)
    print(args.out)
if __name__=="__main__":
    main()
