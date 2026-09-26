#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--results-dir",type=Path,required=True)
    ap.add_argument("--matrix",type=Path,default=None)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    matrix=args.matrix or (args.package/"run_matrix_core.csv")
    runner=args.package/"formal_rerun_core.py"
    args.results_dir.mkdir(parents=True,exist_ok=True)
    with matrix.open(newline="") as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        rid=r["run_id"]
        od=args.results_dir/rid
        result=od/"result.json"
        if result.is_file():
            print("SKIP completed",rid); continue
        cmd=[sys.executable,str(runner),"--data-dir",str(args.data_dir),
             "--out-dir",str(od),"--model",r["model"],"--modal",r["modal"],
             "--seed",r["seed"],"--label-col",r["label_col"],
             "--epochs",r["epochs"],"--patience",r["patience"],
             "--hidden-dim",r["hidden_dim"],"--lr",r["lr"],"--device",args.device]
        print("RUN",rid)
        print(" ".join(cmd))
        if not args.dry_run:
            od.mkdir(parents=True,exist_ok=True)
            subprocess.run(cmd,check=True)
if __name__=="__main__":
    main()
