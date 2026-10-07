#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path("/path/to/private_workspace/peerj_141707_v28_b0gnn_extension")
CODE=ROOT/"code"
RUNNER=CODE/"extension_runner.py"
MATRIX=CODE/"extension_run_matrix.json"
BINDING=ROOT/"bindings/extension_source_binding.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["T","E"],required=True)
    args=ap.parse_args()
    runs=json.loads(MATRIX.read_text())
    logdir=ROOT/("logs_stage_t" if args.stage=="T" else "logs_stage_e")
    logdir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    for i,r in enumerate(runs,1):
        rid=r["run_id"];log=logdir/f"{rid}.log"
        print(f"[{i}/10] START {args.stage} {rid}",flush=True)
        with log.open("w",encoding="utf-8") as f:
            cp=subprocess.run([sys.executable,str(RUNNER),"--stage",args.stage,
                               "--run-id",rid,"--binding",str(BINDING)],
                              stdout=f,stderr=subprocess.STDOUT,env=env)
        if cp.returncode!=0:
            print(f"FAILED {args.stage} {rid} log={log}",flush=True)
            raise SystemExit(cp.returncode)
        print(f"[{i}/10] DONE {args.stage} {rid}",flush=True)
    print(f"ALL_10_STAGE_{args.stage}_PROCESSES_COMPLETED",flush=True)

if __name__=="__main__":main()
