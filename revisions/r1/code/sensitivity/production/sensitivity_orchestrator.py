#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path("/path/to/private_workspace/peerj_141707_v27")
MATRIX=ROOT/"bindings/sensitivity_run_matrix.json"
RUNNER=ROOT/"code/sensitivity_runner.py"
LOCKER=ROOT/"code/sensitivity_lock_training.py"
PY=Path("/path/to/private_workspace/peerj_141707_rerun_v1/.venv/bin/python")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["T","LOCK","E"],required=True);a=ap.parse_args()
 env=os.environ.copy();env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
 if a.stage=="LOCK":
  subprocess.run([str(PY),str(LOCKER)],check=True,env=env);return
 runs=json.loads(MATRIX.read_text())
 for i,r in enumerate(runs,1):
  print(f"[{i}/105] {a.stage} {r['run_id']}",flush=True)
  subprocess.run([str(PY),str(RUNNER),"--stage",a.stage,"--run-id",r["run_id"]],check=True,env=env)
if __name__=="__main__":main()
