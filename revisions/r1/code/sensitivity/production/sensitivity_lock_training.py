#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path("/path/to/private_workspace/peerj_141707_v27")
MATRIX=ROOT/"bindings/sensitivity_run_matrix.json"
BINDING=ROOT/"bindings/RUN_BINDINGS_v2.json"
OUT=ROOT/"stage_t/ALL_105_LOCKED.json"
def sha256(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
runs=json.loads(MATRIX.read_text());binding=json.loads(BINDING.read_text())
if len(runs)!=105 or not binding.get("training_authorized"):raise SystemExit("BINDING_OR_MATRIX_BAD")
entries=[]
for r in runs:
 d=ROOT/"stage_t"/r["run_id"];done=d/"STAGE_T_COMPLETE.json";iid=d/"FIT_IDENTITY.json"
 if not done.is_file() or not iid.is_file():raise SystemExit("MISSING_STAGE_T:"+r["run_id"])
 dj=json.loads(done.read_text())
 if dj.get("status")!="LOCKED" or dj.get("test_labels_read") is not False or dj.get("test_scores_computed") is not False:
  raise SystemExit("BAD_STAGE_T:"+r["run_id"])
 for name,meta in dj["artifacts"].items():
  p=d/name
  if not p.is_file() or sha256(p)!=meta["sha256"] or p.stat().st_size!=meta["size_bytes"]:
   raise SystemExit("ARTIFACT_IDENTITY:"+r["run_id"]+":"+name)
 entries.append({"run_id":r["run_id"],"complete_sha256":sha256(done),"fit_identity_sha256":sha256(iid)})
obj={"status":"ALL_105_LOCKED","fit_count":105,"final_binding_sha256":sha256(BINDING),
     "run_matrix_sha256":sha256(MATRIX),"entries":entries,"test_metrics_computed":False}
tmp=OUT.with_name(OUT.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n");os.replace(tmp,OUT)
print(OUT);print("STATUS=ALL_105_LOCKED")
