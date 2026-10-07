#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
import pandas as pd

ROOT=Path("/path/to/private_workspace/peerj_141707_v28_b0gnn_extension")
CODE=ROOT/"code"
MATRIX=CODE/"extension_run_matrix.json"
RUNNER=CODE/"extension_runner.py"
BINDING=ROOT/"bindings/extension_source_binding.json"

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def writej(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f".tmp.{os.getpid()}")
    t.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n");os.replace(t,p)

def main():
    runs=json.loads(MATRIX.read_text());ids=[r["run_id"] for r in runs]
    if len(ids)!=10 or len(set(ids))!=10:raise RuntimeError("BAD_MATRIX")
    stage_e=ROOT/"stage_e"
    if stage_e.exists() and any(stage_e.iterdir()):raise RuntimeError("STAGE_E_PRESENT_BEFORE_GLOBAL_LOCK")
    fails=ROOT/"stage_t_failures"
    if fails.exists() and any(fails.iterdir()):raise RuntimeError("STAGE_T_FAILURES_PRESENT")
    st=ROOT/"stage_t"
    actual=sorted([p.name for p in st.iterdir() if p.is_dir()]) if st.exists() else []
    if actual!=sorted(ids):raise RuntimeError(f"STAGE_T_DIR_SET:{len(actual)}")
    bj=json.loads(BINDING.read_text())
    if bj.get("training_authorized") is not True or bj.get("stage_e_authorized") is not False:
        raise RuntimeError("BAD_BINDING_AUTH_STATE")
    rows=[];manifest={}
    for r in runs:
        rid=r["run_id"];d=st/rid
        done=json.loads((d/"STAGE_T_COMPLETE.json").read_text())
        res=json.loads((d/"stage_t_result.json").read_text())
        ident=json.loads((d/"FIT_IDENTITY.json").read_text())
        if done.get("status")!="LOCKED" or done.get("test_labels_read") is not False or done.get("test_scores_computed") is not False:
            raise RuntimeError(f"BAD_COMPLETE:{rid}")
        if res.get("test_evaluated") is not False or res.get("modality")!="M11_no_FIN87" or int(res.get("input_dim",-1))!=42:
            raise RuntimeError(f"BAD_RESULT:{rid}")
        if (int(res.get("train_neg",-1)),int(res.get("train_pos",-1)))!=(23852,254):
            raise RuntimeError(f"BAD_TRAIN_COUNTS:{rid}")
        if r["model"]!=res["model"] or int(r["model_seed"])!=int(res["model_seed"]):
            raise RuntimeError(f"BAD_RUN_IDENTITY:{rid}")
        if not (1<=int(res["best_epoch"])<=100):raise RuntimeError(f"BAD_EPOCH:{rid}")
        if not (0<=float(res["best_val_auc"])<=1):raise RuntimeError(f"BAD_VAL_AUC:{rid}")
        if not (0<=float(res["val_threshold"])<=1):raise RuntimeError(f"BAD_THRESHOLD:{rid}")
        vp=pd.read_parquet(d/"validation_predictions.parquet")
        if (len(vp),int(vp.y_true.sum()))!=(7309,40):raise RuntimeError(f"BAD_VAL_PRED:{rid}")
        for fn,meta in done["artifacts"].items():
            p=d/fn
            if not p.is_file() or sha256(p)!=meta["sha256"] or p.stat().st_size!=meta["size_bytes"]:
                raise RuntimeError(f"ARTIFACT_MISMATCH:{rid}:{fn}")
        identity=ident["identity"]
        if identity.get("binding_sha256")!=sha256(BINDING) or identity.get("runner_sha256")!=sha256(RUNNER):
            raise RuntimeError(f"FIT_IDENTITY_BINDING:{rid}")
        if identity.get("test_labels_read") is not False or identity.get("test_scores_computed") is not False:
            raise RuntimeError(f"FIT_IDENTITY_TEST_FLAG:{rid}")
        manifest[rid]={fn:{"sha256":sha256(d/fn),"size_bytes":(d/fn).stat().st_size}
                       for fn in ["checkpoint.pt","scaler.npz","validation_predictions.parquet",
                                  "stage_t_result.json","FIT_IDENTITY.json","STAGE_T_COMPLETE.json"]}
        rows.append({"run_id":rid,"model":r["model"],"seed":r["model_seed"],
                     "best_epoch":res["best_epoch"],"best_val_auc":res["best_val_auc"],
                     "val_threshold":res["val_threshold"]})
    lock={
        "status":"ALL_10_LOCKED","fit_count":10,"created_at_unix":time.time(),
        "binding_sha256":sha256(BINDING),"runner_sha256":sha256(RUNNER),
        "run_matrix_sha256":sha256(MATRIX),"stage_e_files_present":0,
        "training_failures_present":0,"artifacts":manifest}
    lockp=st/"ALL_10_LOCKED.json"
    if lockp.exists():raise RuntimeError("GLOBAL_LOCK_ALREADY_EXISTS")
    writej(lockp,lock)
    share={"status":"PASS_STAGE_T_ALL_10_LOCKED","fit_count_checked":10,
           "global_lock_path":str(lockp),"global_lock_sha256":sha256(lockp),
           "binding_sha256":sha256(BINDING),"runner_sha256":sha256(RUNNER),
           "stage_e_authorized":False,"stage_e_files_present":0,
           "training_failures_present":0,"runs":rows}
    sharep=ROOT/"stage_t/V28_B0GNN_STAGE_T_QA_SHARE.json";writej(sharep,share)
    print(json.dumps(share,indent=2,sort_keys=True))

if __name__=="__main__":main()
