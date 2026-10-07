#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,os,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve

ROOT=Path("/path/to/private_workspace/peerj_141707_v28_b0gnn_extension")
CODE=ROOT/"code"
MATRIX=CODE/"extension_run_matrix.json"; PROTOCOL=CODE/"extension_protocol.json"
RUNNER=CODE/"extension_runner.py"; BINDING=ROOT/"bindings/extension_source_binding.json"
AUTH=ROOT/"bindings/STAGE_E_AUTHORIZATION_v1.json"; TLOCK=ROOT/"stage_t/ALL_10_LOCKED.json"
MASKS=Path("/path/to/private_workspace/peerj_141707_v27/inputs/case/evaluation_masks_v27.parquet")
EXP={"binding_sha256":"c3087180c87a62dc1211fa9a42794d0e3c5a3338d9cae9999b1a1818bdbaf4b4",
"global_stage_t_lock_sha256":"c45ead04b1c1280930e56babb62441a0a43809e7e91d9261118478edab3406d6",
"runner_sha256":"3e4ad39d85428780e74287918d2660d9c9d9027fb3a0167df681778b6377927a",
"run_matrix_sha256":"0b0ae312e3a8af28ff8b50303fdedb3826b697c1bc82c0790bd3a73273ed211e",
"protocol_sha256":"537cf3b88ee866a648df5255ae7f6c8f4a69b34035806f503e272fadf6ef2205"}
POPS={"full_test":(8435,20),"D_frozen":(8422,7),"O_frozen":(8428,13),"S_frozen":(7757,6),"J_frozen":(7753,2)}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def writej(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+".tmp."+str(os.getpid()))
 t.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(t,p)
def norm(s):
 x=s.astype("string").str.strip().str.replace(r"^'","",regex=True)
 x=x.str.replace(r"\.0$","",regex=True).str.replace(r"\.(SZ|SH|BJ)$","",regex=True)
 return x.str.zfill(6)
def ordx(cc,yy,s):
 cc=np.asarray(cc).astype(str);yy=np.asarray(yy).astype(int);s=np.asarray(s,float)
 b=np.lexsort((yy,cc));return b[np.argsort(-s[b],kind="mergesort")]
def met(d,thr):
 y=d.y_true.to_numpy(int);s=d.score.to_numpy(float);ix=ordx(d.company_code,d.fiscal_year,s);ys=y[ix];ss=s[ix]
 n=len(y);budget=max(1,int(math.ceil(.05*n)));fpr,tpr,_=roc_curve(y,s,drop_intermediate=True);ok=fpr<=.10+1e-12
 cut=ss[budget-1];above=ss>cut;equal=ss==cut;slots=budget-int(above.sum());pos=int(ys[equal].sum());neg=int(equal.sum())-pos;base=int(ys[above].sum())
 return {"roc_auc":float(roc_auc_score(y,s)),"ap":float(average_precision_score(y,s)),
 "f1_val_threshold":float(f1_score(y,(s>=thr).astype(int),zero_division=0)),
 "p_at_5pct":float(ys[:budget].mean()),"r_at_10fpr":float(np.max(tpr[ok])) if ok.any() else 0.0,
 "hits_at_budget":int(ys[:budget].sum()),"n":int(n),"n_pos":int(y.sum()),"n_neg":int(n-y.sum()),
 "budget":int(budget),"cutoff_score":float(cut),"cutoff_tie_rows":int(equal.sum()),
 "selected_from_cutoff_tie":int(slots),"tie_hits_min":int(base+max(0,slots-neg)),
 "tie_hits_max":int(base+min(slots,pos))}
def eq(a,b):
 if isinstance(a,(int,np.integer)) and isinstance(b,(int,np.integer)):return int(a)==int(b)
 if a is None or b is None:return a is b
 return abs(float(a)-float(b))<=1e-12
def main():
 paths={"binding_sha256":BINDING,"global_stage_t_lock_sha256":TLOCK,"runner_sha256":RUNNER,
        "run_matrix_sha256":MATRIX,"protocol_sha256":PROTOCOL}
 for k,p in paths.items():
  if not p.is_file() or sha(p)!=EXP[k]:raise RuntimeError("IDENTITY:"+k)
 a=json.loads(AUTH.read_text())
 if a.get("status")!="AUTHORIZED" or a.get("stage_e_authorized") is not True:raise RuntimeError("AUTH")
 for k,v in EXP.items():
  if a.get(k)!=v:raise RuntimeError("AUTH_ID:"+k)
 runs=json.loads(MATRIX.read_text());ids=[r["run_id"] for r in runs]
 if len(ids)!=10 or len(set(ids))!=10:raise RuntimeError("MATRIX")
 fail=ROOT/"stage_e_failures"
 if fail.exists() and any(fail.iterdir()):raise RuntimeError("STAGE_E_FAILURES_PRESENT")
 se=ROOT/"stage_e";actual=sorted(p.name for p in se.iterdir() if p.is_dir()) if se.exists() else []
 if actual!=sorted(ids):raise RuntimeError("STAGE_E_DIR_SET:"+str(len(actual)))
 if (se/"ALL_10_STAGE_E_LOCKED.json").exists():raise RuntimeError("GLOBAL_STAGE_E_LOCK_EXISTS")
 masks=pd.read_parquet(MASKS);masks["company_code"]=norm(masks.company_code);masks["fiscal_year"]=masks.fiscal_year.astype(int)
 results=[];manifest={};testid=None
 for r in runs:
  rid=r["run_id"];d=se/rid;done=json.loads((d/"STAGE_E_COMPLETE.json").read_text());res=json.loads((d/"stage_e_result.json").read_text())
  if done.get("status")!="LOCKED" or done.get("strict_seed_explicit_before_inference") is not True:raise RuntimeError("COMPLETE:"+rid)
  if res.get("strict_seed_explicit_before_inference") is not True:raise RuntimeError("RESULT_STRICT:"+rid)
  for fn,meta in done.get("artifacts",{}).items():
   p=d/fn
   if not p.is_file() or sha(p)!=meta["sha256"] or p.stat().st_size!=meta["size_bytes"]:raise RuntimeError("ARTIFACT:"+rid+":"+fn)
  pred=pd.read_parquet(d/"test_predictions.parquet")
  if list(pred.columns)!=["company_code","fiscal_year","y_true","score"]:raise RuntimeError("PRED_COLS:"+rid)
  pred["company_code"]=norm(pred.company_code);pred["fiscal_year"]=pred.fiscal_year.astype(int);pred["y_true"]=pred.y_true.astype(int)
  pred["score"]=pd.to_numeric(pred.score,errors="coerce")
  if len(pred)!=8435 or int(pred.y_true.sum())!=20 or not np.isfinite(pred.score.to_numpy()).all():raise RuntimeError("PRED:"+rid)
  keys=pred[["company_code","fiscal_year","y_true"]].sort_values(["company_code","fiscal_year"]).reset_index(drop=True)
  ident=hashlib.sha256(keys.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
  if testid is None:testid=ident
  elif ident!=testid:raise RuntimeError("TEST_ID:"+rid)
  z=pred.merge(masks,on=["company_code","fiscal_year"],how="left",validate="one_to_one")
  thr=float(res["validation_threshold"]);recomp={}
  for pop,(n,p) in POPS.items():
   sub=z[z[pop].astype(bool)][["company_code","fiscal_year","y_true","score"]].copy()
   if (len(sub),int(sub.y_true.sum()))!=(n,p):raise RuntimeError("POP:"+rid+":"+pop)
   mm=met(sub,thr);recomp[pop]=mm
   for k,v in mm.items():
    if k not in res["metrics"][pop] or not eq(v,res["metrics"][pop][k]):raise RuntimeError("METRIC:"+rid+":"+pop+":"+k)
  ix=ordx(z.company_code,z.fiscal_year,z.score);top=z.iloc[ix[:422]]
  attr={"budget":422,"full_hits":int(top.y_true.sum()),"D_positive_hits_in_full_top422":int((top.y_true.eq(1)&top.D_positive.astype(bool)).sum()),"O_positive_hits_in_full_top422":int((top.y_true.eq(1)&top.O_positive.astype(bool)).sum())}
  if attr!=res["full_budget_attribution"]:raise RuntimeError("ATTR:"+rid)
  if attr["full_hits"]!=attr["D_positive_hits_in_full_top422"]+attr["O_positive_hits_in_full_top422"]:raise RuntimeError("ATTR_ID:"+rid)
  af,ao,ad=recomp["full_test"]["roc_auc"],recomp["O_frozen"]["roc_auc"],recomp["D_frozen"]["roc_auc"];err=abs(20*af-(13*ao+7*ad))
  if err>1e-10:raise RuntimeError("AUC_ID:"+rid)
  manifest[rid]={fn:{"sha256":sha(d/fn),"size_bytes":(d/fn).stat().st_size} for fn in ("test_predictions.parquet","stage_e_result.json","STAGE_E_COMPLETE.json")}
  results.append({"run_id":rid,"model":r["model"],"model_seed":int(r["model_seed"]),"modality":r["modality"],"validation_threshold":thr,"metrics":recomp,"full_budget_attribution":attr,"auc_DO_identity_abs_error":err})
 lock={"status":"ALL_10_STAGE_E_LOCKED","fit_count":10,"created_at_unix":time.time(),**EXP,
       "stage_e_failure_dirs_present":0,"test_identity_sha256":testid,"all_metrics_independently_recomputed":True,
       "all_full_budget_attribution_identities":True,"all_D_O_auc_identities":True,
       "strict_seed_explicit_before_inference_all_10":True,"artifacts":manifest,
       "interpretation_performed":False,"selection_performed":False}
 lockp=se/"ALL_10_STAGE_E_LOCKED.json";writej(lockp,lock)
 resultp=se/"EXTENSION_RESULTS.json"
 writej(resultp,{"status":"FROZEN_FOR_ANALYSIS","fit_count":10,"extension_family":"B0GNN42",
   "result_aware_extension":True,"original_105_fit_program_unchanged":True,"test_identity_sha256":testid,
   "runs":results,"interpretation_performed":False,"selection_performed":False})
 sharep=se/"EXTENSION_EXECUTION_CHECKS.json"
 share={"status":"PASS_STAGE_E_ALL_10_FROZEN","fit_count_checked":10,
  "global_stage_e_lock_path":str(lockp),"global_stage_e_lock_sha256":sha(lockp),
  "results_for_analysis_sha256":sha(resultp),"binding_sha256":sha(BINDING),"stage_t_global_lock_sha256":sha(TLOCK),
  "runner_sha256":sha(RUNNER),"run_matrix_sha256":sha(MATRIX),"protocol_sha256":sha(PROTOCOL),
  "all_test_prediction_key_label_identity":"PASS","all_metrics_independently_recomputed":"PASS",
  "all_full_budget_attribution_identities":"PASS","all_D_O_auc_identities":"PASS",
  "strict_seed_explicit_before_inference_all_10":"PASS","stage_e_failure_dirs_present":0,
  "row_level_predictions_in_return_zip":False,"interpretation_performed":False,"selection_performed":False,
  "next_gate":"READY_FOR_RESULT_INTERPRETATION"}
 writej(sharep,share)
 ret=ROOT/"V28_B0GNN_STAGE_E_RETURN.zip"
 if ret.exists():raise RuntimeError("RETURN_ZIP_ALREADY_EXISTS")
 with zipfile.ZipFile(ret,"w",compression=zipfile.ZIP_DEFLATED) as z:
  for p,arc in ((sharep,"EXTENSION_EXECUTION_CHECKS.json"),(resultp,"EXTENSION_RESULTS.json"),(lockp,"ALL_10_STAGE_E_LOCKED.json"),(AUTH,"STAGE_E_AUTHORIZATION_v1.json")):
   z.write(p,arc)
 print(json.dumps({**share,"return_zip":str(ret),"return_zip_sha256":sha(ret),"return_zip_size_bytes":ret.stat().st_size},indent=2,sort_keys=True))
if __name__=="__main__":main()
