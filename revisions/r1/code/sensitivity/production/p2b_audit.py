#!/usr/bin/env python3
from __future__ import annotations
import ast,hashlib,importlib.util,json,os,platform,shutil,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime
import numpy as np

CYQ=Path("/path/to/private_workspace")
ROOT=CYQ/"peerj_141707_v27"
HERE=Path(__file__).resolve().parent
PY=CYQ/"peerj_141707_rerun_v1/.venv/bin/python"
DATA_BIND=ROOT/"bindings/RUN_BINDINGS_v2_DATA_STAGE.json"
P2A_BIND=ROOT/"bindings/P2A_REFERENCE_BINDING.json"
P2A_PRIV=ROOT/"bindings/P2A_PRIVATE_REFERENCE_PATHS.json"
MATRIX=ROOT/"bindings/sensitivity_run_matrix.json"
ENVFREEZE=ROOT/"bindings/ENVIRONMENT_FREEZE_v27.txt"
FINAL=ROOT/"bindings/RUN_BINDINGS_v2.json"
OUTDIR=ROOT/"p2b_audit"
SHARE=OUTDIR/"V27_P2B_SHARE.json"
DETAIL=OUTDIR/"V27_P2B_DETAIL.json"

EXPECTED={
 "data_binding":"71ebdb3ec4521b9c09773ce2a8a7cec64e240b0fb092a113a8281a49c745aaa6",
 "p2a_binding":"8cb66579086c6211beef55d8ecca65846f044cfd4787083f02926ef730ed5e81",
 "p2a_private":"de484348197dd4bd95e87ae18de2b7f0579b5a06f8772ebfa4850dea89731492",
 "p2a_share":"1a143c96fe5bf73c1b95e7deec9884d2ad9d5de7e98aa25bc3d3c30f2feaf654",
 "matrix":"4d78371987154fa5aee660479c447e30f56608d28616359f04d8c472759f8923",
 "protocol":"71665491010526f2bb598efbdda0b2ea3639184c80c58b1bfa6329b77323311a",
 "core":"6b7d1188097a6e9a287b865a20ed4d5824a5e35de2d926c4da43b2ceb1c523d7",
}
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def import_file(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

checks=[];errors=[]
def check(name,fn):
 try:
  d=fn();checks.append({"name":name,"pass":True,"detail":d});return d
 except Exception as e:
  checks.append({"name":name,"pass":False,"detail":{"error":repr(e)}});errors.append({"name":name,"traceback":traceback.format_exc()});return None

def main():
 OUTDIR.mkdir(parents=True,exist_ok=True)
 def identities():
  require(sha256(DATA_BIND)==EXPECTED["data_binding"],"DATA_BINDING")
  require(sha256(P2A_BIND)==EXPECTED["p2a_binding"],"P2A_BINDING")
  require(sha256(P2A_PRIV)==EXPECTED["p2a_private"],"P2A_PRIVATE")
  require(sha256(HERE/"sensitivity_input_binding.json")==EXPECTED["p2a_share"],"P2A_SHARE")
  require(sha256(HERE/"sensitivity_run_matrix_frozen.json")==EXPECTED["matrix"],"PACKAGE_MATRIX")
  require(sha256(MATRIX)==EXPECTED["matrix"],"ROOT_MATRIX")
  require(sha256(HERE/"sensitivity_protocol.json")==EXPECTED["protocol"],"PROTOCOL")
  require(sha256(HERE/"formal_rerun_core_REFERENCE.py")==EXPECTED["core"],"CORE")
  return {"PASS":True}
 check("frozen_identity_chain",identities)

 def no_results():
  hits=[]
  for d in [ROOT/"stage_t",ROOT/"stage_e",ROOT/"results"]:
   if d.exists():hits += [str(p) for p in d.rglob("*") if p.is_file()]
  require(not hits,f"PREEXISTING_EMPIRICAL:{hits[:20]}")
  require(not FINAL.exists(),"FINAL_BINDING_ALREADY_EXISTS")
  return {"empirical_files":0}
 check("pretraining_empty_state",no_results)

 def source_contract():
  src=(HERE/"sensitivity_runner.py").read_text()
  tree=ast.parse(src)
  funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  require("run_stage_t" in funcs and "run_stage_e" in funcs,"STAGE_FUNCS")
  tseg=ast.get_source_segment(src,funcs["run_stage_t"]) or ""
  require("read_test_labels" not in tseg,"STAGE_T_CALLS_TEST_LABELS")
  require("EVAL_MASKS" not in tseg,"STAGE_T_REFERENCES_EVAL_MASKS")
  require("stable_metrics(" not in tseg,"STAGE_T_CALLS_TEST_METRICS")
  require("ALL_105_LOCKED" in src,"STAGE_E_GLOBAL_LOCK_GUARD")
  require("warn_only=False" in src,"STRICT_WARN_ONLY_FALSE_MISSING")
  require("append_reverse" in src,"REVERSE_ADAPTER_MISSING")
  require("kind=\"mergesort\"" in src,"STABLE_TIE_SORT_MISSING")
  return {"stage_t_test_reference_static":"PASS","strict_determinism_static":"PASS"}
 check("runner_static_stage_separation",source_contract)

 # Import runner after strict env is set by shell wrapper.
 runner=import_file(HERE/"sensitivity_runner.py","v27_prod")
 def functional():
  return runner.self_test()
 selftest=check("synthetic_cuda_determinism_and_reverse",functional)

 def threshold_parity():
  core=import_file(HERE/"formal_rerun_core_REFERENCE.py","core_ref")
  rng=np.random.default_rng(141707);mx=0.0
  for i in range(100):
   y=rng.integers(0,2,50);y[:2]=[0,1];s=np.round(rng.random(50),2)
   a=runner.f1_val_threshold(y,s);b=core.f1_val_threshold(y,s);mx=max(mx,abs(a-b))
   require(a==b,f"THRESHOLD_MISMATCH:{a}:{b}")
  return {"cases":100,"max_abs_error":mx}
 check("validation_threshold_parity",threshold_parity)

 def rf_params():
  from sklearn.ensemble import RandomForestClassifier
  m=RandomForestClassifier(n_estimators=500,criterion="gini",max_depth=None,min_samples_split=2,min_samples_leaf=1,
      min_weight_fraction_leaf=0.0,max_features="sqrt",max_leaf_nodes=None,min_impurity_decrease=0.0,bootstrap=True,
      oob_score=False,n_jobs=8,random_state=42,verbose=0,warm_start=False,class_weight=None,ccp_alpha=0.0,max_samples=None)
  p=m.get_params(deep=True)
  exp={"n_estimators":500,"criterion":"gini","max_depth":None,"min_samples_split":2,"min_samples_leaf":1,
       "min_weight_fraction_leaf":0.0,"max_features":"sqrt","max_leaf_nodes":None,"min_impurity_decrease":0.0,
       "bootstrap":True,"oob_score":False,"n_jobs":8,"random_state":42,"verbose":0,"warm_start":False,
       "class_weight":None,"ccp_alpha":0.0,"max_samples":None}
  for k,v in exp.items():require(p.get(k)==v,f"RF_PARAM:{k}:{p.get(k)}")
  q=OUTDIR/"RF_GET_PARAMS_PINNED_ENV.json";q.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
  return {"params_sha256":sha256(q),"parameter_count":len(p)}
 rf=check("rf_full_get_params",rf_params)

 def checkpoint_reload():
  import torch, tempfile
  runner.strict_seed(456)
  m=runner.make_model("MLP",3,h=4).cuda().eval();x=torch.tensor([[1.,2.,3.],[3.,2.,1.]],device="cuda")
  with torch.no_grad():a=m(x).detach().cpu()
  td=Path(tempfile.mkdtemp(dir=OUTDIR));p=td/"c.pt";torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},p)
  m2=runner.make_model("MLP",3,h=4);m2.load_state_dict(torch.load(p,map_location="cpu",weights_only=True));m2.cuda().eval()
  with torch.no_grad():b=m2(x).detach().cpu()
  require(torch.equal(a,b),"CHECKPOINT_RELOAD_NOT_EXACT");shutil.rmtree(td)
  return {"exact":True}
 check("checkpoint_reload_parity",checkpoint_reload)

 # If all checks passed, freeze production code and final binding.
 passed=all(c["pass"] for c in checks)
 if passed:
  code=ROOT/"code";code.mkdir(parents=True,exist_ok=True)
  for name in ["sensitivity_runner.py","sensitivity_orchestrator.py","sensitivity_lock_training.py"]:
   src=HERE/name;dst=code/name
   if dst.exists():require(sha256(dst)==sha256(src),f"CODE_DEST_COLLISION:{name}")
   else:shutil.copy2(src,dst)
  code_hash={n:sha256(code/n) for n in ["sensitivity_runner.py","sensitivity_orchestrator.py","sensitivity_lock_training.py"]}
  data=json.loads(DATA_BIND.read_text())
  p2a=json.loads(P2A_BIND.read_text())
  binding={
   "binding_id":"RUN_BINDINGS_v2","protocol_id":"V27_PROTOCOL_FREEZE_v2",
   "created_at":datetime.now().astimezone().isoformat(),
   "training_authorized":True,"stage_t_authorized":True,
   "stage_e_authorized_only_after_all_105_locked":True,
   "score_blind_binding":True,"v27_training_run_before_binding":False,"v27_test_metrics_before_binding":False,
   "data_stage_binding_sha256":EXPECTED["data_binding"],"p2a_reference_binding_sha256":EXPECTED["p2a_binding"],
   "p2a_private_reference_paths_sha256":EXPECTED["p2a_private"],
   "run_matrix_sha256":EXPECTED["matrix"],"fit_count":105,
   "family_counts":{"A":20,"B":60,"B0":20,"C":5},
   "runner_identity":code_hash,
   "p2b_audit_checks":checks,
   "p2b_selftest_summary":selftest,
   "rf_get_params_summary":rf,
   "environment":data["environment"],
   "environment_inventory_sha256":data["environment"]["environment_inventory_sha256"],
   "input_files":data["files"],
   "case_diagnostics":data["case_diagnostics"],
   "direct_key_intersections_after_purge":data["direct_key_intersections_after_purge"],
   "evaluation_mask_counts":data["evaluation_mask_counts"],
   "feature_dimensions":data["feature_dimensions"],
   "donor_worlds":data["donor_worlds"],
   "reference_binding_status":p2a["status"],
   "execution_contract":{
     "stage_T":"train/validation only; no test labels or test scores; every fit writes immutable checkpoint/preprocessor/validation artifacts",
     "global_lock":"sensitivity_lock_training.py must verify all 105 Stage-T artifacts before Stage E",
     "stage_E":"test labels/scores/metrics only after ALL_105_LOCKED.json exists",
     "stale_output":"reject partial/mismatched outputs; reuse only exact locked Stage-T identity",
     "determinism":"CUBLAS_WORKSPACE_CONFIG=:4096:8; torch strict deterministic warn_only=False after every seed reset",
     "ranking":"stable score descending after canonical company/year ascending base order"
   }
  }
  tmp=FINAL.with_name(FINAL.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n");os.replace(tmp,FINAL)
  final_sha=sha256(FINAL)
 else:
  final_sha=None

 detail={"component":"V27_P2B_DETAIL","status":"PASS_RUNNER_READY_FOR_STAGE_T" if passed else "HOLD_P2B",
         "checks":checks,"errors":errors,"final_binding_created":passed,"final_binding_sha256":final_sha}
 DETAIL.write_text(json.dumps(detail,indent=2,sort_keys=True)+"\n")
 share={"component":"V27_P2B_RUNNER_AUDIT","status":detail["status"],"failed_checks":[c["name"] for c in checks if not c["pass"]],
        "final_RUN_BINDINGS_v2_created":passed,"final_RUN_BINDINGS_v2_sha256":final_sha,
        "training_authorized":passed,"stage_e_authorized_now":False,
        "next_gate":"STAGE_T_105_FITS" if passed else "P2B_REVIEW",
        "write_root":str(ROOT)}
 SHARE.write_text(json.dumps(share,indent=2,sort_keys=True)+"\n")
 print(json.dumps(share,indent=2,sort_keys=True))
 print(f"DETAIL={DETAIL}")
 print(f"SHARE={SHARE}")
 raise SystemExit(0 if passed else 2)

if __name__=="__main__":
    main()
