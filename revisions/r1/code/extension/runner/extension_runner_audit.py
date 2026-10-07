#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path

ROOT=Path("/path/to/private_workspace/peerj_141707_v28_b0gnn_extension")
V27=Path("/path/to/private_workspace/peerj_141707_v27")
CODE=ROOT/"code"
PREFLIGHT=ROOT/"preflight/V28_B0GNN_EXTENSION_BINDING_PRELIM_v1.json"
RUNNER=CODE/"extension_runner.py"
ORCH=CODE/"extension_orchestrator.py"
LOCKER=CODE/"extension_lock_training.py"
PROTOCOL=CODE/"extension_protocol.json"
MATRIX=CODE/"extension_run_matrix.json"
BINDING=ROOT/"bindings/extension_source_binding.json"
SHARE=ROOT/"bindings/V28_B0GNN_RUNNER_AUDIT_SHARE.json"
DETAIL=ROOT/"bindings/V28_B0GNN_RUNNER_AUDIT_DETAIL.json"
RETURN=ROOT/"V28_B0GNN_RUNNER_AUDIT_RETURN.zip"

EXPECTED_PREFLIGHT_SHA="4674c1cb19de7d22e23a42f4da81903729c6513236cbc05b6af455072117f654"
EXPECTED_PROTOCOL_SHA="537cf3b88ee866a648df5255ae7f6c8f4a69b34035806f503e272fadf6ef2205"
EXPECTED_MATRIX_SHA="0b0ae312e3a8af28ff8b50303fdedb3826b697c1bc82c0790bd3a73273ed211e"
EXPECTED_V27_BINDING_SHA="cebbfada1bf30ace4f2c95d8fc3b2e60fe9f084294026ea0071213b88a7f4325"
EXPECTED_V27_MATRIX_SHA="4d78371987154fa5aee660479c447e30f56608d28616359f04d8c472759f8923"
EXPECTED_V27_RUNNER_SHA="9e8efa8caf9145b748c01cb6fe86868136afd9f4cd33907fb886668eb854c5b2"
EXPECTED_V27_STAGEE_MANIFEST_SHA="4a0086b676a5df8c7a7c085f44dcfd3722ae4d5f8b3a081bae841d907de95c06"
EXPECTED_COMPONENT_SHA="c8b6c4ec082d41e6264ae5667b4c3982fac43f02a6c8d53f267f824edebb102c"

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def writej(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f".tmp.{os.getpid()}")
    t.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(t,p)

def calls_in(fn):
    out=set()
    for n in ast.walk(fn):
        if isinstance(n,ast.Call):
            f=n.func
            if isinstance(f,ast.Name):out.add(f.id)
    return out

def callgraph(tree):
    funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    graph={k:{c for c in calls_in(v) if c in funcs} for k,v in funcs.items()}
    return funcs,graph

def reachable(graph,start):
    seen=set();stack=[start]
    while stack:
        x=stack.pop()
        if x in seen:continue
        seen.add(x);stack.extend(graph.get(x,()))
    return seen

def main():
    checks={}
    def ck(name,cond,detail=None):
        checks[name]={"pass":bool(cond),"detail":detail}
        if not cond:raise RuntimeError(f"CHECK_FAIL:{name}:{detail}")

    ck("preflight_exists",PREFLIGHT.is_file())
    ck("preflight_sha",sha256(PREFLIGHT)==EXPECTED_PREFLIGHT_SHA,sha256(PREFLIGHT))
    pre=json.loads(PREFLIGHT.read_text())
    ck("preflight_status",pre.get("score_blind") is True and pre.get("extension_training_authorized") is False and pre.get("stage_e_authorized") is False)
    ck("preflight_fit_count",pre.get("fit_count")==10)
    ck("preflight_component",pre.get("prefit_group_membership_sha256")==EXPECTED_COMPONENT_SHA)

    for p in [RUNNER,ORCH,LOCKER,PROTOCOL,MATRIX]:
        ck(f"file_{p.name}",p.is_file())
    ck("protocol_sha",sha256(PROTOCOL)==EXPECTED_PROTOCOL_SHA,sha256(PROTOCOL))
    ck("matrix_sha",sha256(MATRIX)==EXPECTED_MATRIX_SHA,sha256(MATRIX))
    mat=json.loads(MATRIX.read_text())
    ck("matrix_10",len(mat)==10 and len({r["run_id"] for r in mat})==10)
    ck("matrix_models",sum(r["model"]=="GCN_reverse" for r in mat)==5 and sum(r["model"]=="GraphSAGE_reverse" for r in mat)==5)
    ck("matrix_42_original",all(r["modality"]=="M11_no_FIN87" and r["supervision"]=="original" for r in mat))

    # Exact V27 identities retained from preflight.
    v27_id={
      "final_binding":V27/"bindings/RUN_BINDINGS_v2.json",
      "v27_matrix":V27/"bindings/sensitivity_run_matrix.json",
      "v27_runner":V27/"code/sensitivity_runner.py",
      "stage_e_manifest":V27/"stage_e_freeze_v1/SENSITIVITY_ARTIFACT_IDENTITIES.json"}
    exp={"final_binding":EXPECTED_V27_BINDING_SHA,"v27_matrix":EXPECTED_V27_MATRIX_SHA,
         "v27_runner":EXPECTED_V27_RUNNER_SHA,"stage_e_manifest":EXPECTED_V27_STAGEE_MANIFEST_SHA}
    for k,p in v27_id.items():ck(f"v27_{k}",p.is_file() and sha256(p)==exp[k],sha256(p) if p.is_file() else None)

    # Revalidate all bound inputs from score-blind preflight and frozen B0 references.
    for k,rec in pre["input_ledger"].items():
        p=Path(rec["path"])
        ck(f"preflight_input_{k}",p.is_file() and sha256(p)==rec["sha256"] and p.stat().st_size==rec["size_bytes"])
    comp=Path(pre["prefit_group_membership_matches"][0])
    ck("component_membership_exists_and_hash",comp.is_file() and sha256(comp)==EXPECTED_COMPONENT_SHA)
    for i,rec in enumerate(pre["reference_records"]):
        for kind in ["stage_e_complete","stage_e_result","test_predictions"]:
            m=rec[kind];p=Path(m["path"])
            ck(f"reference_{i}_{kind}",p.is_file() and sha256(p)==m["sha256"] and p.stat().st_size==m["size_bytes"])

    # Exact 42-column identity.
    fl=json.loads((V27/"inputs/financial/FEATURE_LISTS_v27.json").read_text())
    expected=[x for x in fl["M11"] if x not in set(fl["FIN87"])]
    ck("feature_identity_42",fl["M11_no_FIN87"]==expected and len(expected)==42 and len(set(expected))==42)

    # Static runner gate.
    src=RUNNER.read_text(encoding="utf-8")
    tree=ast.parse(src);funcs,graph=callgraph(tree)
    rt=reachable(graph,"run_stage_t");re_=reachable(graph,"run_stage_e")
    ck("stage_t_no_test_label_reach","read_test_labels" not in rt,sorted(rt))
    ck("stage_t_no_test_metric_reach","stable_metrics" not in rt and "infer_test" not in rt and "run_stage_e" not in rt,sorted(rt))
    ck("stage_e_strict_seed_reachable","strict_seed" in re_,sorted(re_))
    ck("stage_e_infer_reachable","infer_test" in re_)
    ck("stage_e_auth_reachable","verify_stage_e_authorization" in re_)
    ck("stage_e_metrics_reachable","stable_metrics" in re_)
    ck("write_root_literal",'/path/to/private_workspace/peerj_141707_v28_b0gnn_extension' in src)
    ck("read_v27_root_literal",'/path/to/private_workspace/peerj_141707_v27' in src)
    ck("no_home_user_root_write_literal",'/path/to/private_account/' not in src.replace('/path/to/private_workspace/',''))

    # Stage directories must still be clean at runner-audit time.
    for name in ["stage_t","stage_e","stage_t_failures","stage_e_failures"]:
        p=ROOT/name
        ck(f"clean_{name}",not p.exists() or not any(p.iterdir()))

    # CUDA strict synthetic runner self-test.
    env=os.environ.copy();env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    cp=subprocess.run([sys.executable,str(RUNNER),"--self-test"],capture_output=True,text=True,env=env)
    ck("runner_self_test_returncode",cp.returncode==0,cp.stdout+"\n"+cp.stderr)
    st=json.loads(cp.stdout)
    ck("runner_self_test_status",st.get("status")=="PASS" and st.get("strict") is True and st.get("input_dim_contract")==42,st)

    # Final binding is generated only after all checks pass. No model fit has run.
    input_keys=["asof_features","feature_lists","labels","mapping","edges","evaluation_masks"]
    ledger={k:pre["input_ledger"][k] for k in input_keys}
    binding={
      "binding_id":"V28_B0GNN_EXTENSION_BINDING_v1",
      "protocol_id":"V28_B0GNN_EXTENSION_PROTOCOL_v1",
      "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "result_awareness":"post_V27_results_extension",
      "prospective_preregistration_claim":False,
      "write_root":str(ROOT),"v27_root_read_only":str(V27),
      "training_authorized":True,"stage_e_authorized":False,
      "preflight_binding_path":str(PREFLIGHT),"preflight_binding_sha256":sha256(PREFLIGHT),
      "protocol_sha256":sha256(PROTOCOL),"run_matrix_sha256":sha256(MATRIX),
      "runner_sha256":sha256(RUNNER),"orchestrator_sha256":sha256(ORCH),"locker_sha256":sha256(LOCKER),
      "fit_count":10,"models":{"GCN_reverse":5,"GraphSAGE_reverse":5},
      "modality":"M11_no_FIN87","input_dim":42,
      "supervision_counts":pre["supervision_counts"],
      "input_ledger":ledger,
      "v27_provenance":{"final_binding_sha256":EXPECTED_V27_BINDING_SHA,
                        "run_matrix_sha256":EXPECTED_V27_MATRIX_SHA,
                        "runner_sha256":EXPECTED_V27_RUNNER_SHA,
                        "stage_e_manifest_sha256":EXPECTED_V27_STAGEE_MANIFEST_SHA},
      "prefit_component_membership":{"path":str(comp),"sha256":EXPECTED_COMPONENT_SHA},
      "frozen_reference_records":pre["reference_records"],
      "strict_determinism_selftest":st,
      "next_gate":"INDEPENDENT_REVIEW_THEN_STAGE_T_ONLY"
    }
    if BINDING.exists():raise RuntimeError("FINAL_BINDING_ALREADY_EXISTS")
    writej(BINDING,binding)
    detail={"component":"V28_B0GNN_EXTENSION_RUNNER_AUDIT","checks":checks,
            "stage_t_reachable_helpers":sorted(rt),"stage_e_reachable_helpers":sorted(re_),
            "runner_self_test":st,"no_training_run":True,"test_metrics_computed":False}
    writej(DETAIL,detail)
    share={"component":"V28_B0GNN_EXTENSION_RUNNER_AUDIT",
           "status":"PASS_RUNNER_READY_FOR_STAGE_T",
           "fit_count":10,"training_authorized":True,"stage_e_authorized":False,
           "binding_path":str(BINDING),"binding_sha256":sha256(BINDING),
           "runner_sha256":sha256(RUNNER),"orchestrator_sha256":sha256(ORCH),
           "locker_sha256":sha256(LOCKER),"protocol_sha256":sha256(PROTOCOL),
           "run_matrix_sha256":sha256(MATRIX),"preflight_binding_sha256":sha256(PREFLIGHT),
           "strict_determinism_selftest":st,"no_training_run":True,
           "test_metrics_computed":False,"stage_t_outputs_present":False,
           "stage_e_outputs_present":False,
           "next_gate":"REVIEW_AUDIT_SHARE_BEFORE_RUNNING_STAGE_T"}
    writej(SHARE,share)
    if RETURN.exists():RETURN.unlink()
    with zipfile.ZipFile(RETURN,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in [SHARE,BINDING,DETAIL]:
            z.write(p,p.name)
    print(json.dumps(share,indent=2,sort_keys=True))

if __name__=="__main__":main()
