#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/private_workspace/peerj_141707_v28_b0gnn_extension")
V27_ROOT = Path("/path/to/private_workspace/peerj_141707_v27")
BINDING_DEFAULT = ROOT/"bindings/extension_source_binding.json"
STAGE_E_AUTH = ROOT/"bindings/STAGE_E_AUTHORIZATION_v1.json"
MATRIX_PATH = ROOT/"code/extension_run_matrix.json"
PROTOCOL_PATH = ROOT/"code/extension_protocol.json"

FEATURE_LISTS = V27_ROOT/"inputs/financial/FEATURE_LISTS_v27.json"
LABELS = V27_ROOT/"inputs/base/fraud_labels_v1_0.parquet"
ASOF_FEATURES = V27_ROOT/"inputs/financial/node_features_asof_financial_v1.parquet"
EVAL_MASKS = V27_ROOT/"inputs/case/evaluation_masks_v27.parquet"
MAPPING = V27_ROOT/"inputs/base/node_id_index_label_v1_0_fiverel_derived.parquet"
EDGES = V27_ROOT/"inputs/base/global_edge_index.parquet"

TRAIN_SPLIT="train_2010_2018"
VAL_SPLIT="validation_2019_2020"
TEST_SPLIT="test_2021_2022"
TRAIN_YEARS=list(range(2010,2019))
VAL_YEARS=[2019,2020]
TEST_YEARS=[2021,2022]
REL_MAP={"E1":0,"E1_HOLDS_BY":0,"E2":1,"E2_FLOAT_HELD_BY":1,
         "E3":2,"E3_HAS_MANAGER":2,"E4":3,"E4_CO_HELD":3,
         "E5":4,"E5_CO_MGR":4}
ALLOWED_MODELS={"GCN_reverse","GraphSAGE_reverse"}
ALLOWED_SEEDS={42,123,456,789,1024}

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def write_json_atomic(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def normalize_code(s):
    x=s.astype("string").str.strip().str.replace(r"^'","",regex=True)
    x=x.str.replace(r"\.0$","",regex=True).str.replace(r"\.(SZ|SH|BJ)$","",regex=True)
    x=x.str.replace(r"^C:","",regex=True).str.zfill(6)
    if not (x.notna().all() and x.str.fullmatch(r"\d{6}").all()):
        raise RuntimeError("INVALID_COMPANY_CODE")
    return x.astype(str)

def strict_seed(seed:int):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_NOT_STRICT")
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("DETERMINISTIC_ALGORITHMS_NOT_ENABLED")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("DETERMINISTIC_WARN_ONLY_TRUE")

def append_reverse(edge_index,edge_type):
    import torch
    if edge_index.ndim!=2 or edge_index.shape[0]!=2:
        raise RuntimeError("EDGE_INDEX_SHAPE")
    if edge_type.ndim!=1 or edge_type.shape[0]!=edge_index.shape[1]:
        raise RuntimeError("EDGE_TYPE_SHAPE")
    return torch.cat([edge_index,edge_index.flip(0)],dim=1),torch.cat([edge_type,edge_type],dim=0)

def f1_val_threshold(y,s):
    from sklearn.metrics import precision_recall_curve
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    p,r,t=precision_recall_curve(y,s)
    if len(t)==0:return 0.5
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return float(t[int(np.nanargmax(f))])

def stable_order(company_code,fiscal_year,score):
    cc=np.asarray(company_code).astype(str)
    yy=np.asarray(fiscal_year).astype(int)
    ss=np.asarray(score,dtype=float)
    base=np.lexsort((yy,cc))
    return base[np.argsort(-ss[base],kind="mergesort")]

def stable_metrics(df,threshold):
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
    if not {"company_code","fiscal_year","y_true","score"}<=set(df.columns):
        raise RuntimeError("METRIC_COLUMNS")
    d=df.copy()
    y=d.y_true.to_numpy(int);s=d.score.to_numpy(float)
    if len(y)==0 or not np.isfinite(s).all():
        raise RuntimeError("METRIC_VALUES")
    if len(np.unique(y))<2:
        return {"roc_auc":None,"ap":None,"f1_val_threshold":None,
                "p_at_5pct":None,"r_at_10fpr":None,"hits_at_budget":None,
                "n":int(len(y)),"n_pos":int(y.sum()),"n_neg":int(len(y)-y.sum()),
                "reason":"single_class"}
    order=stable_order(d.company_code,d.fiscal_year,s)
    ys=y[order];ss=s[order]
    n=len(y);budget=max(1,int(math.ceil(.05*n)))
    selected=ys[:budget]
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=True)
    ok=fpr<=.10+1e-12
    cutoff=ss[budget-1]
    above=ss>cutoff;equal=ss==cutoff
    slots=budget-int(above.sum())
    positives=int(ys[equal].sum());negatives=int(equal.sum())-positives
    base_hits=int(ys[above].sum())
    tie_min=base_hits+max(0,slots-negatives)
    tie_max=base_hits+min(slots,positives)
    return {
        "roc_auc":float(roc_auc_score(y,s)),
        "ap":float(average_precision_score(y,s)),
        "f1_val_threshold":float(f1_score(y,(s>=threshold).astype(int),zero_division=0)),
        "p_at_5pct":float(selected.mean()),
        "r_at_10fpr":float(np.max(tpr[ok])) if ok.any() else 0.0,
        "hits_at_budget":int(selected.sum()),
        "n":int(n),"n_pos":int(y.sum()),"n_neg":int(n-y.sum()),
        "budget":int(budget),"cutoff_score":float(cutoff),
        "cutoff_tie_rows":int(equal.sum()),"selected_from_cutoff_tie":int(slots),
        "tie_hits_min":int(tie_min),"tie_hits_max":int(tie_max),
    }

def load_binding(path:Path):
    j=json.loads(path.read_text())
    if j.get("binding_id")!="V28_B0GNN_EXTENSION_BINDING_v1":
        raise RuntimeError("WRONG_EXTENSION_BINDING")
    if not j.get("training_authorized",False):
        raise RuntimeError("TRAINING_NOT_AUTHORIZED")
    if j.get("stage_e_authorized",False):
        raise RuntimeError("STAGE_E_MUST_USE_SEPARATE_AUTH_FILE")
    me=Path(__file__).resolve()
    if sha256(me)!=j.get("runner_sha256"):
        raise RuntimeError("RUNNER_HASH_MISMATCH")
    if sha256(MATRIX_PATH)!=j.get("run_matrix_sha256"):
        raise RuntimeError("RUN_MATRIX_HASH_MISMATCH")
    if sha256(PROTOCOL_PATH)!=j.get("protocol_sha256"):
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    for rec in j.get("input_ledger",{}).values():
        p=Path(rec["path"])
        if not p.is_file() or sha256(p)!=rec["sha256"] or p.stat().st_size!=rec["size_bytes"]:
            raise RuntimeError(f"BOUND_INPUT_MISMATCH:{p}")
    return j

def load_matrix():
    j=json.loads(MATRIX_PATH.read_text())
    if not isinstance(j,list) or len(j)!=10:
        raise RuntimeError("RUN_MATRIX_NOT_10")
    ids=[r.get("run_id") for r in j]
    if len(set(ids))!=10:
        raise RuntimeError("RUN_ID_NOT_UNIQUE_MATRIX")
    for r in j:
        if r.get("extension_family")!="B0GNN42" or r.get("modality")!="M11_no_FIN87":
            raise RuntimeError("BAD_EXTENSION_MATRIX_ROW")
        if r.get("model") not in ALLOWED_MODELS or int(r.get("model_seed")) not in ALLOWED_SEEDS:
            raise RuntimeError("UNAUTHORIZED_EXTENSION_ROW")
        if r.get("supervision")!="original" or r.get("input_features")!="approved_asof_minus_FIN87":
            raise RuntimeError("BAD_EXTENSION_CONTRACT")
    return j

def get_run(run_id):
    matches=[r for r in load_matrix() if r["run_id"]==run_id]
    if len(matches)!=1:raise RuntimeError("RUN_ID_NOT_UNIQUE")
    return matches[0]

def feature_columns(run):
    fl=json.loads(FEATURE_LISTS.read_text())
    cols=list(fl["M11_no_FIN87"])
    m11=list(fl["M11"]);fin87=set(fl["FIN87"])
    expected=[x for x in m11 if x not in fin87]
    if cols!=expected or len(cols)!=42 or len(set(cols))!=42:
        raise RuntimeError("M11_NO_FIN87_IDENTITY")
    return cols

def read_stage_t_labels():
    import pyarrow.parquet as pq
    cols=["company_code","fiscal_year","split_v1","label_v1_strict_ab_primary"]
    t=pq.read_table(LABELS,columns=cols,
                    filters=[("split_v1","in",[TRAIN_SPLIT,VAL_SPLIT])]).to_pandas()
    t["company_code"]=normalize_code(t["company_code"])
    t["fiscal_year"]=t["fiscal_year"].astype(int)
    t=t.rename(columns={"label_v1_strict_ab_primary":"target"})
    if t.duplicated(["company_code","fiscal_year"]).any():raise RuntimeError("LABEL_DUP")
    return t

def read_test_labels(include_null=False):
    import pyarrow.parquet as pq
    cols=["company_code","fiscal_year","split_v1","label_v1_strict_ab_primary"]
    t=pq.read_table(LABELS,columns=cols,filters=[("split_v1","==",TEST_SPLIT)]).to_pandas()
    t["company_code"]=normalize_code(t["company_code"])
    t["fiscal_year"]=t["fiscal_year"].astype(int)
    t=t.rename(columns={"label_v1_strict_ab_primary":"target"})
    if len(t)!=9357:
        raise RuntimeError(f"TEST_CONTEXT_ROWS:{len(t)}")
    eligible=t.target.notna()
    if (int(eligible.sum()),int(pd.to_numeric(t.loc[eligible,"target"],errors="raise").sum()))!=(8435,20):
        raise RuntimeError("TEST_LABEL_COUNTS")
    if not include_null:t=t[eligible].copy()
    return t

def read_features_for_run(run,stage):
    import pyarrow.parquet as pq
    cols=feature_columns(run)
    src=ASOF_FEATURES
    schema=set(pq.ParquetFile(src).schema_arrow.names)
    if {"company_code","fiscal_year"}<=schema:
        keycols=["company_code","fiscal_year"];yearcol="fiscal_year"
    elif {"firm_id","year"}<=schema:
        keycols=["firm_id","year"];yearcol="year"
    else:raise RuntimeError("FEATURE_KEY_SCHEMA")
    years=TRAIN_YEARS+VAL_YEARS if stage=="T" else TEST_YEARS if stage=="E" else None
    if years is None:raise RuntimeError("BAD_FEATURE_STAGE")
    readcols=list(dict.fromkeys(keycols+["node_id"]+cols))
    d=pq.read_table(src,columns=readcols,filters=[(yearcol,"in",years)]).to_pandas()
    if "company_code" not in d:
        d["company_code"]=normalize_code(d["firm_id"]);d["fiscal_year"]=d["year"].astype(int)
    else:
        d["company_code"]=normalize_code(d["company_code"]);d["fiscal_year"]=d["fiscal_year"].astype(int)
        d["year"]=d["fiscal_year"]
    if d.duplicated(["company_code","fiscal_year"]).any():raise RuntimeError("FEATURE_DUP_KEYS")
    return d

def build_stage_t_df(run):
    labels=read_stage_t_labels()
    feat=read_features_for_run(run,"T")
    d=feat.merge(labels[["company_code","fiscal_year","split_v1","target"]],
                 on=["company_code","fiscal_year"],how="inner",validate="one_to_one")
    if len(d)!=24527+7764:raise RuntimeError(f"STAGE_T_JOIN_ROWS:{len(d)}")
    d["partition"]=np.where(d.split_v1==TRAIN_SPLIT,"train","validation")
    d["supervise"]=d.target.notna()
    tr=d[(d.partition=="train")&d.supervise];va=d[(d.partition=="validation")&d.supervise]
    got=(len(tr),int(tr.target.sum()),len(va),int(va.target.sum()))
    exp=(24106,254,7309,40)
    if got!=exp:raise RuntimeError(f"SUPERVISION_COUNTS:{got}!={exp}")
    return d

def fit_scaler(d,cols):
    tr=d[(d.partition=="train")&d.supervise]
    x=tr[cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(np.float32)
    mu=x.mean(0,keepdims=True).astype(np.float32)
    sd=x.std(0,keepdims=True).astype(np.float32)
    sd=np.where(sd<1e-6,1.0,sd).astype(np.float32)
    return mu,sd

def transform(d,cols,mu,sd):
    x=d[cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(np.float32)
    return ((x-mu)/sd).astype(np.float32)

def make_model(name,in_dim,h=64):
    import torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import GCNConv,SAGEConv
    if name=="GCN_reverse":
        class M(nn.Module):
            def __init__(self):
                super().__init__();self.c1=GCNConv(in_dim,h);self.c2=GCNConv(h,h);self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.relu(self.c1(x,edge_index));x=F.dropout(x,.3,training=self.training);x=F.relu(self.c2(x,edge_index));return self.head(x).squeeze(-1)
        return M()
    if name=="GraphSAGE_reverse":
        class M(nn.Module):
            def __init__(self):
                super().__init__();self.c1=SAGEConv(in_dim,h);self.c2=SAGEConv(h,h);self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.relu(self.c1(x,edge_index));x=F.dropout(x,.3,training=self.training);x=F.relu(self.c2(x,edge_index));return self.head(x).squeeze(-1)
        return M()
    raise RuntimeError("UNKNOWN_MODEL")

def pos_weight(d):
    y=d[(d.partition=="train")&d.supervise].target.astype(int).to_numpy()
    p=int((y==1).sum());n=int((y==0).sum())
    if (n,p)!=(23852,254):raise RuntimeError(f"TRAIN_CLASS_COUNTS:{n},{p}")
    return n/p,n,p

def save_npz_atomic(path,**kwargs):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}.npz")
    np.savez(tmp,**kwargs);os.replace(tmp,path)

def graph_static(years):
    import pyarrow.parquet as pq
    mapping=pd.read_parquet(MAPPING,columns=["node_id","node_idx"])
    node_to_idx=dict(zip(mapping.node_id.astype(str),mapping.node_idx.astype(int)))
    n_total=len(mapping)
    edges=pq.read_table(EDGES,columns=["src_idx","dst_idx","edge_type","year"],
                        filters=[("year","in",list(years))]).to_pandas()
    unknown=set(edges.edge_type.astype(str))-set(REL_MAP)
    if unknown:raise RuntimeError(f"UNKNOWN_EDGE_TYPES:{sorted(unknown)}")
    return node_to_idx,n_total,edges

def graph_year_cache(d,cols,mu,sd,years):
    import torch
    node_to_idx,n_total,edges=graph_static(years)
    X=transform(d,cols,mu,sd)
    rowpos={idx:i for i,idx in enumerate(d.index)}
    cache={}
    for year in years:
        sub=d[d.fiscal_year==year]
        pos=np.array([rowpos[i] for i in sub.index],dtype=int)
        idx=np.array([node_to_idx[str(x)] for x in sub.node_id],dtype=np.int64)
        es=edges[edges.year==year]
        ei=np.vstack([es.src_idx.to_numpy(np.int64),es.dst_idx.to_numpy(np.int64)])
        et=np.array([REL_MAP[str(x)] for x in es.edge_type],dtype=np.int64)
        eii,ett=append_reverse(torch.from_numpy(ei),torch.from_numpy(et))
        sup=sub.supervise.to_numpy(bool)
        cache[year]={
            "company_idx":torch.from_numpy(idx),
            "company_x":torch.from_numpy(X[pos]),
            "edge_index":eii,"edge_type":ett,
            "sup_idx":torch.from_numpy(idx[sup]),
            "sup_y":torch.from_numpy(sub.loc[sub.supervise,"target"].astype(np.float32).to_numpy().copy()),
            "sup_keys":sub.loc[sub.supervise,["company_code","fiscal_year"]].reset_index(drop=True)
        }
    return cache,n_total

def eval_graph(model,cache,years,n_total,in_dim,device):
    import torch
    ys=[];ss=[];keys=[]
    model.eval()
    with torch.no_grad():
        for year in years:
            c=cache[year]
            x=torch.zeros((n_total,in_dim),dtype=torch.float32,device=device)
            x[c["company_idx"].to(device)]=c["company_x"].to(device)
            logits=model(x,c["edge_index"].to(device),c["edge_type"].to(device))
            score=torch.sigmoid(logits[c["sup_idx"].to(device)]).cpu().numpy()
            ys.append(c["sup_y"].numpy());ss.append(score);keys.append(c["sup_keys"])
    return np.concatenate(ys).astype(int),np.concatenate(ss),pd.concat(keys,ignore_index=True)

def fit_gnn_stage_t(run,d,tmp):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    strict_seed(int(run["model_seed"]))
    cols=feature_columns(run);mu,sd=fit_scaler(d,cols)
    cache,n_total=graph_year_cache(d,cols,mu,sd,TRAIN_YEARS+VAL_YEARS)
    device=torch.device("cuda");model=make_model(run["model"],len(cols)).to(device)
    pw,n,p=pos_weight(d)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device))
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    best=-np.inf;state=None;best_epoch=0;stale=0
    for epoch in range(1,101):
        model.train()
        for year in TRAIN_YEARS:
            c=cache[year]
            if len(c["sup_y"])==0:continue
            x=torch.zeros((n_total,len(cols)),dtype=torch.float32,device=device)
            x[c["company_idx"].to(device)]=c["company_x"].to(device)
            opt.zero_grad()
            logits=model(x,c["edge_index"].to(device),c["edge_type"].to(device))
            loss=crit(logits[c["sup_idx"].to(device)],c["sup_y"].to(device))
            loss.backward();opt.step()
        vy,vs,_=eval_graph(model,cache,VAL_YEARS,n_total,len(cols),device)
        auc=float(roc_auc_score(vy,vs))
        if auc>best+1e-8:
            best=auc;best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale=0
        else:stale+=1
        if stale>=10:break
    if state is None:raise RuntimeError("NO_GNN_CHECKPOINT")
    model.load_state_dict(state);model.to(device)
    vy,vs,vkeys=eval_graph(model,cache,VAL_YEARS,n_total,len(cols),device)
    if (len(vy),int(vy.sum()))!=(7309,40):raise RuntimeError("VALIDATION_PRED_COUNTS")
    thr=f1_val_threshold(vy,vs)
    torch.save(state,tmp/"checkpoint.pt")
    save_npz_atomic(tmp/"scaler.npz",mu=mu,sd=sd)
    vp=vkeys.copy();vp["y_true"]=vy;vp["score"]=vs
    vp.to_parquet(tmp/"validation_predictions.parquet",index=False)
    return {"best_epoch":best_epoch,"best_val_auc":best,"val_threshold":thr,
            "pos_weight":pw,"train_neg":n,"train_pos":p,
            "model_artifact":"checkpoint.pt","preprocessor_artifact":"scaler.npz",
            "validation_artifact":"validation_predictions.parquet",
            "direction":"append_reverse_same_relation_preserve_multiplicity",
            "input_dim":len(cols)}

def stage_t_identity(run,binding_path):
    return {
        "protocol_id":"V28_B0GNN_EXTENSION_PROTOCOL_v1",
        "binding_sha256":sha256(binding_path),
        "runner_sha256":sha256(Path(__file__).resolve()),
        "run":run,"stage":"T","test_labels_read":False,
        "test_scores_computed":False,"input_dim":42
    }

def run_stage_t(run_id,binding_path):
    load_binding(binding_path);run=get_run(run_id)
    out=ROOT/"stage_t"/run_id
    ident=stage_t_identity(run,binding_path)
    ident_hash=hashlib.sha256((json.dumps(ident,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
    if out.exists():
        done=out/"STAGE_T_COMPLETE.json";iid=out/"FIT_IDENTITY.json"
        if done.is_file() and iid.is_file():
            old=json.loads(iid.read_text())
            if old.get("identity_sha256")==ident_hash and json.loads(done.read_text()).get("status")=="LOCKED":
                print(f"STAGE_T_ALREADY_LOCKED_EXACT {run_id}");return
        raise RuntimeError(f"STALE_OR_PARTIAL_STAGE_T_OUTPUT:{out}")
    tmp=out.with_name(f".tmp_{run_id}_{os.getpid()}")
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        d=build_stage_t_df(run)
        res=fit_gnn_stage_t(run,d,tmp)
        write_json_atomic(tmp/"stage_t_result.json",{
            "run_id":run_id,"extension_family":run["extension_family"],
            "model":run["model"],"modality":run["modality"],
            "model_seed":run["model_seed"],"imputation_seed":None,
            "supervision":"original","test_evaluated":False,**res})
        write_json_atomic(tmp/"FIT_IDENTITY.json",{"identity":ident,"identity_sha256":ident_hash})
        artifacts={}
        for p in sorted(tmp.iterdir()):
            if p.is_file():artifacts[p.name]={"sha256":sha256(p),"size_bytes":p.stat().st_size}
        write_json_atomic(tmp/"STAGE_T_COMPLETE.json",{
            "status":"LOCKED","run_id":run_id,"artifacts":artifacts,
            "test_labels_read":False,"test_scores_computed":False})
        os.replace(tmp,out)
        print(f"STAGE_T_LOCKED {run_id}")
    except Exception:
        if tmp.exists():
            fail=ROOT/"stage_t_failures"/f"{run_id}__{int(time.time())}__{os.getpid()}"
            fail.parent.mkdir(parents=True,exist_ok=True);os.replace(tmp,fail)
        raise

def load_stage_t_artifacts(run_id):
    out=ROOT/"stage_t"/run_id
    done=json.loads((out/"STAGE_T_COMPLETE.json").read_text())
    if done.get("status")!="LOCKED":raise RuntimeError("STAGE_T_NOT_LOCKED")
    return out,json.loads((out/"stage_t_result.json").read_text())

def verify_stage_e_authorization(binding_path):
    lock=ROOT/"stage_t/ALL_10_LOCKED.json"
    if not lock.is_file():raise RuntimeError("ALL_10_STAGE_T_NOT_LOCKED")
    lj=json.loads(lock.read_text())
    if lj.get("status")!="ALL_10_LOCKED" or int(lj.get("fit_count",-1))!=10:
        raise RuntimeError("BAD_GLOBAL_STAGE_T_LOCK")
    if not STAGE_E_AUTH.is_file():raise RuntimeError("STAGE_E_NOT_AUTHORIZED")
    a=json.loads(STAGE_E_AUTH.read_text())
    if a.get("status")!="AUTHORIZED" or not a.get("stage_e_authorized",False):
        raise RuntimeError("BAD_STAGE_E_AUTH")
    checks={
        "binding_sha256":sha256(binding_path),
        "global_stage_t_lock_sha256":sha256(lock),
        "runner_sha256":sha256(Path(__file__).resolve()),
        "run_matrix_sha256":sha256(MATRIX_PATH),
        "protocol_sha256":sha256(PROTOCOL_PATH)}
    for k,v in checks.items():
        if a.get(k)!=v:raise RuntimeError(f"STAGE_E_AUTH_IDENTITY:{k}")
    return lj,a

def infer_test(run,tout,tres):
    import torch
    cols=feature_columns(run)
    labels=read_test_labels(include_null=True)
    feat=read_features_for_run(run,"E")
    d=feat.merge(labels[["company_code","fiscal_year","target"]],
                 on=["company_code","fiscal_year"],how="inner",validate="one_to_one")
    if len(d)!=9357:raise RuntimeError(f"GNN_TEST_CONTEXT_JOIN_ROWS:{len(d)}")
    d=d.sort_values(["fiscal_year","company_code"]).reset_index(drop=True)
    d["partition"]="test";d["supervise"]=d["target"].notna()
    z=np.load(tout/"scaler.npz");mu=z["mu"];sd=z["sd"]
    cache,n_total=graph_year_cache(d,cols,mu,sd,TEST_YEARS)
    model=make_model(run["model"],len(cols))
    state=torch.load(tout/"checkpoint.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state);model.cuda().eval()
    y,score,keys=eval_graph(model,cache,TEST_YEARS,n_total,len(cols),torch.device("cuda"))
    q=keys.copy();q["y_true"]=y;q["score"]=score
    if (len(q),int(q.y_true.sum()))!=(8435,20):raise RuntimeError("GNN_TEST_SUPERVISION_COUNTS")
    return q

def run_stage_e(run_id,binding_path):
    load_binding(binding_path)
    verify_stage_e_authorization(binding_path)
    run=get_run(run_id);tout,tres=load_stage_t_artifacts(run_id)
    out=ROOT/"stage_e"/run_id
    if out.exists():raise RuntimeError("STAGE_E_OUTPUT_EXISTS_STOP")
    tmp=out.with_name(f".tmp_{run_id}_{os.getpid()}")
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        strict_seed(int(run["model_seed"]))
        pred=infer_test(run,tout,tres)
        pred["company_code"]=normalize_code(pred.company_code);pred["fiscal_year"]=pred.fiscal_year.astype(int)
        masks=pd.read_parquet(EVAL_MASKS)
        masks["company_code"]=normalize_code(masks.company_code);masks["fiscal_year"]=masks.fiscal_year.astype(int)
        z=pred.merge(masks,on=["company_code","fiscal_year"],how="left",validate="one_to_one")
        if len(z)!=8435:raise RuntimeError("TEST_PRED_ROWS")
        expected={"full_test":(8435,20),"D_frozen":(8422,7),"O_frozen":(8428,13),
                  "S_frozen":(7757,6),"J_frozen":(7753,2)}
        thr=float(tres["val_threshold"]);metrics={}
        for pop,(n,p) in expected.items():
            sub=z[z[pop].astype(bool)][["company_code","fiscal_year","y_true","score"]].copy()
            if (len(sub),int(sub.y_true.sum()))!=(n,p):
                raise RuntimeError(f"POP_COUNTS:{pop}")
            metrics[pop]=stable_metrics(sub,thr)
        order=stable_order(z.company_code,z.fiscal_year,z.score);top=z.iloc[order[:422]]
        attribution={"budget":422,
                     "full_hits":int(top.y_true.sum()),
                     "D_positive_hits_in_full_top422":int((top.y_true.eq(1)&top.D_positive.astype(bool)).sum()),
                     "O_positive_hits_in_full_top422":int((top.y_true.eq(1)&top.O_positive.astype(bool)).sum())}
        if attribution["full_hits"]!=attribution["D_positive_hits_in_full_top422"]+attribution["O_positive_hits_in_full_top422"]:
            raise RuntimeError("FULL_BUDGET_ATTRIBUTION_IDENTITY")
        af=metrics["full_test"]["roc_auc"];ao=metrics["O_frozen"]["roc_auc"];ad=metrics["D_frozen"]["roc_auc"]
        auc_identity=abs(20*af-(13*ao+7*ad))
        if auc_identity>1e-10:raise RuntimeError(f"AUC_DO_IDENTITY:{auc_identity}")
        pred.to_parquet(tmp/"test_predictions.parquet",index=False)
        write_json_atomic(tmp/"stage_e_result.json",{
            "run_id":run_id,"validation_threshold":thr,"metrics":metrics,
            "full_budget_attribution":attribution,"auc_DO_identity_abs_error":auc_identity,
            "strict_seed_explicit_before_inference":True})
        arts={p.name:{"sha256":sha256(p),"size_bytes":p.stat().st_size}
              for p in tmp.iterdir() if p.is_file()}
        write_json_atomic(tmp/"STAGE_E_COMPLETE.json",{
            "status":"LOCKED","run_id":run_id,"artifacts":arts,
            "strict_seed_explicit_before_inference":True})
        os.replace(tmp,out);print(f"STAGE_E_LOCKED {run_id}")
    except Exception:
        if tmp.exists():
            fail=ROOT/"stage_e_failures"/f"{run_id}__{int(time.time())}__{os.getpid()}"
            fail.parent.mkdir(parents=True,exist_ok=True);os.replace(tmp,fail)
        raise

def self_test():
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA_NOT_AVAILABLE")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_ENV")
    strict_seed(42);strict_seed(42)
    if torch.is_deterministic_algorithms_warn_only_enabled():raise RuntimeError("WARN_ONLY")
    ei=torch.tensor([[0,1,1],[1,2,2]],dtype=torch.long);et=torch.tensor([0,1,1],dtype=torch.long)
    e2,t2=append_reverse(ei,et)
    if not torch.equal(e2,torch.cat([ei,ei.flip(0)],1)) or not torch.equal(t2,torch.cat([et,et],0)):
        raise RuntimeError("REVERSE_TEST")
    o=stable_order(np.array(["000002","000001","000003","000001"]),
                   np.array([2022,2022,2021,2021]),np.array([.5,.5,.7,.5]))
    if o.tolist()!=[2,3,1,0]:raise RuntimeError(f"STABLE_ORDER:{o.tolist()}")
    def one(kind):
        strict_seed(123)
        x=torch.tensor([[1.,2.,3.],[2.,1.,0.],[0.,1.,2.]],device="cuda",requires_grad=True)
        edge=torch.tensor([[0,1,1,2],[1,0,2,1]],device="cuda")
        m=make_model(kind,3,h=4).cuda();opt=torch.optim.Adam(m.parameters(),lr=5e-4)
        opt.zero_grad();out=m(x,edge,None);loss=out.square().mean();loss.backward();opt.step()
        torch.cuda.synchronize()
        vec=torch.cat([v.detach().flatten().cpu() for v in m.state_dict().values()])
        return out.detach().cpu(),vec
    for kind in ["GCN_reverse","GraphSAGE_reverse"]:
        a,av=one(kind);b,bv=one(kind)
        if not torch.equal(a,b) or not torch.equal(av,bv):
            raise RuntimeError(f"DETERMINISM_FAIL:{kind}")
    return {"status":"PASS","cuda_device":torch.cuda.get_device_name(0),
            "strict":True,"input_dim_contract":42,
            "tested":["seed_reset","reverse_edge","stable_tie_order","GCN_reverse","GraphSAGE_reverse"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binding",type=Path,default=BINDING_DEFAULT)
    ap.add_argument("--stage",choices=["T","E"])
    ap.add_argument("--run-id")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True));return
    if not args.stage or not args.run_id:ap.error("--stage and --run-id required")
    if args.stage=="T":run_stage_t(args.run_id,args.binding)
    else:run_stage_e(args.run_id,args.binding)

if __name__=="__main__":
    main()
