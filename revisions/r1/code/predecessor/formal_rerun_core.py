#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, roc_curve,
)

SEEDS = [42,123,456,789,1024]
MODAL_PREFIXES = {
    "M5": ("feat_","fin_","fini_"),
    "M10": ("feat_","fin_","fini_","audit_","pld_","ctrl_"),
    "M11": ("feat_","fin_","fini_","audit_","pld_","ctrl_","rpt_"),
}
EXPECTED_DIMS = {"M5":104,"M10":122,"M11":129,"G6":1}
TRAIN_YEARS = list(range(2010,2019))
VAL_YEARS = [2019,2020]
TEST_YEARS = [2021,2022]
REL_MAP = {
    "E1":0,"E1_HOLDS_BY":0,
    "E2":1,"E2_FLOAT_HELD_BY":1,
    "E3":2,"E3_HAS_MANAGER":2,
    "E4":3,"E4_CO_HELD":3,
    "E5":4,"E5_CO_MGR":4,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def select_cols(df, modal):
    if modal=="G6":
        return []
    cols=sorted(c for c in df.columns if any(c.startswith(p) for p in MODAL_PREFIXES[modal]))
    if len(cols)!=EXPECTED_DIMS[modal]:
        raise RuntimeError(f"MODAL_DIM_MISMATCH:{modal}:got={len(cols)} expected={EXPECTED_DIMS[modal]}")
    return cols

def load_joined(data_dir, label_col):
    labels=pd.read_parquet(data_dir/"fraud_labels_v1_0.parquet")
    nf=pd.read_parquet(data_dir/"node_features_v1_1.parquet")
    labels=labels[[
        "company_code","fiscal_year","split_v1",label_col
    ]].rename(columns={label_col:"target"})
    nf=nf.copy()
    nf["company_code"]=nf["firm_id"].astype(str).str.zfill(6)
    out=nf.merge(labels,left_on=["company_code","year"],
                 right_on=["company_code","fiscal_year"],
                 how="left",validate="one_to_one")
    if len(out)!=51675:
        raise RuntimeError(f"JOIN_ROWCOUNT:{len(out)}")
    return out

def fit_scaler(df, cols):
    tr=df[df["year"].isin(TRAIN_YEARS) & df["target"].notna()]
    x=tr[cols].fillna(0).to_numpy(np.float32)
    mu=x.mean(0,keepdims=True)
    sd=x.std(0,keepdims=True)
    sd=np.where(sd<1e-6,1.0,sd).astype(np.float32)
    return mu.astype(np.float32),sd

def transformed(df, cols, mu, sd):
    x=df[cols].fillna(0).to_numpy(np.float32)
    return ((x-mu)/sd).astype(np.float32)

def dynamic_pos_weight(df):
    y=df[df["year"].isin(TRAIN_YEARS) & df["target"].notna()]["target"].astype(int)
    pos=int((y==1).sum()); neg=int((y==0).sum())
    if pos<=0: raise RuntimeError("NO_TRAIN_POSITIVES")
    return neg/pos,neg,pos

def f1_val_threshold(y,s):
    p,r,t=precision_recall_curve(y,s)
    if len(t)==0: return 0.5
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return float(t[int(np.nanargmax(f))])

def p_at_budget(y,s,budget=.05):
    n=max(1,int(math.ceil(len(y)*budget)))
    idx=np.argsort(-s)[:n]
    return float(np.mean(np.asarray(y)[idx]))

def r_at_fpr(y,s,max_fpr=.10):
    fpr,tpr,_=roc_curve(y,s)
    ok=np.where(fpr<=max_fpr+1e-12)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0

def metrics(y,s,thr):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    return {
        "roc_auc":float(roc_auc_score(y,s)),
        "ap":float(average_precision_score(y,s)),
        "f1_val_threshold":float(f1_score(y,(s>=thr).astype(int))),
        "p_at_5pct":p_at_budget(y,s,.05),
        "r_at_10fpr":r_at_fpr(y,s,.10),
        "n":int(len(y)),"n_pos":int(y.sum()),"n_neg":int(len(y)-y.sum()),
    }

def make_model(name,in_dim,h,n_rel=5):
    import torch.nn as nn
    import torch.nn.functional as F
    if name=="MLP":
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.f1=nn.Linear(in_dim,h); self.f2=nn.Linear(h,h); self.head=nn.Linear(h,1)
            def forward(self,x,edge_index=None,edge_type=None):
                x=F.relu(self.f1(x)); x=F.dropout(x,.3,training=self.training)
                x=F.relu(self.f2(x)); return self.head(x).squeeze(-1)
        return MLP()
    from torch_geometric.nn import GCNConv,GATConv,SAGEConv,RGCNConv
    if name=="GCN":
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.c1=GCNConv(in_dim,h); self.c2=GCNConv(h,h); self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.relu(self.c1(x,edge_index)); x=F.dropout(x,.3,training=self.training)
                x=F.relu(self.c2(x,edge_index)); return self.head(x).squeeze(-1)
        return M()
    if name=="GAT":
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1=GATConv(in_dim,h,heads=4,concat=True,dropout=.3)
                self.c2=GATConv(h*4,h,heads=1,concat=False,dropout=.3)
                self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.elu(self.c1(x,edge_index)); x=F.elu(self.c2(x,edge_index))
                return self.head(x).squeeze(-1)
        return M()
    if name=="SAGE":
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.c1=SAGEConv(in_dim,h); self.c2=SAGEConv(h,h); self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.relu(self.c1(x,edge_index)); x=F.dropout(x,.3,training=self.training)
                x=F.relu(self.c2(x,edge_index)); return self.head(x).squeeze(-1)
        return M()
    if name=="RGCN":
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1=RGCNConv(in_dim,h,num_relations=n_rel)
                self.c2=RGCNConv(h,h,num_relations=n_rel)
                self.head=nn.Linear(h,1)
            def forward(self,x,edge_index,edge_type=None):
                x=F.relu(self.c1(x,edge_index,edge_type)); x=F.dropout(x,.3,training=self.training)
                x=F.relu(self.c2(x,edge_index,edge_type)); return self.head(x).squeeze(-1)
        return M()
    raise ValueError(name)

def run_mlp(df,modal,seed,args,outdir):
    import torch
    import torch.nn as nn
    set_seed(seed)
    cols=select_cols(df,modal)
    if modal=="G6": raise RuntimeError("MLP_G6_NOT_DEFINED")
    mu,sd=fit_scaler(df,cols)
    X=transformed(df,cols,mu,sd)
    eligible=df["target"].notna().to_numpy()
    years=df["year"].to_numpy()
    yall=df["target"].fillna(0).astype(int).to_numpy()
    tr=np.where(eligible & np.isin(years,TRAIN_YEARS))[0]
    va=np.where(eligible & np.isin(years,VAL_YEARS))[0]
    te=np.where(eligible & np.isin(years,TEST_YEARS))[0]
    device=torch.device(args.device)
    model=make_model("MLP",len(cols),args.hidden_dim).to(device)
    posw,neg,pos=dynamic_pos_weight(df)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw,device=device))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    xt=torch.from_numpy(X[tr]).to(device); yt=torch.from_numpy(yall[tr].astype(np.float32)).to(device)
    xv=torch.from_numpy(X[va]).to(device)
    best_auc=-1.; best_state=None; best_epoch=0; stale=0
    for epoch in range(1,args.epochs+1):
        model.train(); opt.zero_grad()
        loss=crit(model(xt),yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vs=torch.sigmoid(model(xv)).cpu().numpy()
        auc=roc_auc_score(yall[va],vs)
        if auc>best_auc+1e-8:
            best_auc=float(auc); best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale=0
        else:
            stale+=1
        if stale>=args.patience: break
    model.load_state_dict(best_state); model.to(device); model.eval()
    with torch.no_grad():
        vs=torch.sigmoid(model(xv)).cpu().numpy()
    thr=f1_val_threshold(yall[va],vs)
    if args.no_test:
        return {"model":"MLP","modal":modal,"seed":seed,"best_epoch":best_epoch,
                "best_val_auc":best_auc,"val_threshold":thr,"pos_weight":posw,
                "train_neg":neg,"train_pos":pos,"test_evaluated":False}
    xtest=torch.from_numpy(X[te]).to(device)
    with torch.no_grad(): ts=torch.sigmoid(model(xtest)).cpu().numpy()
    met=metrics(yall[te],ts,thr)
    pred=pd.DataFrame({"company_code":df.iloc[te]["company_code"].to_numpy(),
                       "fiscal_year":df.iloc[te]["year"].to_numpy(),
                       "y_true":yall[te],"score":ts})
    pred.to_parquet(outdir/"predictions.parquet",index=False)
    return {"model":"MLP","modal":modal,"seed":seed,"best_epoch":best_epoch,
            "best_val_auc":best_auc,"val_threshold":thr,"pos_weight":posw,
            "train_neg":neg,"train_pos":pos,"test_evaluated":True,"metrics":met}

def prepare_graph_cache(df,modal,data_dir):
    import torch
    cols=select_cols(df,modal)
    if modal=="G6":
        mu=sd=None; d=1
    else:
        mu,sd=fit_scaler(df,cols); d=len(cols)
    mapping=pd.read_parquet(data_dir/"node_id_index_label_v1_0_fiverel_derived.parquet")
    node_to_idx=dict(zip(mapping["node_id"].astype(str),mapping["node_idx"].astype(int)))
    n_total=len(mapping)
    edges=pd.read_parquet(data_dir/"global_edge_index.parquet",
                          columns=["src_idx","dst_idx","edge_type","year"])
    unknown=set(edges["edge_type"].astype(str))-set(REL_MAP)
    if unknown: raise RuntimeError(f"UNKNOWN_EDGE_TYPES:{sorted(unknown)}")
    cache={}
    for year in TRAIN_YEARS+VAL_YEARS+TEST_YEARS:
        sub=df[df["year"]==year].copy()
        idx=np.array([node_to_idx[str(x)] for x in sub["node_id"]],dtype=np.int64)
        if modal=="G6":
            cx=np.ones((len(sub),1),dtype=np.float32)
        else:
            cx=transformed(sub,cols,mu,sd)
        es=edges[edges["year"]==year]
        ei=np.vstack([es["src_idx"].to_numpy(np.int64),es["dst_idx"].to_numpy(np.int64)])
        et=np.array([REL_MAP[str(x)] for x in es["edge_type"]],dtype=np.int64)
        elig=sub["target"].notna().to_numpy()
        sup=np.where(elig)[0]
        cache[year]={
            "company_idx":torch.from_numpy(idx),
            "company_x":torch.from_numpy(cx),
            "edge_index":torch.from_numpy(ei),
            "edge_type":torch.from_numpy(et),
            "sup_idx":torch.from_numpy(idx[sup]),
            "sup_y":torch.from_numpy(sub.iloc[sup]["target"].to_numpy(dtype=np.float32, copy=True)),
            "sup_keys":sub.iloc[sup][["company_code","year"]].reset_index(drop=True),
        }
    return cache,n_total,d

def eval_graph(model,cache,years,n_total,d,device):
    import torch
    ys=[]; ss=[]; keys=[]
    model.eval()
    with torch.no_grad():
        for year in years:
            c=cache[year]
            x=torch.zeros((n_total,d),dtype=torch.float32,device=device)
            x[c["company_idx"].to(device)]=c["company_x"].to(device)
            ei=c["edge_index"].to(device); et=c["edge_type"].to(device)
            logits=model(x,ei,et)
            score=torch.sigmoid(logits[c["sup_idx"].to(device)]).cpu().numpy()
            ys.append(c["sup_y"].numpy()); ss.append(score); keys.append(c["sup_keys"])
            del x,ei,et,logits
    return np.concatenate(ys).astype(int),np.concatenate(ss),pd.concat(keys,ignore_index=True)

def run_gnn(df,model_name,modal,seed,args,outdir,data_dir):
    import torch
    import torch.nn as nn
    set_seed(seed)
    cache,n_total,d=prepare_graph_cache(df,modal,data_dir)
    device=torch.device(args.device)
    model=make_model(model_name,d,args.hidden_dim,5).to(device)
    posw,neg,pos=dynamic_pos_weight(df)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw,device=device))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    best_auc=-1.; best_state=None; best_epoch=0; stale=0
    for epoch in range(1,args.epochs+1):
        model.train()
        for year in TRAIN_YEARS:
            c=cache[year]
            x=torch.zeros((n_total,d),dtype=torch.float32,device=device)
            x[c["company_idx"].to(device)]=c["company_x"].to(device)
            ei=c["edge_index"].to(device); et=c["edge_type"].to(device)
            opt.zero_grad()
            logits=model(x,ei,et)
            loss=crit(logits[c["sup_idx"].to(device)],c["sup_y"].to(device))
            loss.backward(); opt.step()
            del x,ei,et,logits,loss
        vy,vs,_=eval_graph(model,cache,VAL_YEARS,n_total,d,device)
        auc=roc_auc_score(vy,vs)
        if auc>best_auc+1e-8:
            best_auc=float(auc); best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale=0
        else:
            stale+=1
        if stale>=args.patience: break
    model.load_state_dict(best_state); model.to(device)
    vy,vs,_=eval_graph(model,cache,VAL_YEARS,n_total,d,device)
    thr=f1_val_threshold(vy,vs)
    if args.no_test:
        return {"model":model_name,"modal":modal,"seed":seed,"best_epoch":best_epoch,
                "best_val_auc":best_auc,"val_threshold":thr,"pos_weight":posw,
                "train_neg":neg,"train_pos":pos,"test_evaluated":False}
    ty,ts,tkeys=eval_graph(model,cache,TEST_YEARS,n_total,d,device)
    met=metrics(ty,ts,thr)
    pred=tkeys.rename(columns={"year":"fiscal_year"}).copy()
    pred["y_true"]=ty; pred["score"]=ts
    pred.to_parquet(outdir/"predictions.parquet",index=False)
    return {"model":model_name,"modal":modal,"seed":seed,"best_epoch":best_epoch,
            "best_val_auc":best_auc,"val_threshold":thr,"pos_weight":posw,
            "train_neg":neg,"train_pos":pos,"test_evaluated":True,"metrics":met}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--model",choices=["MLP","GCN","GAT","SAGE","RGCN"],required=True)
    ap.add_argument("--modal",choices=["M5","M10","M11","G6"],required=True)
    ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    ap.add_argument("--label-col",default="label_v1_strict_ab_primary",
                    choices=["label_v1_strict_ab_primary",
                             "label_v1_strict_aonly_sensitivity",
                             "label_v1_loose_ab_sensitivity"])
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--hidden-dim",type=int,default=64)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--no-test",action="store_true")
    args=ap.parse_args()
    if args.model=="MLP" and args.modal=="G6":
        raise SystemExit("MLP has no G6 topology-only condition")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed)
    df=load_joined(args.data_dir,args.label_col)
    pw,neg,pos=dynamic_pos_weight(df)
    if (neg,pos)!=(23852,254) and args.label_col=="label_v1_strict_ab_primary":
        raise RuntimeError(f"PRIMARY_TRAIN_COUNTS_MISMATCH neg={neg} pos={pos}")
    started=time.time()
    if args.model=="MLP":
        res=run_mlp(df,args.modal,args.seed,args,args.out_dir)
    else:
        res=run_gnn(df,args.model,args.modal,args.seed,args,args.out_dir,args.data_dir)
    res.update({
        "label_col":args.label_col,"epochs_max":args.epochs,"patience":args.patience,
        "hidden_dim":args.hidden_dim,"lr":args.lr,"device":args.device,
        "runtime_seconds":time.time()-started,
        "protocol":{
            "train_years":TRAIN_YEARS,"val_years":VAL_YEARS,"test_years":TEST_YEARS,
            "scaler_fit":"eligible training rows only; missing->0 before mean/std",
            "graph_edges":"edge year == target fiscal year",
            "sage":"full-neighborhood full-batch SAGEConv; no neighbor sampling",
            "gat_heads":"4 then 1",
            "checkpoint":"max validation ROC-AUC; patience 10; test not used",
            "f1_threshold":"selected on validation only, frozen for test",
            "pos_weight":"dynamic Nneg/Npos from eligible training risk set",
        }
    })
    (args.out_dir/"result.json").write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
    print(json.dumps(res,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
