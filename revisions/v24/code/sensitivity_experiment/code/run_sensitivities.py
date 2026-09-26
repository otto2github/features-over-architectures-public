#!/usr/bin/env python3
from __future__ import annotations

# Must be set before torch/CUDA is imported or initialized.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

SEEDS=[42,123,456,789,1024]

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ccode(s):return s.astype('string').str.extract(r'(\d{6})',expand=False).str.zfill(6)

def load_joined(labels_path,nf_path,label_col):
    labels=pd.read_parquet(labels_path);nf=pd.read_parquet(nf_path)
    if label_col not in labels:raise RuntimeError('LABEL_COLUMN_MISSING:'+label_col)
    labels=labels[['company_code','fiscal_year',label_col]].copy();labels['company_code']=ccode(labels.company_code);labels['fiscal_year']=labels.fiscal_year.astype(int);labels=labels.rename(columns={label_col:'target'})
    nf=nf.copy();nf['company_code']=ccode(nf.firm_id);nf['fiscal_year']=nf.year.astype(int)
    out=nf.merge(labels,on=['company_code','fiscal_year'],how='left',validate='one_to_one')
    if len(out)!=51675:raise RuntimeError('JOIN_ROWCOUNT')
    return out

def append_reverse(ei,et):
    import torch
    return torch.cat((ei,ei.flip(0)),dim=1),torch.cat((et,et),dim=0)

def reverse_adapter(core):
    original=core.prepare_graph_cache
    def prepare(df,modal,data_dir):
        cache,n,d=original(df,modal,data_dir)
        for c in cache.values():c['edge_index'],c['edge_type']=append_reverse(c['edge_index'],c['edge_type'])
        return cache,n,d
    return prepare

def run_rf(core,df,modal,seed,n_jobs,outdir):
    cols=core.select_cols(df,modal)
    X=df[cols].fillna(0).to_numpy(np.float32)
    eligible=df.target.notna().to_numpy();years=df.year.to_numpy();yall=df.target.fillna(0).astype(int).to_numpy()
    tr=np.where(eligible & np.isin(years,core.TRAIN_YEARS))[0];va=np.where(eligible & np.isin(years,core.VAL_YEARS))[0];te=np.where(eligible & np.isin(years,core.TEST_YEARS))[0]
    model=RandomForestClassifier(n_estimators=500,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,class_weight=None,n_jobs=n_jobs,random_state=seed)
    started=time.time();model.fit(X[tr],yall[tr]);vs=model.predict_proba(X[va])[:,1];thr=core.f1_val_threshold(yall[va],vs);ts=model.predict_proba(X[te])[:,1];met=core.metrics(yall[te],ts,thr)
    pred=pd.DataFrame({'company_code':df.iloc[te].company_code.to_numpy(),'fiscal_year':df.iloc[te].year.to_numpy(),'y_true':yall[te],'score':ts});pred.to_parquet(outdir/'predictions.parquet',index=False)
    return {'model':'RandomForest','modal':modal,'seed':seed,'best_val_auc':float(roc_auc_score(yall[va],vs)),'val_threshold':thr,'train_neg':int((yall[tr]==0).sum()),'train_pos':int((yall[tr]==1).sum()),'test_evaluated':True,'metrics':met,'runtime_seconds':time.time()-started,'config':{'n_estimators':500,'max_features':'sqrt','class_weight':None,'n_jobs':n_jobs}}

def run_one(core,scenario,model,modal,seed,df,orig_data,outdir,n_jobs):
    outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'result.json').is_file():return json.loads((outdir/'result.json').read_text())
    args=SimpleNamespace(device='cuda',hidden_dim=64,epochs=100,patience=10,lr=5e-4,no_test=False)
    started=time.time()
    if model=='RandomForest':res=run_rf(core,df,modal,seed,n_jobs,outdir)
    elif model=='MLP':
        core.set_seed(seed);res=core.run_mlp(df,modal,seed,args,outdir);res['runtime_seconds']=time.time()-started
    elif model in {'GCN','SAGE'}:
        core.set_seed(seed);orig=core.prepare_graph_cache;core.prepare_graph_cache=reverse_adapter(core)
        try:res=core.run_gnn(df,model,modal,seed,args,outdir,orig_data)
        finally:core.prepare_graph_cache=orig
        res['runtime_seconds']=time.time()-started;res['direction']='append_reverse_same_relation_preserve_multiplicity'
    else:raise ValueError(model)
    res.update({'scenario':scenario,'model_display':'GraphSAGE' if model=='SAGE' else model,'label_counts':{p:{'n':int(len(g)),'positive':int(g.target.sum())} for p,g in [('train',df[df.year.isin(core.TRAIN_YEARS)&df.target.notna()]),('validation',df[df.year.isin(core.VAL_YEARS)&df.target.notna()]),('test',df[df.year.isin(core.TEST_YEARS)&df.target.notna()])]}})
    (outdir/'result.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    return res

def aggregate(results,outroot):
    rows=[]
    for r in results:
        m=r['metrics'];rows.append({'scenario':r['scenario'],'model':r['model_display'],'modal':r['modal'],'seed':r['seed'],'roc_auc':m['roc_auc'],'ap':m['ap'],'f1_val_threshold':m['f1_val_threshold'],'p_at_5pct':m['p_at_5pct'],'r_at_10fpr':m['r_at_10fpr'],'n':m['n'],'n_pos':m['n_pos'],'best_val_auc':r['best_val_auc'],'runtime_seconds':r['runtime_seconds']})
    df=pd.DataFrame(rows).sort_values(['scenario','model','modal','seed']);df.to_csv(outroot/'per_seed_metrics.csv',index=False)
    summ=df.groupby(['scenario','model','modal'],as_index=False).agg(roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),ap_mean=('ap','mean'),ap_sd=('ap','std'),f1_mean=('f1_val_threshold','mean'),p5_mean=('p_at_5pct','mean'),r10_mean=('r_at_10fpr','mean'),seeds=('seed','count'))
    summ.to_csv(outroot/'condition_summary.csv',index=False)
    return df,summ

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-root',type=Path,required=True);ap.add_argument('--prep-root',type=Path,required=True);ap.add_argument('--package-root',type=Path,required=True);ap.add_argument('--out-root',type=Path,required=True);ap.add_argument('--n-jobs',type=int,default=8)
    args=ap.parse_args();P=json.loads((args.package_root/'config/protocol.json').read_text())
    core_path=args.package_root/'code/formal_rerun_core.py'
    if sha256(core_path)!=P['frozen_hashes']['formal_rerun_core.py']:raise RuntimeError('CORE_HASH_MISMATCH')
    sys.path.insert(0,str(args.package_root/'code'));import formal_rerun_core as core

    # Strict deterministic guard.  The frozen core uses warn_only=True for
    # compatibility; the sensitivity orchestrator strengthens that policy
    # without modifying the frozen training/model code or its pinned hash.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError(
            'CUBLAS_WORKSPACE_CONFIG_NOT_STRICT:'+
            repr(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        )
    import torch
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=False)

    frozen_set_seed=core.set_seed
    def strict_set_seed(seed):
        frozen_set_seed(seed)
        # The frozen core requests warn_only=True; restore strict mode
        # immediately after each seed reset so unsupported nondeterministic
        # CUDA operations fail instead of merely warning.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True, warn_only=False)
    core.set_seed=strict_set_seed

    data=args.original_root/'data'
    for name in ['fraud_labels_v1_0.parquet','node_features_v1_1.parquet','global_edge_index.parquet','node_id_index_label_v1_0_fiverel_derived.parquet']:
        p=data/name;exp=P['frozen_hashes'][name]
        if not p.is_file() or sha256(p)!=exp:raise RuntimeError('ORIGINAL_DATA_HASH_MISMATCH:'+name)
    asof_nf=args.prep_root/'asof/node_features_asof_financial_v1.parquet';fixed_labels=args.prep_root/'fixed1095/fraud_labels_fixed1095_v1.parquet'
    if not asof_nf.is_file() or not fixed_labels.is_file():raise RuntimeError('PREP_INPUT_MISSING')

    # Bind the sensitivity runs to the exact inputs approved after the
    # Mac/Tencent preparation audits, not merely to whatever files happen
    # to be present under prep-root.
    approved=P.get('approved_sensitivity_input_hashes',{})
    exp_asof=approved.get('node_features_asof_financial_v1.parquet')
    exp_fixed=approved.get('fraud_labels_fixed1095_v1.parquet')
    got_asof=sha256(asof_nf); got_fixed=sha256(fixed_labels)
    if not exp_asof or got_asof!=exp_asof:
        raise RuntimeError(f'ASOF_APPROVED_HASH_MISMATCH:{got_asof}!={exp_asof}')
    if not exp_fixed or got_fixed!=exp_fixed:
        raise RuntimeError(f'FIXED_APPROVED_HASH_MISMATCH:{got_fixed}!={exp_fixed}')

    # Use prep manifests to bind exact transferred bytes.
    prep_manifest=args.prep_root/'PREP_TRANSFER_MANIFEST.json'
    if not prep_manifest.is_file():raise RuntimeError('PREP_TRANSFER_MANIFEST_MISSING')
    pm=json.loads(prep_manifest.read_text())
    if sha256(asof_nf)!=pm['asof_node_features_sha256']:raise RuntimeError('ASOF_TRANSFER_HASH_MISMATCH')
    if sha256(fixed_labels)!=pm['fixed1095_labels_sha256']:raise RuntimeError('FIXED_TRANSFER_HASH_MISMATCH')

    original_labels=data/'fraud_labels_v1_0.parquet';original_nf=data/'node_features_v1_1.parquet'
    df_asof=load_joined(original_labels,asof_nf,P['primary_label'])
    df_fixed=load_joined(fixed_labels,original_nf,P['fixed_label_column'])
    for df,key in [(df_asof,'expected_common_cutoff_counts'),(df_fixed,'expected_fixed1095_counts')]:
        for part,years in [('train',core.TRAIN_YEARS),('validation',core.VAL_YEARS),('test',core.TEST_YEARS)]:
            g=df[df.year.isin(years)&df.target.notna()];got=(len(g),int(g.target.sum()));exp=tuple(P[key][part])
            if got!=exp:raise RuntimeError(f'COUNT_MISMATCH:{key}:{part}:{got}!={exp}')

    # Fail before the first fit if GPU or strict CuBLAS determinism is unusable.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_NOT_AVAILABLE')
    print('CUDA_DEVICE=',torch.cuda.get_device_name(0),flush=True)
    print('CUBLAS_WORKSPACE_CONFIG=',os.environ.get('CUBLAS_WORKSPACE_CONFIG'),flush=True)
    print('DETERMINISTIC_ALGORITHMS=',torch.are_deterministic_algorithms_enabled(),flush=True)
    print('DETERMINISTIC_WARN_ONLY=',torch.is_deterministic_algorithms_warn_only_enabled(),flush=True)

    # Exact CUDA linear forward/backward preflight. With strict mode and the
    # CuBLAS workspace config missing/ineffective, this fails before any fit.
    core.set_seed(42)
    a=torch.randn((64,32),device='cuda',requires_grad=True)
    lin=torch.nn.Linear(32,8).to('cuda')
    loss=lin(a).square().mean()
    loss.backward()
    del a,lin,loss
    torch.cuda.synchronize()
    print('STRICT_CUBLAS_PREFLIGHT_PASS',flush=True)

    args.out_root.mkdir(parents=True,exist_ok=True);results=[];matrix=[]
    for seed in SEEDS:
        for modal in ['M5','M11']:
            for model in ['MLP','RandomForest']:matrix.append(('asof',model,modal,seed,df_asof))
        for model in ['MLP','RandomForest','GCN','SAGE']:matrix.append(('fixed1095',model,'M11',seed,df_fixed))
    print('AUTHORIZED_FITS=',len(matrix),flush=True)
    if len(matrix)!=40:raise RuntimeError('RUN_MATRIX_NOT_40')
    for i,(sc,model,modal,seed,df) in enumerate(matrix,1):
        runid=f'{sc}__{model}__{modal}__seed{seed}';print(f'[{i}/40] START {runid}',flush=True)
        r=run_one(core,sc,model,modal,seed,df,data,args.out_root/'runs'/runid,args.n_jobs);results.append(r)
        print(f'[{i}/40] DONE {runid} auc={r["metrics"]["roc_auc"]:.6f}',flush=True)
    per,summ=aggregate(results,args.out_root)
    manifest={
        'status':'PASS',
        'fits':len(results),
        'package_version':P.get('package_version'),
        'determinism':{
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch_deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),
            'warn_only':bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
            'strict_cublas_preflight':'PASS'
        },
        'protocol_sha256':sha256(args.package_root/'config/protocol.json'),
        'runner_sha256':sha256(Path(__file__)),
        'original_data_hashes':{
            n:sha256(data/n) for n in P['frozen_hashes']
            if n in [
                'fraud_labels_v1_0.parquet',
                'node_features_v1_1.parquet',
                'global_edge_index.parquet',
                'node_id_index_label_v1_0_fiverel_derived.parquet'
            ]
        },
        'asof_node_features_sha256':sha256(asof_nf),
        'fixed1095_labels_sha256':sha256(fixed_labels),
        'core_sha256':sha256(core_path),
        'training_matrix':P['training_matrix'],
    }
    (args.out_root/'sensitivity_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nSENSITIVITY_40_FITS_PASS')
    print(summ.to_string(index=False))

if __name__=='__main__':main()
