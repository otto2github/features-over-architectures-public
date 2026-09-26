"""Paired re-evaluation of saved predictions. No fit(), checkpoint load, or torch import."""
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import math
import numpy as np
import pandas as pd
from .common import *
from .metrics import Plan,NAMES

MODELS=['GCN','GAT','SAGE','RGCN']

def canonical(ctx):
    f=load_labels(ctx)
    f=f[(f.partition=='test')&f.target.notna()].sort_values(['company_code','fiscal_year']).reset_index(drop=True)
    require(f.company_code.nunique()==ctx.p['expected_test_companies'],'TEST_CLUSTER_COUNT_MISMATCH')
    return f

def read_prediction(ctx,p,role,expected_sha,canonical_df):
    pp=ctx.use(p,role,expected_sha)
    d=pd.read_parquet(pp,columns=['company_code','fiscal_year','y_true','score'])
    d['company_code']=company_codes(d.company_code)
    yr=pd.to_numeric(d.fiscal_year,errors='coerce')
    require(yr.notna().all() and np.equal(yr,yr.astype(int)).all(),'PREDICTION_YEAR_FORMAT')
    d['fiscal_year']=yr.astype(int)
    require(not d.duplicated(['company_code','fiscal_year']).any(),'DUPLICATE_PREDICTION_KEYS')
    d=d.sort_values(['company_code','fiscal_year']).reset_index(drop=True)
    require(len(d)==len(canonical_df),'PREDICTION_ROW_COUNT')
    require(d[['company_code','fiscal_year']].equals(canonical_df[['company_code','fiscal_year']]),'PREDICTION_KEYS_NOT_IDENTICAL')
    require(np.array_equal(d.y_true.to_numpy(),canonical_df.target.to_numpy()),'PREDICTION_LABEL_MISMATCH')
    s=d.score.to_numpy(float)
    require(np.isfinite(s).all() and (s>=0).all() and (s<=1).all(),'INVALID_SAVED_SCORES')
    return s

def validate_run(ctx,j,model,modal,seed):
    require(j.get('model')==model and j.get('modal')==modal and int(j.get('seed',-1))==seed,'RUN_IDENTITY_MISMATCH')
    require(j.get('label_col')==ctx.p['primary_label'],'RUN_LABEL_PROTOCOL_MISMATCH')
    require(j.get('test_evaluated') is True,'RUN_HAS_NO_FINAL_TEST_EVALUATION')
    require(math.isfinite(float(j.get('val_threshold',float('nan')))),'NO_FIXED_VALIDATION_THRESHOLD')
    require(int(j.get('train_pos',-1))==ctx.p['expected_counts']['train'][1],'RUN_TRAIN_POSITIVES_MISMATCH')

def load_runs(ctx,f):
    ref=read_json(ctx.package/'reference/PREDICTION_IDENTITY.json')
    rows=[]
    # Exact paths and hashes from v16's archived direction-reference manifest.
    for path,rec in sorted(ref['original'].items()):
        if not path.endswith('/result.json'):continue
        p=ctx.project/path
        ctx.use(p,'stored_result_'+p.parent.name,rec['sha256']);j=read_json(p)
        model,modal,seed=j['model'],j['modal'],int(j['seed'])
        validate_run(ctx,j,model,modal,seed)
        predrel=str(Path(path).parent/'predictions.parquet')
        s=read_prediction(ctx,ctx.project/predrel,'stored_prediction_'+p.parent.name,ref['original'][predrel]['sha256'],f)
        rows.append((f'STORED_{model}_{modal}',seed,j,s))
    # Do not choose the newest attempt. Require v16's exact retained completed attempt.
    reverse_root=Path(ctx.args.reverse_root).expanduser().resolve() if ctx.args.reverse_root else ctx.project/ctx.p['reverse_relative']
    for runid,rec in sorted(ref['reverse'].items()):
        runroot=reverse_root/'runs'/runid
        done=ctx.use(runroot/'DONE.json','reverse_completion_'+runid)
        dj=read_json(done)
        require(dj.get('attempt')==rec['attempt'],'REVERSE_ATTEMPT_DIFFERS_FROM_V16')
        require(Path(rec['attempt']).name==rec['attempt'],'UNSAFE_REVERSE_ATTEMPT')
        attempt=runroot/rec['attempt'];p=attempt/'result.json'
        ctx.use(p,'reverse_result_'+runid,rec['files']['result.json']['sha256']);j=read_json(p)
        model,modal,seed=j['model'],j['modal'],int(j['seed'])
        validate_run(ctx,j,model,modal,seed)
        require(j.get('direction')=='append_reverse_same_relation_preserve_multiplicity','REVERSE_DIRECTION_RULE_MISMATCH')
        s=read_prediction(ctx,attempt/'predictions.parquet','reverse_prediction_'+runid,rec['files']['predictions.parquet']['sha256'],f)
        rows.append((f'REVERSE_{model}_{modal}',seed,j,s))
    # Same retained primary unweighted tabular family. Other output families are never promoted.
    base=ctx.project/'results/tabular_primary_v1'
    for model in ['Lasso','Ridge','RandomForest','XGBoost']:
        for modal in ['M5','M10','M11']:
            for seed in ctx.p['seeds']:
                runid=f'tabular_{model}_{modal}_seed{seed}';p=base/runid/'result.json'
                ctx.use(p,'tabular_result_'+runid);j=read_json(p)
                validate_run(ctx,j,model,modal,seed)
                require(j.get('class_weight_mode')=='none','TABULAR_IS_NOT_PRIMARY_UNWEIGHTED')
                if model=='RandomForest':require(j.get('model_config',{}).get('class_weight') is None,'RF_WEIGHT_STATUS')
                s=read_prediction(ctx,p.parent/'predictions.parquet','tabular_prediction_'+runid,None,f)
                rows.append((f'TABUW_{model}_{modal}',seed,j,s))
    require(len({(c,seed) for c,seed,j,s in rows})==len(rows),'DUPLICATE_CONDITION_SEED')
    return rows

_WORK={}
def _init(plans,cluster_idx,nclusters,y,group_indices,seedcount,company_seed,fit_seed):
    global _WORK
    _WORK=dict(plans=plans,cluster_idx=cluster_idx,nclusters=nclusters,y=y,group_indices=group_indices,seedcount=seedcount,company_seed=company_seed,fit_seed=fit_seed)

def _rep(b):
    g=_WORK;n=g['nclusters'];rng=np.random.default_rng(g['company_seed']+b*10);redrawn=0
    while True:
        counts=np.bincount(rng.integers(0,n,size=n),minlength=n)
        w=counts[g['cluster_idx']].astype(float)
        pos=float(np.dot(w,g['y']));neg=float(w.sum()-pos)
        if pos>0 and neg>0:break
        redrawn+=1
        require(redrawn<10000,'TOO_MANY_SINGLE_CLASS_RESAMPLES')
    fit=np.array([p.compute(w) for p in g['plans']])
    by_condition=np.array([fit[ii] for ii in g['group_indices']]) # condition, seed, metric
    conditional=by_condition.mean(axis=1)
    # Crossed company x matched seed-label resampling. Same seed draw in every condition.
    srng=np.random.default_rng(g['fit_seed']+b*10)
    seed_draw=srng.integers(0,g['seedcount'],size=g['seedcount'])
    seed_company=by_condition[:,seed_draw,:].mean(axis=1)
    return b,conditional,seed_company,redrawn

def bootstrap(plans,cluster_idx,y,groups,nboot,workers,company_seed,fit_seed):
    nclusters=int(cluster_idx.max())+1
    args=(plans,cluster_idx,nclusters,y,groups,len(groups[0]),company_seed,fit_seed)
    result=[]
    if workers==1:
        _init(*args)
        for b in range(nboot):
            result.append(_rep(b))
            if (b+1)%100==0:print(f'  bootstrap {b+1}/{nboot}',flush=True)
    else:
        # Spawn is supported on both macOS and Linux; never fork a CUDA runtime.
        with mp.get_context('spawn').Pool(workers,initializer=_init,initargs=args) as pool:
            for row in pool.imap(_rep,range(nboot),chunksize=5):
                result.append(row)
                if len(result)%100==0:print(f'  bootstrap {len(result)}/{nboot}',flush=True)
    result.sort(key=lambda x:x[0])
    return np.stack([x[1] for x in result]),np.stack([x[2] for x in result]),[x[3] for x in result]

def prediction_audit(ctx):
    f=canonical(ctx);y=f.target.to_numpy(int);runs=load_runs(ctx,f)
    ones=np.ones(len(f));plans=[Plan(y,s,j['val_threshold']) for c,seed,j,s in runs]
    point=np.array([p.compute(ones) for p in plans])
    records=[];ties=[]
    expected=pd.read_csv(ctx.package/'reference/source_v13_per_seed_unified_metrics.csv').set_index(['condition','seed'])
    for i,(c,seed,j,s) in enumerate(runs):
        p=plans[i];m=point[i]
        # Check the saved values without overriding old P@5% ties.
        for k,name in enumerate(NAMES):
            old_key='f1' if name=='f1_val_threshold' and 'f1' in j.get('metrics',{}) else name
            old=j.get('metrics',{}).get(old_key)
            if name!='p_at_5pct':
                require(old is not None and abs(m[k]-float(old))<1e-8,'SAVED_METRIC_RECONCILIATION:'+c+':'+name)
        if c.startswith(('STORED_','REVERSE_')):
            ec=c.replace('STORED_','ORIGINAL_',1)
            require((ec,seed) in expected.index,'V16_SEED_REFERENCE_MISSING')
            for k,name in enumerate(NAMES):
                require(abs(m[k]-float(expected.loc[(ec,seed),name]))<1e-8,'V16_REFERENCE_POINT_MISMATCH:'+c+':'+name)
        row={'condition':c,'seed':seed,**{name:float(m[k]) for k,name in enumerate(NAMES)}}
        records.append(row)
        tie={'condition':c,'seed':seed,**p.tie_report(),'p5_unified':float(m[3]),'p5_archived':j.get('metrics',{}).get('p_at_5pct')}
        ties.append(tie)
    published=pd.read_csv(ctx.package/'reference/source_v3_observed_condition_metrics.csv')
    ptab=pd.DataFrame(records)
    for c,g in ptab[ptab.condition.str.startswith('TABUW_')].groupby('condition'):
        ref=published[(published.condition==c)&(published.label==ctx.p['primary_label'])]
        require(len(ref)==1,'TABULAR_PUBLISHED_REFERENCE_MISSING')
        require(abs(g.roc_auc.mean()-float(ref.roc_auc_mean.iloc[0]))<1e-8,'TABULAR_MEAN_DIFFERS_FROM_V16')
        require(abs(g.ap.mean()-float(ref.ap_mean.iloc[0]))<1e-8,'TABULAR_AP_DIFFERS_FROM_V16')
    ctx.export_csv('20_saved_prediction_point_checks.csv',records)
    ctx.export_csv('21_unified_top5pct_tie_audit.csv',ties)
    # Only M11 contrasts requested for the first-stage inferential postprocessing.
    conditions=['STORED_MLP_M11','TABUW_RandomForest_M11']+[f'STORED_{m}_M11' for m in MODELS]+[f'REVERSE_{m}_M11' for m in MODELS]
    selected=[];selected_runs=[];groups=[]
    for c in conditions:
        indices=[]
        for seed in ctx.p['seeds']:
            found=[i for i,(cc,ss,j,s) in enumerate(runs) if cc==c and ss==seed]
            require(len(found)==1,'M11_CONDITION_SEED_NOT_UNIQUE')
            indices.append(len(selected));selected.append(plans[found[0]]);selected_runs.append(runs[found[0]])
        groups.append(indices)
    observed=np.array([p.compute(ones) for p in selected]).reshape(len(conditions),len(ctx.p['seeds']),len(NAMES))
    _,codes=np.unique(f.company_code.to_numpy(),return_inverse=True)
    cc,cs,rejections=bootstrap(selected,codes,y,groups,ctx.args.bootstrap,ctx.args.workers,ctx.p['bootstrap_company_seed'],ctx.p['bootstrap_fit_seed'])
    contrast=[('TABUW_RandomForest_M11',x) for x in ['STORED_MLP_M11','REVERSE_GCN_M11','REVERSE_SAGE_M11']]
    contrast += [(f'REVERSE_{m}_M11',f'STORED_{m}_M11') for m in MODELS]
    contrast += [('STORED_MLP_M11',f'REVERSE_{m}_M11') for m in MODELS]
    contrast += [('STORED_MLP_M11',f'STORED_{m}_M11') for m in MODELS]
    rows=[];condition_rows=[]
    for scheme,arr in [('company_only_fixed_five_fits',cc),('company_x_matched_seed',cs)]:
        for c in conditions:
            i=conditions.index(c)
            lo,hi=np.quantile(arr[:,i,:],[.025,.975],axis=0)
            for k,name in enumerate(NAMES):
                condition_rows.append({'condition':c,'resampling_scheme':scheme,'metric':name,'mean_of_five_fit_metrics':observed[i,:,k].mean(),'seed_sd':observed[i,:,k].std(ddof=1),'ci_low':lo[k],'ci_high':hi[k],'replicates':len(arr)})
        for left,right in contrast:
            a,b=conditions.index(left),conditions.index(right)
            delta=arr[:,a,:]-arr[:,b,:];d=observed[a,:,:]-observed[b,:,:]
            lo,hi=np.quantile(delta,[.025,.975],axis=0)
            for k,name in enumerate(NAMES):
                rows.append({'left_condition':left,'right_condition':right,'delta_definition':'left_minus_right','resampling_scheme':scheme,'metric':name,'observed_mean_paired_delta':d[:,k].mean(),'seed_paired_delta_sd':d[:,k].std(ddof=1),'ci_low':lo[k],'ci_high':hi[k],'n_matched_seeds':len(ctx.p['seeds']),'n_company_clusters':int(codes.max())+1,'n_test_rows':len(y),'n_test_positives':int(y.sum()),'bootstrap_replicates':len(delta)})
    ctx.export_csv('22_M11_paired_bootstrap_comparisons.csv',rows)
    ctx.export_csv('23_M11_bootstrap_condition_intervals.csv',condition_rows)
    ctx.export_json('24_bootstrap_design_and_diagnostics.json',{'bootstrap_replicates':ctx.args.bootstrap,'company_rng_seed':ctx.p['bootstrap_company_seed'],'fit_rng_seed':ctx.p['bootstrap_fit_seed'],'replicate_rng_rule':'seed + replicate_index * 10','single_class_rejected_draws':int(sum(rejections)),'replicates_requiring_redraw':int(np.count_nonzero(rejections)),'max_redraws_for_one_replicate':int(max(rejections)),'no_failed_draws_silently_counted':True,'same_company_draw_for_every_condition_and_seed':True,'same_seed_label_draw_for_every_condition':True,'resampling_design':'crossed company x seed-label; companies are not nested independently inside seeds','estimand':'difference in arithmetic means of per-fit metrics, not metric of an ensemble-mean prediction','interval':'percentile 2.5 and 97.5; marginal, exploratory','captures':'variation across the five retained fitted seeds and sampled test companies','does_not_capture':['new training samples','hyperparameter search','new test windows','unseen seed population guarantees','graph/case dependence beyond companies','full repeated-study uncertainty'],'bootstrap_tail_fraction_not_reported_as_hypothesis_test_p_value':True,'new_training':False,'all_saved_input_files_preserved':True})
    ctx.record('saved_prediction_postprocessing','PASS',note='Point estimates reconciled with v16; paired company and crossed company×seed intervals computed without training.')

def history_baseline(ctx):
    f=load_labels(ctx);tr=f[(f.partition=='train')&f.target.notna()]
    mean=tr.groupby('company_code').target.mean()
    rows=[]
    for part in ['validation','test']:
        g=f[(f.partition==part)&f.target.notna()].sort_values(['company_code','fiscal_year'])
        y=g.target.to_numpy(int);s=g.company_code.map(mean).fillna(0).to_numpy(float)
        m=Plan(y,s,.5).compute(np.ones(len(g)))
        rows.append({'partition':part,'rows':len(g),'positive_rows':int(y.sum()),'companies_without_eligible_training_history':g.loc[~g.company_code.isin(mean.index),'company_code'].nunique(),'roc_auc':m[0],'ap':m[1],'p_at_5pct':m[3],'r_at_10fpr':m[4],'score_definition':'eligible 2010-2018 training label mean per company; unknown=0','scope':'retrospective training-label-memory diagnostic, NOT an as-of or deployable baseline'})
    ctx.export_csv('25_history_mean_no_propagation_DIAGNOSTIC.csv',rows)
    ctx.record('history_no_propagation','PASS',note='No model fit; retrospective fixed training-label means only, not claimed historically available.')
