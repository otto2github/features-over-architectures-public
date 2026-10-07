#!/usr/bin/env python3
"""Finish the ALREADY DECLARED V28 fixed-score analyses. Never import model code.

Frozen prediction files are inputs, not outputs. All output goes to one new
attempt under fixedscore_completion_v1. No checkpoints are loaded or modified.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, traceback, zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fs_math import MetricPlan, METRICS, SEEDS, bootstrap_draw, require, ContractError

HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'EXECUTION_SPEC.json').read_text())
ROOT=Path(SPEC['extension_root']);OLD=Path(SPEC['v27_root_read_only'])
BASE=Path(SPEC['write_root']);KEYS=['company_code','fiscal_year']
POPS=['full_test','D_frozen','O_frozen','S_frozen','J_frozen']
POP_COUNTS={'full_test':(8435,20),'D_frozen':(8422,7),'O_frozen':(8428,13),'S_frozen':(7757,6),'J_frozen':(7753,2)}
MODELS=SPEC['conditions']
CONTRASTS=[('MLP_minus_GCN',0,2),('MLP_minus_GraphSAGE',0,3),('RF_minus_GCN',1,2),('RF_minus_GraphSAGE',1,3)]

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe_output(p,base=BASE):
    p=Path(p);actual=p.resolve();allowed=Path(base).resolve()
    require(str(allowed).startswith('/path/to/private_workspace/'),'OUTPUT_ROOT_OUTSIDE_CYQ')
    require(actual==allowed or allowed in actual.parents,'WRITE_OUTSIDE_OUTPUT_ROOT')
    return p

def finite_json(v):
    if isinstance(v,np.generic):return finite_json(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float):return v if math.isfinite(v) else None
    if isinstance(v,dict):return {str(k):finite_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [finite_json(x) for x in v]
    if v is None or isinstance(v,(str,int,bool)):return v
    raise TypeError('NON_AGGREGATE_JSON_TYPE:'+type(v).__name__)

def writej(p,obj):
    p=safe_output(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(finite_json(obj),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:f.write(b)
    os.replace(tmp,p)

def writecsv(p,rows):
    p=safe_output(p);p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows([finite_json(r) for r in rows])

def strict_keys(d):
    d=d.copy();require(set(KEYS)<=set(d),'KEY_COLUMNS')
    cc=d.company_code.astype('string').str.strip()
    cc=cc.str.replace(r"^'",'',regex=True).str.replace(r'^C:','',regex=True).str.replace(r'\.(SZ|SH|BJ)$','',regex=True).str.replace(r'\.0$','',regex=True).str.zfill(6)
    require(cc.notna().all() and cc.str.fullmatch(r'\d{6}').all(),'COMPANY_DOMAIN')
    yy=pd.to_numeric(d.fiscal_year,errors='raise')
    require(yy.notna().all() and np.equal(yy,np.floor(yy)).all(),'YEAR_DOMAIN')
    d['company_code']=cc.astype(str);d['fiscal_year']=yy.astype(np.int64)
    require(not d.duplicated(KEYS).any(),'DUPLICATE_KEYS')
    return d

def binary(s,code):
    require(s.notna().all(),'MISSING_BINARY:'+code)
    require(s.isin([0,1,False,True]).all(),'BINARY_DOMAIN:'+code)
    return s.astype(np.int8).to_numpy()

def aligned(source,canonical,columns,label):
    source=strict_keys(source);require(len(source)==len(canonical),'ROWS:'+label)
    result=canonical[KEYS].merge(source[KEYS+columns],on=KEYS,how='left',sort=False,validate='one_to_one',indicator=True)
    require(result['_merge'].eq('both').all(),'KEY_SET:'+label)
    require(result[KEYS].equals(canonical[KEYS]),'LEFT_ORDER:'+label)
    return result.drop(columns='_merge')

class InputLedger:
    def __init__(self):self.rows={}
    def check(self,p,rec,role):
        p=Path(p);require(p.is_file(),'MISSING_INPUT:'+role)
        stat=p.stat();h=sha256(p)
        size=rec.get('size_bytes',rec.get('bytes'))
        require(h==rec['sha256'],'INPUT_HASH:'+role)
        if size is not None:require(stat.st_size==int(size),'INPUT_SIZE:'+role)
        self.rows[str(p)]=dict(path=str(p),role=role,sha256=h,size_bytes=stat.st_size)
        return p
    def check_relative(self,p,rec,role):
        require(Path(p).name not in ('','..'),'ARTIFACT_NAME:'+role)
        return self.check(p,rec,role)
    def json(self,p,rec,role):return json.loads(self.check(p,rec,role).read_text())
    def recheck(self):
        for r in list(self.rows.values()):
            require(Path(r['path']).stat().st_size==r['size_bytes'] and sha256(r['path'])==r['sha256'],'INPUT_CHANGED_DURING_POSTPROCESS:'+r['role'])
    def aggregate(self):
        # Relative project paths, no company/document identifiers.
        out=[]
        for r in self.rows.values():
            p=Path(r['path'])
            try:rel='V28/'+str(p.relative_to(ROOT))
            except ValueError:
                try:rel='V27/'+str(p.relative_to(OLD))
                except ValueError:rel=r['role']
            out.append(dict(role=r['role'],relative_path=rel,sha256=r['sha256'],size_bytes=r['size_bytes']))
        return sorted(out,key=lambda x:(x['role'],x['relative_path']))

def read_predictions(p,canonical,label):
    d=pd.read_parquet(p,columns=KEYS+['y_true','score'])
    d=aligned(d,canonical,['y_true','score'],label)
    y=binary(d.y_true,label)
    require(np.array_equal(y,canonical.y_true.to_numpy()),'LABEL_IDENTITY:'+label)
    s=pd.to_numeric(d.score,errors='raise').to_numpy(np.float64)
    require(np.isfinite(s).all() and (s>=0).all() and (s<=1).all(),'SCORE_DOMAIN:'+label)
    return s

def verify_tstage(ledger,root,rid,tlock):
    rd=root/'stage_t'/rid
    if 'artifacts' in tlock:
        entries=tlock['artifacts'][rid]
        for fn,rec in entries.items():
            require(Path(fn).name==fn,'UNSAFE_ARTIFACT_NAME')
            ledger.check(rd/fn,rec,rid+':T:'+fn)
    else:
        entry=next(x for x in tlock['entries'] if x['run_id']==rid)
        ledger.check(rd/'STAGE_T_COMPLETE.json',{'sha256':entry['complete_sha256']},rid+':T:COMPLETE')
        ledger.check(rd/'FIT_IDENTITY.json',{'sha256':entry['fit_identity_sha256']},rid+':T:IDENTITY')
    done=json.loads((rd/'STAGE_T_COMPLETE.json').read_text())
    require(done['status']=='LOCKED' and done.get('test_labels_read') is False and done.get('test_scores_computed') is False,'T_STAGE_SEPARATION:'+rid)
    require({'stage_t_result.json','FIT_IDENTITY.json','validation_predictions.parquet'}<=set(done['artifacts']),'T_REQUIRED_ARTIFACTS:'+rid)
    for fn,rec in done['artifacts'].items():
        require(Path(fn).name==fn,'T_UNSAFE_ARTIFACT_NAME')
        ledger.check(rd/fn,rec,rid+':T:'+fn)
    res=json.loads((rd/'stage_t_result.json').read_text())
    require(res.get('test_evaluated') is False,'T_TEST_EVALUATED:'+rid)
    thr=float(res['val_threshold']);require(np.isfinite(thr) and 0<=thr<=1,'THRESHOLD:'+rid)
    return thr

def load_inputs():
    ledger=InputLedger();roots={}
    for role,rec in SPEC['source_roots'].items():
        if role=='flags':continue  # ancillary status reported separately if unavailable
        p=ledger.check(rec['path'],rec,role)
        if role!='runner':roots[role]=json.loads(p.read_text())
    bind=roots['binding'];tlock=roots['tlock'];elock=roots['elock'];result=roots['results'];matrix=roots['matrix']
    require(bind['fit_count']==10 and bind['input_dim']==42,'EXTENSION_CONTRACT')
    require(tlock.get('fit_count')==10 and elock.get('fit_count')==10,'GLOBAL_LOCK_COUNTS')
    require(elock['status']=='ALL_10_STAGE_E_LOCKED','E_NOT_LOCKED')
    require(elock.get('strict_seed_explicit_before_inference_all_10') is True,'E_STRICT_FLAG')
    require(len(matrix)==10 and len({x['run_id'] for x in matrix})==10,'MATRIX_COUNT')
    for k in ('protocol_sha256','run_matrix_sha256','runner_sha256'):
        require(bind[k]==elock[k] and bind[k]==roots['authorization'][k],'BINDING_CHAIN:'+k)
    require(SPEC['prefit_membership']==bind['prefit_component_membership'],'COMPONENT_PREFIT_BINDING')
    # Revalidate frozen input bytes without importing model definitions.
    for role,rec in SPEC['input_ledger'].items():ledger.check(rec['path'],rec,'input:'+role)
    lab=pd.read_parquet(SPEC['input_ledger']['labels']['path'],columns=KEYS+['split_v1','label_v1_strict_ab_primary'])
    lab=strict_keys(lab);yy=pd.to_numeric(lab.label_v1_strict_ab_primary,errors='raise')
    keep=lab.split_v1.eq('test_2021_2022') & yy.notna()
    can=lab.loc[keep,KEYS].copy();can['y_true']=binary(yy[keep],'canonical')
    can=can.sort_values(KEYS).reset_index(drop=True)
    require((len(can),int(can.y_true.sum()))==(8435,20),'CANONICAL_COUNTS')
    require(hashlib.sha256(can.to_csv(index=False,lineterminator='\n').encode()).hexdigest()==elock['test_identity_sha256'],'CANONICAL_LOCK_IDENTITY')
    masks=strict_keys(pd.read_parquet(SPEC['input_ledger']['evaluation_masks']['path']))
    m=can[KEYS].merge(masks,on=KEYS,how='left',validate='one_to_one',sort=False,indicator=True)
    require(m['_merge'].eq('both').all(),'MASK_JOIN')
    masks={p:binary(m[p],p).astype(bool) for p in POPS+['D_positive','O_positive']}
    y=can.y_true.to_numpy(np.int8)
    for p,(n,pos) in POP_COUNTS.items():require((int(masks[p].sum()),int(y[masks[p]].sum()))==(n,pos),'MASK_COUNTS:'+p)
    require(np.array_equal(masks['D_positive']|masks['O_positive'],y==1),'POSITIVE_MASK_UNION')
    require(not (masks['D_positive'] & masks['O_positive']).any(),'POSITIVE_MASK_OVERLAP')
    cprec=SPEC['prefit_membership'];cp=ledger.check(cprec['path'],cprec,'pre-extension-component-membership')
    gm=pd.read_csv(cp,dtype={'company_code':'string'})
    gm=aligned(gm,can,['company_cluster','document_component'],'component-membership')
    _,company=np.unique(can.company_code.to_numpy(str),return_inverse=True)
    for col in ['company_cluster','document_component']:
        v=pd.to_numeric(gm[col],errors='raise').to_numpy();require(np.isfinite(v).all() and np.equal(v,np.floor(v)).all(),'GROUP_ID_DOMAIN')
        v=v.astype(np.int64);require(np.array_equal(np.unique(v),np.arange(v.max()+1)),'GROUP_ID_NONCONTIGUOUS')
        gm[col]=v
    require(np.array_equal(gm.company_cluster.to_numpy(),company),'COMPANY_GROUP_CONTRACT')
    component=gm.document_component.to_numpy(np.int64)
    require(len(np.unique(company))==4531 and len(np.unique(component))==4530,'GROUP_COUNTS')
    require(pd.DataFrame({'cc':can.company_code,'g':component}).groupby('cc').g.nunique().max()==1,'COMPANY_SPLIT_IN_COMPONENT')
    resultby={r['run_id']:r for r in result['runs']};fits=[]
    for r in matrix:
        rid=r['run_id'];rd=ROOT/'stage_e'/rid
        require(rid in elock['artifacts'] and rid in resultby,'E_RUN_MISSING:'+rid)
        for fn,rec in elock['artifacts'][rid].items():ledger.check(rd/fn,rec,rid+':E:'+fn)
        done=json.loads((rd/'STAGE_E_COMPLETE.json').read_text());stored=json.loads((rd/'stage_e_result.json').read_text())
        require(done['status']=='LOCKED' and done.get('strict_seed_explicit_before_inference') is True and stored.get('strict_seed_explicit_before_inference') is True,'E_FLAGS:'+rid)
        thr=verify_tstage(ledger,ROOT,rid,tlock)
        require(float(stored['validation_threshold'])==thr,'E_T_THRESHOLD:'+rid)
        for p in POPS:
            for k in METRICS:require(abs(float(stored['metrics'][p][k])-float(resultby[rid]['metrics'][p][k]))<2e-12,'EXPORTED_RESULT_PARITY')
        score=read_predictions(rd/'test_predictions.parquet',can,rid)
        fits.append(dict(run_id=rid,model=r['model'],seed=int(r['model_seed']),threshold=thr,score=score,stored=stored,origin='V28_B0GNN42'))
    oldrefs=json.loads((HERE/'reference/matched_feature_reference_results.json').read_text())
    for r in bind['frozen_reference_records']:
        rid=r['run_id']
        for key in ['stage_e_complete','stage_e_result','test_predictions']:
            meta=r[key];ledger.check(meta['path'],meta,rid+':'+key)
        stored=json.loads(Path(r['stage_e_result']['path']).read_text())
        require(stored==oldrefs[rid],'OLD_REFERENCE_RESULT:'+rid)
        thr=verify_tstage(ledger,OLD,rid,roots['old_tlock'])
        require(float(stored['validation_threshold'])==thr,'OLD_THRESHOLD:'+rid)
        score=read_predictions(r['test_predictions']['path'],can,rid)
        fits.append(dict(run_id=rid,model=r['model'],seed=int(r['model_seed']),threshold=thr,score=score,stored=stored,origin='V27_B0_42_REFERENCE'))
    require(len(fits)==20,'MATCHED_FIT_COUNT')
    fits.sort(key=lambda f:(MODELS.index(f['model']),list(SEEDS).index(f['seed'])))
    require([(f['model'],f['seed']) for f in fits]==[(m,s) for m in MODELS for s in SEEDS],'MATCHED_FIT_LAYOUT')
    return ledger,can,y,masks,company,component,fits

def point_metrics(y,masks,fits):
    rows=[];cube={}
    for pop in POPS:
        mask=masks[pop];values=[]
        for f in fits:
            plan=MetricPlan(y[mask],f['score'][mask],f['threshold']);v=plan.compute(np.ones(mask.sum(),dtype=np.int64))
            for k,a in zip(METRICS,v):require(abs(float(f['stored']['metrics'][pop][k])-a)<2e-12,'POINT_PARITY:'+f['run_id']+':'+pop+':'+k)
            tr=plan.tie_report();stored=f['stored']['metrics'][pop]
            for k in ('budget','cutoff_tie_rows','selected_from_cutoff_tie','tie_hits_min','tie_hits_max'):require(stored[k]==tr[k],'TIE_PARITY:'+k)
            rows.append(dict(run_id=f['run_id'],model=f['model'],model_seed=f['seed'],origin=f['origin'],population=pop,n_positive=int(y[mask].sum()),validation_threshold=f['threshold'],**dict(zip(METRICS,v)),**tr))
            values.append(v)
        cube[pop]=np.asarray(values).reshape(4,5,6)
    return rows,cube

_BOOT=None

def init_boot(y,scores,thresholds,groups,group_seed,seed_seed):
    global _BOOT
    _BOOT=(y,[MetricPlan(y,s,t) for s,t in zip(scores,thresholds)],groups,group_seed,seed_seed)

def one_boot(b):
    y,plans,g,gs,ss=_BOOT
    weights,seeds,rejects=bootstrap_draw(b,g,y,gs,ss,1000)
    if weights is None:return b,None,dict(replicate=b,status='NON_ESTIMABLE',rejections=rejects)
    fit=np.stack([p.compute(weights) for p in plans]).reshape(4,5,6)
    value=fit[:,seeds,:].mean(axis=1)
    audit=dict(replicate=b,status='VALID',rejections=rejects,resampled_N=int(weights.sum()),positive=int(np.dot(weights,y)),seed_indices=seeds.tolist(),weights_sha256=hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest())
    return b,value,audit

def compute_bootstraps(y,fits,groups,gs,ss,reps,workers=2):
    scores=np.stack([f['score'] for f in fits]);thresholds=np.array([f['threshold'] for f in fits])
    arr=np.full((reps,4,6),np.nan);audit=[None]*reps
    if workers==1:
        init_boot(y,scores,thresholds,groups,gs,ss)
        iterator=map(one_boot,range(reps))
        for b,value,a in iterator:
            if value is not None:arr[b]=value
            audit[b]=a
            if (b+1)%100==0:print(f'BOOTSTRAP {gs}: {b+1}/{reps}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_boot,initargs=(y,scores,thresholds,groups,gs,ss)) as ex:
            for b,value,a in ex.map(one_boot,range(reps),chunksize=10):
                if value is not None:arr[b]=value
                audit[b]=a
                if (b+1)%100==0:print(f'BOOTSTRAP {gs}: {b+1}/{reps}',flush=True)
    delta=np.stack([arr[:,l]-arr[:,r] for _,l,r in CONTRASTS],axis=1)
    return arr,delta,audit

def interval_rows(arr,delta,point,scheme):
    valid=np.isfinite(arr).all(axis=(1,2));estimable=bool(valid.all());n=int(valid.sum())
    cond=[];contrast=[]
    for ci,name in enumerate(MODELS):
        for mi,metric in enumerate(METRICS):
            lo,hi=np.quantile(arr[:,ci,mi],[.025,.975],method='linear') if estimable else (None,None)
            cond.append(dict(scheme=scheme,model=name,metric=metric,point_estimate=float(point[ci,:,mi].mean()),lower=lo,upper=hi,valid_replicates=n,expected_replicates=len(arr),status='ESTIMABLE' if estimable else 'NA_INSUFFICIENT_VALID_DRAWS'))
    for ci,(name,l,r) in enumerate(CONTRASTS):
        for mi,metric in enumerate(METRICS):
            lo,hi=np.quantile(delta[:,ci,mi],[.025,.975],method='linear') if estimable else (None,None)
            contrast.append(dict(scheme=scheme,contrast=name,metric=metric,point_estimate=float((point[l,:,mi]-point[r,:,mi]).mean()),lower=lo,upper=hi,valid_replicates=n,expected_replicates=len(arr),status='ESTIMABLE' if estimable else 'NA_INSUFFICIENT_VALID_DRAWS',scope='exploratory_marginal_conditional_no_p_value'))
    return cond,contrast

def influence(y,masks,fits):
    positions=np.flatnonzero(masks['D_positive']);require(len(positions)==7,'D_SEVEN')
    plans=[MetricPlan(y,f['score'],f['threshold']) for f in fits]
    values=[]
    for row in positions:
        w=masks['D_frozen'].astype(np.int64).copy();w[row]=0
        require(int(w[y==0].sum())==8415 and int(np.dot(w,y))==6,'LOPO_COUNTS')
        values.append(np.stack([p.compute(w) for p in plans]).reshape(4,5,6))
    v=np.asarray(values);avg=v.mean(axis=2);out=[];contr=[];perfit=[]
    for c,name in enumerate(MODELS):
        for k,metric in enumerate(METRICS):
            out.append(dict(model=name,metric=metric,deletions=7,remaining_positives=6,negatives_retained=8415,min_estimate=float(avg[:,c,k].min()),max_estimate=float(avg[:,c,k].max()),scope='descriptive_influence_not_CI'))
            for si,seed in enumerate(SEEDS):perfit.append(dict(model=name,model_seed=seed,metric=metric,min_estimate=float(v[:,c,si,k].min()),max_estimate=float(v[:,c,si,k].max())))
    for name,l,r in CONTRASTS:
        d=avg[:,l]-avg[:,r]
        for k,metric in enumerate(METRICS):contr.append(dict(contrast=name,metric=metric,min_paired_mean=float(d[:,k].min()),max_paired_mean=float(d[:,k].max()),deletions=7,scope='descriptive_influence_not_CI'))
    return out,contr,perfit

def ancillary_flags(ledger,canonical,y,masks):
    sets_by_ranking={p:[] for p in ['D_subset_reranked','full_top422_D_members']}
    for rec in SPEC['legacy_A']:
        p=ledger.check(rec['prediction']['path'],rec['prediction'],rec['run_id']+':legacy_scores')
        s=read_predictions(p,canonical,rec['run_id'])
        for name in sets_by_ranking:
            ix=np.flatnonzero(masks['D_frozen']) if name=='D_subset_reranked' else np.arange(len(y))
            top=ix[np.argsort(-s[ix],kind='mergesort')[:422]]
            hits=set(int(i) for i in top if y[i]==1 and masks['D_positive'][i])
            require(len(hits)==1,'LEGACY_EXPECTED_ONE_HIT')
            sets_by_ranking[name].append(hits)
    common=[]
    for name,sets in sets_by_ranking.items():
        require(len(sets)==5,'LEGACY_SEEDS')
        both=set.intersection(*sets);union=set.union(*sets)
        require(len(both)==len(union)==1,'LEGACY_HIT_NOT_SHARED')
        common.append(next(iter(both)))
    require(common[0]==common[1],'LEGACY_RANKING_IDENTITY')
    flagrec=SPEC['source_roots']['flags'];p=ledger.check(flagrec['path'],flagrec,'financial_source_flags')
    fn=['asof_any_source_missing','asof_any_final_financial_nonfinite','fin_any_no_prior_candidate']
    f=strict_keys(pd.read_parquet(p,columns=KEYS+fn))
    f=canonical[KEYS].merge(f,on=KEYS,how='left',sort=False,validate='one_to_one',indicator=True)
    require(f['_merge'].eq('both').all(),'FLAG_JOIN')
    flags={n:bool(binary(f[n],n)[common[0]]) for n in fn}
    return dict(status='COMPLETE',n_shared_positive_rows=1,n_seeds=5,rankings_agree=True,source_flags=flags,company_codes_exported=False,document_keys_exported=False,interpretation='co-occurrence only; does not establish what caused the GCN prediction')

def package_return(attempt,status):
    share=attempt/'SHARE';files=sorted(p for p in share.iterdir() if p.is_file())
    manifest={'status':status,'files':{p.name:{'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in files},'row_level_predictions_included':False,'model_artifacts_included':False,'private_membership_included':False}
    writej(share/'RETURN_MANIFEST.json',manifest)
    target=attempt/'V28_FIXED_SCORE_RETURN.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(share.iterdir()):
            if p.is_file():z.write(p,p.name)
    with zipfile.ZipFile(target) as z:require(z.testzip() is None,'RETURN_CRC')
    published=BASE/'V28_FIXED_SCORE_RETURN.zip'
    # No silent replacement of an earlier returned attempt.
    if not published.exists():
        with target.open('rb') as src,published.open('xb') as dst:
            for b in iter(lambda:src.read(4*1024*1024),b''):dst.write(b)
    writej(BASE/'LATEST_COMPLETED.json',dict(status=status,attempt=str(attempt),return_zip=str(target),sha256=sha256(target),size_bytes=target.stat().st_size))
    return target

def verify_package():
    j=json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
    for name,meta in j['files'].items():
        p=HERE/name;require(p.is_file() and sha256(p)==meta['sha256'] and p.stat().st_size==meta['size_bytes'],'PACKAGE_IDENTITY:'+name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    attempt=safe_output(a.attempt);require(attempt.parent==BASE/'attempts','ATTEMPT_PARENT');require(a.workers in (1,2),'CPU_WORKERS')
    share=attempt/'SHARE';share.mkdir(parents=True,exist_ok=False)
    modules=[];status='HOLD_FIXED_SCORE_COMPLETION';error=None
    ledger=None;non_estimable=False
    def complete(name,detail=None):
        modules.append(dict(name=name,status='COMPLETE',detail=detail));print('MODULE_COMPLETE '+name,flush=True)
        writej(share/'MODULE_STATUS.json',modules)
    try:
        verify_package()
        import sklearn,pyarrow
        runtime_versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__)
        expected_versions=dict(python='3.12.3',numpy='2.5.3',pandas='3.0.5',sklearn='1.9.1',pyarrow='25.0.1')
        require(runtime_versions==expected_versions,'PINNED_CPU_ENVIRONMENT_MISMATCH')
        from selftest import run_tests
        selftest=run_tests(require_parquet=True,temp_root=attempt/'SELFTEST_PRIVATE')
        writej(share/'EXECUTION_SELFTESTS.json',selftest);complete('synthetic_numeric_and_real_parquet_interfaces')
        ledger,canonical,y,masks,company,component,fits=load_inputs();complete('frozen_input_and_reference_identity',{'matched_fits':20})
        writej(share/'INPUT_HASH_LEDGER.json',ledger.aggregate())
        writej(share/'EXECUTION_SPEC.json',SPEC)
        import sklearn,pyarrow
        writej(share/'ENVIRONMENT.json',dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__,workers=a.workers,GPU_used=False))
        perfit,cube=point_metrics(y,masks,fits);writecsv(share/'01_MATCHED_PER_FIT_METRICS.csv',perfit)
        rows=[];contr=[]
        for pop,v in cube.items():
            for ci,model in enumerate(MODELS):
                for mi,m in enumerate(METRICS):
                    x=v[ci,:,mi];rows.append(dict(model=model,population=pop,metric=m,estimate=float(x.mean()),seed_sd=float(x.std(ddof=1)),seed_min=float(x.min()),seed_max=float(x.max()),uncertainty='seed_description_not_test_CI'))
            for name,l,r in CONTRASTS:
                for mi,m in enumerate(METRICS):
                    x=v[l,:,mi]-v[r,:,mi];contr.append(dict(contrast=name,population=pop,metric=m,estimate=float(x.mean()),seed_sd=float(x.std(ddof=1)),min_seed=float(x.min()),max_seed=float(x.max())))
        writecsv(share/'02_MATCHED_DESCRIPTIVES.csv',rows);writecsv(share/'03_PAIRED_POINT_ESTIMATES.csv',contr);complete('all_population_points')
        for scheme,g,seeds in [('company_x_matched_seed',company,SPEC['primary_seeds']),('company_document_x_matched_seed',component,SPEC['secondary_seeds'])]:
            arr,delta,draws=compute_bootstraps(y,fits,g,*seeds,SPEC['replicates'],a.workers)
            cond,ct=interval_rows(arr,delta,cube['full_test'],scheme)
            writecsv(share/(scheme+'_CONDITION_INTERVALS.csv'),cond);writecsv(share/(scheme+'_CONTRAST_INTERVALS.csv'),ct)
            np.savez_compressed(safe_output(share/(scheme+'_AGGREGATE_REPLICATES.npz')),condition_replicates=arr,contrast_replicates=delta,models=np.array(MODELS),contrasts=np.array([x[0] for x in CONTRASTS]),metrics=np.array(METRICS))
            valid=int(np.isfinite(arr).all(axis=(1,2)).sum());non_estimable|=valid!=SPEC['replicates']
            writej(share/(scheme+'_DRAW_AUDIT.json'),draws)
            writej(share/(scheme+'_DESIGN_STATUS.json'),dict(valid_replicates=valid,planned_replicates=SPEC['replicates'],rejected_draws=sum(x['rejections'] for x in draws),company_or_component_count=len(np.unique(g)),group_seed=seeds[0],seed_label_seed=seeds[1],replicate_rule='base+10*b',scope='conditional_exploratory_marginal_not_new_training_or_new_test_cohort',membership_sha256=SPEC['prefit_membership']['sha256'] if scheme.startswith('company_document') else None))
            complete(scheme,{'valid_replicates':valid})
        x,c,f=influence(y,masks,fits);writecsv(share/'04_D_LOPO_CONDITION_RANGES.csv',x);writecsv(share/'05_D_LOPO_CONTRAST_RANGES.csv',c);writecsv(share/'06_D_LOPO_PER_FIT_RANGES.csv',f);complete('D_leave_one_positive_out')
        try:
            flags=ancillary_flags(ledger,canonical,y,masks);writej(share/'07_LEGACY_A_GCN_COMMON_D_HIT_FLAGS.json',flags);complete('legacy_A_GCN_unique_D_hit_flags')
        except Exception as e:
            code=e.code if isinstance(e,ContractError) else type(e).__name__
            writej(share/'07_LEGACY_A_GCN_COMMON_D_HIT_FLAGS.json',dict(status='UNAVAILABLE',reason=code,no_substitution=True))
            modules.append(dict(name='legacy_A_GCN_unique_D_hit_flags',status='UNAVAILABLE',reason=code));non_estimable=True
        ledger.recheck();writej(share/'INPUT_HASH_LEDGER.json',ledger.aggregate());complete('frozen_inputs_unchanged_after_readonly_postprocess')
        status='COMPLETE_WITH_UNAVAILABLE_OR_NON_ESTIMABLE_MODULES' if non_estimable else 'COMPLETE_V28_FIXED_SCORE_FOR_FINAL_REVIEW'
    except Exception as e:
        error={'type':type(e).__name__,'code':e.code if isinstance(e,ContractError) else 'RUNTIME_'+type(e).__name__}
        # Detailed traceback retained on server only; not part of public return.
        p=safe_output(attempt/'ERROR_LOCAL_ONLY.txt');p.write_text(traceback.format_exc(),encoding='utf-8')
        if ledger is not None:writej(share/'INPUT_HASH_LEDGER.json',ledger.aggregate())
        print('HOLD '+str(error),flush=True)
    writej(share/'MODULE_STATUS.json',modules)
    state=dict(status=status,completed_at_utc=datetime.now(timezone.utc).isoformat(),modules=modules,error=error,new_fits=0,new_inference=0,new_thresholds=0,source_mutation_operations_performed=False,input_hash_recheck_status=('PASS' if any(m['name']=='frozen_inputs_unchanged_after_readonly_postprocess' and m['status']=='COMPLETE' for m in modules) else 'NOT_ESTABLISHED'),authorized_write_root=str(BASE),primary_comparisons=4,metrics_per_comparison=6,planned_schemes=2,planned_replicates_each=2000,next_gate='INDEPENDENT_REVIEW_AND_V28_FINALIZATION')
    writej(share/'RUN_STATUS.json',state)
    ret=package_return(attempt,status)
    print('STATUS='+status+'\nRETURN_ZIP='+str(ret),flush=True)
    raise SystemExit(0 if status=='COMPLETE_V28_FIXED_SCORE_FOR_FINAL_REVIEW' else 2)

if __name__=='__main__':main()
