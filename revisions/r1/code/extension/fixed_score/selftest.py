"""Synthetic execution tests, with no model fitting and no empirical scores."""
from __future__ import annotations
import math,json,tempfile,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from fs_math import MetricPlan,METRICS,SEEDS,bootstrap_draw,require,ContractError

def explicit(y,s,w,t):
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
    ix=np.repeat(np.arange(len(y)),w);y=np.asarray(y)[ix];s=np.asarray(s)[ix]
    order=np.argsort(-s,kind='mergesort');b=max(1,math.ceil(.05*len(y)))
    fp,tp,_=roc_curve(y,s,drop_intermediate=True)
    ok=fp<=.10+1e-12
    return np.array([roc_auc_score(y,s),average_precision_score(y,s),f1_score(y,s>=t,zero_division=0),float(y[order[:b]].mean()),float(tp[ok].max()),float(y[order[:b]].sum())])

def synthetic():
    import extension_fixedscore as main
    n=180;can=pd.DataFrame({'company_code':[f'{i//2:06d}' for i in range(n)],'fiscal_year':np.tile([2021,2022],n//2)})
    y=np.zeros(n,dtype=np.int8);y[:20]=1;can['y_true']=y
    dp=np.zeros(n,bool);dp[:7]=True;op=(y==1)&~dp
    sm=np.ones(n,bool);sm[:20]=False;sm[[0,1,7,8,9,10]]=True;sm[-12:]=False
    masks={'full_test':np.ones(n,bool),'D_positive':dp,'O_positive':op,'D_frozen':(y==0)|dp,'O_frozen':(y==0)|op,'S_frozen':sm,'J_frozen':sm&((y==0)|dp)}
    fits=[];rng=np.random.default_rng(280141707)
    for m in main.MODELS:
        for seed in SEEDS:
            s=np.round(rng.random(n),3);th=.67
            stored={}
            for p in main.POPS:
                a=masks[p];mp=MetricPlan(y[a],s[a],th)
                stored[p]={**dict(zip(METRICS,mp.compute(np.ones(a.sum(),int)))),**mp.tie_report()}
            fits.append(dict(run_id=m+'_'+str(seed),model=m,seed=seed,threshold=th,score=s,origin='synthetic',stored={'metrics':stored}))
    return can,y,masks,fits

def run_tests(require_parquet=False,temp_root=None):
    import extension_fixedscore as v
    rows=[];skipped=[]
    def check(name,fn):
        detail=fn();json.dumps(v.finite_json(detail),allow_nan=False);rows.append(dict(name=name,pass_=True,detail=detail))
    def reject(fn):
        try:fn()
        except (ContractError,ValueError,TypeError):return {'rejected':True}
        raise AssertionError('INVALID_INPUT_ACCEPTED')
    def numeric():
        rng=np.random.default_rng(14170728);err=0
        for i in range(400):
            n=int(rng.integers(10,150));y=rng.integers(0,2,n);y[:2]=[0,1]
            s=np.round(rng.random(n),i%4);w=rng.integers(0,6,n);w[:2]=1;t=float(rng.random())
            a=MetricPlan(y,s,t).compute(w);b=explicit(y,s,w,t)
            err=max(err,float(np.max(abs(a-b))));require(np.allclose(a,b,rtol=0,atol=2e-12),'TEST_DUPLICATES')
        return dict(cases=400,max_abs_error=err)
    check('weighted_kernel_vs_explicit_duplicated_samples',numeric)
    def rare():
        rng=np.random.default_rng(90123);err=0
        for i in range(10):
            y=np.zeros(8435,int);y[rng.choice(8435,20,replace=False)]=1;s=np.round(rng.random(8435),i%3)
            w=np.ones(8435,int);a=MetricPlan(y,s,.5).compute(w);b=explicit(y,s,w,.5)
            err=max(err,float(np.max(abs(a-b))));require(np.allclose(a,b,rtol=0,atol=2e-12),'TEST_RARE')
        return dict(cases=10,N=8435,positive=20,max_abs_error=err)
    check('rare_event_actual_population_size',rare)
    check('nonfinite_score_rejection',lambda:reject(lambda:MetricPlan([0,1],[.2,np.nan],.5)))
    check('noninteger_multiplicity_rejection',lambda:reject(lambda:MetricPlan([0,1],[.2,.8],.5).compute([1,.5])))
    check('single_class_returns_NA',lambda:require(np.isnan(MetricPlan([0,1],[.2,.8],.5).compute([1,0])).all(),'TEST_SINGLE_CLASS'))
    can=pd.DataFrame({'company_code':['000001','000002'],'fiscal_year':[2021,2022],'y_true':[0,1]})
    def keys():
        d=can.iloc[::-1].copy();d['score']=[.8,.2]
        got=v.aligned(d,can,['score','y_true'],'synthetic')
        require(got.score.tolist()==[.2,.8],'TEST_ALIGNMENT')
        reject(lambda:v.aligned(pd.concat([d,d.iloc[:1]]),can,['score'],'duplicate'))
        reject(lambda:v.aligned(d.iloc[:1],can,['score'],'missing'))
        reject(lambda:v.strict_keys(pd.DataFrame({'company_code':['000001'],'fiscal_year':[2021.4]})))
        reject(lambda:v.binary(pd.Series([0,None]),'missing'))
        reject(lambda:v.binary(pd.Series(['False','True']),'string_bool'))
        return dict(shuffled_join=True,duplicates_missing_and_fractional_year_rejected=True)
    check('key_and_binary_domain_guards',keys)
    check('runtime_dataframe_cannot_enter_json',lambda:reject(lambda:v.finite_json(can)))
    check('numpy_json_scalars',lambda:v.finite_json({'i':np.int64(1),'b':np.bool_(True),'f':np.float64(.3)}))
    def draws():
        idx=np.repeat(np.arange(30),2);y=np.zeros(60,int);y[:10]=1
        w,s,rejects=bootstrap_draw(5,idx,y,1807141,1807142)
        expected=np.random.default_rng(1807142+50).integers(0,5,5)
        require(np.array_equal(s,expected),'SEED_DRAW_TEST')
        w,s,n=bootstrap_draw(0,np.zeros(2,int),np.ones(2),1,2,max_attempts=3)
        require(w is None and s is None and n==3,'BOUNDED_REJECTION')
        return {'matched_rng':True,'maximum_attempts':True}
    check('bootstrap_rng_and_rejection',draws)
    def integration():
        can,y,m,f=synthetic();rows,cube=v.point_metrics(y,m,f)
        require(len(rows)==100,'SYNTH_POINT_ROWS')
        groups=np.repeat(np.arange(90),2)
        a,d,dr=v.compute_bootstraps(y,f,groups,1807141,1807142,6,1)
        b,dd,db=v.compute_bootstraps(y,f,groups,1807141,1807142,6,2)
        require(np.array_equal(a,b) and np.array_equal(d,dd) and dr==db,'WORKER_PARITY')
        cond,cts=v.interval_rows(a,d,cube['full_test'],'synthetic')
        require(len(cond)==len(cts)==24,'INTERVAL_COUNTS')
        for ci,(_,l,r) in enumerate(v.CONTRASTS):require(np.array_equal(d[:,ci],a[:,l]-a[:,r]),'CONTRAST_PAIRING')
        # D influence on full-size synthetic arrays obeys protocol negative-count assertions.
        n=8435;y=np.zeros(n,np.int8);y[:20]=1;dp=np.zeros(n,bool);dp[:7]=True
        rng=np.random.default_rng(330);ff=[]
        for model in v.MODELS:
            for seed in SEEDS:ff.append({'model':model,'seed':seed,'score':rng.random(n),'threshold':.5})
        x,c,p=v.influence(y,{'D_positive':dp,'D_frozen':(y==0)|dp},ff)
        require((len(x),len(c),len(p))==(24,24,120),'LOPO_OUTPUT_SHAPE')
        return dict(synthetic_fits=20,conditions=4,populations=5,matched_contrasts=4,metrics=6,worker_arrays_exact=True,LOPO_range_rows=168)
    check('twenty_fit_numeric_pipeline_integration',integration)
    root=Path(temp_root) if temp_root else Path(tempfile.mkdtemp(prefix='v28_test_'))
    root.mkdir(parents=True,exist_ok=True)
    def ledger():
        p=root/'frozen_test.bin';p.write_bytes(b'abc');meta={'sha256':v.sha256(p),'size_bytes':3}
        l=v.InputLedger();l.check(p,meta,'synthetic');l.recheck();p.write_bytes(b'abd')
        reject(l.recheck);p.unlink()
        return {'mutation_detected':True}
    check('before_after_source_hash_audit',ledger)
    def reporting():
        import zipfile
        testbase=root/'output_fixture';testbase.mkdir(exist_ok=False)
        attempt=testbase/'attempt';share=attempt/'SHARE';share.mkdir(parents=True)
        oldsafe,oldbase=v.safe_output,v.BASE
        def fixture_safe(p,base=None):
            q=Path(p).resolve();b=testbase.resolve()
            require(q==b or b in q.parents,'SYNTHETIC_OUTPUT_ESCAPE')
            return Path(p)
        try:
            v.safe_output=fixture_safe;v.BASE=testbase
            v.writej(share/'RUN_STATUS.json',{'status':'SYNTHETIC','scalar':np.int64(2)})
            v.writecsv(share/'test.csv',[{'count':np.int64(2),'NA':None}])
            (attempt/'ERROR_LOCAL_ONLY.txt').write_text('synthetic local detail')
            z=v.package_return(attempt,'SYNTHETIC')
            with zipfile.ZipFile(z) as f:
                require(f.testzip() is None,'SYNTH_CRC')
                require(set(f.namelist())=={'RUN_STATUS.json','test.csv','RETURN_MANIFEST.json'},'RETURN_WHITELIST')
                mf=json.loads(f.read('RETURN_MANIFEST.json'))
                for name,rec in mf['files'].items():
                    import hashlib
                    b=f.read(name);require(len(b)==rec['size_bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],'RETURN_MANIFEST_PARITY')
            require((testbase/'LATEST_COMPLETED.json').is_file(),'POINTER_WRITTEN')
            return {'json_csv_zip_roundtrip':True,'manifest_parity':True,'local_only_file_excluded':True}
        finally:
            v.safe_output=oldsafe;v.BASE=oldbase
            shutil.rmtree(testbase)
    check('closed_loop_return_manifest_and_private_exclusion',reporting)
    def parquet():
        import pyarrow
        p=root/'synthetic_predictions.parquet'
        d=can.iloc[::-1].copy();d['score']=[.8,.2];d.to_parquet(p,index=False)
        s=v.read_predictions(p,can,'synthetic');require(np.array_equal(s,[.2,.8]),'PARQUET_ALIGN')
        d['y_true']=[.5,0];d.to_parquet(p,index=False);reject(lambda:v.read_predictions(p,can,'bad_label'))
        p.unlink();return dict(pyarrow=pyarrow.__version__,real_parquet_roundtrip=True)
    try:import pyarrow
    except ImportError:
        if require_parquet:raise RuntimeError('PYARROW_REQUIRED_IN_SERVER_ENVIRONMENT')
        skipped.append('real_parquet_roundtrip: local dependency not installed')
    else:check('real_parquet_roundtrip_and_shuffled_key_parity',parquet)
    if not temp_root:shutil.rmtree(root)
    return dict(status='PASS' if not skipped else 'PASS_AVAILABLE_TESTS_WITH_DECLARED_LOCAL_SKIP',checks=rows,skipped=skipped,new_fits=0,new_inference=0)

if __name__=='__main__':
    print(json.dumps(run_tests(),indent=2,default=str))
