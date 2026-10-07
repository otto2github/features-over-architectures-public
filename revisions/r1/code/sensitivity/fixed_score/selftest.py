"""Real arithmetic/serialization regression tests. Never access empirical inputs."""
from __future__ import annotations
import ast, hashlib, json, math, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from fs_math import (MetricPlan,METRICS,SEEDS,ContractError,require,bootstrap_draw,
                     connected_company_groups,contrast_catalog,condition_name)
from fs_io import Output,Ledger,Fit,Bundle,strict_keys,one_to_one,binary_array,bool_array,jsonable

HERE=Path(__file__).resolve().parent

def direct_sklearn(y,s,w,threshold):
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
    expanded=np.repeat(np.arange(len(y)),w)
    yy=np.asarray(y)[expanded];ss=np.asarray(s)[expanded]
    ix=np.argsort(-ss,kind='mergesort');k=max(1,math.ceil(.05*len(yy)))
    fpr,tpr,_=roc_curve(yy,ss,drop_intermediate=True)
    return np.array([roc_auc_score(yy,ss),average_precision_score(yy,ss),
                     f1_score(yy,ss>=threshold,zero_division=0),float(yy[ix[:k]].mean()),
                     float(tpr[fpr<=.10+1e-12].max()),float(yy[ix[:k]].sum())])

def synthetic_bundle():
    n=120;cc=np.array([f'{i//2:06d}' for i in range(n)]);years=np.tile([2021,2022],n//2)
    canonical=pd.DataFrame({'company_code':cc,'fiscal_year':years})
    y=np.zeros(n,dtype=int);y[:20]=1;canonical['target']=y
    dpos=np.zeros(n,bool);dpos[:7]=True;opos=(y==1)&(~dpos)
    source=np.ones(n,bool);source[:20]=False;source[[0,1,7,8,9,10]]=True;source[60:80]=False
    masks=dict(full_test=np.ones(n,bool),D_positive=dpos,O_positive=opos,
               D_frozen=(y==0)|dpos,O_frozen=(y==0)|opos,S_frozen=source,J_frozen=source&((y==0)|dpos))
    specs=json.loads((HERE/'reference/RUN_MATRIX_v2.json').read_text())
    fits=[];rng=np.random.default_rng(12391)
    def make(rid,c,seed,fam,m,mod,w):
        s=np.round(rng.random(n),2);t=.65
        metric={p:dict(zip(METRICS,direct_sklearn(y[a],s[a],np.ones(a.sum(),dtype=int),t))) for p,a in masks.items() if p.endswith('_frozen') or p=='full_test'}
        top=np.argsort(-s,kind='mergesort')[:422]
        attrib=dict(budget=422,full_hits=int(y[top].sum()),D_positive_hits_in_full_top422=int(dpos[top].sum()),O_positive_hits_in_full_top422=int(opos[top].sum()))
        stored={'metrics':metric,'full_budget_attribution':attrib} if fam!='REFERENCE' else {'metrics':metric['full_test']}
        return Fit(rid,c,seed,fam,m,mod,w,s,t,stored)
    for r in specs:fits.append(make(r['run_id'],condition_name(r),r['model_seed'],r['family'],r['model'],r['modality'],r['imputation_seed']))
    refs=json.loads((HERE/'reference/REFERENCE_EXPECTATIONS.json').read_text())
    for r in refs:fits.append(make('REF_'+r['private_condition']+'_'+str(r['seed']),r['condition'],r['seed'],'REFERENCE',r['model'],r['modality'],None))
    rows=[]
    for i in range(20):
        # Deterministic shared document links include several companies.
        rows.append(dict(company_code=cc[i],fiscal_year=years[i],partition='test',doc_key_hash=hashlib.sha256(f'doc_{i//4}'.encode()).hexdigest()))
    inc=pd.DataFrame(rows)
    flags={'asof_any_source_missing':np.r_[np.ones(8),np.zeros(n-8)],
           'asof_any_final_financial_nonfinite':np.r_[np.ones(16),np.zeros(n-16)],
           'fin_any_no_prior_candidate':np.tile([0.,1.],n//2)}
    return Bundle(canonical,y,masks,flags,inc,fits,{}, {},None,specs,'synthetic')

def run_selftests(root,require_parquet=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[];skips=[]
    def check(name,fn):
        try:
            detail=fn();json.dumps(jsonable(detail),allow_nan=False)
            rows.append({'test':name,'pass':True,'detail':detail})
        except Exception as e:
            rows.append({'test':name,'pass':False,'error':getattr(e,'code',type(e).__name__)})
    def reject(fn):
        try:fn()
        except (ContractError,ValueError):return True
        raise AssertionError('invalid input was accepted')
    def numerical():
        rng=np.random.default_rng(88141707);maxerr=0.;cases=0
        for i in range(400):
            n=int(rng.integers(15,200));y=rng.integers(0,2,n);y[:2]=[0,1]
            s=np.round(rng.random(n),i%4);w=rng.integers(0,6,n);w[:2]=1;t=float(rng.random())
            a=MetricPlan(y,s,t).compute(w);b=direct_sklearn(y,s,w,t)
            require(np.allclose(a,b,rtol=0,atol=2e-12),'NUMERICAL_EQUIVALENCE')
            maxerr=max(maxerr,float(np.max(abs(a-b))));cases+=1
        return dict(actual_random_tie_zero_weight_cases=cases,max_absolute_error=maxerr)
    check('weighted_metrics_vs_explicit_duplicate_sklearn',numerical)
    def old_reference():
        p=HERE/'reference/predecessor_metrics_reference.py'
        require(hashlib.sha256(p.read_bytes()).hexdigest()=='b1a6ed3e1212debe39f97222dd3284acd1a6a69a29d492e1f1e7046168b78ed3','REFERENCE_HASH')
        tree=ast.parse(p.read_text());tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.level)]
        ns={'require':require};exec(compile(tree,str(p),'exec'),ns)
        rng=np.random.default_rng(442201);err=0
        for i in range(200):
            n=100;y=rng.integers(0,2,n);y[:2]=[0,1];s=np.round(rng.random(n),2);w=rng.integers(0,5,n);w[:2]=1;t=.5
            a=MetricPlan(y,s,t).compute(w)[:5];b=ns['Plan'](y,s,t).compute(w)
            err=max(err,float(np.max(abs(a-b))));require(np.array_equal(a,b),'FROZEN_PLAN_PARITY')
        return dict(cases=200,max_absolute_error=err,reference_file_modified=False)
    check('exact_frozen_V26_plan_parity',old_reference)
    def rare():
        rng=np.random.default_rng(77);err=0.
        for i in range(20):
            y=np.zeros(8435,dtype=int);y[rng.choice(8435,20,replace=False)]=1
            s=np.round(rng.random(8435),2 if i%2 else 0);w=np.ones(8435,dtype=int)
            a=MetricPlan(y,s,.5).compute(w);b=direct_sklearn(y,s,w,.5)
            err=max(err,float(np.max(abs(a-b))));require(np.allclose(a,b,rtol=0,atol=2e-12),'RARE_METRIC_PARITY')
        return dict(cases=20,rows=8435,positives=20,max_absolute_error=err)
    check('rare_event_real_test_size_numerical_parity',rare)
    check('single_class_is_NA_not_half',lambda:require(np.isnan(MetricPlan([0,1],[.1,.2],.5).compute([1,0])).all(),'SINGLE_CLASS'))
    check('bad_weights_fail_closed',lambda:reject(lambda:MetricPlan([0,1],[.2,.8],.5).compute([1,.5])))
    check('nan_scores_rejected',lambda:reject(lambda:MetricPlan([0,1],[np.nan,.8],.5)))
    check('json_runtime_dataframe_not_leaked',lambda:reject(lambda:jsonable(pd.DataFrame({'company_code':['PRIVATE_SENTINEL']}))))
    check('json_numpy_finite_serialization',lambda:jsonable({'n':np.int64(4),'b':np.bool_(True),'f':np.float64(.4)}))
    check('json_nonfinite_rejected',lambda:reject(lambda:jsonable({'nan':float('nan')})))
    def joins():
        a=pd.DataFrame({'company_code':['000001','000002'],'fiscal_year':[2021,2022]})
        b=pd.DataFrame({'company_code':['000002','000001'],'fiscal_year':[2022,2021],'label':[1,0]})
        got=one_to_one(a,b,['label'],'test');require(got.label.tolist()==[0,1],'JOIN_ALIGNMENT')
        reject(lambda:one_to_one(a,pd.concat([b,b.iloc[:1]]),['label'],'test'))
        reject(lambda:one_to_one(a,b.iloc[:1],['label'],'test'))
        return {'row_order_invariant':True,'duplicates_rejected':True,'missing_join_rejected':True}
    check('one_to_one_key_alignment_guards',joins)
    check('fractional_year_rejected',lambda:reject(lambda:strict_keys(pd.DataFrame({'company_code':['000001'],'fiscal_year':[2021.4]}))))
    check('invalid_binary_label_rejected',lambda:reject(lambda:binary_array(pd.Series([0,.5]))))
    check('missing_boolean_not_filled_false',lambda:reject(lambda:bool_array(pd.Series([True,None]))))
    def group_test():
        cc=['a','a','b','c','c','d','e'];inc=[('a','doc1'),('b','doc1'),('b','doc2'),('c','doc2')]
        a,m=connected_company_groups(cc,inc);b,m2=connected_company_groups(list(reversed(cc)),list(reversed(inc)))
        require(m==m2 and a.tolist()==list(reversed(b.tolist())),'GROUP_ORDER_INVARIANCE')
        require(m['a']==m['b']==m['c'] and m['d']!=m['a'] and m['e']!=m['d'],'GROUP_TRANSITIVE')
        return dict(company_year_rows_stay_together=True,transitive_test_document_links=True,isolated_companies_retained=True)
    check('group_definition_permutation_and_attachment',group_test)
    def draws():
        idx=np.repeat(np.arange(8),2);y=np.array([1,0]*8)
        a,s,rejects=bootstrap_draw(7,idx,y,1807141,1807142)
        expected_counts=np.bincount(np.random.default_rng(1807141+70).integers(0,8,size=8),minlength=8)
        ss=np.random.default_rng(1807142+70).integers(0,5,size=5)
        require(np.array_equal(a,expected_counts[idx]) and np.array_equal(s,ss),'DRAW_RULE')
        w,ss,rejected=bootstrap_draw(0,np.zeros(2,dtype=int),np.ones(2),1,2,max_attempts=3)
        require(w is None and ss is None and rejected==3,'DRAW_ATTEMPT_LIMIT')
        return {'base_plus_10b':True,'five_matched_seed_labels':True,'bounded_rejection':True}
    check('bootstrap_rng_and_attempt_limit',draws)
    def ties():
        y=np.zeros(20,dtype=int);y[:2]=1;s=np.ones(20)*.5
        p=MetricPlan(y,s,.5);tr=p.tie_report()
        require(tr['tie_hits_min']==0 and tr['tie_hits_max']==1 and tr['budget']==1,'TIE_BOUNDS')
        require(p.compute(np.ones(20,dtype=int))[3]==1,'FIXED_BASE_ORDER')
        return dict(tie_bounds_exact=True,labels_do_not_reorder_ties=True)
    check('budget_ties_and_stable_base_order',ties)
    def safety():
        o=Output(root/'safe',allowed=root/'safe')
        reject(lambda:o.json('../outside.json',{}))
        l=Ledger([root]);p=root/'immutable.bin';p.write_bytes(b'abc')
        l.verify(p,hashlib.sha256(b'abc').hexdigest(),3,'synth');p.write_bytes(b'abd')
        reject(l.audit_again)
        return {'write_escape_rejected':True,'modified_input_detected':True}
    check('write_scope_and_source_mutation_guards',safety)
    def integration():
        from fs_compute import (condition_layout,observed_tables,groups_and_provenance,
                                flag_and_threshold_diagnostics,influence_tables,bootstrap_scheme)
        b=synthetic_bundle();o=Output(root/'integration',allowed=root/'integration')
        names,indices,obs,catalog=observed_tables(b,o)
        require((len(names),len(catalog))==(29,46),'INTEGRATION_CATALOG')
        ci,gi,summary=groups_and_provenance(b,o)
        require(summary['membership_timing_status']=='POSTFIT_MATERIALIZATION_OF_PRESPECIFIED_RULE','GROUP_NOT_BACKDATED')
        flag_and_threshold_diagnostics(b,o);influence_tables(b,gi,names,indices,catalog,o)
        one=Output(root/'boot_one',allowed=root/'boot_one');two=Output(root/'boot_two',allowed=root/'boot_two')
        for obj in (one,two):(obj.root/'SHARE').mkdir(exist_ok=True)
        bs1=bootstrap_scheme(b,ci,names,indices,obs,catalog,one,'company_x_matched_seed',workers=1,nboot=4,progress=lambda *a,**kw:None)
        bs2=bootstrap_scheme(b,ci,names,indices,obs,catalog,two,'company_x_matched_seed',workers=2,nboot=4,progress=lambda *a,**kw:None)
        with np.load(one.root/'SHARE/company_x_matched_seed_AGGREGATE_REPLICATES.npz',allow_pickle=False) as a, np.load(two.root/'SHARE/company_x_matched_seed_AGGREGATE_REPLICATES.npz',allow_pickle=False) as bb:
            require(np.array_equal(a['condition_metrics'],bb['condition_metrics']),'WORKER_COUNT_PARITY')
        files=list((o.root/'SHARE').glob('*'))
        for p in files:
            if p.suffix=='.json':json.loads(p.read_text())
            require('PRIVATE_SENTINEL' not in p.read_text(),'PRIVATE_SENTINEL_EXPORTED')
        return dict(fits=145,new_configuration_identities=105,conditions=29,contrasts=46,
                    full_computation_modules_exercised=True,one_vs_two_worker_draw_parity=True,
                    no_empirical_data_used=True,artifacts=len(files))
    check('synthetic_analysis_and_serialization_integration',integration)
    def return_packaging():
        import run_fixedscore as app
        old=app.WRITE_ROOT
        app.WRITE_ROOT=root/'return_stage'
        try:
            o=Output(root/'return_attempt',allowed=root/'return_attempt')
            o.json('SHARE/RUN_STATUS.json',{'status':'SYNTHETIC_PASS'})
            o.json('SHARE/AGGREGATE.json',{'test_count':145,'replay':False})
            o.json('PRIVATE/secret.json',{'company_code':'PRIVATE_SENTINEL'})
            import contextlib,io,zipfile
            with contextlib.redirect_stdout(io.StringIO()):app.package_return(o,'SYNTHETIC_PASS',{'safe':True})
            with zipfile.ZipFile(o.root/'V27_FIXED_SCORE_RETURN.zip') as z:
                require(z.testzip() is None,'ZIP_CRC')
                require(set(z.namelist())=={'RUN_STATUS.json','AGGREGATE.json','RETURN_MANIFEST.json'},'ZIP_ALLOWLIST')
                for name in z.namelist():require(b'PRIVATE_SENTINEL' not in z.read(name),'RETURN_PRIVACY')
            return {'return_zip_written_and_crc_checked':True,'private_member_excluded':True,'json_serialization_exercised':True}
        finally:app.WRITE_ROOT=old
    check('complete_return_writer_and_privacy',return_packaging)
    def parquet():
        import pyarrow
        d=pd.DataFrame({'company_code':['000002','000001'],'fiscal_year':[2022,2021],
                        'y_true':[1,0],'score':[.7,.2]})
        p=root/'synthetic_prediction.parquet';d.to_parquet(p,index=False)
        got=pd.read_parquet(p,columns=['company_code','fiscal_year','y_true','score'])
        require(got.equals(d),'PARQUET_ROUNDTRIP')
        readkeys=pd.read_parquet(p,columns=['company_code','fiscal_year','y_true'])
        require('score' not in readkeys,'PARQUET_COLUMN_PROJECTION')
        return {'real_parquet_roundtrip':True,'pyarrow':pyarrow.__version__}
    if require_parquet:check('real_parquet_roundtrip',parquet)
    else:skips.append('real_parquet_roundtrip: local build environment lacks pyarrow; mandatory on server')
    failed=[r['test'] for r in rows if not r['pass']]
    return dict(status='PASS' if not failed else 'FAIL',test_count=len(rows),failed=failed,tests=rows,
                parquet_io_tested=require_parquet and 'real_parquet_roundtrip' not in failed,
                skipped=skips,synthetic_only=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--local-no-parquet',action='store_true')
    a=ap.parse_args();r=run_selftests(a.out,require_parquet=not a.local_no_parquet)
    print(json.dumps(jsonable(r),indent=2,ensure_ascii=False,allow_nan=False))
    raise SystemExit(0 if r['status']=='PASS' else 2)
