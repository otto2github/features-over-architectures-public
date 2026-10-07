#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, traceback
from pathlib import Path
import numpy as np
import pandas as pd

CYQ=Path('/path/to/private_workspace')
ROOT=CYQ/'peerj_141707_v27'
RUNNER=ROOT/'code/sensitivity_runner.py'
MATRIX=ROOT/'bindings/sensitivity_run_matrix.json'
BINDING=ROOT/'bindings/RUN_BINDINGS_v2.json'
STAGE_T=ROOT/'stage_t'
STAGE_E=ROOT/'stage_e'
EVAL_MASKS=ROOT/'inputs/case/evaluation_masks_v27.parquet'
STAGE_E_MANIFEST=ROOT/'stage_e_freeze_v1/SENSITIVITY_ARTIFACT_IDENTITIES.json'
STAGE_T_LOCK=ROOT/'stage_t/ALL_105_LOCKED.json'

EXPECTED={
 'binding':'cebbfada1bf30ace4f2c95d8fc3b2e60fe9f084294026ea0071213b88a7f4325',
 'matrix':'4d78371987154fa5aee660479c447e30f56608d28616359f04d8c472759f8923',
 'runner':'9e8efa8caf9145b748c01cb6fe86868136afd9f4cd33907fb886668eb854c5b2',
 'stage_e_manifest':'4a0086b676a5df8c7a7c085f44dcfd3722ae4d5f8b3a081bae841d907de95c06',
 'stage_t_lock':'72dbfdeeadad9742a3c80a15e1bb96abebe88be67461b53539932c195dcf805b',
}
NEURAL={'MLP','GCN_reverse','GraphSAGE_reverse'}
POPS=['full_test','D_frozen','O_frozen','S_frozen','J_frozen']
KEYS=['company_code','fiscal_year']

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json_atomic(p:Path,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,p)

def normalize_code(s):
    x=s.astype('string').str.strip().str.replace(r"^'",'',regex=True)
    x=x.str.replace(r'\.0$','',regex=True).str.replace(r'\.(SZ|SH|BJ)$','',regex=True)
    x=x.str.replace(r'^C:','',regex=True).str.zfill(6)
    if not (x.notna().all() and x.str.fullmatch(r'\d{6}').all()):
        raise RuntimeError('INVALID_COMPANY_CODE')
    return x.astype(str)

def import_runner():
    if sha256(RUNNER)!=EXPECTED['runner']:raise RuntimeError('RUNNER_HASH_MISMATCH')
    spec=importlib.util.spec_from_file_location('v27_frozen_runner',RUNNER)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def digest_array(a)->str:
    a=np.ascontiguousarray(np.asarray(a))
    h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(b'|');h.update(json.dumps(list(a.shape)).encode());h.update(b'|');h.update(a.tobytes())
    return h.hexdigest()

def digest_key_sequence(df:pd.DataFrame,order)->str:
    h=hashlib.sha256()
    for i in np.asarray(order,dtype=int):
        row=df.iloc[int(i)]
        h.update(f"{row['company_code']}|{int(row['fiscal_year'])}\n".encode())
    return h.hexdigest()

def metric_diffs(a:dict,b:dict):
    keys=sorted(set(a)|set(b));maxdiff=0.0;exact=True;details={}
    for k in keys:
        x=a.get(k);y=b.get(k)
        numeric=lambda z:isinstance(z,(int,float,np.integer,np.floating)) and not isinstance(z,bool)
        if x is None or y is None:
            same=(x is None and y is None);exact=exact and same;details[k]={'original':x,'replay':y,'exact':same}
        elif numeric(x) and numeric(y):
            d=abs(float(x)-float(y));maxdiff=max(maxdiff,d);same=(float(x)==float(y));exact=exact and same
            details[k]={'original':float(x),'replay':float(y),'abs_diff':d,'exact':same}
        else:
            same=(x==y);exact=exact and same;details[k]={'original':x,'replay':y,'exact':same}
    return exact,maxdiff,details

def population_compare(runner,orig,replay,masks,threshold,pop):
    base=orig.merge(masks[KEYS+[pop]],on=KEYS,how='left',validate='one_to_one')
    rb=replay.merge(masks[KEYS+[pop]],on=KEYS,how='left',validate='one_to_one')
    a=base[base[pop].astype(bool)][KEYS+['y_true','score']].copy().sort_values(KEYS).reset_index(drop=True)
    b=rb[rb[pop].astype(bool)][KEYS+['y_true','score']].copy().sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS]) or not np.array_equal(a.y_true.to_numpy(),b.y_true.to_numpy()):raise RuntimeError(f'{pop}:POP_IDENTITY')
    oa=runner.stable_order(a.company_code,a.fiscal_year,a.score);ob=runner.stable_order(b.company_code,b.fiscal_year,b.score)
    budget=max(1,int(math.ceil(.05*len(a))))
    rank_a=digest_key_sequence(a,oa);rank_b=digest_key_sequence(b,ob)
    topseq_a=digest_key_sequence(a,oa[:budget]);topseq_b=digest_key_sequence(b,ob[:budget])
    top_set_a=set(zip(a.iloc[oa[:budget]].company_code.astype(str),a.iloc[oa[:budget]].fiscal_year.astype(int)))
    top_set_b=set(zip(b.iloc[ob[:budget]].company_code.astype(str),b.iloc[ob[:budget]].fiscal_year.astype(int)))
    ma=runner.stable_metrics(a,threshold);mb=runner.stable_metrics(b,threshold)
    mex,mxd,md=metric_diffs(ma,mb)
    return {
      'population':pop,'rows':int(len(a)),'positive':int(a.y_true.sum()),'budget':budget,
      'rank_order_identical':rank_a==rank_b,'rank_order_original_sha256':rank_a,'rank_order_replay_sha256':rank_b,
      'top_budget_sequence_identical':topseq_a==topseq_b,'top_budget_set_identical':top_set_a==top_set_b,
      'top_budget_original_sha256':topseq_a,'top_budget_replay_sha256':topseq_b,
      'metrics_exact':mex,'metrics_max_abs_diff':mxd,'metric_details':md,
      'original_metrics':ma,'replay_metrics':mb,
    }

def graph_signatures(runner,run,tout):
    if run['model'] not in {'GCN_reverse','GraphSAGE_reverse'}:return None
    cols=runner.feature_columns(run)
    labels=runner.read_test_labels(include_null=True);feat=runner.read_features_for_run(run,'E')
    d=feat.merge(labels[['company_code','fiscal_year','target']],on=['company_code','fiscal_year'],how='inner',validate='one_to_one')
    if len(d)!=9357:raise RuntimeError(f'GNN_CONTEXT_ROWS:{len(d)}')
    d=d.sort_values(['fiscal_year','company_code']).reset_index(drop=True);d['partition']='test';d['supervise']=d.target.notna()
    z=np.load(tout/'scaler.npz');cache,n_total=runner.graph_year_cache(d,cols,z['mu'],z['sd'],runner.TEST_YEARS)
    edges=pd.read_parquet(runner.EDGES,columns=['src_idx','dst_idx','edge_type','year'])
    out={'n_total_nodes':int(n_total),'years':{}}
    for year in runner.TEST_YEARS:
        c=cache[year];es=edges[edges.year==year]
        raw_ei=np.vstack([es.src_idx.to_numpy(np.int64),es.dst_idx.to_numpy(np.int64)])
        raw_et=np.array([runner.REL_MAP[str(x)] for x in es.edge_type],dtype=np.int64)
        exp_ei=np.concatenate([raw_ei,raw_ei[::-1,:]],axis=1);exp_et=np.concatenate([raw_et,raw_et])
        got_ei=c['edge_index'].cpu().numpy();got_et=c['edge_type'].cpu().numpy()
        if not np.array_equal(got_ei,exp_ei) or not np.array_equal(got_et,exp_et):raise RuntimeError(f'REVERSE_EDGE_FORMULA:{year}')
        out['years'][str(year)]={
          'raw_edge_records':int(raw_ei.shape[1]),'replay_edge_records':int(got_ei.shape[1]),
          'exact_double_count':bool(got_ei.shape[1]==2*raw_ei.shape[1]),'reverse_formula_pass':True,
          'edge_index_sha256':digest_array(got_ei),'edge_type_sha256':digest_array(got_et),
          'company_idx_sha256':digest_array(c['company_idx'].cpu().numpy()),'company_x_sha256':digest_array(c['company_x'].cpu().numpy()),
          'supervised_idx_sha256':digest_array(c['sup_idx'].cpu().numpy()),'supervised_rows':int(len(c['sup_y'])),'supervised_positive':int(c['sup_y'].sum().item()),
        }
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_WORKSPACE_CONFIG_NOT_STRICT')
        for p,k in [(BINDING,'binding'),(MATRIX,'matrix'),(RUNNER,'runner'),(STAGE_E_MANIFEST,'stage_e_manifest'),(STAGE_T_LOCK,'stage_t_lock')]:
            if not p.is_file() or sha256(p)!=EXPECTED[k]:raise RuntimeError(f'IDENTITY:{k}:{p}')
        matrix=json.loads(MATRIX.read_text());matches=[r for r in matrix if r['run_id']==args.run_id]
        if len(matches)!=1:raise RuntimeError('RUN_ID_NOT_UNIQUE')
        run=matches[0]
        if run['model'] not in NEURAL:raise RuntimeError('NON_NEURAL_RUN_FORBIDDEN')
        manifest=json.loads(STAGE_E_MANIFEST.read_text());mrs=[r for r in manifest['records'] if r['run_id']==args.run_id]
        if len(mrs)!=1:raise RuntimeError('MANIFEST_RUN_ID')
        mr=mrs[0];tout=STAGE_T/args.run_id;eout=STAGE_E/args.run_id
        for p in [tout/'STAGE_T_COMPLETE.json',tout/'stage_t_result.json',eout/'STAGE_E_COMPLETE.json',eout/'stage_e_result.json',eout/'test_predictions.parquet']:
            if not p.is_file():raise RuntimeError(f'MISSING:{p}')
        if sha256(eout/'test_predictions.parquet')!=mr['test_predictions_sha256']:raise RuntimeError('ORIGINAL_PRED_HASH')
        if sha256(eout/'stage_e_result.json')!=mr['stage_e_result_sha256']:raise RuntimeError('ORIGINAL_RESULT_HASH')
        if sha256(eout/'STAGE_E_COMPLETE.json')!=mr['stage_e_complete_sha256']:raise RuntimeError('ORIGINAL_COMPLETE_HASH')
        st_done=json.loads((tout/'STAGE_T_COMPLETE.json').read_text())
        lockj=json.loads(STAGE_T_LOCK.read_text());locks=[x for x in lockj.get('entries',[]) if x.get('run_id')==args.run_id]
        if len(locks)!=1:raise RuntimeError('STAGE_T_LOCK_RUN_ENTRY')
        if sha256(tout/'STAGE_T_COMPLETE.json')!=locks[0]['complete_sha256']:raise RuntimeError('STAGE_T_COMPLETE_LOCK_HASH')
        if sha256(tout/'FIT_IDENTITY.json')!=locks[0]['fit_identity_sha256']:raise RuntimeError('STAGE_T_FIT_IDENTITY_LOCK_HASH')
        for name,meta in st_done['artifacts'].items():
            p=tout/name
            if not p.is_file() or sha256(p)!=meta['sha256'] or p.stat().st_size!=meta['size_bytes']:raise RuntimeError(f'STAGE_T_ARTIFACT:{name}')
        tres=json.loads((tout/'stage_t_result.json').read_text());stored=json.loads((eout/'stage_e_result.json').read_text())
        orig=pd.read_parquet(eout/'test_predictions.parquet',columns=['company_code','fiscal_year','y_true','score'])
        orig['company_code']=normalize_code(orig.company_code);orig['fiscal_year']=orig.fiscal_year.astype(int);orig['y_true']=orig.y_true.astype(int)
        orig=orig.sort_values(KEYS).reset_index(drop=True)
        if (len(orig),int(orig.y_true.sum()))!=(8435,20):raise RuntimeError('ORIGINAL_COUNTS')
        runner=import_runner();runner.strict_seed(int(run['model_seed']))
        import torch
        strict_state={'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'deterministic_algorithms_enabled':bool(torch.are_deterministic_algorithms_enabled()),'deterministic_warn_only':bool(torch.is_deterministic_algorithms_warn_only_enabled()),'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if not strict_state['deterministic_algorithms_enabled'] or strict_state['deterministic_warn_only'] or strict_state['cudnn_benchmark'] or not strict_state['cudnn_deterministic']:
            raise RuntimeError(f'STRICT_STATE:{strict_state}')
        replay=runner.infer_test(run,tout,tres);torch.cuda.synchronize()
        replay['company_code']=normalize_code(replay.company_code);replay['fiscal_year']=replay.fiscal_year.astype(int);replay['y_true']=replay.y_true.astype(int)
        replay=replay.sort_values(KEYS).reset_index(drop=True)
        if not orig[KEYS].equals(replay[KEYS]) or not np.array_equal(orig.y_true.to_numpy(),replay.y_true.to_numpy()):raise RuntimeError('REPLAY_KEY_LABEL_IDENTITY')
        oscore=orig.score.to_numpy();rscore=replay.score.to_numpy();diff=np.abs(oscore.astype(np.float64)-rscore.astype(np.float64))
        score_cmp={'original_dtype':str(oscore.dtype),'replay_dtype':str(rscore.dtype),'rows':int(len(orig)),'exact_elementwise':bool(np.array_equal(oscore,rscore)),'nonzero_difference_count':int(np.count_nonzero(diff)),'max_abs_diff':float(diff.max(initial=0.0)),'mean_abs_diff':float(diff.mean()),'original_score_bytes_sha256':digest_array(oscore),'replay_score_bytes_sha256':digest_array(rscore)}
        masks=pd.read_parquet(EVAL_MASKS);masks['company_code']=normalize_code(masks.company_code);masks['fiscal_year']=masks.fiscal_year.astype(int)
        threshold=float(tres['val_threshold']);pops=[]
        for pop in POPS:
            pc=population_compare(runner,orig,replay,masks,threshold,pop);st=stored['metrics'][pop];ex,mx,_=metric_diffs(st,pc['original_metrics'])
            pc['stored_vs_original_recompute_exact']=ex;pc['stored_vs_original_recompute_max_abs_diff']=mx;pops.append(pc)
        graph=graph_signatures(runner,run,tout)
        exact_all=score_cmp['exact_elementwise'] and all(p['rank_order_identical'] and p['top_budget_sequence_identical'] and p['top_budget_set_identical'] and p['metrics_exact'] and p['stored_vs_original_recompute_exact'] for p in pops)
        decision_equiv=all(p['rank_order_identical'] and p['top_budget_set_identical'] and p['metrics_max_abs_diff']<=1e-12 for p in pops)
        obj={'component':'V27_STAGE_E_STRICT_REPLAY_PER_FIT','run_id':args.run_id,'family':run['family'],'model':run['model'],'modality':run['modality'],'model_seed':run['model_seed'],'imputation_seed':run['imputation_seed'],'strict_state':strict_state,'original_test_predictions_sha256':mr['test_predictions_sha256'],'score_comparison':score_cmp,'population_comparisons':pops,'graph_input_signatures':graph,'exact_reproduction_all_checks':bool(exact_all),'decision_equivalent_all_populations':bool(decision_equiv),'checkpoint_replay_performed':True,'training_performed':False,'original_artifacts_modified':False}
        write_json_atomic(args.out,obj);print(f"REPLAY_OK {args.run_id} exact={exact_all} maxdiff={score_cmp['max_abs_diff']:.9g}")
    except Exception as e:
        fail={'component':'V27_STAGE_E_STRICT_REPLAY_PER_FIT','run_id':args.run_id,'status':'ERROR','error':repr(e),'traceback':traceback.format_exc(),'training_performed':False,'original_artifacts_modified':False}
        write_json_atomic(args.out,fail);print(json.dumps(fail,indent=2));raise
if __name__=='__main__':main()
