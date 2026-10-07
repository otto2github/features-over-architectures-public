#!/usr/bin/env python3
"""20 prescribed source-bound fits; global checkpoint lock before test inference."""
from __future__ import annotations
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
from e7_runtime_common import KIT,require,safe_path,sha,verify_package,verify_records,write_json,zip_exact
from e7_cache_adapter import strict_seed,load_train_validation_df,validate_supervision,check_scaler,make_cache,ValidationObserver

BOUNDARY=Path('/path/to/private_workspace')
RETURN_FILES=['EXECUTION_CHECKS.json','RESULTS.json','PER_SEED_METRICS.csv','GROUP_MEANS.csv',
              'PAIRED_CONTRASTS.json','ALL_20_LOCKED.json','INPUT_IDENTITIES.json','ENVIRONMENT.json',
              'RUN_MATRIX.json','RUNNER_IDENTITIES.json','README_RETURN.txt']


def configure_paths():
    safe_path(KIT,BOUNDARY)
    os.environ['E7_WRITE_BOUNDARY']=str(BOUNDARY)
    for name,relative in [('TMPDIR','runtime/tmp'),('XDG_CACHE_HOME','runtime/cache'),
             ('CUDA_CACHE_PATH','runtime/cache/cuda'),('TORCH_HOME','runtime/cache/torch'),
             ('TORCH_EXTENSIONS_DIR','runtime/cache/torch_extensions'),('TRITON_CACHE_DIR','runtime/cache/triton')]:
        path=safe_path(KIT/relative,BOUNDARY);path.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.environ[name]=str(path)
    tempfile.tempdir=os.environ['TMPDIR']
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def status(state,**extra):
    write_json(KIT/'JOB_STATUS.json',{'status':state,'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                  'worker_pid':os.getpid(),**extra})
    print('JOB_STATUS '+state+' '+json.dumps(extra,ensure_ascii=False),flush=True)


def validate_matrix(matrix):
    expected={(m,c,s) for m in ['GCN','SAGE'] for c in ['control','E7'] for s in [42,123,456,789,1024]}
    require(len(matrix)==20 and len({x['run_id'] for x in matrix})==20,'RUN_MATRIX_NOT_20_UNIQUE_FITS')
    require({(r['model'],r['condition'],r['seed']) for r in matrix}==expected,'RUN_MATRIX_SCOPE_MISMATCH')
    for row in matrix:
        require(row['run_id']==f'{row["model"]}__M11__{row["condition"]}__seed{row["seed"]}',
                'RUN_ID_MISMATCH')


def resolve_evaluation_mask(record):
    """If the archived locator moved, accept only exact pinned bytes under the private workspace."""
    expected=Path(record['path'])
    if expected.is_file():
        verify_records([record])
        return dict(record)
    matches=[]
    for candidate in sorted(BOUNDARY.rglob(expected.name)):
        if candidate.is_file() and not any(p.is_symlink() for p in [candidate,*candidate.parents]):
            if candidate.stat().st_size==record['size_bytes'] and sha(candidate)==record['sha256']:
                matches.append(candidate)
    require(matches,'BOUND_EVALUATION_MASK_MISSING; no exact-hash copy found under /path/to/private_workspace/')
    resolved={**record,'path':str(matches[0])}
    print('EVALUATION_MASK_EXACT_HASH_LOCATOR '+str(matches[0]),flush=True)
    return resolved


def verify_environment(reference,operator_reference):
    import torch
    import torch_geometric.backend
    import torch_geometric.typing
    actual={}
    for name,expected in reference['versions'].items():
        try:value=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:value=None
        require(value==expected,'ENVIRONMENT_VERSION_CHANGED:'+name)
        actual[name]=value
    require(sys.version==reference['python'] and sys.executable==reference['executable'],'PYTHON_RUNTIME_CHANGED')
    require(torch.cuda.is_available(),'CUDA_UNAVAILABLE_NO_FALLBACK')
    require(torch.cuda.get_device_name(0)==reference['gpu_name']
            and torch.cuda.get_device_properties(0).total_memory==reference['gpu_total_memory'],'CUDA_DEVICE_CHANGED')
    require(torch.version.cuda==reference['torch_cuda']
            and torch.backends.cudnn.version()==reference['cudnn_version'],'CUDA_OR_CUDNN_CHANGED')
    require(torch.backends.cuda.matmul.allow_tf32==reference['cuda_matmul_allow_tf32']
            and torch.backends.cudnn.allow_tf32==reference['cudnn_allow_tf32'],'TF32_FLAGS_CHANGED')
    flags={k:bool(getattr(torch_geometric.typing,k,False)) for k in reference['optional_backend_flags']}
    require(flags==reference['optional_backend_flags'],'OPTIONAL_PYG_BACKEND_CHANGED')
    torch_geometric.backend.use_segment_matmul=False
    from torch_geometric.nn import GCNConv,SAGEConv
    for cls in [GCNConv,SAGEConv]:
        require(sha(Path(inspect.getsourcefile(cls)))==operator_reference['operator_source_identities'][cls.__name__]['sha256'],
                'GRAPH_OPERATOR_SOURCE_CHANGED:'+cls.__name__)
    strict_seed(42)
    return {**reference,'versions':actual,'strict_determinism':True,'warn_only':False,
            'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'segment_matmul_effective':False}


def load_inputs():
    import numpy as np
    import pandas as pd
    verify_package()
    binding=json.loads((KIT/'INPUT_BINDINGS.json').read_text())
    require(binding['training_authorized'] is True,'TRAINING_AUTHORIZATION_MISSING')
    binding['evaluation_masks']=resolve_evaluation_mask(binding['evaluation_masks'])
    matrix=json.loads((KIT/'RUN_MATRIX.json').read_text());validate_matrix(matrix)
    prep=Path(binding['graph_prep_kit'])
    safe_path(prep,BOUNDARY)
    require(sha(prep/'E7_PREP_RETURN.zip')==binding['graph_prep_return_sha256'],'PREP_RETURN_CHANGED')
    latest=json.loads((prep/'LATEST_PRIVATE_RUN.json').read_text())
    require(latest['status']=='GRAPH_PREP_COMPLETE_TRAINING_PROTOCOL_PENDING','LATEST_PREP_RUN_NOT_SUCCESSFUL')
    run=safe_path(Path(latest['run_dir']),BOUNDARY)
    require(run.is_relative_to(prep/'runs'),'PRIVATE_PREP_RUN_LOCATOR_REJECTED')
    ledger=list(binding['baseline_inputs'])+[binding['evaluation_masks']]
    for name,rec in binding['private_e7_artifacts'].items():
        ledger.append({'path':str(safe_path(run/name,BOUNDARY)),**rec})
    verify_records(ledger)
    environment=verify_environment(json.loads((KIT/'ENVIRONMENT_REFERENCE.json').read_text()),
                                   json.loads((KIT/'OPERATOR_CHECKS_REFERENCE.json').read_text()))
    spec=importlib.util.spec_from_file_location('e7_original_core',KIT/'formal_rerun_core_REFERENCE.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    require(sha(KIT/'formal_rerun_core_REFERENCE.py')==binding['baseline_inputs'][-1]['sha256'],
            'PACKAGED_ORIGINAL_CORE_CHANGED')
    e7=pd.read_parquet(run/'PRIVATE_e7_forward_edges.parquet')
    support=pd.read_parquet(run/'PRIVATE_e7_support_index.parquet')
    require(len(e7)==binding['expected_forward_e7_edges'] and list(e7.columns)==['src_idx','dst_idx','edge_type','year'],
            'PRIVATE_E7_EDGE_SCHEMA_OR_COUNT_CHANGED')
    require(len(support)==binding['expected_n_support'] and support.node_id.str.startswith('E7R:').all()
            and support.node_id.is_unique and support.node_idx.is_unique,'E7_SUPPORT_INDEX_CHANGED')
    require(np.array_equal(support.node_idx.to_numpy(),np.arange(binding['expected_n_original'],binding['expected_n_extended'])),
            'SUPPORT_INDICES_NOT_FROZEN_CONTIGUOUS')
    require((e7.src_idx>=0).all() and (e7.src_idx<binding['expected_n_original']).all()
            and (e7.dst_idx>=binding['expected_n_original']).all() and (e7.dst_idx<binding['expected_n_extended']).all()
            and e7.edge_type.eq('E7_RPT_RECORD').all() and not e7.duplicated(['year','src_idx','dst_idx']).any(),
            'PRIVATE_E7_EDGE_VALUES_CHANGED')
    data=Path(binding['baseline_root'])/'data'
    train=load_train_validation_df(data)
    columns=json.loads((KIT/'M11_COLUMN_ORDER.json').read_text())['columns']
    validate_supervision(core,train)
    mu,sd=check_scaler(core,train,binding,columns)
    context={'binding_file_sha256':sha(KIT/'INPUT_BINDINGS.json'),
             'protocol_sha256':sha(KIT/'E7_TRAINING_PROTOCOL_v1.md'),'matrix_sha256':sha(KIT/'RUN_MATRIX.json'),
             'package_manifest_sha256':sha(KIT/'PACKAGE_SHA256.json'),
             'source_ledger':ledger,'scaler_mean_sha256':hashlib.sha256(mu.tobytes()).hexdigest(),
             'scaler_std_sha256':hashlib.sha256(sd.tobytes()).hexdigest()}
    frozen=KIT/'FROZEN_RUNTIME_BINDING.json'
    if frozen.exists():require(json.loads(frozen.read_text())==context,'RESUME_RUNTIME_BINDING_CHANGED')
    else:write_json(frozen,context)
    return binding,matrix,core,train,e7,context,environment


def file_ledger(folder,names):
    return {name:{'sha256':sha(folder/name),'size_bytes':(folder/name).stat().st_size} for name in names}


def verify_completed(folder,context,stage):
    require(folder.is_dir() and not folder.is_symlink(),'STAGE_OUTPUT_PATH_REJECTED')
    done=json.loads((folder/(stage+'_COMPLETE.json')).read_text())
    require(done['status']=='LOCKED' and done['context']==context,'COMPLETED_STAGE_BINDING_CHANGED')
    for name,rec in done['artifacts'].items():
        path=safe_path(folder/name,BOUNDARY)
        require(path.is_file() and path.stat().st_size==rec['size_bytes'] and sha(path)==rec['sha256'],
                'LOCKED_ARTIFACT_CHANGED:'+name)
    return done


def run_fit(row,core,train,e7,binding,context):
    import numpy as np
    import torch
    out=safe_path(KIT/'stage_t'/row['run_id'],BOUNDARY)
    if out.exists():
        verify_completed(out,context,'STAGE_T')
        print('REUSE_LOCKED_FIT '+row['run_id'],flush=True)
        return
    failures=list((KIT/'stage_t_failures').glob(row['run_id']+'__*')) if (KIT/'stage_t_failures').exists() else []
    require(not failures,'FAILED_FIT_REQUIRES_TECHNICAL_REVIEW:'+row['run_id'])
    pending=list((KIT/'stage_t').glob('.attempt_'+row['run_id']+'_*')) if (KIT/'stage_t').exists() else []
    require(not pending,'INTERRUPTED_FIT_REQUIRES_TECHNICAL_REVIEW:'+row['run_id'])
    temporary=safe_path(out.with_name('.attempt_'+row['run_id']+'_'+str(os.getpid())),BOUNDARY)
    temporary.mkdir(parents=True,mode=0o700)
    original_seed,original_cache,original_eval=core.set_seed,core.prepare_graph_cache,core.eval_graph
    observer=ValidationObserver(original_eval,core.roc_auc_score,core.VAL_YEARS,row['run_id'])
    core.set_seed=strict_seed
    core.prepare_graph_cache=make_cache(original_cache,row['condition'],e7,binding,core.TRAIN_YEARS+core.VAL_YEARS)
    core.eval_graph=observer
    started=time.monotonic()
    try:
        args=SimpleNamespace(device='cuda',hidden_dim=64,lr=5e-4,epochs=100,patience=10,no_test=True)
        # The frozen training loop is called verbatim, not reimplemented.
        result=core.run_gnn(train,row['model'],'M11',row['seed'],args,temporary,Path(binding['baseline_root'])/'data')
        require(result['test_evaluated'] is False and observer.last is not None,'TRAINING_STAGE_TEST_READ')
        require(abs(observer.history[-1]-result['best_val_auc'])<=1e-12,'BEST_CHECKPOINT_VALIDATION_REPLAY')
        require(max(observer.history[:-1])<=result['best_val_auc']+1e-8,'CHECKPOINT_SELECTION_MISMATCH')
        require(1<=result['best_epoch']<=len(observer.history)-1<=100,'EPOCH_COUNT_MISMATCH')
        y,s,keys=observer.last
        require(core.f1_val_threshold(y,s)==result['val_threshold'],'VALIDATION_THRESHOLD_MISMATCH')
        state={k:v.detach().cpu().clone() for k,v in observer.model.state_dict().items()}
        torch.save(state,temporary/'checkpoint.pt')
        predictions=keys.rename(columns={'year':'fiscal_year'}).copy()
        predictions['y_true']=y;predictions['score']=s
        predictions.to_parquet(temporary/'PRIVATE_validation_predictions.parquet',index=False)
        mu,sd=core.fit_scaler(train,core.select_cols(train,'M11'))
        np.savez(temporary/'scaler.npz',mu=mu,sd=sd)
        result.update({**row,'epochs_max':100,'patience':10,'hidden_dim':64,'lr':5e-4,
            'dropout':0.3,'optimizer':'Adam','strict_determinism':True,'warn_only':False,
            'train_context_rows':32291,'train_label_positive':254,'validation_label_positive':40,
            'common_node_count':binding['expected_n_extended'],'runtime_seconds':time.monotonic()-started,
            'original_training_loop_sha256':sha(KIT/'formal_rerun_core_REFERENCE.py')})
        write_json(temporary/'training_result.json',result)
        write_json(temporary/'validation_history.json',{'epoch_validation_auc':observer.history[:-1],
                         'restored_best_checkpoint_validation_auc':observer.history[-1]})
        artifacts=file_ledger(temporary,['checkpoint.pt','PRIVATE_validation_predictions.parquet','scaler.npz',
                                       'training_result.json','validation_history.json'])
        write_json(temporary/'STAGE_T_COMPLETE.json',{'status':'LOCKED','run_id':row['run_id'],'context':context,
                       'artifacts':artifacts,'test_evaluated':False})
        os.replace(temporary,out)
        print(f'FIT_LOCKED {row["run_id"]} best_epoch={result["best_epoch"]} seconds={result["runtime_seconds"]:.1f}',flush=True)
    except Exception as exc:
        write_json(temporary/'FAILURE.json',{'run_id':row['run_id'],'type':type(exc).__name__,
                         'message':str(exc),'test_evaluated':False})
        failed=safe_path(KIT/'stage_t_failures'/(row['run_id']+'__'+str(int(time.time()))+'_'+str(os.getpid())),BOUNDARY)
        failed.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.replace(temporary,failed)
        raise
    finally:
        core.set_seed,core.prepare_graph_cache,core.eval_graph=original_seed,original_cache,original_eval
        observer.model=None
        torch.cuda.empty_cache()


def lock_all(matrix,context):
    locks={}
    for row in matrix:
        folder=KIT/'stage_t'/row['run_id']
        verify_completed(folder,context,'STAGE_T')
        locks[row['run_id']]={'complete_sha256':sha(folder/'STAGE_T_COMPLETE.json')}
    value={'status':'ALL_20_LOCKED','fit_count':20,'context':context,'runs':locks,
           'test_performance_evaluated_before_lock':False}
    path=KIT/'ALL_20_LOCKED.json'
    if path.exists():require(json.loads(path.read_text())==value,'GLOBAL_CHECKPOINT_LOCK_CHANGED')
    else:write_json(path,value)
    return value


def verify_all_lock(matrix,context):
    require((KIT/'ALL_20_LOCKED.json').is_file(),'ALL_20_LOCK_REQUIRED_BEFORE_TEST')
    expected=json.loads((KIT/'ALL_20_LOCKED.json').read_text())
    require(expected['status']=='ALL_20_LOCKED' and expected['fit_count']==20
            and expected['context']==context,'GLOBAL_CHECKPOINT_LOCK_INVALID')
    require(set(expected['runs'])=={r['run_id'] for r in matrix},'GLOBAL_CHECKPOINT_RUN_SET_CHANGED')
    for row in matrix:
        p=KIT/'stage_t'/row['run_id']
        verify_completed(p,context,'STAGE_T')
        require(sha(p/'STAGE_T_COMPLETE.json')==expected['runs'][row['run_id']]['complete_sha256'],
                'GLOBAL_LOCK_MEMBER_CHANGED')


def run_evaluation(row,core,test_cache,binding,context,masks):
    # Protect this entry point as well as the main orchestration path.
    matrix=json.loads((KIT/'RUN_MATRIX.json').read_text())
    validate_matrix(matrix)
    verify_all_lock(matrix,context)
    import numpy as np
    import torch
    from e7_statistics import align_masks,point_metrics
    out=safe_path(KIT/'stage_e'/row['run_id'],BOUNDARY)
    if out.exists():
        verify_completed(out,context,'STAGE_E');return
    temporary=safe_path(out.with_name('.attempt_'+row['run_id']+'_'+str(os.getpid())),BOUNDARY)
    require(not temporary.exists(),'EVALUATION_ATTEMPT_EXISTS')
    temporary.mkdir(parents=True,mode=0o700)
    tout=KIT/'stage_t'/row['run_id']
    training=json.loads((tout/'training_result.json').read_text())
    strict_seed(row['seed'])
    model=core.make_model(row['model'],129,64,5)
    state=torch.load(tout/'checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state);model.cuda().eval()
    cache,n,d=test_cache[row['condition']]
    y,s,keys=core.eval_graph(model,cache,core.TEST_YEARS,n,d,torch.device('cuda'))
    require(len(y)==8435 and int(y.sum())==20 and np.isfinite(s).all()
            and ((s>=0)&(s<=1)).all(),'TEST_SCORE_OR_LABEL_COUNTS_INVALID')
    pred=keys.rename(columns={'year':'fiscal_year'}).copy()
    pred['y_true']=y;pred['score']=s
    joined=align_masks(pred,masks,binding['population_contract'])
    threshold=float(training['val_threshold'])
    metrics={pop:point_metrics(joined[joined[pop]],threshold) for pop in binding['population_contract']}
    require(abs(20*metrics['full_test']['roc_auc']-7*metrics['D_frozen']['roc_auc']-13*metrics['O_frozen']['roc_auc'])<=1e-10,
            'FULL_D_O_AUC_IDENTITY')
    top=joined.iloc[np.argsort(-joined.score.to_numpy(),kind='mergesort')[:422]]
    attribution={'budget':422,'full_positive_hits':int(top.y_true.sum()),
                 'D_positive_hits':int((top.y_true.eq(1)&top.D_positive).sum()),
                 'O_positive_hits':int((top.y_true.eq(1)&top.O_positive).sum())}
    require(attribution['full_positive_hits']==attribution['D_positive_hits']+attribution['O_positive_hits'],
            'FULL_BUDGET_ATTRIBUTION_IDENTITY')
    pred.to_parquet(temporary/'PRIVATE_test_predictions.parquet',index=False)
    write_json(temporary/'evaluation_result.json',{**row,'validation_threshold':threshold,
                    'metrics':metrics,'full_budget_attribution':attribution,
                    'test_evaluated_after_all_20_locked':True})
    artifacts=file_ledger(temporary,['PRIVATE_test_predictions.parquet','evaluation_result.json'])
    write_json(temporary/'STAGE_E_COMPLETE.json',{'status':'LOCKED','context':context,'run_id':row['run_id'],'artifacts':artifacts})
    os.replace(temporary,out)
    print(f'TEST_EVALUATED {row["run_id"]} Full_AUC={metrics["full_test"]["roc_auc"]:.6f}',flush=True)
    del model,state
    torch.cuda.empty_cache()


def analyze_and_return(matrix,binding,context,environment,started):
    import numpy as np
    import pandas as pd
    from e7_statistics import summarize
    rows,payload,keys,attributions=[],{},None,[]
    for row in matrix:
        out=KIT/'stage_e'/row['run_id'];verify_completed(out,context,'STAGE_E')
        result=json.loads((out/'evaluation_result.json').read_text())
        attributions.append({**row,**result['full_budget_attribution']})
        for pop,met in result['metrics'].items():
            rows.append({**row,'population':pop,**met})
        pred=pd.read_parquet(out/'PRIVATE_test_predictions.parquet').sort_values(['company_code','fiscal_year'],kind='stable').reset_index(drop=True)
        key=pred[['company_code','fiscal_year','y_true']]
        if keys is None:keys=key
        else:require(key.equals(keys),'PAIRED_TEST_KEY_OR_LABEL_ALIGNMENT')
        payload[(row['model'],row['condition'],row['seed'])]=(pred.y_true.to_numpy(int),pred.score.to_numpy(float),result['validation_threshold'])
    summary,draws=summarize(rows,payload,keys,binding['bootstrap'])
    return_dir=safe_path(KIT/'return',BOUNDARY);return_dir.mkdir(exist_ok=True,mode=0o700)
    for name in RETURN_FILES:
        safe_path(return_dir/name,BOUNDARY)
    pd.DataFrame(rows).to_csv(return_dir/'PER_SEED_METRICS.csv',index=False)
    pd.DataFrame(summary['group_means']).to_csv(return_dir/'GROUP_MEANS.csv',index=False)
    write_json(return_dir/'PAIRED_CONTRASTS.json',{'contrast':'E7 minus contemporaneous control',
                'values':summary['paired_contrasts'],'bootstrap':summary['bootstrap']})
    np.save(safe_path(KIT/'PRIVATE_full_bootstrap_differences.npy',BOUNDARY),draws)
    trained=[json.loads((KIT/'stage_t'/r['run_id']/'training_result.json').read_text()) for r in matrix]
    results={'status':'E7_20_FITS_AND_PAIRED_ANALYSIS_COMPLETE','fit_count':20,
             'models':['GCN','SAGE'],'seeds':[42,123,456,789,1024],'M11_dimensions':129,
             'group_means':summary['group_means'],'paired_contrasts':summary['paired_contrasts'],
             'bootstrap':summary['bootstrap'],'training_summaries':trained,
             'full_budget_D_O_attributions':attributions,
             'graph_scope':'Conservative RPT-derived ID/name record incidence; global legal-entity ID semantics unverified',
             'prescribed_fit_count_completed':True,'performance_used_to_select_graph_or_seeds':False,
             'small_subsets_descriptive_only':True,'runtime_seconds':time.monotonic()-started}
    write_json(return_dir/'RESULTS.json',results)
    checks={'status':'PASS','all_fits':20,'validation_only_checkpoint_and_threshold_selection':True,
      'global_checkpoint_lock_before_any_test_performance':True,'strict_determinism':True,'warn_only':False,
      'same_extended_node_universe_both_conditions':366695,'M11_dimensions':129,
      'full_D_O_auc_identity_all_20':True,'full_budget_D_O_attribution_identity_all_20':True,
      'source_rechecked_after_runs':True,'test_population_hash':binding['evaluation_masks']['sha256'],
      'raw_names_ids_labels_scores_checkpoints_in_return':False,'context':context}
    write_json(return_dir/'EXECUTION_CHECKS.json',checks)
    write_json(return_dir/'ALL_20_LOCKED.json',json.loads((KIT/'ALL_20_LOCKED.json').read_text()))
    write_json(return_dir/'INPUT_IDENTITIES.json',context['source_ledger'])
    write_json(return_dir/'ENVIRONMENT.json',environment)
    write_json(return_dir/'RUN_MATRIX.json',matrix)
    manifest=json.loads((KIT/'PACKAGE_SHA256.json').read_text())
    write_json(return_dir/'RUNNER_IDENTITIES.json',{name:{'sha256':sha(KIT/name),'size_bytes':(KIT/name).stat().st_size}
              for name in [*manifest['files'],'PACKAGE_SHA256.json']})
    (return_dir/'README_RETURN.txt').write_text(
      'Aggregate author audit of the prescribed 20 E7 supplemental fits; no raw predictions or model weights.\n'
      'All checkpoints and validation thresholds were locked before any test evaluation.\n'
      'Full-population paired contrasts use shared company and seed draws; D/O/S/J are descriptive.\n'
      'This post-results extension tests conservative RPT ID/name record incidence, not a validated legal-entity transaction graph.\n'
      'Do not interpret an interval containing zero as equivalence or evidence of universal RPT performance.\n',encoding='utf-8')
    verify_package()
    verify_records(context['source_ledger'])
    verify_all_lock(matrix,context)
    pending=safe_path(KIT/('E7_TRAIN_RETURN.pending.'+str(os.getpid())+'.zip'),BOUNDARY)
    zip_exact(pending,return_dir,RETURN_FILES)
    return pending


def main():
    configure_paths()
    # Exclusive local worker lock; no overlapping jobs, duplicate fits or race to test evaluation.
    with safe_path(KIT/'WORKER.lock',BOUNDARY).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('ANOTHER_E7_WORKER_IS_RUNNING')
        started=time.monotonic()
        try:
            status('CHECKING_INPUT_BINDINGS',fits_locked=0,total_fits=20)
            binding,matrix,core,train,e7,context,environment=load_inputs()
            if (KIT/'E7_TRAIN_RETURN.zip').exists():
                verify_all_lock(matrix,context)
                completed=json.loads((KIT/'RETURN_COMPLETE.json').read_text())
                require(completed['context']==context and completed['sha256']==sha(KIT/'E7_TRAIN_RETURN.zip'),
                        'COMPLETED_RETURN_CONTEXT_OR_HASH_CHANGED')
                status('COMPLETE',fits_locked=20,return_zip=str(KIT/'E7_TRAIN_RETURN.zip'));return
            for index,row in enumerate(matrix):
                status('TRAINING',current_run=row['run_id'],fits_locked=index,total_fits=20)
                run_fit(row,core,train,e7,binding,context)
            verify_records(context['source_ledger'])
            verify_package()
            lock_all(matrix,context)
            status('ALL_20_CHECKPOINTS_LOCKED',fits_locked=20,test_evaluated=False)
            verify_all_lock(matrix,context)
            # First read of test labels and construction of test-score inference happens here.
            joined=core.load_joined(Path(binding['baseline_root'])/'data','label_v1_strict_ab_primary')
            check_scaler(core,joined,binding,json.loads((KIT/'M11_COLUMN_ORDER.json').read_text())['columns'])
            test_cache={condition:make_cache(core.prepare_graph_cache,condition,e7,binding,core.TEST_YEARS)(
                              joined,'M11',Path(binding['baseline_root'])/'data') for condition in ['control','E7']}
            import pandas as pd
            masks=pd.read_parquet(binding['evaluation_masks']['path'])
            for row in matrix:
                status('TEST_EVALUATION',current_run=row['run_id'],fits_locked=20)
                run_evaluation(row,core,test_cache,binding,context,masks)
            verify_records(context['source_ledger'])
            verify_all_lock(matrix,context)
            status('PAIRED_BOOTSTRAP',fits_locked=20,test_fits_evaluated=20)
            pending=analyze_and_return(matrix,binding,context,environment,started)
            verify_records(context['source_ledger'])
            verify_package()
            verify_all_lock(matrix,context)
            target=safe_path(KIT/'E7_TRAIN_RETURN.zip',BOUNDARY)
            require(not target.exists(),'COMPLETED_RETURN_ZIP_ALREADY_EXISTS')
            os.replace(pending,target)
            write_json(KIT/'RETURN_COMPLETE.json',{'status':'COMPLETE','context':context,'sha256':sha(target)})
            status('COMPLETE',fits_locked=20,test_fits_evaluated=20,return_zip=str(KIT/'E7_TRAIN_RETURN.zip'))
        except Exception as exc:
            status('FAILED',exception_type=type(exc).__name__,message=str(exc))
            raise


if __name__=='__main__':
    main()
