#!/usr/bin/env python3
"""Authorized fixed-score postprocessing; no fit, predict or checkpoint replay."""
from __future__ import annotations
import os
# Restrict only this process and its children, never the friend's other jobs.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, fcntl, importlib.metadata, json, platform, shutil, sys, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
from fs_io import (ROOT, WRITE_ROOT, Output, sha256, read_json, load_bundle, jsonable)
from fs_math import ContractError, require
from fs_compute import (groups_and_provenance, execution_deviation, observed_tables,
                        flag_and_threshold_diagnostics, influence_tables, bootstrap_scheme)

HERE=Path(__file__).resolve().parent

def package_verify():
    index=read_json(HERE/'PACKAGE_MANIFEST.json')
    for name,meta in index['files'].items():
        p=HERE/name
        require(p.is_file() and sha256(p)==meta['sha256'] and p.stat().st_size==meta['size_bytes'],'PACKAGE_FILE_IDENTITY',name)
    return {'file_count':len(index['files']),'manifest_sha256':sha256(HERE/'PACKAGE_MANIFEST.json')}

def package_return(out,status,summary):
    """Strict allowlist: nothing under PRIVATE, no score Parquets or checkpoints."""
    p=out.check(out.root/'V27_FIXED_SCORE_RETURN.zip')
    files=sorted(x for x in (out.root/'SHARE').iterdir() if x.is_file())
    allowed={'.json','.csv','.npz','.md'}
    require(all(x.suffix in allowed for x in files),'RETURN_FILE_TYPE')
    members=[]
    for f in files:members.append(dict(name=f.name,sha256=sha256(f),size_bytes=f.stat().st_size))
    man=out.json('SHARE/RETURN_MANIFEST.json',{'status':status,'files':members,'row_level_scores_included':False,
                         'row_membership_or_company_document_identifiers_included':False})
    with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED) as z:
        for f in files+[man]:z.write(f,arcname=f.name)
    with zipfile.ZipFile(p) as z:require(z.testzip() is None,'RETURN_ZIP_CRC')
    info={'status':status,'return_zip':str(p),'return_zip_sha256':sha256(p),'return_zip_bytes':p.stat().st_size,
          'summary':summary,'source_artifacts_written_by_this_program':False}
    out.json('RETURN_READY.json',info)
    # Stable return locations under the same allowed postprocessing root.
    global_out=Output(WRITE_ROOT,allowed=WRITE_ROOT)
    for src,dest in [(p,WRITE_ROOT/'V27_FIXED_SCORE_RETURN.zip'),(out.root/'SHARE/RUN_STATUS.json',WRITE_ROOT/'V27_FIXED_SCORE_STATUS.json')]:
        global_out.check(dest)
        tmp=dest.with_name(dest.name+f'.tmp.{os.getpid()}');global_out.check(tmp)
        shutil.copyfile(src,tmp);os.replace(tmp,dest)
    global_out.json('LATEST_RETURN.json',info)
    print(json.dumps(info,indent=2,ensure_ascii=False,sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,choices=[1,2,3,4],default=2)
    args=ap.parse_args()
    Output(WRITE_ROOT).check(HERE)
    out=Output(args.run_dir)
    require(not (out.root/'RETURN_READY.json').exists(),'COMPLETED_ATTEMPT_REFUSE_OVERWRITE')
    for d in ['SHARE','PRIVATE','TMP']:(out.root/d).mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(out.root/'TMP')
    import tempfile
    tempfile.tempdir=str(out.root/'TMP')
    lockfile=Output(WRITE_ROOT).check(WRITE_ROOT/'EXECUTION.lock')
    with lockfile.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('ANOTHER_FIXED_SCORE_PROCESS_RUNNING')
        modules=[];bundle=None;failure=None;scheme_results=[];deviations=[]
        def complete(name,detail):
            modules.append(dict(module=name,status='COMPLETE',detail=detail))
            out.json('SHARE/MODULE_STATUS.json',modules)
            print('MODULE_COMPLETE '+name,flush=True)
        started=datetime.now(timezone.utc).isoformat()
        try:
            complete('package_integrity',package_verify())
            # Mandatory real-Parquet self-test on the user's existing environment.
            from selftest import run_selftests
            tests=run_selftests(out.root/'TMP/selftests',require_parquet=True)
            out.json('SHARE/EXECUTION_ENV_SELFTESTS.json',tests)
            require(tests['status']=='PASS','LOCAL_SELFTEST_FAILED',tests.get('failed'))
            complete('synthetic_tests',{'checks':tests['test_count'],'actual_parquet_io_tested':True})
            import pyarrow as pa
            pa.set_cpu_count(2);pa.set_io_thread_count(2)
            versions={m:importlib.metadata.version(m) for m in ['numpy','pandas','pyarrow','scikit-learn']}
            expected={'numpy':'2.5.3','pandas':'3.0.5','pyarrow':'25.0.1','scikit-learn':'1.9.1'}
            require(versions==expected and sys.version.split()[0]=='3.12.3','FROZEN_CPU_DEPENDENCIES_CHANGED',versions)
            out.json('SHARE/EXECUTION_ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                          versions=versions,requested_workers=args.workers,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          no_dependency_installation=True,all_thread_pools_requested_single_thread=True))
            print('Binding frozen inputs and all artifacts; no model objects will be loaded.',flush=True)
            bundle=load_bundle(HERE,progress=print)
            complete('input_and_artifact_identity',{'fits':len(bundle.fits),'new_fits':105,'V26_reference_fits':40,'bound_files':len(bundle.ledger.entries)})
            company_idx,component_idx,gs=groups_and_provenance(bundle,out)
            complete('group_rule_and_provenance',{'status':gs['membership_timing_status'],'secondary_allowed':gs['secondary_computation_allowed']})
            if gs['timing_deviation_requires_reporting']:deviations.append('COMPONENT_MAP_TIME_BINDING_REVIEW')
            ex=execution_deviation(bundle,out)
            complete('execution_deviation_static_review',{'status':ex['status'],'replay_performed':False})
            if not ex['stage_e_explicit_strict_algorithms_call_reachable']:deviations.append('STAGE_E_STRICT_ENFORCEMENT_GAP')
            names,indices,observations,catalog=observed_tables(bundle,out)
            complete('fixed_score_points_and_contrasts',{'conditions':len(names),'contrasts':len(catalog),'populations':5})
            flag_and_threshold_diagnostics(bundle,out)
            complete('single_flags_fixed_threshold_and_D_hit_overlap',{'single_flags':3,'models_trained':0})
            influences=influence_tables(bundle,component_idx,names,indices,catalog,out)
            complete('fixed_score_positive_deletion_influence',influences)
            primary=bootstrap_scheme(bundle,company_idx,names,indices,observations,catalog,out,'company_x_matched_seed',workers=args.workers)
            scheme_results.append(primary);complete('primary_company_bootstrap',primary)
            if gs['secondary_computation_allowed']:
                sec=bootstrap_scheme(bundle,component_idx,names,indices,observations,catalog,out,'company_document_x_matched_seed',workers=args.workers)
                scheme_results.append(sec);complete('secondary_component_bootstrap',sec)
            else:
                modules.append({'module':'secondary_component_bootstrap','status':'HOLD','detail':'Existing bound component candidate requires review; no substitute chosen.'})
            complete('end_to_start_source_hash_parity',bundle.ledger.audit_again())
            out.json('SHARE/INPUT_HASH_LEDGER.json',bundle.ledger.public_rows())
            out.json('PRIVATE/INPUT_PATH_LEDGER.json',bundle.ledger.entries)
            enough=all(s['valid_replicates']==2000 for s in scheme_results) and len(scheme_results)==2
            if enough:
                status='COMPLETE_FIXED_SCORE_RESULTS_REVIEW_REQUIRED' if deviations else 'COMPLETE_FIXED_SCORE_RESULTS'
            else:
                status='COMPLETE_WITH_UNAVAILABLE_OR_NON_ESTIMABLE_MODULES'
        except Exception as exc:
            status='HOLD_FIXED_SCORE_POSTPROCESS'
            failure={'code':getattr(exc,'code',type(exc).__name__),
                     'module_after':modules[-1]['module'] if modules else 'startup'}
            # Tracebacks are retained locally; not included in aggregate return archive.
            out.json('PRIVATE/FAILURE_TRACEBACK.json',{'error_type':type(exc).__name__,'traceback':traceback.format_exc()})
            print('HOLD '+failure['code']+'; writing aggregate status return.',flush=True)
        summary=dict(status=status,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
            modules=modules,failed=failure,unresolved_reporting_items=deviations,
            bootstrap_required_per_scheme=2000,bootstrap_schemes_completed=len(scheme_results),
            new_model_fits=0,new_model_inferences=0,checkpoints_replayed=0,
            frozen_predictions_replaced=False,scientific_design_changed=False,
            manuscript_or_author_declaration_changed=False,
            next_action='Review fixed-score estimates, intervals and the disclosed execution qualifications before interpretation.',
            execution_specification='FIXED_SCORE_EXECUTION_SPECIFICATION',private_rows_exported=False)
        out.json('SHARE/RUN_STATUS.json',summary)
        note='''# 返回包范围\n\n本包仅含冻结预测的聚合/逐fit指标、配对重抽区间与逐replicate聚合值、影响范围、来源哈希及模块状态。\n\n不含公司代码、文书号、逐公司score、checkpoint、原始特征或private group membership。\n\n计算完成不等于Stage E推理确定性缺口已解决，也不等于所有协议时间证据齐备。请同时读取EXECUTION_DEVIATION_REVIEW.json和GROUP_PROVENANCE.json。\n\n没有新增或筛选fit、seed、donor world、模型、时间窗、测试群体或阈值。bootstrap使用每fit指标的平均，不是集成平均score的指标。\n'''
        (out.root/'SHARE/README_RETURN_ZH.md').write_text(note,encoding='utf-8')
        package_return(out,status,summary)
        raise SystemExit(0 if not status.startswith('HOLD') else 2)

if __name__=='__main__':main()
