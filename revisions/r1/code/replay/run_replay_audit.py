#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time, traceback, zipfile
from collections import Counter
from pathlib import Path

CYQ=Path('/path/to/private_workspace');ROOT=CYQ/'peerj_141707_v27';BASE=ROOT/'stage_e_replay_audit_v1'
MATRIX=ROOT/'bindings/sensitivity_run_matrix.json';BINDING=ROOT/'bindings/RUN_BINDINGS_v2.json';RUNNER=ROOT/'code/sensitivity_runner.py'
STAGE_T_LOCK=ROOT/'stage_t/ALL_105_LOCKED.json';STAGE_E_MANIFEST=ROOT/'stage_e_freeze_v1/SENSITIVITY_ARTIFACT_IDENTITIES.json';STAGE_E_FREEZE=ROOT/'stage_e_freeze_v1/FREEZE_COMPLETE.json'
EXPECTED={'binding':'cebbfada1bf30ace4f2c95d8fc3b2e60fe9f084294026ea0071213b88a7f4325','matrix':'4d78371987154fa5aee660479c447e30f56608d28616359f04d8c472759f8923','runner':'9e8efa8caf9145b748c01cb6fe86868136afd9f4cd33907fb886668eb854c5b2','stage_t_lock':'72dbfdeeadad9742a3c80a15e1bb96abebe88be67461b53539932c195dcf805b','stage_e_manifest':'4a0086b676a5df8c7a7c085f44dcfd3722ae4d5f8b3a081bae841d907de95c06'}
NEURAL={'MLP','GCN_reverse','GraphSAGE_reverse'}

def sha256(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def import_runner():
 spec=importlib.util.spec_from_file_location('v27_frozen_runner_audit',RUNNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def preflight():
 for p,k in [(BINDING,'binding'),(MATRIX,'matrix'),(RUNNER,'runner'),(STAGE_T_LOCK,'stage_t_lock'),(STAGE_E_MANIFEST,'stage_e_manifest')]:
  if not p.is_file() or sha256(p)!=EXPECTED[k]:raise RuntimeError(f'PREFLIGHT_IDENTITY:{k}:{p}')
 if not STAGE_E_FREEZE.is_file():raise RuntimeError('STAGE_E_FREEZE_COMPLETE_MISSING')
 fc=json.loads(STAGE_E_FREEZE.read_text())
 if fc.get('status')!='LOCKED' or fc.get('fit_count')!=105:raise RuntimeError('STAGE_E_FREEZE_COMPLETE_BAD')
 if fc.get('final_binding_sha256')!=EXPECTED['binding'] or fc.get('run_matrix_sha256')!=EXPECTED['matrix'] or fc.get('production_runner_sha256')!=EXPECTED['runner'] or fc.get('stage_t_global_lock_sha256')!=EXPECTED['stage_t_lock'] or fc.get('artifact_manifest_sha256')!=EXPECTED['stage_e_manifest']:raise RuntimeError('STAGE_E_FREEZE_CHAIN')
 matrix=json.loads(MATRIX.read_text());runs=[r for r in matrix if r['model'] in NEURAL]
 if len(runs)!=55 or Counter(r['model'] for r in runs)!=Counter({'MLP':45,'GCN_reverse':5,'GraphSAGE_reverse':5}):raise RuntimeError('NEURAL_RUN_CONTRACT')
 manifest=json.loads(STAGE_E_MANIFEST.read_text());mids={r['run_id'] for r in manifest['records']}
 if any(r['run_id'] not in mids for r in runs):raise RuntimeError('MANIFEST_NEURAL_COVERAGE')
 return runs

def synthetic_selftest(outdir):
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_WORKSPACE_CONFIG_NOT_STRICT')
 runner=import_runner();res=runner.self_test();res['production_runner_sha256']=sha256(RUNNER);write_json(outdir/'SYNTHETIC_STRICT_SELFTEST.json',res);return res

def write_csv(path,rows):
 if not rows:return
 keys=list(rows[0])
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize(attempt,worker_results,failures,selftest,started):
 perfit=[];poprows=[];graph=[]
 for obj in worker_results:
  sc=obj['score_comparison'];perfit.append({'run_id':obj['run_id'],'family':obj['family'],'model':obj['model'],'modality':obj['modality'],'model_seed':obj['model_seed'],'imputation_seed':obj['imputation_seed'],'score_exact_elementwise':sc['exact_elementwise'],'score_nonzero_difference_count':sc['nonzero_difference_count'],'score_max_abs_diff':sc['max_abs_diff'],'score_mean_abs_diff':sc['mean_abs_diff'],'exact_reproduction_all_checks':obj['exact_reproduction_all_checks'],'decision_equivalent_all_populations':obj['decision_equivalent_all_populations']})
  for p in obj['population_comparisons']:
   poprows.append({'run_id':obj['run_id'],'family':obj['family'],'model':obj['model'],'population':p['population'],'rows':p['rows'],'positive':p['positive'],'budget':p['budget'],'rank_order_identical':p['rank_order_identical'],'top_budget_sequence_identical':p['top_budget_sequence_identical'],'top_budget_set_identical':p['top_budget_set_identical'],'metrics_exact':p['metrics_exact'],'metrics_max_abs_diff':p['metrics_max_abs_diff'],'stored_vs_original_recompute_exact':p['stored_vs_original_recompute_exact'],'stored_vs_original_recompute_max_abs_diff':p['stored_vs_original_recompute_max_abs_diff']})
  if obj.get('graph_input_signatures'):
   for year,g in obj['graph_input_signatures']['years'].items():graph.append({'run_id':obj['run_id'],'model':obj['model'],'year':year,**g})
 write_csv(attempt/'01_REPLAY_PER_FIT.csv',perfit);write_csv(attempt/'02_REPLAY_POPULATION_COMPARISON.csv',poprows);write_csv(attempt/'03_GNN_GRAPH_INPUT_SIGNATURES.csv',graph)
 all_scores_exact=(len(worker_results)==55 and not failures and all(r['score_comparison']['exact_elementwise'] for r in worker_results))
 all_exact=(len(worker_results)==55 and not failures and all(r['exact_reproduction_all_checks'] for r in worker_results));all_decision=(len(worker_results)==55 and not failures and all(r['decision_equivalent_all_populations'] for r in worker_results))
 maxdiff=max((r['score_comparison']['max_abs_diff'] for r in worker_results),default=None);nonzero=sum(1 for r in worker_results if not r['score_comparison']['exact_elementwise'])
 if failures:status='HOLD_STRICT_REPLAY_ERRORS'
 elif all_scores_exact and all_exact:status='PASS_STRICT_REPLAY_EXACT_55_OF_55'
 elif all_decision:status='COMPLETE_STRICT_REPLAY_NUMERIC_DIFFERENCES_DECISION_EQUIVALENT'
 else:status='COMPLETE_STRICT_REPLAY_DIFFERENCES_REVIEW_REQUIRED'
 summary={'component':'V27_STAGE_E_STRICT_REPLAY_AUDIT','status':status,'expected_neural_fits':55,'completed_neural_fits':len(worker_results),'failed_neural_fits':len(failures),'model_counts':dict(Counter(r['model'] for r in worker_results)),'all_scores_exact_elementwise':bool(all_scores_exact),'all_exact_reproduction_checks':bool(all_exact),'all_decision_equivalent':bool(all_decision),'fits_with_any_score_difference':int(nonzero),'max_abs_score_difference_across_fits':maxdiff,'synthetic_strict_selftest_status':selftest.get('status'),'training_performed':False,'new_scientific_fit_performed':False,'checkpoint_replay_performed':True,'original_artifacts_modified':False,'strict_replay_is_post_results_technical_audit':True,'original_stage_e_enforcement_omission_erased':False,'started_at_epoch':started,'finished_at_epoch':time.time(),'failures':failures}
 write_json(attempt/'RUN_STATUS.json',summary);write_json(attempt/'REPLAY_AUDIT_DETAIL.json',{'summary':summary,'per_fit_records':worker_results})
 ledger={'final_binding':{'path':str(BINDING),'sha256':sha256(BINDING)},'run_matrix':{'path':str(MATRIX),'sha256':sha256(MATRIX)},'production_runner':{'path':str(RUNNER),'sha256':sha256(RUNNER)},'stage_t_global_lock':{'path':str(STAGE_T_LOCK),'sha256':sha256(STAGE_T_LOCK)},'stage_e_artifact_manifest':{'path':str(STAGE_E_MANIFEST),'sha256':sha256(STAGE_E_MANIFEST)},'stage_e_freeze_complete':{'path':str(STAGE_E_FREEZE),'sha256':sha256(STAGE_E_FREEZE)}};write_json(attempt/'INPUT_HASH_LEDGER.json',ledger)
 tech={'finding_id':'EXEC-01-REPLAY','original_fact':'The frozen Stage-E code did not explicitly call strict_seed/use_deterministic_algorithms in each inference process.','audit_action':'Post-results technical replay only: load the frozen Stage-T checkpoint and original inputs, explicitly establish the frozen strict deterministic settings, run the exact frozen infer_test path once, and compare against immutable frozen Stage-E predictions.','original_files_overwritten':False,'scientific_retraining':False,'interpretation_rule':'Exact replay supports empirical reproducibility of the frozen predictions in the pinned environment but does not retroactively make the original Stage-E process explicitly strict. Any non-exact replay is preserved and reviewed; replay never replaces the frozen result.'};write_json(attempt/'TECHNICAL_DEVIATION_AND_REPLAY_SCOPE.json',tech)
 readme=("# V27 Stage-E strict replay audit return\n\n"+f"Status: `{status}`\n\n"+"This is a post-results technical reproducibility audit, not a new scientific fit. It never overwrites the frozen Stage-T or Stage-E artifacts.\n\n"+f"- Frozen neural checkpoints replayed: {len(worker_results)}/55\n- Worker failures: {len(failures)}\n- Fits with any score difference: {nonzero}\n- Maximum absolute score difference: {maxdiff}\n- All population rankings/top-budget sets/metrics decision-equivalent: {all_decision}\n\n"+"The original execution omission remains a provenance fact even if strict replay is exact; exact replay would show that the stored outputs are reproduced under the stricter setting in the pinned environment.\n")
 (attempt/'README_RETURN_ZH.md').write_text(readme,encoding='utf-8');return summary

def make_return(attempt):
 include=['RUN_STATUS.json','README_RETURN_ZH.md','INPUT_HASH_LEDGER.json','TECHNICAL_DEVIATION_AND_REPLAY_SCOPE.json','SYNTHETIC_STRICT_SELFTEST.json','01_REPLAY_PER_FIT.csv','02_REPLAY_POPULATION_COMPARISON.csv','03_GNN_GRAPH_INPUT_SIGNATURES.csv','REPLAY_AUDIT_DETAIL.json'];manifest={}
 for n in include:
  p=attempt/n
  if p.is_file():manifest[n]={'sha256':sha256(p),'size_bytes':p.stat().st_size}
 write_json(attempt/'RETURN_MANIFEST.json',manifest);include.append('RETURN_MANIFEST.json');z=attempt/'V27_STAGE_E_REPLAY_RETURN.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
  for n in include:
   p=attempt/n
   if p.is_file():zz.write(p,arcname=n)
 return z

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);ap.add_argument('--worker',type=Path,required=True);args=ap.parse_args();attempt=args.attempt.resolve();base=BASE.resolve()
 if base not in attempt.parents:raise SystemExit('ATTEMPT_OUTSIDE_REPLAY_ROOT')
 attempt.mkdir(parents=True,exist_ok=False);(attempt/'per_run').mkdir();started=time.time();failures=[];worker_results=[]
 try:runs=preflight();selftest=synthetic_selftest(attempt)
 except Exception as e:
  failures.append({'stage':'preflight_or_selftest','error':repr(e),'traceback':traceback.format_exc()});summary=summarize(attempt,[],failures,{'status':'ERROR'},started);z=make_return(attempt);print(json.dumps(summary,indent=2));print('UPLOAD_THIS_ZIP='+str(z));raise SystemExit(2)
 env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';py=Path(sys.executable)
 for i,run in enumerate(runs,1):
  rid=run['run_id'];out=attempt/'per_run'/f'{rid}.json';print(f'[{i}/55] REPLAY {rid}',flush=True)
  cp=subprocess.run([str(py),str(args.worker),'--run-id',rid,'--out',str(out)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(cp.stdout,end='',flush=True)
  if not out.is_file():failures.append({'run_id':rid,'returncode':cp.returncode,'error':'WORKER_OUTPUT_MISSING'});continue
  obj=json.loads(out.read_text())
  if cp.returncode!=0 or obj.get('status')=='ERROR':failures.append({'run_id':rid,'returncode':cp.returncode,'error':obj.get('error'),'worker_output_tail':cp.stdout[-4000:]});continue
  worker_results.append(obj)
 summary=summarize(attempt,worker_results,failures,selftest,started);z=make_return(attempt);canonical=BASE/'V27_STAGE_E_REPLAY_RETURN.zip'
 if canonical.exists():
  print('NOTE_CANONICAL_RETURN_ALREADY_EXISTS='+str(canonical),flush=True);upload=z
 else:
  canonical.write_bytes(z.read_bytes());upload=canonical
 write_json(BASE/'LATEST_COMPLETED.json',{'attempt':str(attempt),'status':summary['status'],'attempt_return_zip':str(z),'canonical_return_zip':str(canonical) if canonical.exists() else None,'canonical_return_sha256':sha256(canonical) if canonical.exists() else None,'upload_this_zip':str(upload)})
 print(json.dumps(summary,indent=2,sort_keys=True));print('UPLOAD_THIS_ZIP='+str(upload))
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
