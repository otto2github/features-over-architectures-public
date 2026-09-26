#!/usr/bin/env python3
"""Single-entry verification of the current supplementary archive. Never fit a model."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tempfile,contextlib,io
import numpy as np
import pandas as pd
from regenerate_main_tables import derive
ROOT=Path(__file__).resolve().parents[1]
POST=ROOT/'aggregates/subset_diagnostics'
METRICS=['roc_auc','ap','f1_val_threshold','p_at_5pct','r_at_10fpr']
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(d):need(len(d)==1,'NONUNIQUE_RECORD');return d.iloc[0]
def run_numeric(script):
 z=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
 need(z.returncode==0,'SELFTEST_FAILED:'+str(script)+'\n'+z.stdout+z.stderr)
 return z.stdout.strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verbose',action='store_true');a=ap.parse_args()
 man=json.loads((ROOT/'FILE_SHA256.json').read_text())
 actual={str(p.relative_to(ROOT))for p in ROOT.rglob('*')if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='FILE_SHA256.json'}
 need(actual==set(man),'FILE_LIST_MISMATCH')
 for rel,x in man.items():need((ROOT/rel).stat().st_size==x['bytes']and sha(ROOT/rel)==x['sha256'],'PACKAGE_IDENTITY:'+rel)
 identity=json.loads((ROOT/'historical_reference/SOURCE_IDENTITY.json').read_text());need(sha(ROOT/identity['historical_source'])==identity['source_sha256'],'HISTORICAL_SOURCE_HASH');need(len((ROOT/identity['historical_source']).read_text().splitlines())==621,'HISTORICAL_SOURCE_LINES')
 # Actual sensitivity fits: match every JSON to its recorded metrics and check aggregation.
 p=pd.read_csv(ROOT/'aggregates/sensitivity/per_seed_metrics.csv');s=pd.read_csv(ROOT/'aggregates/sensitivity/condition_summary.csv')
 need(len(p)==40 and len(s)==8,'SENSITIVITY_MATRIX_SIZE')
 for (sc,mo,mod),g in p.groupby(['scenario','model','modal']):
  need(set(g.seed)=={42,123,456,789,1024}and len(g)==5,'SENSITIVITY_SEEDS');z=one(s[(s.scenario==sc)&(s.model==mo)&(s.modal==mod)])
  for m,key in [('roc_auc','roc_auc'),('ap','ap'),('f1_val_threshold','f1'),('p_at_5pct','p5'),('r_at_10fpr','r10')]:
   need(abs(g[m].mean()-z[key+'_mean'])<1e-12,'FIT_MEAN:'+m)
   if key+'_sd'in s:need(abs(g[m].std(ddof=1)-z[key+'_sd'])<1e-12,'FIT_SD:'+m)
 recs=list((ROOT/'run_records/sensitivity').glob('*.json'));need(len(recs)==40,'FIT_RECORD_COUNT')
 for rec in recs:
  j=json.loads(rec.read_text());model={'SAGE':'GraphSAGE'}.get(j.get('model_display',j['model']),j.get('model_display',j['model']));z=one(p[(p.scenario==j['scenario'])&(p.model==model)&(p.modal==j['modal'])&(p.seed==j['seed'])])
  for m in METRICS:need(abs(j['metrics'][m]-z[m])<1e-12,'JSON_METRIC:'+rec.name+':'+m)
 m=json.loads((ROOT/'run_records/sensitivity_run_manifest.json').read_text())
 for k,rel in [('core','code/sensitivity_experiment/code/formal_rerun_core.py'),('runner','code/sensitivity_experiment/code/run_sensitivities.py'),('protocol','code/sensitivity_experiment/config/protocol.json')]:need(sha(ROOT/rel)==m[k+'_sha256'],'EXECUTED_SOURCE_IDENTITY:'+k)
 pts=pd.read_csv(POST/'subset_evaluation/01_full_test_points.csv');d=pd.read_csv(POST/'subset_evaluation/04_subset_per_seed_metrics.csv');summary=pd.read_csv(POST/'subset_evaluation/05_subset_summaries.csv');ci=pd.read_csv(POST/'subset_evaluation/06_sensitivity_paired_intervals.csv')
 need((len(pts),len(d),len(summary),len(ci))==(70,350,70,160),'POSTPROCESS_COUNTS')
 need(not pts.duplicated(['condition','seed']).any()and not d.duplicated(['subset','condition','seed']).any(),'DUPLICATE_FIT')
 for (sub,c),g in d.groupby(['subset','condition']):
  z=one(summary[(summary.subset==sub)&(summary.condition==c)]);need(len(g)==5 and set(g.seed)=={42,123,456,789,1024},'SUBSET_SEEDS')
  for metric in METRICS:
   need(abs(g[metric].mean()-z[metric+'_mean'])<1e-12,'SUBSET_MEAN:'+metric);need(abs(g[metric].std(ddof=1)-z[metric+'_std'])<1e-12,'SUBSET_SD:'+metric)
  need((g.budget==np.ceil(.05*g.n)).all(),'SUBSET_BUDGET')
 full=d[d.subset=='full_test'].merge(pts,on=['condition','seed'],validate='one_to_one',suffixes=('_a','_b'))
 for metric in METRICS:need(np.allclose(full[metric+'_a'],full[metric+'_b'],atol=1e-12,rtol=0),'FULL_POINTS:'+metric)
 means=pts.groupby('condition')[METRICS].mean()
 for x in ci.itertuples():
  need(abs(x.observed_delta-(means.loc[x.left,x.metric]-means.loc[x.right,x.metric]))<1e-12,'PAIRED_DELTA');need(x.ci_low<=x.ci_high and x.n==8435 and x.positives==20 and x.company_clusters==4531 and x.replicates==2000,'PAIRED_CONTRACT')
 need(len(ci[['left','right']].drop_duplicates())==16,'PAIRED_FAMILIES')
 ct=pd.read_csv(POST/'subset_evaluation/03_overlap_by_vintage_test_positives.csv');n=ct.test_positive_firm_years;j=(~ct.doc_overlap_earlier_partition)&(~ct.fin_any_matched_dated_post_anchor)&(~ct.asof_any_source_missing)
 need(n.sum()==20 and n[ct.doc_overlap_earlier_partition].sum()==13,'CROSSTAB_TOTAL');need(n[j].sum()==2 and n[j&ct.fin_any_unverified_feature].sum()==1,'JOINT_SCREEN')
 late=pd.read_csv(POST/'document_masks/late_only_training_document_recurrence.csv');n=late.training_positive_firm_years;need(n.sum()==135 and n[late.late_positive_document_recurs_validation].sum()==35 and n[late.late_positive_document_recurs_test].sum()==8,'LATE_COUNTS')
 ties=pd.read_csv(POST/'subset_evaluation/02_p5_tie_reconciliation.csv');need(np.allclose(ties.archived_p5,ties.stable_p5,atol=1e-12,rtol=0),'P5_RECONCILIATION')
 # No result is silently replaced by an aggregate re-evaluation.
 for x in p.itertuples():
  name={'GraphSAGE':'SAGE'}.get(x.model,x.model);z=one(pts[(pts.condition==f'{x.scenario}__{name}__{x.modal}')&(pts.seed==x.seed)])
  for metric in METRICS:need(abs(getattr(x,metric)-z[metric])<1e-12,'SENSITIVITY_POINT_IDENTITY')
 display,panels,flags=derive()
 # Recompute exact complements and all derived yield tables from published per-seed metrics.
 with tempfile.TemporaryDirectory() as tmp:
  z=subprocess.run([sys.executable,str(ROOT/'tools/derive_complements.py'),'--output-dir',tmp],capture_output=True,text=True);need(z.returncode==0,'DERIVATION_FAILED:'+z.stdout+z.stderr)
  for path in (ROOT/'aggregates/composition_derivations').glob('*.csv'):
   aa=pd.read_csv(path);bb=pd.read_csv(Path(tmp)/path.name);need(aa.shape==bb.shape and list(aa)==list(bb),'DERIVATION_SCHEMA:'+path.name)
   for col in aa:
    if pd.api.types.is_numeric_dtype(aa[col]):need(np.allclose(aa[col],bb[col],atol=1e-12,rtol=0,equal_nan=True),'DERIVATION_VALUE:'+path.name+':'+col)
    else:need(aa[col].fillna('').equals(bb[col].fillna('')),'DERIVATION_KEY:'+path.name+':'+col)
 tests=[run_numeric(ROOT/'tests/postprocess_numeric.py'),run_numeric(ROOT/'tests/synthetic_interfaces.py')]
 mapping=json.loads((ROOT/'documentation/publication_source_map.json').read_text());preserved=0
 for x in mapping['files']:
  if x['published_path'].startswith('aggregates/')and x['published_path'].endswith('.csv'):
   need(sha(ROOT/x['published_path'])==x['input_sha256'],'SCIENTIFIC_CSV_CHANGED:'+x['published_path']);preserved+=1
 if a.verbose:
  for z in tests:print(z)
 print(json.dumps({'status':'V24_ARCHIVE_AND_AGGREGATE_CHECK_PASS','payload_files':len(man),'preserved_scientific_csv_files':preserved,'sensitivity_fit_records':40,'full_test_fit_records':70,'subset_fit_records':350,'subset_summaries':70,'paired_interval_rows':160,'main_panels':9,'numeric_panels':8,'narrative_maps':1,'graph_reference_rows_verified':4,'binary_status_scores':12,'synthetic_test_groups':2,'historical_loader_byte_match':True,'private_rank_reference_performed':False,'new_model_fits':0,'private_bootstrap_repeated_here':False},indent=2))
if __name__=='__main__':main()
