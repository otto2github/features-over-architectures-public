from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True,type=Path);args=ap.parse_args()
R=Path(__file__).resolve().parents[1];P=R/'aggregates/subset_diagnostics/subset_evaluation';O=args.output_dir;O.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(P/'04_subset_per_seed_metrics.csv'); s=pd.read_csv(P/'05_subset_summaries.csv')
print(d.subset.unique());print(d.condition.unique())
print(pd.read_csv(P/'02_p5_tie_reconciliation.csv').columns.tolist())
print(s.columns.tolist())
# Named screens retrieved from actual artifact, not assumed strings.
for sub,g in d.groupby('subset'):print(sub,g[['n','positive','negative','budget']].drop_duplicates().to_dict('records'))
FULL='full_test';D='positive_no_detected_earlier_doc_vs_all_negatives';S='source_screen_no_detected_exposure_or_absence';J='intersection_doc_and_source_screen'
keys=['condition','seed']; idx=d.set_index(['subset']+keys)
# Recover only pairwise AUC statistics; no complement AP or retrieval is inferred.
rr=[]
for (cond,seed),row in d[d.subset==FULL].set_index(keys).iterrows():
 for tag,whole,part,np_whole,np_part in [('O',FULL,D,20,7),('S_overlap',S,J,6,2)]:
  a=idx.loc[(whole,cond,seed)];b=idx.loc[(part,cond,seed)]
  assert a.negative==b.negative
  value=(np_whole*a.roc_auc-np_part*b.roc_auc)/(np_whole-np_part)
  assert 0<=value<=1
  rr.append(dict(population=tag,condition=cond,seed=int(seed),n=int(a.negative+np_whole-np_part),positive=np_whole-np_part,negative=int(a.negative),budget=int(np.ceil(.05*(a.negative+np_whole-np_part))),roc_auc=float(value),reference_subset=part,reference_auc=float(b.roc_auc),overlap_minus_reference=float(value-b.roc_auc)))
comp=pd.DataFrame(rr);comp.to_csv(O/'01_overlap_auc_per_seed_derived.csv',index=False)
cs=comp.groupby(['population','condition'],sort=False).agg(n=('n','first'),positive=('positive','first'),negative=('negative','first'),budget=('budget','first'),roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),gap_mean=('overlap_minus_reference','mean'),gap_sd=('overlap_minus_reference','std')).reset_index()
cs.to_csv(O/'02_overlap_auc_summary_derived.csv',index=False)
retr=d.copy();retr['retrieved_positives']=np.rint(retr.p_at_5pct*retr.budget).astype(int)
assert np.allclose(retr.retrieved_positives,retr.p_at_5pct*retr.budget,atol=1e-11)
rs=retr.groupby(['subset','condition'],sort=False).agg(n=('n','first'),positive=('positive','first'),negative=('negative','first'),budget=('budget','first'),retrieved_mean=('retrieved_positives','mean'),retrieved_min=('retrieved_positives','min'),retrieved_max=('retrieved_positives','max')).reset_index()
rs.to_csv(O/'03_retrieved_counts_by_subset.csv',index=False)
# full budget overlap-positive retrieval can be recovered exactly for zero-D pipelines only.
records=[]
for model in ['MLP','RandomForest']:
 c='primary__'+model+'__M11'
 a=retr[(retr.subset==FULL)&(retr.condition==c)].sort_values('seed'); b=retr[(retr.subset==D)&(retr.condition==c)].sort_values('seed')
 assert (b.retrieved_positives==0).all() and (b.tie_p5_min==0).all() and (b.tie_p5_max==0).all()
 assert (a.budget.values==b.budget.values).all()
 for ar,br in zip(a.itertuples(),b.itertuples()):
  records.append(dict(condition=c,seed=ar.seed,full_budget=ar.budget,full_retrieved=ar.retrieved_positives,full_retrieved_no_detected_overlap=0,full_retrieved_overlap=ar.retrieved_positives,derivation='All negatives and fixed scores retained in D; zero retrieval and zero tie bounds in D imply zero in full top-k.'))
pd.DataFrame(records).to_csv(O/'04_full_budget_document_yield_derived.csv',index=False)
# Difference on subsets and feature increments, descriptive five fits only.
r=[]
for sub in d.subset.unique():
 for left,right,label in [('primary__GCN__M11','primary__MLP__M11','reverse_GCN_minus_MLP'),('primary__MLP__M11','primary__MLP__M5','MLP_M11_minus_M5'),('primary__RandomForest__M11','primary__RandomForest__M5','RF_M11_minus_M5')]:
  a=d[(d.subset==sub)&(d.condition==left)].sort_values('seed');b=d[(d.subset==sub)&(d.condition==right)].sort_values('seed')
  ds=a.roc_auc.to_numpy()-b.roc_auc.to_numpy()
  r.append(dict(subset=sub,contrast=label,mean_delta=ds.mean(),seed_sd=ds.std(ddof=1),positive_seed_pairs=int((ds>0).sum()),seed_differences=json.dumps(ds.tolist())))
for sub in ['O','S_overlap']:
 for m in ['MLP','RandomForest']:
  a=comp[(comp.population==sub)&(comp.condition==f'primary__{m}__M11')].sort_values('seed');b=comp[(comp.population==sub)&(comp.condition==f'primary__{m}__M5')].sort_values('seed');ds=a.roc_auc.to_numpy()-b.roc_auc.to_numpy()
  r.append(dict(subset=sub,contrast=f'{m}_M11_minus_M5',mean_delta=ds.mean(),seed_sd=ds.std(ddof=1),positive_seed_pairs=int((ds>0).sum()),seed_differences=json.dumps(ds.tolist())))
pd.DataFrame(r).to_csv(O/'05_descriptive_subset_contrasts.csv',index=False)
tie=pd.read_csv(P/'02_p5_tie_reconciliation.csv'); tie[tie.tie_p5_max>tie.tie_p5_min].to_csv(O/'06_variable_cutoff_ties.csv',index=False)
checks={'status':'CURRENT_AGGREGATE_DERIVATIONS_PASS','overlap_complement_seed_records':len(comp),'overlap_summary_records':len(cs),'retrieval_summary_records':len(rs),'source_fit_records':len(d),'full_budget_yield_records':len(records),'new_model_fits':0,'private_scores_used':False,'new_subset_reference_distributions_computed':False,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'04_subset_per_seed_metrics.csv',P/'05_subset_summaries.csv',P/'02_p5_tie_reconciliation.csv']}}
(O/'CHECKS.json').write_text(json.dumps(checks,indent=2))
print('\nPRIMARY M11 COMPLEMENT\n',cs[cs.condition.str.match('primary.*M11')].to_string(index=False))
print('\nPRIMARY RETRIEVALS\n',rs[rs.condition.str.match('primary.*M11')].to_string(index=False))
print('\nTIES\n',tie[tie.tie_p5_max>tie.tie_p5_min].to_string(index=False))
print('\nCONTRASTS\n',pd.DataFrame(r).to_string(index=False))
