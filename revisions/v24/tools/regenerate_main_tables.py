#!/usr/bin/env python3
"""Derive eight numeric panels from scientific aggregates and reproduce the declared narrative map. No training."""
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
POST=ROOT/'aggregates/subset_diagnostics'
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def one(d):need(len(d)==1,'NONUNIQUE_SOURCE');return d.iloc[0]
def f(x,n=4,sg=False):return (f'{float(x):+.{n}f}'if sg else f'{float(x):.{n}f}').replace('-','−')
def ci(lo,hi,n=4):return '['+f(lo,n,True)+', '+f(hi,n,True)+']'
def binary(p,n,a,b):return .5*(a/p+1-b/n),((a/p)*(a/(a+b))+(1-a/p)*p/(p+n))if a+b else p/(p+n)
def derive():
 display={str(x['table']):x for x in json.loads((ROOT/'display_tables/main_tables.json').read_text())}
 out={};count=pd.read_csv(ROOT/'aggregates/audits/01_cohort_counts.csv');fixed=pd.read_csv(ROOT/'aggregates/preparation/fixed1095_counts.csv');rows=[]
 for label,part in [('Training (2010–2018)','train'),('Validation (2019–2020)','validation'),('Test (2021–2022)','test'),('Deferred (2023–2024)','outside_primary')]:
  z=one(count[count.partition==part]);w=one(fixed[fixed.partition==part]);vals=[z.panel_rows,z.eligible_rows,z.positive_rows,z.negative_rows,z.null_rows,w.fixed1095_positive];rows.append([label,*[f'{int(x):,}' for x in vals]])
 rows.append(['Total',*[f'{sum(int(r[j].replace(",","")) for r in rows):,}'for j in range(1,7)]])
 out['1']=rows
 obs=pd.read_csv(ROOT/'aggregates/primary/source_v3_observed_condition_metrics.csv');obs=obs[obs.label=='label_v1_strict_ab_primary'];t2=[];t3=[]
 for label,n in [('MLP','MLP'),('L1 logistic','TABUW_Lasso'),('L2 logistic','TABUW_Ridge'),('Random Forest','TABUW_RandomForest'),('XGBoost','TABUW_XGBoost')]:
  z=one(obs[obs.condition==n+'_M11']);t2.append([label,f(z.roc_auc_mean,3)+' ± '+f(z.roc_auc_sd,3),f(z.ap_mean),f(z.f1_mean),f(z.p_at_5pct_mean),f(z.r_at_10fpr_mean,2)])
  m={k:one(obs[obs.condition==n+'_'+k]).roc_auc_mean for k in ['M5','M10','M11']};t3.append([label,f(m['M10']-m['M5']),f(m['M11']-m['M10']),f(m['M11']-m['M5'])])
 # Add the documented reverse-message GNN references from unrounded five-fit values.
 direction=pd.read_csv(ROOT/'aggregates/direction/source_v13_per_seed_unified_metrics.csv')
 for label,model in [('GCN (reverse)','GCN'),('GAT (reverse)','GAT'),('GraphSAGE (reverse)','SAGE'),('RGCN (reverse)','RGCN')]:
  z=direction[direction.condition==f'REVERSE_{model}_M11'].sort_values('seed')
  need(z.seed.tolist()==[42,123,456,789,1024],'GRAPH_REFERENCE_SEEDS:'+model)
  t2.append([label,f(z.roc_auc.mean(),3)+' ± '+f(z.roc_auc.std(ddof=1),3),f(z.ap.mean()),f(z.f1_val_threshold.mean()),f(z.p_at_5pct.mean()),f(z.r_at_10fpr.mean(),2)])
 out['2A']=t2;out['3']=t3
 counts=pd.read_csv(ROOT/'aggregates/source_status_diagnostics/binary_status_counts.csv');src=pd.read_csv(POST/'financial_masks/source_mask_counts.csv',keep_default_na=False);extra=[]
 for part in ['train','validation','test']:
  p=one(src[(src.partition==part)&(src.label_group=='positive')]);n=one(src[(src.partition==part)&(src.label_group=='negative')]);extra.append({'flag':'asof_final_financial_nonfinite','partition':part,'positives':int(p['rows']),'negatives':int(n['rows']),'flagged_positives':int(p.final_feature_nonfinite),'flagged_negatives':int(n.final_feature_nonfinite)})
 flags=pd.concat([counts,pd.DataFrame(extra)],ignore_index=True)
 for m,i in [('roc_auc',0),('ap',1)]:flags[m]=[binary(z.positives,z.negatives,z.flagged_positives,z.flagged_negatives)[i]for z in flags.itertuples()]
 rows=[]
 for flag,label in [('post_anchor_fin','Compatible post-anchor fin_ input'),('post_anchor_fin_no_prior','Post-anchor input; no pre-anchor candidate'),('asof_source_missing','Any selected source missing after filtering'),('asof_final_financial_nonfinite','Any final financial nonfinite value')]:
  vals=[]
  for part in ['validation','test']:
   z=one(flags[(flags.flag==flag)&(flags.partition==part)]);vals.extend([f(z.roc_auc),f(z.ap,5)])
  rows.append([label,*vals])
 out['2B']=rows
 pts=pd.read_csv(POST/'subset_evaluation/04_subset_per_seed_metrics.csv');ss=pd.read_csv(POST/'subset_evaluation/05_subset_summaries.csv');pairs=pd.read_csv(POST/'subset_evaluation/06_sensitivity_paired_intervals.csv')
 def summary(sub,c):return one(ss[(ss.subset==sub)&(ss.condition==c)])
 def pair(l,r,m='roc_auc'):return one(pairs[(pairs.left==l)&(pairs.right==r)&(pairs.metric==m)&(pairs.resampling_scheme=='company_x_matched_seed')])
 D='positive_no_detected_earlier_doc_vs_all_negatives';S='source_screen_no_detected_exposure_or_absence';B='source_screen_plus_complete_fin_binding';J='intersection_doc_and_source_screen';rows=[]
 for label,sub in [('Full test','full_test'),('D: no detected earlier document match',D),('O: earlier document match','O'),('S: source screen',S),('B: S + raw fin_ binding screen',B),('J: D positive rule + S',J)]:
  vals=[]
  for m in ['MLP','RandomForest','GCN','SAGE']:
   c='primary__'+m+'__M11'
   if sub=='O':
    a=pts[(pts.subset=='full_test')&(pts.condition==c)].sort_values('seed');b=pts[(pts.subset==D)&(pts.condition==c)].sort_values('seed');need(a.seed.tolist()==b.seed.tolist(),'SEED_ALIGNMENT');auc=np.mean((20*a.roc_auc.to_numpy()-7*b.roc_auc.to_numpy())/13);n=8428;pos=13;budget=422;hits='—'
    if m in ['MLP','RandomForest']:
     need((b.p_at_5pct==0).all()and(b.tie_p5_max==0).all(),'ZERO_D_BOUNDS');hits=f(np.mean(a.p_at_5pct*a.budget),1)
   else:
    z=summary(sub,c);a=pts[(pts.subset==sub)&(pts.condition==c)];auc=z.roc_auc_mean;n=int(z.n);pos=int(z.positive);budget=math.ceil(.05*n);hits=f(np.mean(a.p_at_5pct*a.budget),1)
   vals.append(f(auc)+' / '+hits)
  rows.append([label,f'{n:,} / {pos}',str(budget),*vals])
 out['2C']=rows
 mainpairs=pd.read_csv(ROOT/'aggregates/audits/22_M11_paired_bootstrap_comparisons.csv');mainpairs=mainpairs[(mainpairs.metric=='roc_auc')&(mainpairs.resampling_scheme!='company_only_fixed_five_fits')]
 identities=[('TABUW_RandomForest_M11','STORED_MLP_M11'),('TABUW_RandomForest_M11','REVERSE_GCN_M11'),('TABUW_RandomForest_M11','REVERSE_SAGE_M11'),('STORED_MLP_M11','REVERSE_GCN_M11'),('STORED_MLP_M11','REVERSE_GAT_M11'),('STORED_MLP_M11','REVERSE_SAGE_M11'),('STORED_MLP_M11','REVERSE_RGCN_M11'),('REVERSE_GCN_M11','STORED_GCN_M11'),('REVERSE_GAT_M11','STORED_GAT_M11'),('REVERSE_SAGE_M11','STORED_SAGE_M11'),('REVERSE_RGCN_M11','STORED_RGCN_M11')]
 rows=[]
 for i,(l,r) in enumerate(identities):
  z=one(mainpairs[(mainpairs.left_condition==l)&(mainpairs.right_condition==r)]);iv=ci(z.ci_low,z.ci_high)
  if (l,r)==('REVERSE_SAGE_M11','STORED_SAGE_M11'):need(2.6e-6<z.ci_low<2.8e-6,'SAGE_SMALL_BOUND');iv='[+0.0000027, '+f(z.ci_high,4,True)+']'
  rows.append([display['4A']['rows'][i][0],f(z.observed_mean_paired_delta,4,True),iv])
 out['4A']=rows
 # This panel is an explicitly declared narrative map, not a computed numerical statistic.
 out['4B']=display['4B']['rows']
 ids=['asof__MLP__M5','asof__MLP__M11','asof__RandomForest__M5','asof__RandomForest__M11','fixed1095__MLP__M11','fixed1095__RandomForest__M11','fixed1095__GCN__M11','fixed1095__SAGE__M11'];rows=[]
 for i,c in enumerate(ids):
  z=summary('full_test',c);q=summary('full_test','primary__'+c.split('__',1)[1]);v=pair(c,'primary__'+c.split('__',1)[1]);need(abs(v.observed_delta-(z.roc_auc_mean-q.roc_auc_mean))<1e-12,'TEMPORAL_DELTA')
  rows.append([display['5A']['rows'][i][0],f(q.roc_auc_mean),f(z.roc_auc_mean)+' ± '+f(z.roc_auc_std),f(v.observed_delta,4,True),ci(v.ci_low,v.ci_high),f(z.ap_mean,5),f(z.p_at_5pct_mean,5),f(z.r_at_10fpr_mean,2)])
 out['5A']=rows
 ids=[('asof__MLP__M11','asof__RandomForest__M11'),('asof__MLP__M11','asof__MLP__M5'),('asof__RandomForest__M11','asof__RandomForest__M5'),('fixed1095__MLP__M11','fixed1095__RandomForest__M11'),('fixed1095__MLP__M11','fixed1095__GCN__M11'),('fixed1095__MLP__M11','fixed1095__SAGE__M11'),('fixed1095__RandomForest__M11','fixed1095__GCN__M11'),('fixed1095__RandomForest__M11','fixed1095__SAGE__M11')];rows=[]
 for i,(l,r) in enumerate(ids):
  a=pair(l,r);b=pair(l,r,'ap');rows.append([display['5B']['rows'][i][0],f(a.observed_delta,4,True),ci(a.ci_low,a.ci_high),f(b.observed_delta,5,True),ci(b.ci_low,b.ci_high,5)])
 out['5B']=rows
 for k,rows in out.items():need(rows==display[k]['rows'],'DISPLAY_MISMATCH:'+k+'\n'+str(rows))
 return display,out,flags

def main():
 a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,required=True);args=a.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);disp,rows,flags=derive()
 for k,t in disp.items():
  with (args.output_dir/f'Table{k}.csv').open('w',encoding='utf-8',newline='')as f1:
   w=csv.writer(f1);w.writerow(t['headers']);w.writerows(rows[k])
 flags.to_csv(args.output_dir/'BinaryStatus_AllPartitions.csv',index=False)
 print(json.dumps({'status':'MAIN_TABLES_REGENERATED_AND_CHECKED','numbered_tables':5,'numeric_panels':8,'narrative_maps':1,'panels':9,'binary_scores':12,'model_fitting':False,'private_resampling_repeated':False},indent=2))
if __name__=='__main__':main()
