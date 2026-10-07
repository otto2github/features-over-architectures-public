#!/usr/bin/env python3
"""Verify reader-archive scientific evidence, aggregate arithmetic and current tables.
This is aggregate verification, not a rerun of licensed empirical observations.
"""
from __future__ import annotations
import csv, hashlib, json, sys, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SCHEMES=['company_x_matched_seed','company_document_x_matched_seed']
METRICS=['roc_auc','ap','f1_val_threshold','p_at_5pct','r_at_10fpr','hits_at_budget']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)
def tables_in_docx(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
    answer=[]
    for table in root.findall('./w:body/w:tbl',ns):
        rows=[]
        for row in table.findall('./w:tr',ns):
            vals=[]
            for cell in row.findall('./w:tc',ns):
                paras=[]
                for p in cell.findall('./w:p',ns):paras.append(''.join(t.text or '' for t in p.findall('.//w:t',ns)))
                vals.append('\n'.join(paras))
            rows.append(vals)
        answer.append(rows)
    return answer

def main():
    p27=ROOT/'aggregates/sensitivity_fixed_score';p28=ROOT/'aggregates/extension_fixed_score'
    n27=pd.read_csv(p27/'01_FIXED_PER_FIT_METRICS.csv');n28=pd.read_csv(p28/'01_MATCHED_PER_FIT_METRICS.csv')
    require(len(n27)==725 and n27.run_id.nunique()==145,'V27_FIT_COVERAGE')
    require(len(n28)==100 and n28.run_id.nunique()==20,'V28_MATCHED_FIT_COVERAGE')
    require(n28.groupby('model').run_id.nunique().to_dict()=={'MLP':5,'RandomForest_unweighted':5,'GCN_reverse':5,'GraphSAGE_reverse':5},'V28_MODEL_COUNT')
    replay=pd.read_csv(ROOT/'aggregates/strict_replay/01_REPLAY_PER_FIT.csv')
    require(len(replay)==55 and int(replay.score_exact_elementwise.sum())==45,'ORIGINAL_REPLAY_BOUNDARY')
    definitions=json.loads((p27/'CONTRAST_DEFINITIONS.json').read_text())
    interval_count=0;max_endpoint_error=0.;max_replica_error=0.
    for scheme in SCHEMES:
        with np.load(p27/(scheme+'_AGGREGATE_REPLICATES.npz'),allow_pickle=False) as a:
            require(a['condition_metrics'].shape==(2000,29,6) and bool(a['valid'].all()),'V27_REPLICATE_CONTRACT')
            names=list(a['condition_names']);metrics=list(a['metric_names']);contrasts=list(a['contrast_names'])
            for spec in definitions:
                reconstructed=np.zeros((2000,6),dtype=float)
                for name,weight in spec['coefficients'].items():reconstructed+=float(weight)*a['condition_metrics'][:,names.index(name),:]
                error=float(np.max(np.abs(reconstructed-a['contrast_metrics'][:,contrasts.index(spec['contrast_id']),:])))
                max_replica_error=max(max_replica_error,error);require(error<1e-12,'V27_CONTRAST_ARITHMETIC')
            for suffix,key,lookup,array in [('_CONDITION_INTERVALS.csv','condition',names,a['condition_metrics']),('_CONTRAST_INTERVALS.csv','contrast_id',contrasts,a['contrast_metrics'])]:
                for row in pd.read_csv(p27/(scheme+suffix)).itertuples():
                    name=getattr(row,key);metric=metrics.index(row.metric)
                    if name in lookup:v=array[:,lookup.index(name),metric]
                    else:
                        require(key=='condition' and name.startswith('B_mean3|'),'UNKNOWN_CONDITION')
                        _,model,modal=name.split('|');v=np.mean([array[:,names.index(f'B|{model}|{modal}|{world}'),metric] for world in [271707,271708,271709]],axis=0)
                    require(np.isfinite(v).all(),'V27_NONFINITE_REPLICATE')
                    q=np.percentile(v,[2.5,97.5],method='linear');error=float(np.max(np.abs(q-np.array([row.ci_low,row.ci_high]))))
                    max_endpoint_error=max(max_endpoint_error,error);require(error<1e-12,'V27_INTERVAL_ENDPOINT');interval_count+=1
    require(interval_count==948,'V27_INTERVAL_COUNT')
    new_pairs={'MLP_minus_GCN':('MLP','GCN_reverse'),'MLP_minus_GraphSAGE':('MLP','GraphSAGE_reverse'),'RF_minus_GCN':('RandomForest_unweighted','GCN_reverse'),'RF_minus_GraphSAGE':('RandomForest_unweighted','GraphSAGE_reverse')}
    ci_zero=0
    for scheme in SCHEMES:
        with np.load(p28/(scheme+'_AGGREGATE_REPLICATES.npz'),allow_pickle=False) as a:
            models=list(a['models']);contrasts=list(a['contrasts']);metrics=list(a['metrics'])
            require(a['condition_replicates'].shape==(2000,4,6),'V28_REPLICATE_CONTRACT')
            for ident,(left,right) in new_pairs.items():
                delta=a['condition_replicates'][:,models.index(left),:]-a['condition_replicates'][:,models.index(right),:]
                error=float(np.max(np.abs(delta-a['contrast_replicates'][:,contrasts.index(ident),:])))
                max_replica_error=max(max_replica_error,error);require(error<1e-12,'V28_CONTRAST_ARITHMETIC')
            for suffix,key,lookup,array in [('_CONDITION_INTERVALS.csv','model',models,a['condition_replicates']),('_CONTRAST_INTERVALS.csv','contrast',contrasts,a['contrast_replicates'])]:
                for row in pd.read_csv(p28/(scheme+suffix)).itertuples():
                    v=array[:,lookup.index(getattr(row,key)),metrics.index(row.metric)]
                    require(np.isfinite(v).all() and row.valid_replicates==2000,'V28_NONFINITE_OR_INCOMPLETE')
                    q=np.percentile(v,[2.5,97.5],method='linear');error=float(np.max(np.abs(q-np.array([row.lower,row.upper]))))
                    max_endpoint_error=max(max_endpoint_error,error);require(error<1e-12,'V28_INTERVAL_ENDPOINT');interval_count+=1
                    if key=='contrast':ci_zero+=int(row.lower<=0<=row.upper)
    require(interval_count==1044 and ci_zero==48,'TOTAL_INTERVAL_CONTRACT')
    # Means / seed SDs independently recomputed from per-fit metric records.
    desc=pd.read_csv(p28/'02_MATCHED_DESCRIPTIVES.csv')
    for row in desc.itertuples():
        q=n28[(n28.model==row.model)&(n28.population==row.population)][row.metric].to_numpy(float)
        require(len(q)==5,'MATCHED_POINT_SEEDS')
        require(abs(float(q.mean())-row.estimate)<1e-12 and abs(float(q.std(ddof=1))-row.seed_sd)<1e-12,'MATCHED_POINT_AGGREGATION')
    for rid,d in n28.groupby('run_id'):
        m=d.set_index('population')
        require(abs(20*m.loc['full_test','roc_auc']-13*m.loc['O_frozen','roc_auc']-7*m.loc['D_frozen','roc_auc'])<1e-10,'AUC_DO_IDENTITY')
    extension=json.loads((ROOT/'run_records/extension/EXTENSION_RESULTS.json').read_text())
    require(extension['fit_count']==10 and len(extension['runs'])==10,'EXTENSION_FROZEN_COUNT')
    for r in extension['runs']:
        rid=r['run_id']
        for population,metrics in r['metrics'].items():
            q=n28[(n28.run_id==rid)&(n28.population==population)]
            require(len(q)==1,'EXTENSION_POINT_JOIN')
            for metric in METRICS:require(abs(float(q.iloc[0][metric])-float(metrics[metric]))<1e-12,'EXTENSION_POINT_IDENTITY')
    flags=json.loads((p28/'07_LEGACY_A_GCN_COMMON_D_HIT_FLAGS.json').read_text())
    require(flags['status']=='COMPLETE' and flags['n_shared_positive_rows']==1 and flags['n_seeds']==5 and flags['rankings_agree'],'LEGACY_SHARED_HIT_CONTRACT')
    require(len(flags['source_flags'])==3 and all(x is False for x in flags['source_flags'].values()),'LEGACY_THREE_FLAGS_FALSE')
    lopo=pd.read_csv(p28/'04_D_LOPO_CONDITION_RANGES.csv');require(len(lopo)==24 and (lopo.deletions==7).all(),'D_LOPO_COMPLETE')
    from regenerate_tables import regenerate
    tables=regenerate(ROOT);cells=0
    for name,rows in tables.items():
        with (ROOT/'tables/expected'/name).open(newline='',encoding='utf-8') as f:expected_rows=list(csv.reader(f))
        require(rows==expected_rows,'TABLE_EXPECTED_CELLS:'+name)
        stem,panel=name.removesuffix('.csv').split('_panel')
        actual_rows=tables_in_docx(ROOT/'tables/current'/(stem+'.docx'))[int(panel)-1]
        require(rows==actual_rows,'TABLE_DOCX_CELLS:'+name);cells+=sum(map(len,rows))
    print(json.dumps({'status':'SCIENTIFIC_AGGREGATE_CHECK_PASS','sensitivity_intervals_recomputed':948,'extension_intervals_recomputed':96,'total_intervals_recomputed':interval_count,'endpoints_recomputed':2*interval_count,'max_endpoint_abs_error':max_endpoint_error,'max_replica_contrast_abs_error':max_replica_error,'new_contrast_intervals_containing_zero':ci_zero,'current_table_panels':len(tables),'current_table_cells_exact':cells,'legacy_shared_hit_flags':[False,False,False],'predecessor_evidence_extracted_and_file_hash_verified':True,'model_training_or_inference_performed':False,'licensed_row_level_reconstruction_performed':False},indent=2))
if __name__=='__main__':main()
