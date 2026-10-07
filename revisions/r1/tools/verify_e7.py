#!/usr/bin/env python3
"""Verify frozen E7 aggregates; no training, private rows or new bootstrap draws."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[42,123,456,789,1024]
MODELS=['GCN','SAGE']
METRICS=['roc_auc','ap','f1_val_threshold','p_at_5pct','r_at_10fpr','hits_at_budget']
COUNTS={'full_test':(8435,20,8415,422),'D_frozen':(8422,7,8415,422),
        'O_frozen':(8428,13,8415,422),'S_frozen':(7757,6,7751,388),'J_frozen':(7753,2,7751,388)}
PER_SEED_COLUMNS=['run_id','model','condition','seed','population','ap','budget','cutoff_tie_positive',
    'cutoff_tie_rows','f1_val_threshold','hits_at_budget','n','n_neg','n_pos','p_at_5pct','r_at_10fpr',
    'roc_auc','selected_from_cutoff_tie','tie_p5_max','tie_p5_min']

def require(ok,message):
    if not ok:raise AssertionError(message)
def close(a,b,message,tolerance=1e-12):
    aa=np.asarray(a,dtype=float);bb=np.asarray(b,dtype=float)
    require(aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
        and np.all(np.abs(aa-bb)<tolerance),message)
def read(root,name):return json.loads((root/name).read_text())

def verify(root=ROOT):
    root=Path(root)
    p=pd.read_csv(root/'aggregates/e7/PER_SEED_METRICS.csv')
    g=pd.read_csv(root/'aggregates/e7/GROUP_MEANS.csv')
    results=read(root,'aggregates/e7/RESULTS.json')
    contrasts=read(root,'aggregates/e7/PAIRED_CONTRASTS.json')
    lock=read(root,'run_records/e7/ALL_20_LOCKED.json')
    checks=read(root,'run_records/e7/EXECUTION_CHECKS.json')
    env=read(root,'run_records/e7/ENVIRONMENT.json')
    prep=read(root,'run_records/e7/GRAPH_PREP_SUMMARY.json')
    matrix=read(root,'protocol/matrices/e7_run_matrix.json')
    columns=read(root,'config/e7/M11_COLUMN_ORDER.json')['columns']
    expected={(m,c,s) for m in MODELS for c in ['control','E7'] for s in SEEDS}
    run_ids={f'{m}__M11__{c}__seed{s}' for m,c,s in expected}
    require(len(matrix)==20 and {(r['model'],r['condition'],r['seed']) for r in matrix}==expected,'E7_MATRIX')
    require({r['run_id'] for r in matrix}==run_ids,'E7_MATRIX_RUN_IDS')
    require(len(p)==100 and list(p.columns)==PER_SEED_COLUMNS,'E7_POPULATION_ROW_SCHEMA')
    require(set(p.run_id)==run_ids and not p.duplicated(['run_id','population']).any(),'E7_RUN_POPULATION_KEYS')
    require(set(p.population)==set(COUNTS),'E7_POPULATIONS')
    require(np.isfinite(p.select_dtypes(include='number').to_numpy()).all(),'E7_FINITE_POINT_RECORDS')
    for rid,q in p.groupby('run_id'):
        require(len(q)==5 and set(q.population)==set(COUNTS),'E7_RUN_FIVE_POPULATIONS')
        require(q[['model','condition','seed']].drop_duplicates().shape==(1,3),'E7_RUN_ID_METADATA')
        row=q.iloc[0];require(rid==f'{row.model}__M11__{row.condition}__seed{row.seed}','E7_RUN_ID_VALUE')
        for rr in q.itertuples():
            require((rr.n,rr.n_pos,rr.n_neg,rr.budget)==COUNTS[rr.population],'E7_COUNT_BUDGET_CONTRACT')
            require(0<=rr.hits_at_budget<=min(rr.n_pos,rr.budget) and int(rr.hits_at_budget)==rr.hits_at_budget,'E7_INTEGER_HITS')
            close(rr.p_at_5pct,rr.hits_at_budget/rr.budget,'E7_BUDGET_PRECISION')
            for metric in METRICS[:-1]:require(0<=getattr(rr,metric)<=1,'E7_POINT_METRIC_BOUNDS')
            require(0<=rr.tie_p5_min<=rr.p_at_5pct<=rr.tie_p5_max<=1,'E7_TIE_PRECISION_BOUNDS')
            require(0<=rr.cutoff_tie_positive<=rr.cutoff_tie_rows and 0<=rr.selected_from_cutoff_tie<=rr.cutoff_tie_rows,'E7_TIE_COUNTS')
        q=q.set_index('population')
        close(20*q.loc['full_test','roc_auc'],7*q.loc['D_frozen','roc_auc']+13*q.loc['O_frozen','roc_auc'],
            'E7_FULL_D_O_AUC',1e-10)
    require(len(g)==20 and not g.duplicated(['model','condition','population']).any(),'E7_GROUP_COUNT')
    summary=results['group_means']
    require(len(summary)==20,'E7_RESULT_GROUP_COUNT')
    cells=0
    for row in g.itertuples():
        q=p[(p.model==row.model)&(p.condition==row.condition)&(p.population==row.population)]
        require(len(q)==5 and set(q.seed)==set(SEEDS),'E7_GROUP_SEEDS')
        require((row.n,row.n_pos)==COUNTS[row.population][:2],'E7_GROUP_POPULATION')
        stored=[r for r in summary if (r['model'],r['condition'],r['population'])==(row.model,row.condition,row.population)]
        require(len(stored)==1,'E7_GROUP_DUPLICATE_OR_MISSING')
        for metric in METRICS:
            close(q[metric].mean(),getattr(row,metric+'_mean'),'E7_GROUP_MEAN')
            close(q[metric].std(ddof=1),getattr(row,metric+'_sd'),'E7_GROUP_SD')
            close(stored[0][metric+'_mean'],getattr(row,metric+'_mean'),'E7_GROUP_JSON_MEAN')
            close(stored[0][metric+'_sd'],getattr(row,metric+'_sd'),'E7_GROUP_JSON_SD');cells+=2
    require(results['status']=='E7_20_FITS_AND_PAIRED_ANALYSIS_COMPLETE' and results['fit_count']==20
        and results['prescribed_fit_count_completed'] is True and results['M11_dimensions']==129,'E7_COMPLETION')
    require(results['performance_used_to_select_graph_or_seeds'] is False
        and results['small_subsets_descriptive_only'] is True,'E7_REPORTING_SCOPE')
    require(len(columns)==129 and len(set(columns))==129,'E7_M11_129_UNIQUE_COLUMNS')
    training=results['training_summaries']
    require(len(training)==20 and {r['run_id'] for r in training}==run_ids,'E7_TRAINING_COVERAGE')
    for r in training:
        require(r['strict_determinism'] is True and r['warn_only'] is False and r['test_evaluated'] is False,'E7_T_TRAINING_SELECTION')
        require((r['train_pos'],r['train_neg'],r['validation_label_positive'],r['common_node_count'])==(254,23852,40,366695),'E7_TRAINING_COUNTS')
        require((r['hidden_dim'],r['dropout'],r['lr'],r['patience'],r['epochs_max'])==(64,0.3,0.0005,10,100),'E7_TRAINING_HYPERPARAMETERS')
        require(0<=r['val_threshold']<=1 and 0<=r['best_val_auc']<=1,'E7_VALIDATION_BOUNDS')
        require(r['original_training_loop_sha256']=='6b7d1188097a6e9a287b865a20ed4d5824a5e35de2d926c4da43b2ceb1c523d7','E7_ORIGINAL_CORE_BINDING')
    values=contrasts['values']
    require(len(values)==60 and len({(r['model'],r['population'],r['metric']) for r in values})==60,'E7_SIXTY_CONTRASTS')
    require(results['paired_contrasts']==values,'E7_RESULT_CONTRAST_IDENTITY')
    primary_count=0
    for r in values:
        require(r['model'] in MODELS and r['population'] in COUNTS and r['metric'] in METRICS,'E7_CONTRAST_KEY')
        q=p[(p.model==r['model'])&(p.population==r['population'])]
        control=q[q.condition=='control'].set_index('seed').loc[SEEDS,r['metric']].to_numpy()
        treated=q[q.condition=='E7'].set_index('seed').loc[SEEDS,r['metric']].to_numpy()
        diff=treated-control
        close(diff,r['paired_seed_differences'],'E7_PAIRED_SEED_ARITHMETIC')
        close(diff.mean(),r['paired_mean_difference'],'E7_PAIRED_MEAN')
        close(diff.std(ddof=1),r['paired_seed_sd'],'E7_PAIRED_SD')
        if r['population']=='full_test':
            ci=np.asarray(r['paired_company_seed_bootstrap_ci95'],dtype=float)
            require(ci.shape==(2,) and np.isfinite(ci).all() and ci[0]<=ci[1],'E7_RECORDED_CI_CONTRACT')
            require(r['bootstrap_valid_replicates']==2000,'E7_RECORDED_VALID_REPLICATES')
            require(0<=r['descriptive_exact_seed_signflip_p']<=1,'E7_DESCRIPTIVE_SIGNFLIP_RANGE')
            primary_count+=1
        else:require('paired_company_seed_bootstrap_ci95' not in r,'E7_SECONDARY_INFERENTIAL_INTERVAL')
    require(primary_count==12,'E7_PRIMARY_INTERVAL_COUNT')
    arr=np.load(root/'aggregates/e7/FULL_BOOTSTRAP_DIFFERENCES.npy')
    require(arr.shape==(2000,2,6) and np.isfinite(arr).all(),'E7_BOOTSTRAP_ARRAY_SHAPE')
    interval_error=0.0;recomputed=0
    for r in values:
        if r['population']!='full_test':continue
        col=arr[:,MODELS.index(r['model']),METRICS.index(r['metric'])]
        lo,hi=np.quantile(col,[.025,.975],method='linear')
        lo0,hi0=r['paired_company_seed_bootstrap_ci95']
        interval_error=max(interval_error,abs(lo-lo0),abs(hi-hi0));recomputed+=1
    require(recomputed==12 and interval_error<1e-12,'E7_INTERVAL_RECOMPUTATION')
    b=contrasts['bootstrap']
    require(b==results['bootstrap'],'E7_BOOTSTRAP_DESCRIPTION_IDENTITY')
    require((b['replicates'],b['valid_replicates'],b['invalid_single_class_replicates'],b['rng_seed'],b['company_clusters'])==(2000,2000,0,141707,4531),'E7_BOOTSTRAP_COUNTS')
    require(b['company_draw_shared'] is True and b['seed_draw_shared'] is True and b['same_draw_all_conditions_and_models'] is True,'E7_MATCHED_DRAW_CONTRACT')
    attrs=results['full_budget_D_O_attributions']
    require(len(attrs)==20 and {r['run_id'] for r in attrs}==run_ids,'E7_ATTRIBUTION_COVERAGE')
    for r in attrs:
        full=p[(p.run_id==r['run_id'])&(p.population=='full_test')].iloc[0]
        require(r['budget']==422 and r['D_positive_hits']+r['O_positive_hits']==r['full_positive_hits']==full.hits_at_budget,'E7_FULL_BUDGET_ATTRIBUTION')
        require(0<=r['D_positive_hits']<=7 and 0<=r['O_positive_hits']<=13,'E7_ATTRIBUTION_POSITIVE_BOUNDS')
    require(lock['status']=='ALL_20_LOCKED' and lock['fit_count']==20 and set(lock['runs'])==run_ids
        and lock['test_performance_evaluated_before_lock'] is False,'E7_GLOBAL_LOCK')
    require(checks['status']=='PASS' and checks['all_fits']==20 and checks['M11_dimensions']==129
        and checks['strict_determinism'] is True and checks['warn_only'] is False,'E7_EXECUTION_CONTRACT')
    for key in ['global_checkpoint_lock_before_any_test_performance','source_rechecked_after_runs',
          'full_D_O_auc_identity_all_20','full_budget_D_O_attribution_identity_all_20',
          'validation_only_checkpoint_and_threshold_selection']:require(checks[key] is True,'E7_EXECUTION_CHECK:'+key)
    require(checks['raw_names_ids_labels_scores_checkpoints_in_return'] is False,'E7_RETURN_BOUNDARY')
    require(checks['context']==lock['context'],'E7_CONTEXT_LOCK_IDENTITY')
    require(env['strict_determinism'] is True and env['warn_only'] is False and env['segment_matmul_effective'] is False,'E7_ENVIRONMENT_FLAGS')
    require(env['CUBLAS_WORKSPACE_CONFIG']==':4096:8','E7_CUBLAS_SETTING')
    matrix_hash=hashlib.sha256((root/'protocol/matrices/e7_run_matrix.json').read_bytes()).hexdigest()
    require(lock['context']['matrix_sha256']==matrix_hash,'E7_MATRIX_EXECUTION_HASH')
    source_records=read(root,'provenance/source_hashes.json')['records']
    source_map={r['public_path']:r for r in source_records}
    require(source_map['protocol/reader/e7_training_protocol.md']['original_sha256']==lock['context']['protocol_sha256'],
        'E7_ORIGINAL_PROTOCOL_HASH')
    runner=read(root,'run_records/e7/RUNNER_IDENTITIES.json')
    for name in ['e7_cache_adapter.py','e7_statistics.py','e7_weighted_metrics.py','e7_worker.py','e7_runtime_common.py','formal_rerun_core_REFERENCE.py']:
        require(source_map['code/e7/training/'+name]['original_sha256']==runner[name]['sha256'],'E7_ORIGINAL_SOURCE_IDENTITY:'+name)
    require(prep['global_legal_party_id_semantics_verified'] is False,'E7_IDENTITY_LIMITATION_RETAINED')
    require((prep['candidate_edges_before_alignment'],prep['aligned_forward_edges'],prep['dropped_absent_frozen_feature_year_edges'],
        prep['original_node_count'],prep['E7_support_node_count'],prep['common_extended_node_count'])==(274175,265383,8792,303775,62920,366695),'E7_GRAPH_COUNTS')
    annual=pd.read_csv(root/'aggregates/e7/ANNUAL_ALIGNMENT.csv')
    require(annual.to_dict('records')==prep['annual_alignment'],'E7_ANNUAL_SUMMARY_IDENTITY')
    require(list(annual.year)==list(range(2010,2023)) and annual.aligned_forward_edges.sum()==265383,'E7_ANNUAL_ALIGNMENT')
    require((annual.candidate_edges_before_alignment-annual.aligned_forward_edges==annual.dropped_absent_feature_year_edges).all(),'E7_ANNUAL_DROP_ARITHMETIC')
    require((annual.control_message_edges==2*annual.base_forward_edges).all()
        and (annual.treatment_message_edges==annual.control_message_edges+2*annual.aligned_forward_edges).all(),'E7_ANNUAL_MESSAGE_ARITHMETIC')
    return {'status':'E7_AGGREGATE_CHECK_PASS','new_fits':20,'per_fit_population_rows':100,
        'groups':20,'group_mean_sd_values_recomputed':cells,'paired_metric_differences_recomputed':60,
        'recorded_primary_intervals_contract_checked':12,'intervals_independently_recomputed':recomputed,'interval_endpoints_recomputed':2*recomputed,
        'max_interval_endpoint_abs_error':float(interval_error),
        'scope':'Frozen aggregate arithmetic, reported execution contracts and the twelve Full percentile intervals recomputed from the distributed paired bootstrap-difference array; no private score/checkpoint/row reconstruction and no new resampling.',
        'training_or_inference_performed':False,'new_empirical_resampling_performed':False}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
