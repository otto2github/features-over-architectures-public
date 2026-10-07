#!/usr/bin/env python3
"""Private-data postprocessing only. Never imports torch, fits models or edits papers."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse, json, hashlib, shutil, zipfile, traceback
from types import SimpleNamespace
from datetime import datetime
from contextlib import closing
import numpy as np
import pandas as pd
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE/'baseline_audit'
sys.path.insert(0,str(BASE))
from auditlib.common import (Context,load_labels,load_features,read_json,sha256,require,readonly_sqlite,json_clean,company_codes)
from auditlib.endpoints import load_qualified,document_key
from auditlib.financials import financial_audit
from auditlib.predictions import read_prediction,validate_run,bootstrap
from auditlib.metrics import Plan,NAMES,self_test
import asof_rules
P=read_json(BASE/'config/protocol.json')
KEY=['company_code','fiscal_year']
ASOF_HASH='9f15e48a1427111d6f66d017f49bddf18aa4b9d8bb6a73c1285e1f659575eaf0'
FIN_HASH='5cc0550b95c7bbd44fb9d819096ec4b4cb5827839a837af2210d7fc69a3d30cb'
STRICT_HASH='cbd51e39a188df3f036341655a4908a1077a0a4f1c5d161ca30b7754615e15a6'
PRIMARY=P['primary_label']
SEEDS=P['seeds']


def savej(p,obj):
    Path(p).write_text(json.dumps(json_clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def private_write(ctx,df,name,transfer):
    require(not df.duplicated(KEY).any(),'DUPLICATE_MASK_KEYS')
    p=ctx.private/name;df.to_parquet(p,index=False);os.chmod(p,0o600)
    meta={'format_version':'v18-postprocess-1','file_name':name,'sha256':sha256(p),'rows':len(df),
          'columns':list(df),'frozen_label_sha256':P['frozen_hashes']['fraud_labels_v1_0.parquet'],
          'private_not_for_chat_or_public_release':True}
    meta_path=p.with_suffix('.manifest.json');savej(meta_path,meta)
    transfer.mkdir(parents=True,exist_ok=True,mode=0o700)
    for src in [p,meta_path]:
        dst=transfer/src.name;tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copy2(src,tmp);os.chmod(tmp,0o600);os.replace(tmp,dst)
    print('PRIVATE_MASK='+str(transfer/name),flush=True)
    ctx.export_json('private_mask_identity.json',{k:v for k,v in meta.items() if k!='columns'})


def finite_marginals_match(ctx,actual,reference,keys,value_cols):
    # CSV round-trips may parse the literal label_group value "null" as NA.
    # Canonicalize key values BEFORE sorting so in-memory "null" and CSV NA
    # compare as the same semantic label group. Scientific value columns
    # remain subject to exact equality checks below.
    a=actual.copy();b=reference.copy()
    for frame in (a,b):
        for k in keys:
            s=frame[k].astype('string')
            if k=='label_group':
                frame[k]=s.fillna('null')
            else:
                frame[k]=s.fillna('<NA>')
    a=a.sort_values(keys).reset_index(drop=True)
    b=b.sort_values(keys).reset_index(drop=True)
    require(len(a)==len(b),'MARGINAL_ROW_COUNT_MISMATCH')
    require(a[keys].equals(b[keys]),'MARGINAL_KEYS_MISMATCH')
    for c in value_cols:
        require(np.array_equal(a[c].to_numpy(),b[c].to_numpy()),'MARGINAL_MISMATCH:'+c)


def financial_masks(ctx,args):
    """Re-expose exact legacy audit flags + approved as-of source absence. No feature writes."""
    ctx.use(args.financial_db,'approved_financial_db',FIN_HASH)
    ctx.use(args.asof_file,'approved_asof_features',ASOF_HASH)
    financial_audit(ctx)
    rows,blocks=ctx.row_flags
    require('fin' in blocks,'FIN_FLAGS_MISSING')
    got=pd.read_csv(ctx.summary/'14_financial_exposure_by_block.csv')
    ref=pd.read_csv(HERE/'reference/financial_vintage/14_financial_exposure_by_block.csv')
    finite_marginals_match(ctx,got[got.block=='fin'],ref[ref.block=='fin'],['block','partition','label_group'],
        ['denominator_rows','any_matched_dated_post_anchor','any_value_mismatch','any_unverified_feature','any_no_prior_candidate'])
    out=rows.copy()
    for col,mask in blocks['fin'].items():out['fin_'+col]=mask.to_numpy(bool)
    labels_path=ctx.find_pinned('fraud_labels_v1_0.parquet');node_path=ctx.find_pinned('node_features_v1_1.parquet')
    _,nf,panel=asof_rules.load_panel(labels_path,node_path,PRIMARY)
    fields=read_json(BASE/'config/fini_fields.json');raw={}
    print('Read as-of source tables (no financial features will be written)...',flush=True)
    with closing(readonly_sqlite(args.financial_db)) as con:
        for table,cols in {**asof_rules.FIN_FIELDS,'fina_indicator':fields['T0_FIELDS']}.items():
            raw[table]=asof_rules.read_table(con,table,cols)
    absent=np.zeros(len(panel),dtype=bool);post=np.zeros(len(panel),dtype=bool)
    for table in raw:
        for lag in ([0,1] if table=='fina_indicator' else [0,1,2]):
            a=asof_rules.select_for_panel(raw[table],panel,lag,'fini' if table=='fina_indicator' else 'fin',True)
            absent|=a['__sqlite_rowid'].isna().to_numpy()
            dd=a['__ann'] if table=='fina_indicator' else a['__avail']
            post|=(dd.notna()&panel.anchor.notna()&(dd.reset_index(drop=True)>panel.anchor.reset_index(drop=True))).to_numpy()
    require(not post.any(),'POST_ANCHOR_SELECTED_AFTER_FILTER')
    fm=panel[KEY].copy();fm['asof_any_source_missing']=absent
    out=out.merge(fm,on=KEY,validate='one_to_one')
    actual=out.groupby(['partition','label_group'],as_index=False).agg(rows=('target','size'),rows_with_any_asof_source_missing=('asof_any_source_missing','sum'))
    reference=pd.read_csv(HERE/'reference/preparation/asof_change_summary_by_partition_label.csv')
    finite_marginals_match(ctx,actual,reference,['partition','label_group'],['rows','rows_with_any_asof_source_missing'])
    af=pd.read_parquet(args.asof_file);af['company_code']=company_codes(af.firm_id);af['fiscal_year']=af.year.astype(int)
    cols=[c for c in af if c.startswith(('fin_','fini_'))];require(len(cols)==87,'ASOF_87_COLUMN_GUARD')
    absent_final=(~np.isfinite(af[cols].to_numpy(dtype=float,na_value=np.nan))).any(axis=1)
    mm=af[KEY].copy();mm['asof_any_final_financial_nonfinite']=absent_final
    out=out.merge(mm,on=KEY,validate='one_to_one')
    out['source_screen_no_detected_exposure_or_absence']=(~out.fin_any_matched_dated_post_anchor)&(~out.asof_any_source_missing)
    out['source_screen_plus_fin_binding_complete']=out.source_screen_no_detected_exposure_or_absence&(~out.fin_any_unverified_feature)&(~out.fin_any_value_mismatch)
    ctx.export_csv('source_mask_counts.csv',out.groupby(['partition','label_group'],as_index=False).agg(
        rows=('target','size'),post_anchor_fin=('fin_any_matched_dated_post_anchor','sum'),
        no_prior_candidate=('fin_any_no_prior_candidate','sum'),asof_source_missing=('asof_any_source_missing','sum'),
        final_feature_nonfinite=('asof_any_final_financial_nonfinite','sum'),
        source_screen_rows=('source_screen_no_detected_exposure_or_absence','sum'),
        source_screen_binding_complete_rows=('source_screen_plus_fin_binding_complete','sum')))
    private_write(ctx,out,'financial_status_masks_v18.parquet',Path(args.transfer))
    ctx.record('financial_mask_export','PASS',scope='old fin exposure and as-of missing-source marginals exactly reconciled; no clean-input certification')


def case_masks(ctx,args):
    f=load_labels(ctx);q=load_qualified(ctx)
    q['doc_key']=q.apply(document_key,axis=1);order={'train':0,'validation':1,'test':2}
    known=q[q.doc_key.notna()].copy();require(len(known)>0,'NO_USABLE_DOC_KEYS')
    ranks=known.assign(rank=known.partition.map(order)).groupby('doc_key',sort=False)['rank'].min().to_dict()
    known['earlier_partition_match']=known.doc_key.map(ranks)<known.partition.map(order)
    case=known.groupby(KEY,as_index=False).agg(doc_key_identified=('doc_key',lambda s:True),doc_overlap_earlier_partition=('earlier_partition_match','max'))
    first=q.groupby(KEY,as_index=False).agg(first_event=('event_date','min'))
    out=f[KEY+['target','partition','label_group','anchor']].merge(case,on=KEY,how='left',validate='one_to_one').merge(first,on=KEY,how='left',validate='one_to_one')
    out['doc_key_identified']=out.doc_key_identified.fillna(False).astype(bool)
    out['doc_overlap_earlier_partition']=out.doc_overlap_earlier_partition.fillna(False).astype(bool)
    out['positive_no_detected_prior_doc_match']=(out.target==1)&out.doc_key_identified&(~out.doc_overlap_earlier_partition)
    lag=(out.first_event-out.anchor).dt.days
    out['late_only_positive']=(out.target==1)&(lag>1095)
    for part,known_n,over_n,late_n in [('train',None,None,135),('validation',39,21,15),('test',20,13,0)]:
        g=out[(out.partition==part)&(out.target==1)]
        require(int(g.late_only_positive.sum())==late_n,'LATE_ONLY_COUNT:'+part)
        if known_n is not None:
            require(int(g.doc_key_identified.sum())==known_n,'KNOWN_DOC_COUNT:'+part)
            require(int(g.doc_overlap_earlier_partition.sum())==over_n,'OVERLAP_DOC_COUNT:'+part)
    keysets={p:set(known.loc[known.partition==p,'doc_key']) for p in ['validation','test']}
    late=q.merge(out.loc[out.late_only_positive,KEY],on=KEY,how='inner',validate='many_to_one');late['doc_key']=late.apply(document_key,axis=1)
    for p in ['validation','test']:
        hit=late.assign(flag=late.doc_key.map(lambda k:k in keysets[p])).groupby(KEY)['flag'].any()
        vals=pd.MultiIndex.from_frame(out[KEY]).map(hit.to_dict()).fillna(False)
        out['late_positive_document_recurs_'+p]=np.array(vals,dtype=bool)
    g=out[(out.partition=='train')&out.late_only_positive].copy()
    g['usable_document']=g.doc_key_identified
    ct=g.groupby(['usable_document','late_positive_document_recurs_validation','late_positive_document_recurs_test'],dropna=False).size().reset_index(name='training_positive_firm_years')
    require(int(ct.training_positive_firm_years.sum())==135,'LATE_ONLY_CROSSTAB_TOTAL')
    ctx.export_csv('late_only_training_document_recurrence.csv',ct)
    ctx.export_csv('document_mask_counts.csv',out[out.target==1].groupby('partition',as_index=False).agg(positive_rows=('target','size'),identified=('doc_key_identified','sum'),earlier_match=('doc_overlap_earlier_partition','sum'),no_detected_earlier_match=('positive_no_detected_prior_doc_match','sum'),late_only=('late_only_positive','sum')))
    out=out.drop(columns=['anchor','first_event'])
    private_write(ctx,out,'document_status_masks_v18.parquet',Path(args.transfer))
    ctx.record('document_mask_export','PASS',scope='issuing-authority/document-number rule; no claim of independently verified new cases')


def readmask(ctx,path):
    path=Path(path);meta=read_json(path.with_suffix('.manifest.json'))
    require(meta['frozen_label_sha256']==P['frozen_hashes']['fraud_labels_v1_0.parquet'],'MASK_LABEL_HASH')
    ctx.use(path,'private_mask_'+path.stem,meta['sha256']);x=pd.read_parquet(path)
    require(len(x)==51675 and not x.duplicated(KEY).any(),'MASK_KEY_CONTRACT')
    return x


def evaluate(ctx,args):
    allf=load_labels(ctx);f=allf[(allf.partition=='test')&allf.target.notna()].sort_values(KEY).reset_index(drop=True)
    y=f.target.to_numpy(int);runs=[]
    ref=read_json(BASE/'reference/PREDICTION_IDENTITY.json')
    expected_new={}
    for p in (HERE/'reference/sensitivity').glob('*.json'):expected_new[p.stem]=sha256(p)
    # Exact primary fit paths; no search for newest or highest-performing attempt.
    for model,modal in [('MLP','M5'),('MLP','M11'),('RandomForest','M5'),('RandomForest','M11'),('GCN','M11'),('SAGE','M11')]:
        for seed in SEEDS:
            if model in ['GCN','SAGE']:
                rid=f'{model}_{modal}_seed{seed}';rr=ref['reverse'][rid]
                base=ctx.project/'experiments/direction_sensitivity_v13/runs'/rid
                require(read_json(base/'DONE.json')['attempt']==rr['attempt'],'REVERSE_ATTEMPT_MISMATCH')
                base=base/rr['attempt'];exp=rr['files']['result.json']['sha256'];pre=rr['files']['predictions.parquet']['sha256']
            else:
                rel=f'results/core_v1/primary_{model}_{modal}_seed{seed}' if model=='MLP' else f'results/tabular_primary_v1/tabular_{model}_{modal}_seed{seed}'
                base=ctx.project/rel;exp=ref['original'].get(rel+'/result.json',{}).get('sha256');pre=ref['original'].get(rel+'/predictions.parquet',{}).get('sha256')
            ctx.use(base/'result.json','primary_result_'+base.name,exp);j=read_json(base/'result.json');validate_run(ctx,j,model,modal,seed)
            s=read_prediction(ctx,base/'predictions.parquet','primary_scores_'+base.name,pre,f)
            c='primary__'+model+'__'+modal;runs.append((c,seed,j,s))
    for scenario,model,modal in [('asof','MLP','M5'),('asof','MLP','M11'),('asof','RandomForest','M5'),('asof','RandomForest','M11'),('fixed1095','MLP','M11'),('fixed1095','RandomForest','M11'),('fixed1095','GCN','M11'),('fixed1095','SAGE','M11')]:
        for seed in SEEDS:
            rid=f'{scenario}__{model}__{modal}__seed{seed}';base=Path(args.strict_run)/'runs'/rid
            ctx.use(base/'result.json','sensitivity_result_'+rid,expected_new[rid]);j=read_json(base/'result.json')
            require(j.get('scenario')==scenario and j.get('model')==model and j.get('seed')==seed,'SENSITIVITY_RESULT_ID')
            s=read_prediction(ctx,base/'predictions.parquet','sensitivity_scores_'+rid,None,f)
            runs.append((scenario+'__'+model+'__'+modal,seed,j,s))
    point=[];plans=[];ties=[]
    for c,seed,j,s in runs:
        pl=Plan(y,s,j['val_threshold']);m=pl.compute(np.ones(len(y)));plans.append(pl)
        for k,metric in enumerate(NAMES):
            if metric=='p_at_5pct':continue
            old=j['metrics'].get(metric,j['metrics'].get('f1') if metric=='f1_val_threshold' else None)
            require(old is not None and abs(m[k]-float(old))<1e-8,'SCORE_METRIC_MISMATCH:'+c+':'+metric)
        point.append(dict(condition=c,seed=seed,**dict(zip(NAMES,m))))
        ties.append(dict(condition=c,seed=seed,archived_p5=j['metrics']['p_at_5pct'],stable_p5=m[3],**pl.tie_report()))
    pointdf=pd.DataFrame(point);ctx.export_csv('01_full_test_points.csv',pointdf);ctx.export_csv('02_p5_tie_reconciliation.csv',ties)
    refm=pd.read_csv(HERE/'reference/primary_condition_metrics.csv')
    for c,g in pointdf[pointdf.condition.str.startswith('primary__')].groupby('condition'):
        _,model,modal=c.split('__')
        if model in ['GCN','SAGE']:continue
        rc=f'{model}_{modal}' if model=='MLP' else f'TABUW_{model}_{modal}'
        r=refm[(refm.condition==rc)&(refm.label==PRIMARY)]
        require(len(r)==1 and abs(g.roc_auc.mean()-float(r.roc_auc_mean.iloc[0]))<1e-8,'PRIMARY_MEAN_RECONCILIATION')
    # Stable scalar metrics on explicitly limited subsets; no claims of new-case generalization.
    if args.financial_mask and args.document_mask:
        fm=readmask(ctx,args.financial_mask);dm=readmask(ctx,args.document_mask)
        cols_f=[c for c in fm if c not in ['target','partition','label_group']];cols_d=[c for c in dm if c not in ['target','partition','label_group']]
        x=f[KEY+['target']].merge(fm[cols_f],on=KEY,validate='one_to_one').merge(dm[cols_d],on=KEY,validate='one_to_one')
        for col in ['fin_any_matched_dated_post_anchor','asof_any_source_missing','positive_no_detected_prior_doc_match']:
            require(x[col].notna().all(),'MISSING_SUBSET_MASK:'+col)
        positive=x.target==1;no_doc=x.positive_no_detected_prior_doc_match.astype(bool)
        screened=x.source_screen_no_detected_exposure_or_absence.astype(bool)
        strictscreen=x.source_screen_plus_fin_binding_complete.astype(bool)
        masks={'full_test':np.ones(len(x),dtype=bool),
               'positive_no_detected_earlier_doc_vs_all_negatives':((~positive)|no_doc).to_numpy(),
               'source_screen_no_detected_exposure_or_absence':screened.to_numpy(),
               'source_screen_plus_complete_fin_binding':strictscreen.to_numpy(),
               'intersection_doc_and_source_screen':(screened&((~positive)|no_doc)).to_numpy()}
        ct=x[positive].groupby(['doc_overlap_earlier_partition','fin_any_matched_dated_post_anchor','asof_any_source_missing','fin_any_unverified_feature'],dropna=False).size().reset_index(name='test_positive_firm_years')
        require(ct.test_positive_firm_years.sum()==20,'POSITIVE_CROSSTAB_TOTAL');ctx.export_csv('03_overlap_by_vintage_test_positives.csv',ct)
        rows=[]
        for group,mask in masks.items():
            yy=y[mask];pos=int(yy.sum());neg=int(len(yy)-pos)
            for c,seed,j,s in runs:
                row=dict(subset=group,condition=c,seed=seed,n=len(yy),positive=pos,negative=neg,status='DESCRIPTIVE')
                if pos==0 or neg==0:row.update(status='NOT_ESTIMABLE_SINGLE_CLASS',**{m:None for m in NAMES})
                else:
                    mm=Plan(yy,s[mask],j['val_threshold']);row.update(dict(zip(NAMES,mm.compute(np.ones(len(yy))))));row.update(mm.tie_report())
                rows.append(row)
        ctx.export_csv('04_subset_per_seed_metrics.csv',rows)
        dt=pd.DataFrame(rows)
        summary=dt.groupby(['subset','condition','n','positive','negative'],dropna=False)[NAMES].agg(['mean','std']).reset_index()
        summary.columns=[f'{a}_{b}' if b else a for a,b in summary.columns]
        ctx.export_csv('05_subset_summaries.csv',summary)
        # Late-only row recurrence was generated in the Mac report, not inferred here.
        ctx.record('subset_evaluation','PASS',scope='conditional evaluation only; no retraining or verified clean/new-case claim')
    else:ctx.record('subset_evaluation','BLOCKED',reason='PRIVATE_SOURCE_OR_DOCUMENT_MASK_NOT_SUPPLIED; interval computation can still complete')
    conditions=sorted({r[0] for r in runs});groups=[[i for i,r in enumerate(runs) if r[0]==c] for c in conditions]
    require(all([runs[i][1] for i in g]==SEEDS for g in groups),'MATCHED_SEED_ORDER')
    _,idx=np.unique(f.company_code,return_inverse=True)
    cc,cs,reject=bootstrap(plans,idx,y,groups,args.bootstrap,args.workers,1807141,1807142)
    contrasts=[]
    for c in conditions:
        if not c.startswith('primary__'):contrasts.append((c,'primary__'+c.split('__',1)[1]))
    for scenario in ['asof','fixed1095']:
        contrasts.append((scenario+'__MLP__M11',scenario+'__RandomForest__M11'))
    for m in ['MLP','RandomForest']:contrasts.append(('asof__'+m+'__M11','asof__'+m+'__M5'))
    for m in ['GCN','SAGE']:
        for l in ['MLP','RandomForest']:contrasts.append(('fixed1095__'+l+'__M11','fixed1095__'+m+'__M11'))
    observed=np.array([plans[i].compute(np.ones(len(y))) for g in groups for i in g]).reshape(len(conditions),5,5)
    out=[]
    for scheme,arr in [('company_only_fixed_fits',cc),('company_x_matched_seed',cs)]:
        for l,r in contrasts:
            li,ri=conditions.index(l),conditions.index(r);diff=arr[:,li,:]-arr[:,ri,:];lo,hi=np.quantile(diff,[.025,.975],axis=0)
            oo=(observed[li]-observed[ri]).mean(axis=0)
            for k,m in enumerate(NAMES):out.append(dict(left=l,right=r,resampling_scheme=scheme,metric=m,observed_delta=oo[k],ci_low=lo[k],ci_high=hi[k],replicates=args.bootstrap,n=len(y),positives=int(y.sum()),company_clusters=int(idx.max()+1)))
    ctx.export_csv('06_sensitivity_paired_intervals.csv',out)
    ctx.export_json('07_resampling_scope.json',{'company_seed':1807141,'fit_seed':1807142,'replicates':args.bootstrap,'single_class_redraws':int(sum(reject)),
      'scheme':'shared company draw crossed with matched seed-label draw','n_fit_records':len(runs),'new_fits':0,
      'scope':'conditional on fixed datasets, target policies and five observed fits; no case/network bootstrap or training-data uncertainty',
      'top5':'company/year ascending then stable score descending; archived values and tie bounds retained',
      'source_masks':'absence/exposure diagnostics, not certification of clean inputs or new cases'})
    ctx.record('sensitivity_intervals','PASS',scope='evaluation only; no model fits')


def test_logic():
    from sklearn.metrics import roc_auc_score,average_precision_score
    maxerr=0.0
    for Pn,N,a,b in [(20,8415,14,403),(20,8415,12,266),(20,8415,12,527),(40,7269,22,565),(4,7,0,0),(4,7,4,7)]:
        y=np.r_[np.ones(Pn),np.zeros(N)];s=np.r_[np.ones(a),np.zeros(Pn-a),np.ones(b),np.zeros(N-b)]
        auc=.5*(a/Pn+1-b/N);ap=(a/Pn)*(a/(a+b))+(1-a/Pn)*(Pn/(Pn+N)) if a+b else Pn/(Pn+N)
        require(abs(auc-roc_auc_score(y,s))<1e-12,'BINARY_AUC_TEST')
        require(abs(ap-average_precision_score(y,s))<1e-12,'BINARY_AP_TEST')
    out=self_test(100)
    # Parquet schema round trip; no proprietary data or fit.
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'mask.parquet';f=pd.DataFrame({'company_code':['000001','000002'],'fiscal_year':[2021,2022],'flag':[False,True]});f.to_parquet(p,index=False)
        require(pd.read_parquet(p).equals(f),'PARQUET_SCHEMA_ROUND_TRIP')
    print(json.dumps({'status':'SELF_TEST_PASS','binary_count_cases':6,'weighted_metric_tests':out,'private_data_used':False,'trained_models':0},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['mac','tencent','server','selftest'])
    ap.add_argument('--project-root');ap.add_argument('--audit-root',default=str(Path.home()/'peerj_141707_audit'));ap.add_argument('--output-root');ap.add_argument('--transfer',default=str(Path.home()/'peerj_141707_reviewkc_transfer'))
    ap.add_argument('--financial-db');ap.add_argument('--asof-file');ap.add_argument('--financial-mask');ap.add_argument('--document-mask')
    ap.add_argument('--strict-run',default=None)
    ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    manifest_path=HERE/'TOOL_SHA256.json'
    require(manifest_path.is_file(),'TOOL_MANIFEST_MISSING')
    for rel,expected in read_json(manifest_path).items():
        require((HERE/rel).is_file() and sha256(HERE/rel)==expected,'TOOL_HASH_MISMATCH:'+rel)
    if args.role=='selftest':test_logic();return
    require(args.project_root is not None,'PROJECT_ROOT_REQUIRED');args.reverse_root=None
    if args.role=='server':require(args.strict_run is not None,'STRICT_RUN_ROOT_REQUIRED')
    if not args.output_root:args.output_root=str(Path(args.audit_root)/'reviewkc_v18'/args.role)
    ctx=Context(args,P,BASE);ctx.role=args.role
    fn={'mac':case_masks,'tencent':financial_masks,'server':evaluate}[args.role]
    failed=False
    try:
        fn(ctx,args)
    except Exception as e:
        failed=True
        (ctx.private/'failure_traceback.txt').write_text(traceback.format_exc())
        ctx.record('v18_postprocessing','BLOCKED',reason=(str(e) if e.__class__.__name__=='AuditStop' else e.__class__.__name__),note='exact traceback retained locally; no success inferred')
    result=ctx.finish()
    # Publish aggregate package pointer only; do not bundle row-level masks.
    dest=Path.home()/f'peerj_reviewkc_{args.role}_RETURN.zip';tmp=dest.with_suffix('.zip.tmp');shutil.copy2(result,tmp);os.replace(tmp,dest)
    print('RETURN_ZIP='+str(dest),flush=True);print('RETURN_SHA256='+sha256(dest),flush=True)
    if sys.platform=='darwin':
        import subprocess
        subprocess.run(['open','-R',str(dest)],check=False)
    if failed or any(x['state'] in ('INVALID','ERROR','BLOCKED') for x in ctx.status):raise SystemExit(2)
if __name__=='__main__':main()
