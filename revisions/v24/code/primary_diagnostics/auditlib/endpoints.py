"""Use frozen labels and their Stage-E lineage; do not create replacement labels."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .common import *

QUAL={'QUALIFIED_A_POSTFILING_FORMAL_CSRC','QUALIFIED_B_POSTFILING_FORMAL_CSRC'}

def find_manifest(ctx, stage, expected):
    directory=ctx.audit/('label_v1_'+stage)
    files=sorted(directory.glob(stage+'_*/manifest_v1_'+stage+'.json'))
    for p in files:
        if p.is_file() and not p.is_symlink() and sha256(p)==expected:
            return ctx.use(p,stage+'_manifest',expected)
    raise AuditStop('MATCHING_'+stage.upper()+'_MANIFEST_NOT_FOUND')

def load_qualified(ctx):
    f=load_labels(ctx)
    mf=find_manifest(ctx,'stageF',ctx.p['stageF_manifest_sha256'])
    fj=read_json(mf)
    require(fj['label_sha256']==ctx.p['frozen_hashes']['fraud_labels_v1_0.parquet'],'STAGEF_LABEL_CHAIN')
    me=find_manifest(ctx,'stageE',fj['stageE_manifest_sha256'])
    ej=read_json(me)
    risk=ctx.use(me.parent/'candidate_risksets_v1_stageE.parquet','stageE_risksets',fj['riskset_input_sha256'])
    risks=pd.read_parquet(risk)
    r=risks[(risks.scope=='strict')&(risks.policy=='A+B')].copy()
    r['company_code']=company_codes(r.company_code);r['fiscal_year']=r.fiscal_year.astype(int)
    require(not r.duplicated(['company_code','fiscal_year']).any(),'STAGEE_RISK_DUPLICATE')
    rr=r.merge(f[['company_code','fiscal_year','target']],on=['company_code','fiscal_year'],validate='one_to_one')
    candidate=pd.to_numeric(rr.candidate_label,errors='coerce').where(rr.primary_eligible_candidate.astype(bool))
    require(len(rr)==len(f) and np.allclose(candidate.to_numpy(float,na_value=np.nan),rr.target.to_numpy(float,na_value=np.nan),equal_nan=True),'STAGEE_FROZEN_RECONCILIATION')
    pq=ctx.use(me.parent/'pair_qualification_v1_stageE.parquet','stageE_pair_qualification')
    q=pd.read_parquet(pq)
    need={'scope','company_code','raw_violation_year','stageE_pair_status','declare_date','row_ref','annual_report_public_anchor'}
    require(need<=set(q.columns),'STAGEE_PAIR_SCHEMA')
    q=q[(q.scope=='strict') & q.stageE_pair_status.isin(QUAL)].copy()
    q['company_code']=company_codes(q.company_code);q['fiscal_year']=q.raw_violation_year.astype(int)
    q['event_date']=dates(q.declare_date);q['pair_anchor']=dates(q.annual_report_public_anchor)
    q=q.merge(f[['company_code','fiscal_year','target','partition','anchor']],on=['company_code','fiscal_year'],how='inner',validate='many_to_one')
    q=q[q.target.notna()].copy()
    require(q.event_date.notna().all() and (q.event_date>q.anchor).all(),'QUALIFIED_EVENT_DATE_INVALID')
    require((q.event_date<=pd.Timestamp(ctx.p['cutoff'])).all(),'QUALIFIED_EVENT_AFTER_FIXED_CUTOFF')
    require((q.pair_anchor==q.anchor).all(),'EVENT_ANCHOR_MISMATCH')
    require((q.target==1).all(),'QUALIFIED_EVENT_FOR_FROZEN_NEGATIVE')
    q=q.drop_duplicates(['company_code','fiscal_year','row_ref'])
    pos=set(map(tuple,f.loc[f.target==1,['company_code','fiscal_year']].to_numpy()))
    qkeys=set(map(tuple,q[['company_code','fiscal_year']].to_numpy()))
    require(qkeys==pos,'QUALIFIED_EVENTS_DO_NOT_COVER_ALL_FROZEN_POSITIVES')
    # Retrieve original event-level issuer fields. Missing lineage -> document-level overlap remains unknown.
    try:
        ma=find_manifest(ctx,'stageA',ej['stageA_manifest_sha256'])
        pe=ctx.use(ma.parent/'event_ledger_v1_stageA.parquet','stageA_event_ledger')
        e=pd.read_parquet(pe)
        require('row_ref' in e.columns and not e.row_ref.duplicated().any(),'EVENT_LEDGER_DUPLICATE')
        cols=[c for c in ['row_ref','supervisor','promulgator','document_number','declare_date'] if c in e]
        e=e[cols].rename(columns={c:'event_'+c for c in cols if c!='row_ref'})
        q=q.merge(e,on='row_ref',how='left',validate='many_to_one')
    except AuditStop:
        ctx.record('case_issuer_link','PARTIAL',reason='STAGEA_ISSUER_FIELDS_UNAVAILABLE_DOCUMENT_OVERLAP_NOT_ESTABLISHED')
    return q

def cohort_summary(ctx):
    f=load_labels(ctx)
    rows=[]
    for part,g in f.groupby('partition',sort=True):
        rows.append({'partition':part,'panel_rows':len(g),'eligible_rows':int(g.target.notna().sum()),'positive_rows':int((g.target==1).sum()),'negative_rows':int((g.target==0).sum()),'null_rows':int(g.target.isna().sum())})
    ctx.export_csv('01_cohort_counts.csv',rows)
    rows=[]
    for (part,reason),g in f[f.target.isna()].groupby(['partition','freeze_status'],dropna=False):
        rows.append({'partition':part,'exclusion_reason':str(reason),'rows':len(g)})
    ctx.export_csv('02_exclusion_counts.csv',rows)
    rows=[]
    for part,g in f[f.target==1].groupby('partition'):
        years={'train':[],'validation':list(range(2010,2019)),'test':list(range(2010,2021))}[part]
        known=set(f.loc[(f.target==1)&f.fiscal_year.isin(years),'company_code'])
        earlier=f[(f.target==1)&f.fiscal_year.isin(years)]
        rows.append({'partition':part,'positive_firm_years':len(g),'positive_companies':g.company_code.nunique(),'positive_rows_company_positive_in_earlier_partitions':int(g.company_code.isin(known).sum()),'positive_companies_positive_in_earlier_partitions':g.loc[g.company_code.isin(known),'company_code'].nunique(),'earlier_partition_positive_rows':len(earlier),'interpretation':'retrospective label recurrence; not by itself information leakage'})
    ctx.export_csv('03_company_overlap.csv',rows)
    ctx.record('frozen_cohort','PASS',note='Pinned labels and partition counts match; source labels unchanged.')

def document_key(r):
    """Conservative issuer + document-number key. A database row is NOT a legal case."""
    num=r.get('event_document_number',r.get('document_number'))
    if pd.isna(num) or not str(num).strip():return None
    num=str(num).strip()
    if any(c in num for c in [';','；','\n','、']):return None
    # Require number/year-like content and an identified issuing body.
    if not re.search(r'\d',num):return None
    issuer=r.get('event_supervisor')
    if pd.isna(issuer) or not str(issuer).strip():issuer=r.get('event_promulgator')
    if issuer is None or pd.isna(issuer) or not str(issuer).strip():return None
    issuer=re.sub(r'\s+','',str(issuer))
    if any(c in issuer for c in [';','；','、','\n']):return None
    num=re.sub(r'\s+','',num).translate(str.maketrans({'[':'〔',']':'〕','【':'〔','】':'〕','（':'(', '）':')'}))
    return (issuer,num)

def endpoint_audit(ctx):
    f=load_labels(ctx);q=load_qualified(ctx)
    first=q.groupby(['company_code','fiscal_year'],as_index=False).agg(first_event=('event_date','min'),qualified_source_rows=('row_ref','nunique'))
    p=f[f.target==1].merge(first,on=['company_code','fiscal_year'],validate='one_to_one')
    p['lag_days']=(p.first_event-p.anchor).dt.days
    # Annual and split summaries, earliest qualifying event per positive firm-year.
    lag=[]
    for part,g in p.groupby('partition'):
        lag.append({'group':'partition','period':part,**quantiles(g.lag_days)})
    for year,g in p.groupby('fiscal_year'):
        lag.append({'group':'fiscal_year','period':int(year),**quantiles(g.lag_days)})
    ctx.export_csv('04_event_lag_days.csv',lag)
    rows=[]
    for part,g in f[f.target.notna()].groupby('partition'):
        pp=p[p.partition==part];fixed=int((pp.lag_days<=1095).sum())
        rows.append({'group':'partition','period':part,'eligible_rows_unchanged':len(g),'common_cutoff_positive':len(pp),'fixed_1095_positive':fixed,'positive_only_after_1095':len(pp)-fixed,'fixed_1095_negative':len(g)-fixed,'retained_fraction_of_original_positive':fixed/len(pp) if len(pp) else None})
    for yr,g in f[f.target.notna()].groupby('fiscal_year'):
        pp=p[p.fiscal_year==yr];fixed=int((pp.lag_days<=1095).sum())
        rows.append({'group':'fiscal_year','period':int(yr),'eligible_rows_unchanged':len(g),'common_cutoff_positive':len(pp),'fixed_1095_positive':fixed,'positive_only_after_1095':len(pp)-fixed,'fixed_1095_negative':len(g)-fixed,'retained_fraction_of_original_positive':fixed/len(pp) if len(pp) else None})
    ctx.export_csv('05_fixed_1095_counts_ONLY.csv',rows)
    ctx.export_json('05_fixed_window_scope.json',{'classification':'descriptive counterfactual on the existing eligible risk set','time_rule':'0 < earliest_qualified_public_event - anchor <= 1095 days','frozen_labels_written_or_replaced':False,'null_rows_reclassified_as_negative':False,'models_retrained':False,'followup_differences_causally_explain_performance':False})
    overlaps=[]
    q['case_key']=q.apply(document_key,axis=1)
    order={'train':0,'validation':1,'test':2}
    for typ,col in [('source_record','row_ref'),('issuer_document','case_key')]:
        a=q[q[col].notna()].copy();rank=a.partition.map(order)
        minimum={k:int(v) for k,v in a.assign(rank=rank).groupby(col,sort=False)['rank'].min().items()}
        splitsets=defaultdict(set)
        for k,part in zip(a[col],a.partition):splitsets[k].add(part)
        coverage=set(map(tuple,a[['company_code','fiscal_year']].to_numpy()))
        for part in ['validation','test']:
            b=a[a.partition==part]
            shared={tuple(v) for v in b.loc[b[col].map(minimum)<order[part],['company_code','fiscal_year']].to_numpy()}
            gg=p[p.partition==part];keys=set(map(tuple,gg[['company_code','fiscal_year']].to_numpy()))
            overlaps.append({'key_level':typ,'partition':part,'positive_rows_total':len(gg),'positive_rows_with_usable_key':len(keys&coverage),'positive_rows_with_no_usable_key':len(keys-coverage),'positive_rows_with_key_also_in_earlier_partition':len(shared),'unique_keys_here':b[col].nunique(),'unique_keys_spanning_multiple_partitions':sum(1 for k in b[col].unique() if len(splitsets[k])>1),'interpretation':'document overlap lower bound on identifiable keys; source records are not legal cases' if typ=='issuer_document' else 'source-row recurrence only; do not call this case leakage'})
    ctx.export_csv('06_case_and_source_overlap.csv',overlaps)
    # Earlier positive fiscal years versus actually announced by each current anchor.
    bycompany={c:g for c,g in p.groupby('company_code')}
    rr=[]
    for part in ['validation','test']:
        g=p[p.partition==part];earlier_n=known_n=0
        for _,row in g.iterrows():
            prior=bycompany[row.company_code]
            prior=prior[prior.fiscal_year<row.fiscal_year]
            earlier_n+=int(len(prior)>0)
            known_n+=int((prior.first_event<=row.anchor).any())
        rr.append({'partition':part,'positive_rows':len(g),'positive_rows_any_earlier_fiscal_year_positive':earlier_n,'positive_rows_earlier_fiscal_positive_public_by_anchor':known_n,'scope':'observed qualifying outcomes from this panel, not all historical sanctions'})
    ctx.export_csv('07_prior_positive_known_by_anchor.csv',rr)
    documented=q.case_key.notna().sum()
    ctx.record('endpoint_timing','PASS',note='All frozen positive keys matched to qualified events; earliest-event lags and fixed-window counts computed.')
    ctx.record('case_overlap','PASS' if documented==len(q) else 'PARTIAL',note='Issuer-document keys reported separately from source-row keys; unidentified documents remain unknown.',qualified_source_rows=len(q),rows_with_document_key=int(documented))
