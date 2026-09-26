#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

QUAL={'QUALIFIED_A_POSTFILING_FORMAL_CSRC','QUALIFIED_B_POSTFILING_FORMAL_CSRC'}

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def find_by_hash(root:Path,name:str,expected:str):
    for p in root.rglob(name):
        if p.is_file():
            try:
                if sha256(p)==expected:return p
            except Exception:pass
    raise RuntimeError(f'HASH_MATCH_NOT_FOUND:{name}:{expected}')

def date_series(s):
    """Parse both compact YYYYMMDD and ISO-like YYYY-MM-DD dates.

    Stage-E parquet may preserve declare_date / anchor as ISO strings; the
    final-check audit parser already accepts both forms. Keep this prep tool
    consistent with that frozen audit logic.
    """
    x=s.astype('string').str.replace(r'\.0$','',regex=True).str.strip()
    a=pd.to_datetime(x,format='%Y%m%d',errors='coerce')
    b=pd.to_datetime(x.str.slice(0,10),format='%Y-%m-%d',errors='coerce')
    return a.fillna(b).dt.normalize()

def company_codes(s):
    return s.astype('string').str.extract(r'(\d{6})',expand=False).str.zfill(6)

def partition(y):
    y=int(y)
    if 2010<=y<=2018:return 'train'
    if y in (2019,2020):return 'validation'
    if y in (2021,2022):return 'test'
    return 'outside_primary'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit-root',type=Path,default=Path.home()/'peerj_141707_audit')
    ap.add_argument('--out-root',type=Path,default=Path.home()/'peerj_141707_sensitivity_prep/fixed1095')
    ap.add_argument('--protocol',type=Path,required=True)
    args=ap.parse_args()
    P=json.loads(args.protocol.read_text())
    out=args.out_root; private=out/'PRIVATE'; share=out/'SHARE'
    private.mkdir(parents=True,exist_ok=True);share.mkdir(parents=True,exist_ok=True)

    stageF=find_by_hash(args.audit_root,'manifest_v1_stageF.json',P['stageF_manifest_sha256'])
    fj=json.loads(stageF.read_text())
    labels=stageF.parent/'fraud_labels_v1_0.parquet'
    if not labels.is_file() or sha256(labels)!=P['frozen_hashes']['fraud_labels_v1_0.parquet']:
        labels=find_by_hash(args.audit_root,'fraud_labels_v1_0.parquet',P['frozen_hashes']['fraud_labels_v1_0.parquet'])
    me=find_by_hash(args.audit_root,'manifest_v1_stageE.json',fj['stageE_manifest_sha256'])
    pq=me.parent/'pair_qualification_v1_stageE.parquet'
    if not pq.is_file(): raise RuntimeError('PAIR_QUALIFICATION_NOT_FOUND_NEXT_TO_STAGEE_MANIFEST')

    f=pd.read_parquet(labels)
    need={'company_code','fiscal_year',P['primary_label'],'annual_report_public_anchor'}
    if not need<=set(f.columns):raise RuntimeError('LABEL_SCHEMA_MISSING:'+','.join(sorted(need-set(f.columns))))
    f=f.copy();f['company_code']=company_codes(f.company_code);f['fiscal_year']=f.fiscal_year.astype(int)
    f['anchor']=date_series(f.annual_report_public_anchor)
    original=pd.to_numeric(f[P['primary_label']],errors='coerce')

    q=pd.read_parquet(pq)
    q=q[(q['scope']=='strict') & q['stageE_pair_status'].isin(QUAL)].copy()
    q['company_code']=company_codes(q.company_code);q['fiscal_year']=q.raw_violation_year.astype(int)
    q['event_date']=date_series(q.declare_date)
    q=q.merge(f[['company_code','fiscal_year','anchor',P['primary_label']]],on=['company_code','fiscal_year'],how='inner',validate='many_to_one')
    q=q[pd.to_numeric(q[P['primary_label']],errors='coerce').eq(1)].copy()
    if q.event_date.isna().any(): raise RuntimeError('QUALIFIED_EVENT_DATE_MISSING')
    if not (q.event_date>q.anchor).all():raise RuntimeError('QUALIFIED_EVENT_NOT_POST_ANCHOR')
    earliest=q.groupby(['company_code','fiscal_year'],as_index=False)['event_date'].min().rename(columns={'event_date':'earliest_qualified_event'})

    g=f.merge(earliest,on=['company_code','fiscal_year'],how='left',validate='one_to_one')
    fixed=original.copy()
    pos=original.eq(1)
    if g.loc[pos,'earliest_qualified_event'].isna().any():raise RuntimeError('POSITIVE_WITHOUT_QUALIFIED_EVENT')
    lag=(g.earliest_qualified_event-g.anchor).dt.days
    fixed.loc[pos]=lag.loc[pos].le(P['fixed_window_days']).astype(int)
    fixed.loc[original.isna()]=np.nan

    col=P['fixed_label_column']
    f[col]=fixed
    f['label_v1_strict_ab_primary_common_cutoff']=original
    # Do not overwrite primary column.
    outp=private/'fraud_labels_fixed1095_v1.parquet'
    f.drop(columns=['anchor'],errors='ignore').to_parquet(outp,index=False)

    rows=[]
    for part in ['train','validation','test','outside_primary']:
        mask=f.fiscal_year.map(partition).eq(part)
        old=original[mask];new=fixed[mask]
        rows.append({'partition':part,'panel_rows':int(mask.sum()),'eligible_rows':int(new.notna().sum()),'common_cutoff_positive':int(old.eq(1).sum()),'fixed1095_positive':int(new.eq(1).sum()),'reclassified_positive_to_negative':int((old.eq(1)&new.eq(0)).sum()),'null_rows':int(new.isna().sum())})
    counts=pd.DataFrame(rows)
    counts.to_csv(share/'fixed1095_counts.csv',index=False)
    for part,(n,p) in P['expected_fixed1095_counts'].items():
        r=counts[counts.partition==part].iloc[0]
        if (int(r.eligible_rows),int(r.fixed1095_positive))!=(n,p):
            raise RuntimeError(f'FIXED_COUNT_MISMATCH:{part}:{(int(r.eligible_rows),int(r.fixed1095_positive))}!={(n,p)}')

    manifest={
      'status':'PASS','source_labels':str(labels),'source_labels_sha256':sha256(labels),
      'stageF_manifest':str(stageF),'stageF_manifest_sha256':sha256(stageF),
      'stageE_manifest':str(me),'stageE_manifest_sha256':sha256(me),
      'pair_qualification_sha256':sha256(pq),'output_private_path':str(outp),'output_sha256':sha256(outp),
      'fixed_window_days':P['fixed_window_days'],'label_column':col,
      'scope':'existing eligible risk set unchanged; original null rows remain null; only original positives with earliest qualifying event after 1095 days are reclassified negative for sensitivity training'
    }
    (share/'fixed1095_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (share/'README.txt').write_text('PASS. PRIVATE/fraud_labels_fixed1095_v1.parquet contains company identifiers and is for transfer to the private 4090 only. Do not upload it to public release. SHARE contains aggregate checks safe to return for review.\n')
    zpath=out/'peerj_fixed1095_prep_RETURN.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(share.iterdir()):z.write(p,p.name)
    print(counts.to_string(index=False))
    print('FIXED1095_PREP_PASS')
    print('PRIVATE_FILE='+str(outp))
    print('RETURN_ZIP='+str(zpath))

if __name__=='__main__': main()
