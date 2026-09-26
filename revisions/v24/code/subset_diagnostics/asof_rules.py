#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sqlite3, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

FIN_FIELDS={
 'balancesheet':['total_assets','total_cur_assets','total_cur_liab','total_liab','accounts_receiv','oth_receiv','fix_assets','surplus_rese','undistr_porfit','total_hldr_eqy_inc_min_int'],
 'income':['revenue','oper_cost','sell_exp','admin_exp','n_income_attr_p','ebit'],
 'cashflow':['n_cashflow_act','depr_fa_coga_dpba','amort_intang_assets'],
}
META=['ts_code','end_date','ann_date','f_ann_date','comp_type','report_type','update_flag']

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ccode(s):
    return s.astype('string').str.extract(r'(\d{6})',expand=False).str.zfill(6)

def dates(s):
    """Parse both compact YYYYMMDD and ISO-like YYYY-MM-DD dates.

    Frozen v16 label/anchor parquet stores the annual-report anchor in an
    ISO-like representation, while source financial tables may use compact
    YYYYMMDD. Accept both without changing date semantics.
    """
    x=s.astype('string').str.replace(r'\.0$','',regex=True).str.strip()
    a=pd.to_datetime(x,format='%Y%m%d',errors='coerce')
    b=pd.to_datetime(x.str.slice(0,10),format='%Y-%m-%d',errors='coerce')
    return a.fillna(b).dt.normalize()

def safe_div(a,b):
    b=b.replace(0,np.nan)
    return (a/b).replace([np.inf,-np.inf],np.nan)

def load_panel(labels_path,node_path,primary):
    labels=pd.read_parquet(labels_path)
    nf=pd.read_parquet(node_path)
    need={'company_code','fiscal_year',primary,'annual_report_public_anchor'}
    if not need<=set(labels):raise RuntimeError('LABEL_SCHEMA_MISSING:'+','.join(sorted(need-set(labels))))
    labels=labels.copy();labels['company_code']=ccode(labels.company_code);labels['fiscal_year']=labels.fiscal_year.astype(int);labels['anchor']=dates(labels.annual_report_public_anchor)
    nf=nf.copy();nf['company_code']=ccode(nf.firm_id);nf['fiscal_year']=nf.year.astype(int)
    if len(labels)!=51675 or len(nf)!=51675:raise RuntimeError('PANEL_ROWCOUNT')
    if labels.duplicated(['company_code','fiscal_year']).any() or nf.duplicated(['company_code','fiscal_year']).any():raise RuntimeError('PANEL_DUPLICATE_KEYS')
    panel=nf.merge(labels[['company_code','fiscal_year','anchor',primary]],on=['company_code','fiscal_year'],how='left',validate='one_to_one')
    panel=panel.reset_index(drop=True);panel['__rowid']=np.arange(len(panel));panel['target']=pd.to_numeric(panel[primary],errors='coerce')
    return labels,nf,panel

def read_table(con,table,fields):
    info=pd.read_sql_query(f'PRAGMA table_info("{table}")',con)
    cols=set(info['name'])
    req={'ts_code','end_date'}|set(fields)
    miss=req-cols
    if miss:raise RuntimeError(f'{table}_MISSING:'+','.join(sorted(miss)))
    chosen=[]
    for c in META+fields:
        if c in cols and c not in chosen: chosen.append(c)
    q='SELECT rowid AS __sqlite_rowid,'+','.join('"'+c+'"' for c in chosen)+f' FROM "{table}" WHERE CAST(end_date AS TEXT) LIKE \'%1231\''
    if table!='fina_indicator' and 'comp_type' in cols:q+=" AND CAST(comp_type AS TEXT)='1'"
    d=pd.read_sql_query(q,con)
    d['company_code']=ccode(d.ts_code);d['source_year']=pd.to_numeric(d.end_date.astype(str).str[:4],errors='raise').astype(int)
    for c in fields:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['__ann']=dates(d.ann_date) if 'ann_date' in d else pd.NaT
    d['__fann']=dates(d.f_ann_date) if 'f_ann_date' in d else pd.NaT
    d['__avail']=pd.concat([d['__ann'],d['__fann']],axis=1).max(axis=1)
    d['__update']=pd.to_numeric(d.update_flag,errors='coerce') if 'update_flag' in d else np.nan
    return d

def select_for_panel(raw,panel,lag,kind,asof=True):
    keys=panel[['__rowid','company_code','fiscal_year','anchor']].copy();keys['source_year']=keys.fiscal_year-lag
    m=keys.merge(raw,on=['company_code','source_year'],how='left',sort=False)
    if asof:
        if kind=='fini':ok=m.__ann.notna() & m.anchor.notna() & (m.__ann<=m.anchor)
        else:ok=m.__avail.notna() & m.anchor.notna() & (m.__avail<=m.anchor)
        m=m[ok].copy()
    else:
        # Current-snapshot replay for validation, no anchor filter.
        m=m[m.__sqlite_rowid.notna()].copy()
    if len(m)==0:
        return pd.DataFrame(index=panel.__rowid)
    if kind=='fini':
        m=m.sort_values(['__rowid','__ann','__sqlite_rowid'],ascending=[True,False,True],kind='mergesort')
    else:
        # As-of sensitivity: preserve max-update intent, make ties deterministic by latest recorded availability then rowid.
        m=m.sort_values(['__rowid','__update','__avail','__sqlite_rowid'],ascending=[True,False,False,True],kind='mergesort')
    m=m.drop_duplicates('__rowid',keep='first').set_index('__rowid')
    return m.reindex(panel.__rowid)

def compute_fin(aligned):
    period={}
    for lag in [0,1,2]:
        pieces=[]
        for table,fields in FIN_FIELDS.items(): pieces.append(aligned[(table,lag)][fields])
        period[lag]=pd.concat(pieces,axis=1)
    out={}
    def mscore(t,m):
        dsri=safe_div(safe_div(t.accounts_receiv,t.revenue),safe_div(m.accounts_receiv,m.revenue))
        gmi=safe_div(1-safe_div(m.oper_cost,m.revenue),1-safe_div(t.oper_cost,t.revenue))
        aqi=safe_div(1-safe_div(t.total_cur_assets+t.fix_assets,t.total_assets),1-safe_div(m.total_cur_assets+m.fix_assets,m.total_assets))
        sgi=safe_div(t.revenue,m.revenue)
        dt=t.depr_fa_coga_dpba.fillna(0)+t.amort_intang_assets.fillna(0);dm=m.depr_fa_coga_dpba.fillna(0)+m.amort_intang_assets.fillna(0)
        depi=safe_div(safe_div(dm,dm+m.fix_assets),safe_div(dt,dt+t.fix_assets))
        sgai=safe_div(safe_div(t.sell_exp.fillna(0)+t.admin_exp.fillna(0),t.revenue),safe_div(m.sell_exp.fillna(0)+m.admin_exp.fillna(0),m.revenue))
        tata=safe_div(t.n_income_attr_p-t.n_cashflow_act,t.total_assets)
        lvgi=safe_div(safe_div(t.total_liab,t.total_assets),safe_div(m.total_liab,m.total_assets))
        return dsri,-4.84+.92*dsri+.528*gmi+.404*aqi+.892*sgi+.115*depi-.172*sgai+4.679*tata-.327*lvgi
    for lag in [0,1]:
        t=period[lag];m=period[lag+1];ds,ms=mscore(t,m)
        out[f'fin_dsri_t{lag}']=ds;out[f'fin_mscore_t{lag}']=ms
        out[f'fin_ocf_to_ni_t{lag}']=safe_div(t.n_cashflow_act,t.n_income_attr_p).where(t.n_income_attr_p.abs()>=1_000_000)
        out[f'fin_oth_recv_to_ta_t{lag}']=safe_div(t.oth_receiv,t.total_assets.where(t.total_assets>0))
    t=period[0]
    out['fin_zscore_t0']=.717*safe_div(t.total_cur_assets-t.total_cur_liab,t.total_assets)+.847*safe_div(t.surplus_rese.fillna(0)+t.undistr_porfit.fillna(0),t.total_assets)+3.107*safe_div(t.ebit,t.total_assets)+.420*safe_div(t.total_hldr_eqy_inc_min_int,t.total_liab)+.998*safe_div(t.revenue,t.total_assets)
    return pd.DataFrame(out)

def industry_map(stock_db):
    con=sqlite3.connect(f'file:{stock_db}?mode=ro',uri=True)
    info=pd.read_sql_query('PRAGMA table_info(stock_basic)',con);cols=set(info.name)
    choose=['industry']
    if 'firm_id' in cols:choose=['firm_id','industry']
    elif 'ts_code' in cols:choose=['ts_code','industry']
    else:con.close();raise RuntimeError('STOCK_BASIC_NO_FIRM_OR_TS_CODE')
    d=pd.read_sql_query('SELECT '+','.join(choose)+' FROM stock_basic',con);con.close()
    if 'firm_id' in d:d['company_code']=ccode(d.firm_id)
    else:d['company_code']=ccode(d.ts_code)
    # Flag conflicting industries, then choose first nonmissing deterministically.
    conf=d.dropna(subset=['industry']).groupby('company_code').industry.nunique();nconf=int((conf>1).sum())
    d=d.dropna(subset=['company_code']).sort_values(['company_code','industry'],na_position='last').drop_duplicates('company_code')
    return d.set_index('company_code').industry,nconf

def add_zind(panel,features,indmap,fini_fields):
    years=panel.fiscal_year.reset_index(drop=True)

    # Use the industry identity stored in the frozen v16 node-feature rows.
    # The current stock_basic DB is retained only as a diagnostic/hash source.
    # This avoids introducing today's one-industry-per-company mapping into a
    # replay of historical frozen industry-year transforms.
    if 'industry' not in panel.columns:
        raise RuntimeError('FROZEN_PANEL_INDUSTRY_MISSING')
    inds=panel['industry'].astype('string').str.strip().reset_index(drop=True)
    inds=inds.mask(inds.isin(['', 'nan', 'None', '<NA>']))
    # fin zind: original source uses group row length >=10 (std ignores NaN)
    finraw=['fin_dsri_t0','fin_dsri_t1','fin_mscore_t0','fin_mscore_t1','fin_ocf_to_ni_t0','fin_ocf_to_ni_t1','fin_oth_recv_to_ta_t0','fin_oth_recv_to_ta_t1','fin_zscore_t0']
    for c in finraw:
        s=features[c]
        tmp=pd.DataFrame({'year':years,'industry':inds,'x':s})
        grp=tmp.groupby(['year','industry'],dropna=True).x
        mean=grp.transform('mean');std=grp.transform('std');size=grp.transform('size')
        z=(s-mean)/std;z=z.where((size>=10)&std.ne(0)&inds.notna())
        features[c+'_zind']=z
    # fini zind: original source uses valid count >=10.
    for f in fini_fields['ZIND_FIELDS']:
        c=f'fini_{f}_t0';s=features[c]
        tmp=pd.DataFrame({'year':years,'industry':inds,'x':s})
        grp=tmp.groupby(['year','industry'],dropna=False).x
        mean=grp.transform('mean');std=grp.transform('std');cnt=grp.transform('count')
        z=(s-mean)/std;z=z.where((cnt>=10)&std.gt(0)&inds.notna())
        features[c+'_zind']=z
    return features

def make_features(raw_tables,panel,fini_fields,indmap,asof):
    aligned={}
    for table in FIN_FIELDS:
        for lag in [0,1,2]: aligned[(table,lag)]=select_for_panel(raw_tables[table],panel,lag,'fin',asof)
    for lag in [0,1]: aligned[('fina_indicator',lag)]=select_for_panel(raw_tables['fina_indicator'],panel,lag,'fini',asof)
    feat=compute_fin(aligned)
    for lag,key in [(0,'T0_FIELDS'),(1,'T1_FIELDS')]:
        a=aligned[('fina_indicator',lag)]
        for f in fini_fields[key]:feat[f'fini_{f}_t{lag}']=pd.to_numeric(a[f],errors='coerce')
    feat=add_zind(panel,feat,indmap,fini_fields)
    # row exposure summary
    exp=pd.DataFrame(index=panel.index)
    exp['any_source_missing']=False;exp['any_selected_post_anchor']=False
    for (table,lag),a in aligned.items():
        exp['any_source_missing'] |= a.__sqlite_rowid.isna().to_numpy()
        date=a.__ann if table=='fina_indicator' else a.__avail
        exp['any_selected_post_anchor'] |= (date.notna() & panel.anchor.notna() & (date.reset_index(drop=True)>panel.anchor.reset_index(drop=True))).to_numpy()
    return feat,aligned,exp

def compare_replay(frozen,replay,cols):
    rows=[]
    for c in cols:
        old=pd.to_numeric(frozen[c],errors='coerce').to_numpy(float,na_value=np.nan);new=pd.to_numeric(replay[c],errors='coerce').to_numpy(float,na_value=np.nan)
        finite=np.isfinite(old)&np.isfinite(new);eq=finite&np.isclose(old,new,rtol=1e-5,atol=1e-6)
        both=(~np.isfinite(old))&(~np.isfinite(new));zero=(old==0)&(~np.isfinite(new));match=eq|both|zero
        if finite.sum() >= 3:
            oo=old[finite]; nn=new[finite]
            if np.nanstd(oo)>0 and np.nanstd(nn)>0:
                corr=float(np.corrcoef(oo,nn)[0,1])
            else:
                corr=1.0 if np.allclose(oo,nn,rtol=1e-5,atol=1e-6,equal_nan=True) else None
            ad=np.abs(oo-nn)
            med_abs=float(np.nanmedian(ad))
            p95_abs=float(np.nanquantile(ad,0.95))
        else:
            corr=None;med_abs=None;p95_abs=None
        rows.append({
            'feature':c,
            'comparable_finite':int(finite.sum()),
            'finite_match':int(eq.sum()),
            'finite_match_rate':float(eq.sum()/finite.sum()) if finite.sum() else None,
            'all_row_compatible_rate':float(match.mean()),
            'mismatch_rows':int((~match).sum()),
            'pearson_r':corr,
            'median_abs_diff':med_abs,
            'p95_abs_diff':p95_abs,
        })
    return pd.DataFrame(rows)

def part(y):
    if 2010<=y<=2018:return 'train'
    if y in [2019,2020]:return 'validation'
    if y in [2021,2022]:return 'test'
    return 'outside_primary'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--node-features',type=Path,required=True)
    ap.add_argument('--financial-db',type=Path,required=True);ap.add_argument('--stock-basic-db',type=Path,required=True)
    ap.add_argument('--out-root',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--fini-fields',type=Path,required=True)
    args=ap.parse_args();P=json.loads(args.protocol.read_text());fini=json.loads(args.fini_fields.read_text())
    out=args.out_root;private=out/'PRIVATE';share=out/'SHARE';private.mkdir(parents=True,exist_ok=True);share.mkdir(parents=True,exist_ok=True)
    if sha256(args.labels)!=P['frozen_hashes']['fraud_labels_v1_0.parquet']:raise RuntimeError('LABEL_HASH_MISMATCH')
    if sha256(args.node_features)!=P['frozen_hashes']['node_features_v1_1.parquet']:raise RuntimeError('NODE_HASH_MISMATCH')
    finhash=sha256(args.financial_db)
    if finhash!=P['frozen_hashes']['financials.db']:raise RuntimeError('FINANCIAL_DB_HASH_MISMATCH:'+finhash)
    labels,nf,panel=load_panel(args.labels,args.node_features,P['primary_label'])
    if panel.anchor.isna().sum()==len(panel):raise RuntimeError('NO_ANCHORS')
    con=sqlite3.connect(f'file:{args.financial_db.resolve()}?mode=ro&immutable=1',uri=True)
    raw={}
    for t,fields in {**FIN_FIELDS,'fina_indicator':fini['T0_FIELDS']}.items():raw[t]=read_table(con,t,fields)
    con.close();indmap,nconf=industry_map(args.stock_basic_db)

    # Current-snapshot replay validates the recovered feature formulas/rules against frozen values.
    current,_,_=make_features(raw,panel,fini,indmap,False)
    asof,aligned,exposure=make_features(raw,panel,fini,indmap,True)

    # The recovered upstream extractor can generate more financial candidate
    # columns than the frozen model schema retained.  v16's frozen M5 schema
    # contains exactly 18 fin_ + 69 fini_ = 87 financial inputs.
    #
    # Nine raw per-share fini_ fields are generated upstream but are NOT
    # retained as model inputs.  They remain available here as intermediates
    # (e.g. fini_ocfps_t0 is needed to reconstruct its retained z-score), but
    # must not be added to the frozen node-feature schema.
    expected_generated_not_frozen=sorted([
        'fini_bps_t0',
        'fini_cfps_t0',
        'fini_ebit_ps_t0',
        'fini_eps_t0',
        'fini_fcfe_ps_t0',
        'fini_fcff_ps_t0',
        'fini_ocfps_t0',
        'fini_total_revenue_ps_t0',
        'fini_undist_profit_ps_t0',
    ])
    generated_cols=sorted(current.columns)
    frozen_financial_cols=sorted([
        c for c in nf.columns
        if c.startswith('fin_') or c.startswith('fini_')
    ])

    missing_from_replay=[
        c for c in frozen_financial_cols
        if c not in current.columns
    ]
    if missing_from_replay:
        raise RuntimeError(
            'FROZEN_FINANCIAL_COLUMNS_NOT_RECONSTRUCTED:'+
            ','.join(missing_from_replay)
        )

    generated_not_frozen=sorted([
        c for c in generated_cols
        if c not in nf.columns
    ])
    if generated_not_frozen!=expected_generated_not_frozen:
        raise RuntimeError(
            'UNEXPECTED_GENERATED_NOT_FROZEN:'+
            ','.join(generated_not_frozen)
        )

    n_fin=sum(c.startswith('fin_') for c in frozen_financial_cols)
    n_fini=sum(c.startswith('fini_') for c in frozen_financial_cols)
    if (n_fin,n_fini)!=(18,69):
        raise RuntimeError(
            f'FROZEN_FINANCIAL_SCHEMA_EXPECTED_18_69_GOT_{n_fin}_{n_fini}'
        )

    replace_cols=frozen_financial_cols
    if len(replace_cols)!=87:
        raise RuntimeError(
            f'EXPECTED_87_RETAINED_FINANCIAL_FEATURES_GOT_{len(replace_cols)}'
        )

    replay=compare_replay(
        nf.reset_index(drop=True),
        current.reset_index(drop=True),
        replace_cols
    )
    replay.to_csv(share/'asof_current_snapshot_replay_compatibility.csv',index=False)
    finraw=[c for c in replace_cols if c.startswith('fin_') and not c.endswith('_zind')]
    finiraw=[c for c in replace_cols if c.startswith('fini_') and not c.endswith('_zind')]
    fin_min=float(replay[replay.feature.isin(finraw)].finite_match_rate.min())
    fini_min=float(replay[replay.feature.isin(finiraw)].finite_match_rate.min())
    zcols=[c for c in replace_cols if c.endswith('_zind')]
    zr=replay[replay.feature.isin(zcols)].copy()
    z_corr_min=float(zr.pearson_r.dropna().min()) if zr.pearson_r.notna().any() else float('nan')
    z_med_abs_max=float(zr.median_abs_diff.dropna().max()) if zr.median_abs_diff.notna().any() else float('nan')

    # Replay gates. Raw fields must nearly reproduce frozen values.
    if fin_min<0.995:raise RuntimeError(f'FIN_REPLAY_TOO_LOW:{fin_min}')
    if fini_min<0.99:raise RuntimeError(f'FINI_REPLAY_TOO_LOW:{fini_min}')

    # The 24 industry-year transformed fields are allowed tiny propagated
    # differences from a small number of raw-current-snapshot discrepancies,
    # but the reconstructed values must remain numerically very close.
    if (not np.isfinite(z_corr_min)) or z_corr_min<0.99:
        raise RuntimeError(f'ZIND_REPLAY_CORRELATION_TOO_LOW:{z_corr_min}')
    if (not np.isfinite(z_med_abs_max)) or z_med_abs_max>0.05:
        raise RuntimeError(f'ZIND_REPLAY_MEDIAN_ABS_DIFF_TOO_HIGH:{z_med_abs_max}')

    outnf=nf.copy().reset_index(drop=True)
    for c in replace_cols:outnf[c]=asof[c].to_numpy()
    outp=private/'node_features_asof_financial_v1.parquet';outnf.to_parquet(outp,index=False)

    # Summaries without identifiers.
    changed=[]
    for c in replace_cols:
        old=pd.to_numeric(nf[c],errors='coerce').to_numpy(float,na_value=np.nan);new=pd.to_numeric(outnf[c],errors='coerce').to_numpy(float,na_value=np.nan)
        same=(np.isfinite(old)&np.isfinite(new)&np.isclose(old,new,rtol=1e-5,atol=1e-6))|((~np.isfinite(old))&(~np.isfinite(new)))
        changed.append({'feature':c,'changed_rows':int((~same).sum()),'changed_fraction':float((~same).mean()),'asof_missing':int((~np.isfinite(new)).sum()),'frozen_missing':int((~np.isfinite(old)).sum())})
    pd.DataFrame(changed).to_csv(share/'asof_feature_changes.csv',index=False)
    summary=[]
    diffs=pd.DataFrame(index=panel.index)
    for c in replace_cols:
        old=pd.to_numeric(nf[c],errors='coerce').to_numpy(float,na_value=np.nan);new=pd.to_numeric(outnf[c],errors='coerce').to_numpy(float,na_value=np.nan)
        same=(np.isfinite(old)&np.isfinite(new)&np.isclose(old,new,rtol=1e-5,atol=1e-6))|((~np.isfinite(old))&(~np.isfinite(new)))
        diffs[c]=~same
    anychanged=diffs.any(axis=1)
    panel['partition']=panel.fiscal_year.map(part);panel['label_group']=panel.target.map({0.0:'negative',1.0:'positive'}).fillna('null')
    for (p,l),idx in panel.groupby(['partition','label_group']).groups.items():
        ii=np.asarray(list(idx),dtype=int);summary.append({'partition':p,'label_group':l,'rows':len(ii),'rows_with_any_financial_feature_changed':int(anychanged.iloc[ii].sum()),'fraction_changed':float(anychanged.iloc[ii].mean()),'rows_with_any_asof_source_missing':int(exposure.any_source_missing.iloc[ii].sum()),'selected_source_post_anchor_after_asof_filter':int(exposure.any_selected_post_anchor.iloc[ii].sum())})
    pd.DataFrame(summary).to_csv(share/'asof_change_summary_by_partition_label.csv',index=False)
    manifest={'status':'PASS','definition':P['asof_definition'],'source_label_sha256':sha256(args.labels),'source_node_sha256':sha256(args.node_features),'financial_db_sha256':finhash,'stock_basic_sha256':sha256(args.stock_basic_db),'output_private_path':str(outp),'output_sha256':sha256(outp),'replaced_feature_count':len(replace_cols),'retained_fin_count':n_fin,'retained_fini_count':n_fini,'generated_candidate_financial_count':len(generated_cols),'generated_not_frozen_intermediate_fields':generated_not_frozen,'replay_fin_raw_min_finite_match_rate':fin_min,'replay_fini_raw_min_finite_match_rate':fini_min,'replay_zind_min_pearson_r':z_corr_min,'replay_zind_max_median_abs_diff':z_med_abs_max,'industry_reference_for_zind':'frozen_v16_node_features.industry','stock_basic_company_codes_with_conflicting_nonmissing_industry':nconf,'limitations':['Current DB snapshot may not contain every historically available vendor version.','Source date semantics are treated as recorded availability dates but are not independently vendor-certified.','Industry-year z-scores use the frozen v16 row-level industry metadata and each company-year row\'s as-of reconstructed raw value; this is a current-snapshot sensitivity and does not certify a fully contemporaneous cross-company information set at every individual filing timestamp.']}
    (share/'asof_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (share/'README.txt').write_text('PASS. PRIVATE/node_features_asof_financial_v1.parquet is a private sensitivity input for transfer to the 4090 only. It is not a replacement for the frozen v16 features and should not be publicly released without separate review.\n')
    zpath=out/'peerj_asof_prep_RETURN.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(share.iterdir()):z.write(p,p.name)
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('ASOF_PREP_PASS')
    print('PRIVATE_FILE='+str(outp));print('RETURN_ZIP='+str(zpath))

if __name__=='__main__':main()
