"""Replay documented selection/formula rules in memory and quantify their date exposure.
No trained model, rewritten feature file, source DB change, or automatic sensitivity run.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from .common import *

FIN_FIELDS={
 'balancesheet':['total_assets','total_cur_assets','total_cur_liab','total_liab','accounts_receiv','oth_receiv','fix_assets','surplus_rese','undistr_porfit','total_hldr_eqy_inc_min_int'],
 'income':['revenue','oper_cost','sell_exp','admin_exp','n_income_attr_p','ebit'],
 'cashflow':['n_cashflow_act','depr_fa_coga_dpba','amort_intang_assets'],
}
META=['ts_code','end_date','ann_date','f_ann_date','comp_type','report_type','update_flag']

def safe_div(a,b):
    return (a/b.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)

def select_records(raw,table):
    """Same primary keys and sorting variables as the retained source; flag ties."""
    d=raw.copy()
    require({'ts_code','end_date'}<=set(d),'FINANCIAL_TABLE_KEYS:'+table)
    if table=='fina_indicator':
        require('ann_date' in d,'FINI_ANN_DATE_MISSING')
        d['__selection_rank']=pd.to_numeric(d.ann_date,errors='coerce')
        d=d.sort_values('__selection_rank',ascending=False)
        rank='__selection_rank'
    elif 'update_flag' in d:
        # Source leaves this metadata column uncast. Preserve its actual dtype.
        rank='update_flag';d=d.sort_values(['ts_code','end_date',rank],ascending=[True,True,False])
    else:
        require(not d.duplicated(['ts_code','end_date']).any(),'FIN_NO_UPDATE_FLAG_BUT_DUPLICATE_RECORDS:'+table)
        rank=None
    selected=d.drop_duplicates(['ts_code','end_date'],keep='first').copy()
    keys=['ts_code','end_date']
    if rank is not None:
        top=selected[keys+[rank]].rename(columns={rank:'__top_rank'})
        tmp=d.merge(top,on=keys,how='left',validate='many_to_one')
        winners=tmp[(tmp[rank]==tmp.__top_rank)|(tmp[rank].isna()&tmp.__top_rank.isna())]
        cnt=winners.groupby(keys,dropna=False).size().rename('__top_candidates')
        selected=selected.merge(cnt,on=keys,how='left',validate='one_to_one')
    else:selected['__top_candidates']=1
    selected['company_code']=company_codes(selected.ts_code)
    selected['source_year']=pd.to_numeric(selected.end_date.astype(str).str[:4],errors='raise').astype(int)
    require(not selected.duplicated(['company_code','source_year']).any(),'FIN_MULTI_EXCHANGE_OR_PERIOD_AMBIGUITY:'+table)
    for c in ['ann_date','f_ann_date']:
        selected['__'+c]=dates(selected[c]) if c in selected else pd.NaT
    selected['__date_any']=selected[['__ann_date','__f_ann_date']].max(axis=1)
    selected['__present']=True
    selected['__flag1']=pd.to_numeric(selected['update_flag'],errors='coerce').eq(1) if 'update_flag' in selected else False
    selected['__flag_available']='update_flag' in selected
    # Existence of an older candidate in this database snapshot, not a new as-of feature vector.
    earliest=raw[['ts_code','end_date']].copy()
    ds=[]
    for c in ['ann_date','f_ann_date']:
        if c in raw:ds.append(dates(raw[c]))
    if ds:earliest['__available_date']=pd.concat(ds,axis=1).max(axis=1)
    else:earliest['__available_date']=pd.NaT
    mini=earliest.groupby(keys,dropna=False)['__available_date'].min().rename('__earliest_in_snapshot')
    selected=selected.merge(mini,on=keys,how='left',validate='one_to_one')
    return selected

def load_tables(ctx):
    if getattr(ctx.args,'financial_db',None):candidates=[Path(ctx.args.financial_db).expanduser()]
    else:candidates=[ctx.project/'data/raw/financials/financials.db',Path.home()/'code/thesis_project/data/raw/financials/financials.db']
    p=next((p for p in candidates if p.is_file()),None)
    require(p is not None,'FINANCIALS_DB_NOT_FOUND')
    for suffix in ['-wal','-journal']:
        side=Path(str(p)+suffix)
        require(not side.exists() or side.stat().st_size==0,'FINANCIALS_DB_NOT_QUIESCENT')
    ctx.use(p,'financials_database_current_snapshot')
    fields=read_json(ctx.package/'config/fini_fields.json')
    needed={**FIN_FIELDS,'fina_indicator':fields['T0_FIELDS']}
    raw={};selected={};schema=[];scope=[]
    with closing(readonly_sqlite(p)) as con:
        for table,values in needed.items():
            info=pd.read_sql_query(f'PRAGMA table_info("{table}")',con)
            cols=set(info['name']) if len(info) else set()
            if not cols:
                ctx.record('financial_'+table,'PARTIAL',reason='SOURCE_TABLE_MISSING');continue
            require({'ts_code','end_date'}<=cols,'FIN_TABLE_MISSING_KEYS:'+table)
            chosen=[c for c in META+values if c in cols]
            require(len(chosen)==len(set(chosen)),'DUPLICATE_SELECTED_SQL_COLUMN')
            query='SELECT '+','.join('"'+c+'"' for c in chosen)+' FROM "'+table+'" WHERE CAST(end_date AS TEXT) LIKE \'%1231\''
            if table!='fina_indicator' and 'comp_type' in cols:
                query+=' AND CAST(comp_type AS TEXT)=\'1\''
            d=pd.read_sql_query(query,con)
            require(len(d)>0,'EMPTY_ANNUAL_TABLE:'+table)
            for c in values:
                if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
            raw[table]=d;selected[table]=select_records(d,table)
            schema.append({'table':table,'selected_annual_rows_before_dedup':len(d),'retained_rows_after_dedup':len(selected[table]),'missing_required_numeric_fields':'|'.join(sorted(set(values)-cols)),'ann_date_present':'ann_date' in cols,'f_ann_date_present':'f_ann_date' in cols,'update_flag_present':'update_flag' in cols,'selection_rule':'maximum ann_date' if table=='fina_indicator' else 'maximum raw update_flag per (ts_code,end_date)','historical_snapshot_identity_verified':False})
            for col in ['comp_type','report_type','update_flag']:
                if col in d:
                    for val,n in d[col].astype('string').fillna('MISSING').value_counts(dropna=False).items():
                        # Metadata values only, never arbitrary long text.
                        vv=str(val);vv=vv if re.fullmatch(r'(?:[+-]?\d+(?:\.\d+)?|MISSING)',vv) else 'NONSTANDARD_VALUE'
                        scope.append({'table':table,'field':col,'recorded_value':vv,'rows':int(n),'applied_filter':'comp_type=1 for fin; no report_type filter' if table!='fina_indicator' else 'no comp_type filter'})
    ctx.export_csv('10_financial_schema_and_selection.csv',schema)
    ctx.export_csv('11_financial_type_flag_counts.csv',scope)
    return selected,fields

def align_period(selected,panel,lag):
    a=selected.set_index(['company_code','source_year'])
    keys=pd.MultiIndex.from_arrays([panel.company_code,panel.fiscal_year-lag],names=['company_code','source_year'])
    return a.reindex(keys).reset_index(drop=True)

def compute_fin(period):
    out={}
    def mscore(t,m):
        dsri=safe_div(safe_div(t.accounts_receiv,t.revenue),safe_div(m.accounts_receiv,m.revenue))
        gmi=safe_div(1-safe_div(m.oper_cost,m.revenue),1-safe_div(t.oper_cost,t.revenue))
        aqi=safe_div(1-safe_div(t.total_cur_assets+t.fix_assets,t.total_assets),1-safe_div(m.total_cur_assets+m.fix_assets,m.total_assets))
        sgi=safe_div(t.revenue,m.revenue)
        dt=t.depr_fa_coga_dpba.fillna(0)+t.amort_intang_assets.fillna(0)
        dm=m.depr_fa_coga_dpba.fillna(0)+m.amort_intang_assets.fillna(0)
        depi=safe_div(safe_div(dm,dm+m.fix_assets),safe_div(dt,dt+t.fix_assets))
        sgai=safe_div(safe_div(t.sell_exp.fillna(0)+t.admin_exp.fillna(0),t.revenue),safe_div(m.sell_exp.fillna(0)+m.admin_exp.fillna(0),m.revenue))
        tata=safe_div(t.n_income_attr_p-t.n_cashflow_act,t.total_assets)
        lvgi=safe_div(safe_div(t.total_liab,t.total_assets),safe_div(m.total_liab,m.total_assets))
        return dsri,-4.84+.92*dsri+.528*gmi+.404*aqi+.892*sgi+.115*depi-.172*sgai+4.679*tata-.327*lvgi
    for lag in [0,1]:
        t=period[lag];m=period[lag+1]
        ds,ms=mscore(t,m)
        out[f'fin_dsri_t{lag}']=ds;out[f'fin_mscore_t{lag}']=ms
        out[f'fin_ocf_to_ni_t{lag}']=safe_div(t.n_cashflow_act,t.n_income_attr_p).where(t.n_income_attr_p.abs()>=1_000_000)
        out[f'fin_oth_recv_to_ta_t{lag}']=safe_div(t.oth_receiv,t.total_assets.where(t.total_assets>0))
    t=period[0]
    out['fin_zscore_t0']=.717*safe_div(t.total_cur_assets-t.total_cur_liab,t.total_assets)+.847*safe_div(t.surplus_rese.fillna(0)+t.undistr_porfit.fillna(0),t.total_assets)+3.107*safe_div(t.ebit,t.total_assets)+.420*safe_div(t.total_hldr_eqy_inc_min_int,t.total_liab)+.998*safe_div(t.revenue,t.total_assets)
    return out

def financial_audit(ctx):
    f=load_labels(ctx);nf=load_features(ctx)
    panel=f.merge(nf.drop(columns=['year','firm_id'],errors='ignore'),on=['company_code','fiscal_year'],how='left',validate='one_to_one').reset_index(drop=True)
    sel,fields=load_tables(ctx)
    aligned={};date_rows=[]
    for table,s in sel.items():
        for lag in ([0,1] if table=='fina_indicator' else [0,1,2]):
            a=align_period(s,panel,lag);aligned[(table,lag)]=a
            present=a.__present.fillna(False).astype(bool)
            ann=a.__ann_date;fann=a.__f_ann_date;anchor=panel.anchor
            indicators={'source_row_found':present,'anchor_known':anchor.notna(),'ann_date_known':ann.notna(),'f_ann_date_known':fann.notna(),'ann_date_after_anchor':ann.notna()&anchor.notna()&(ann>anchor),'f_ann_date_after_anchor':fann.notna()&anchor.notna()&(fann>anchor),'either_recorded_date_after_anchor':a.__date_any.notna()&anchor.notna()&(a.__date_any>anchor),'selected_update_flag_1':a.__flag1.fillna(False).astype(bool),'tied_max_selection':a.__top_candidates.fillna(0)>1,'prior_dated_record_exists_in_current_snapshot':a.__earliest_in_snapshot.notna()&anchor.notna()&(a.__earliest_in_snapshot<=anchor)}
            for (part,label),ii in panel.groupby(['partition','label_group']).groups.items():
                row={'block':'fini' if table=='fina_indicator' else 'fin','table':table,'source_fiscal_lag':lag,'partition':part,'label_group':label,'denominator_panel_rows':len(ii)}
                row.update({k:int(v.loc[ii].sum()) for k,v in indicators.items()})
                known=present&anchor.notna()&a.__date_any.notna()
                row['denominator_dated_selected_rows']=int(known.loc[ii].sum())
                row['post_anchor_fraction_among_dated']=row['either_recorded_date_after_anchor']/row['denominator_dated_selected_rows'] if row['denominator_dated_selected_rows'] else None
                row['scope']='selection replay on available DB; frozen-value binding is assessed separately'
                date_rows.append(row)
    ctx.export_csv('12_selected_financial_record_dates.csv',date_rows)
    values={};deps={};problems=[]
    # Derived fin values. No originals are imported or executed (they create directories/write DBs).
    if all(k in sel for k in FIN_FIELDS):
        if all(set(v)<=set(sel[k]) for k,v in FIN_FIELDS.items()):
            periods={lag:pd.concat([aligned[(table,lag)][cols] for table,cols in FIN_FIELDS.items()],axis=1) for lag in [0,1,2]}
            values.update(compute_fin(periods))
            for lag in [0,1]:
                deps[f'fin_dsri_t{lag}']=[(t,j) for j in [lag,lag+1] for t in ['balancesheet','income']]
                deps[f'fin_mscore_t{lag}']=[(t,j) for j in [lag,lag+1] for t in FIN_FIELDS]
                deps[f'fin_ocf_to_ni_t{lag}']=[('income',lag),('cashflow',lag)]
                deps[f'fin_oth_recv_to_ta_t{lag}']=[('balancesheet',lag)]
            deps['fin_zscore_t0']=[('balancesheet',0),('income',0)]
        else:problems.append('FIN_FORMULA_INPUT_FIELDS_MISSING')
    else:problems.append('FIN_SOURCE_TABLES_MISSING')
    if 'fina_indicator' in sel:
        for lag,key in [(0,'T0_FIELDS'),(1,'T1_FIELDS')]:
            a=aligned[('fina_indicator',lag)]
            for field in fields[key]:
                name=f'fini_{field}_t{lag}'
                if field in a:
                    values[name]=a[field];deps[name]=[('fina_indicator',lag)]
                else:problems.append('FINI_INPUT_FIELD_MISSING:'+field)
    else:problems.append('FINI_SOURCE_TABLE_MISSING')
    binding_rows=[];block_rows=[];blocks={}
    rtol,atol=1e-5,1e-6
    for name,replayed in values.items():
        if name not in panel:
            problems.append('FROZEN_COLUMN_NOT_FOUND:'+name);continue
        old=pd.to_numeric(panel[name],errors='coerce').to_numpy(dtype=float,na_value=np.nan)
        new=pd.to_numeric(replayed,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
        finite=np.isfinite(old)&np.isfinite(new)
        equal=finite & np.isclose(old,new,rtol=rtol,atol=atol,equal_nan=False)
        missing_equal=(~np.isfinite(old))&(~np.isfinite(new))
        missing_to_zero=(old==0)&(~np.isfinite(new))
        mismatch=~(equal|missing_equal|missing_to_zero)
        known=pd.Series(True,index=panel.index);tie=pd.Series(False,index=panel.index);post=pd.Series(False,index=panel.index);available=pd.Series(True,index=panel.index)
        for dep in deps[name]:
            a=aligned[dep]
            known&=a.__present.fillna(False).astype(bool)&a.__date_any.notna()&panel.anchor.notna()
            tie|=a.__top_candidates.fillna(0)>1
            post|=a.__date_any.notna()&panel.anchor.notna()&(a.__date_any>panel.anchor)
            available&=a.__earliest_in_snapshot.notna()&panel.anchor.notna()&(a.__earliest_in_snapshot<=panel.anchor)
        bound=pd.Series(equal,index=panel.index)&known&~tie
        # Equal finite value is necessary evidence, not proof of exact historical DB version.
        observed_post=bound&post
        block='fini' if name.startswith('fini_') else 'fin'
        b=blocks.setdefault(block,{'any_matched_dated_post_anchor':pd.Series(False,index=panel.index),'any_value_mismatch':pd.Series(False,index=panel.index),'any_unverified_feature':pd.Series(False,index=panel.index),'any_no_prior_candidate':pd.Series(False,index=panel.index)})
        b['any_matched_dated_post_anchor']|=observed_post
        b['any_value_mismatch']|=pd.Series(mismatch,index=panel.index)
        b['any_unverified_feature']|=~bound
        b['any_no_prior_candidate']|=observed_post&~available
        for (part,label),ii in panel.groupby(['partition','label_group']).groups.items():
            ix=np.asarray(ii,dtype=int)
            binding_rows.append({'feature':name,'block':block,'partition':part,'label_group':label,'denominator_rows':len(ix),'finite_values_compared':int(finite[ix].sum()),'finite_values_matching_frozen':int(equal[ix].sum()),'both_missing':int(missing_equal[ix].sum()),'replay_missing_frozen_zero':int(missing_to_zero[ix].sum()),'other_value_mismatch':int(mismatch[ix].sum()),'matching_value_complete_dated_untied_inputs':int(bound.loc[ii].sum()),'matching_value_post_anchor_input':int(observed_post.loc[ii].sum()),'matching_post_anchor_without_prior_candidate_in_snapshot':int((observed_post&~available).loc[ii].sum()),'selection_tied_rows':int(tie.loc[ii].sum()),'rtol':rtol,'atol':atol})
    for block,flags in blocks.items():
        for (part,label),ii in panel.groupby(['partition','label_group']).groups.items():
            block_rows.append({'block':block,'partition':part,'label_group':label,'denominator_rows':len(ii),**{k:int(v.loc[ii].sum()) for k,v in flags.items()}})
    ctx.export_csv('13_financial_value_binding.csv',binding_rows)
    ctx.export_csv('14_financial_exposure_by_block.csv',block_rows)
    # v18 postprocessing hook: capture the existing flags for PRIVATE joins.
    ctx.row_flags=(panel[['company_code','fiscal_year','target','partition','label_group']].copy(),blocks)

    n_zind=sum(c.startswith(('fin_','fini_')) and c.endswith('_zind') for c in panel.columns)
    ctx.export_json('15_financial_interpretation_limits.json',{
        'replayed_fin_nonstandardized_features':len([k for k in values if k.startswith('fin_')]),'replayed_fini_nonstandardized_features':len([k for k in values if k.startswith('fini_')]),'standardized_columns_not_reconstructed':n_zind,
        'selected_update_flag_1_is_not_proof_of_post_anchor_availability':True,
        'record_dates_reported_separately':['ann_date','f_ann_date'],'any_recorded_date_is_conservative_vendor_date_exposure_not_a_new_historical_availability_claim':True,
        'source_date_semantics_independently_verified':False,'full_historical_database_snapshot_provenance_proven':False,
        'value_matches_are_rule_replay_compatibility_not_unique_historical_source_identification':True,
        'ties_not_treated_as_uniquely_identified_retained_rows':True,
        'all_fin_fini_columns_certified_as_of':False,
        'cross_sectional_zind_and_other_feature_blocks_not_cleared_by_this_check':True,
        'first_announced_or_asof_feature_matrix_written':False,'models_trained':0,'problems':sorted(set(problems)),
        'next_decision':'Review date counts, value binding, missingness, tied selections and available historical candidates before authorizing any sensitivity training.'})
    ctx.record('financial_vintage','PARTIAL' if problems else 'PASS',note='Current-snapshot selection replay and numeric compatibility completed; historical as-of correctness is not certified.',problems=sorted(set(problems)),n_unchecked_standardized_columns=n_zind)
