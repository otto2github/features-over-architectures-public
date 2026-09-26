#!/usr/bin/env python3
"""Small invented data; calls actual preparation functions, no private data or network."""
from pathlib import Path
from datetime import date,timedelta
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/sensitivity_experiment/code'))
import build_asof_features as financial
import build_fixed1095_labels as fixed
checks=[]
def ok(name,value):
 if not bool(value):raise AssertionError(name)
 checks.append(name)
examples=pd.Series(['20230517','2023-05-17','2023-05-17 12:00:00',None,'not a date'])
a=financial.dates(examples);b=fixed.date_series(examples)
ok('two_actual_date_parsers_agree',a.equals(b));ok('compact_iso_datetime_match',a.iloc[:3].eq(pd.Timestamp('2023-05-17')).all());ok('invalid_missing_dates_remain_missing',a.iloc[3:].isna().all())
panel=pd.DataFrame({'__rowid':[0,1],'company_code':['000001','000002'],'fiscal_year':[2022,2022],'anchor':pd.to_datetime(['2023-04-01','2023-04-01'])})
raw=pd.DataFrame({'company_code':['000001','000001','000002'],'source_year':[2022]*3,'__sqlite_rowid':[1,2,3],'__update':[0,1,1],'__ann':pd.to_datetime(['2023-03-01','2023-05-01','2023-05-01']),'__avail':pd.to_datetime(['2023-03-01','2023-05-01','2023-05-01']),'example_value':[10.,99.,88.]})
s=financial.select_for_panel(raw,panel,0,'fin',True);u=financial.select_for_panel(raw,panel,0,'fin',False)
ok('post_anchor_candidate_excluded',s.loc[0,'example_value']==10);ok('no_eligible_prior_candidate_stays_missing',pd.isna(s.loc[1,'example_value']));ok('unfiltered_current_selects_updated_record',u.loc[0,'example_value']==99)
# Endpoint illustration is independent logic, explicitly not the full label builder.
anchor=date(2022,4,1)
def window(y,lag):return None if y is None else (int(0<lag<=1095) if y==1 else 0)
ok('fixed_window_inclusive_day1095',window(1,1095)==1);ok('late_only_reclassified',window(1,1096)==0);ok('null_status_not_expanded',window(None,20) is None)
# Actual industry routine: all fields within one invented group, frozen metadata selected.
finraw=['fin_dsri_t0','fin_dsri_t1','fin_mscore_t0','fin_mscore_t1','fin_ocf_to_ni_t0','fin_ocf_to_ni_t1','fin_oth_recv_to_ta_t0','fin_oth_recv_to_ta_t1','fin_zscore_t0']
p=pd.DataFrame({'fiscal_year':[2020]*12,'industry':['Synthetic group']*12,'company_code':[str(i) for i in range(12)]})
f=pd.DataFrame({n:np.arange(12,dtype=float) for n in finraw});z=financial.add_zind(p,f,None,{'ZIND_FIELDS':[]})
ok('frozen_industry_transform_zero_mean',abs(z.fin_dsri_t0_zind.mean())<1e-12);ok('sample_sd_one',abs(z.fin_dsri_t0_zind.std()-1)<1e-12)
print(json.dumps({'status':'SYNTHETIC_INTERFACE_CHECK_PASS','checks':checks,'actual_functions_exercised':['dates','date_series','select_for_panel','add_zind'],'full_raw_pipeline_executed':False,'model_fitting':False,'empirical_or_licensed_data_used':False},indent=2))
