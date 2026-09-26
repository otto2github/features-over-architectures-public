#!/usr/bin/env python3
"""Actual postprocessing numerical functions; invented arrays, no Parquet/no fits."""
from pathlib import Path
import importlib.util, io, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code/subset_diagnostics'
sys.path.insert(0,str(CODE))
import postprocess as pp
from auditlib.common import AuditStop

def main():
 check=[]
 for p,n,a,b in [(20,8415,14,403),(20,8415,12,266),(20,8415,12,527),(20,8415,16,1927),(40,7269,33,2563),(254,23852,218,12959),(4,7,0,0),(4,7,4,7)]:
  y=np.r_[np.ones(p),np.zeros(n)];s=np.r_[np.ones(a),np.zeros(p-a),np.ones(b),np.zeros(n-b)]
  auc=.5*(a/p+1-b/n);ap=(a/p)*(a/(a+b))+(1-a/p)*p/(p+n) if a+b else p/(p+n)
  assert abs(auc-roc_auc_score(y,s))<1e-12
  assert abs(ap-average_precision_score(y,s))<1e-12
  check.append('binary_'+str((p,n,a,b)))
 metric=pp.self_test(100)
 # Actual r2 marginal comparison: in-memory 'null' versus read_csv NA before sorting.
 a=pd.DataFrame({'partition':['test','test','test'],'label_group':['negative','null','positive'],'rows':[8415,922,20]})
 b=pd.read_csv(io.StringIO('partition,label_group,rows\ntest,negative,8415\ntest,null,922\ntest,positive,20\n'))
 pp.finite_marginals_match(None,a,b,['partition','label_group'],['rows']);check.append('null_NA_roundtrip')
 wrong=b.copy();wrong.loc[0,'rows']=8414
 failed=False
 try:pp.finite_marginals_match(None,a,wrong,['partition','label_group'],['rows'])
 except (AuditStop,RuntimeError):failed=True
 assert failed;check.append('marginal_value_mismatch_rejected')
 wrong=b.copy();wrong.loc[0,'partition']='validation';failed=False
 try:pp.finite_marginals_match(None,a,wrong,['partition','label_group'],['rows'])
 except (AuditStop,RuntimeError):failed=True
 assert failed;check.append('marginal_key_mismatch_rejected')
 print(json.dumps({'status':'NUMERIC_SELFTEST_PASS','binary_cases':8,'marginal_regressions':3,'weighted_metric_selftest':metric,'training':False,'private_inputs':False,'parquet_io_tested':False},indent=2))
if __name__=='__main__':main()
