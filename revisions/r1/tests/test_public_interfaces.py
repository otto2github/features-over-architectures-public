#!/usr/bin/env python3
"""Actual public metric/preparation functions on invented inputs only."""
from pathlib import Path
import importlib.util,json,subprocess,sys,math,os
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 modules=[load('metric_sensitivity',ROOT/'code/sensitivity/fixed_score/fs_math.py'),load('metric_extension',ROOT/'code/extension/fixed_score/fs_math.py')]
 rng=np.random.default_rng(141707);largest=0.;cases=0
 for _ in range(80):
  y=np.r_[0,1,rng.integers(0,2,28)];scores=rng.integers(0,11,30)/10.;w=rng.integers(0,4,30);w[:2]=1
  yy=np.repeat(y,w);ss=np.repeat(scores,w);order=np.argsort(-ss,kind='stable');budget=math.ceil(.05*len(yy));hits=float(yy[order[:budget]].sum())
  fpr,tpr,_=roc_curve(yy,ss,drop_intermediate=True)
  expected=np.array([roc_auc_score(yy,ss),average_precision_score(yy,ss),f1_score(yy,ss>=.5,zero_division=0),hits/budget,float(tpr[fpr<=.1+1e-12].max()),hits])
  for mod in modules:
   got=mod.MetricPlan(y,scores,.5).compute(w);err=float(np.max(np.abs(got-expected)));largest=max(largest,err);assert err<1e-12;cases+=1
 results=[]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for name in ['postprocess_numeric.py','synthetic_interfaces.py']:
  q=subprocess.run([sys.executable,str(ROOT/'tests/predecessor'/name)],cwd=ROOT,capture_output=True,text=True,env=env,check=True)
  results.append(json.loads(q.stdout))
 print(json.dumps({'status':'PUBLIC_SYNTHETIC_INTERFACES_PASS','actual_metric_function_cases':cases,'max_metric_error':largest,'predecessor_tests':results,'empirical_data_used':False,'new_model_fits':0,'new_empirical_resampling_draws':0},indent=2))
if __name__=='__main__':main()
