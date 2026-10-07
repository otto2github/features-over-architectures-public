"""Presorted weighted metrics; verified against sklearn/v3 on ties and zero weights."""
from __future__ import annotations
import math
import numpy as np
from .common import require
NAMES=['roc_auc','ap','f1_val_threshold','p_at_5pct','r_at_10fpr']
class Plan:
 def __init__(self,y,s,threshold):
  y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=np.float64)
  require(len(y)==len(s) and len(y)>0,'METRIC_LENGTH')
  require(np.isin(y,[0,1]).all() and np.isfinite(s).all(),'METRIC_VALUES')
  self.order=np.argsort(-s,kind='mergesort')
  self.y=y[self.order].astype(float);self.s=s[self.order]
  self.ends=np.r_[np.flatnonzero(np.diff(self.s)!=0),len(s)-1]
  self.pred=(self.s>=float(threshold))
 def compute(self,w):
  w=np.asarray(w,dtype=float)[self.order]
  require(np.isfinite(w).all() and (w>=0).all(),'METRIC_WEIGHTS')
  pos=np.cumsum(w*self.y);neg=np.cumsum(w*(1-self.y))
  P=float(pos[-1]);N=float(neg[-1]);require(P>0 and N>0,'METRIC_SINGLE_CLASS')
  tp=pos[self.ends];fp=neg[self.ends]
  nz=np.diff(np.r_[0,tp+fp])>0;tp=tp[nz];fp=fp[nz]
  xt=np.r_[0,fp/N];yt=np.r_[0,tp/P]
  auc=float(np.sum(np.diff(xt)*(yt[:-1]+yt[1:])*0.5))
  ap=float(np.sum(np.diff(np.r_[0,tp])/P*tp/(tp+fp)))
  t=float(np.sum(w*self.y*self.pred));f=float(np.sum(w*(1-self.y)*self.pred))
  f1=2*t/(P+t+f) if P+t+f else 0.0
  budget=max(1,int(math.ceil((P+N)*0.05)))
  cumulative=np.cumsum(w);before=cumulative-w
  take=np.clip(budget-before,0,w)
  p5=float(np.sum(take*self.y)/take.sum())
  # sklearn roc_curve default removes collinear intermediate points AFTER zero-weight removal.
  if len(fp)>2:
   keep=np.r_[True,np.logical_or(np.diff(fp,2)!=0,np.diff(tp,2)!=0),True]
   fp, tp=fp[keep],tp[keep]
  allowed=fp/N<=0.10+1e-12
  r10=float(np.max(tp[allowed]/P)) if allowed.any() else 0.0
  return np.array([auc,ap,f1,p5,r10],dtype=float)
 def tie_report(self):
  n=len(self.y);k=max(1,math.ceil(0.05*n));cut=self.s[k-1]
  above=self.s>cut;equal=self.s==cut
  slots=k-int(above.sum());positives=int(self.y[equal].sum());negatives=int(equal.sum())-positives
  base=float(self.y[above].sum())
  return {'n':n,'budget':k,'cutoff_tie_rows':int(equal.sum()),'selected_from_cutoff_tie':slots,
   'cutoff_tie_positive':positives,'tie_p5_min':(base+max(0,slots-negatives))/k,
   'tie_p5_max':(base+min(slots,positives))/k}

def reference_metric(y,s,w,threshold):
 from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
 order=np.argsort(-s,kind='mergesort');ww=np.asarray(w)[order];yy=np.asarray(y)[order]
 k=max(1,math.ceil(float(np.sum(w))*.05));before=np.cumsum(ww)-ww;take=np.clip(k-before,0,ww)
 fpr,tpr,_=roc_curve(y,s,sample_weight=w);r=float(np.max(tpr[fpr<=.10+1e-12]))
 return np.array([roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w),
 f1_score(y,np.asarray(s)>=threshold,sample_weight=w,zero_division=0),np.sum(take*yy)/np.sum(take),r])

def self_test(n=120):
 rng=np.random.default_rng(8841);error=0.
 for i in range(n):
  size=int(rng.integers(15,150));y=rng.integers(0,2,size);y[:2]=[0,1]
  s=np.round(rng.random(size),int(rng.integers(0,4))) if i%3 else rng.random(size)
  w=rng.integers(0,5,size).astype(float);w[:2]=1;thr=float(rng.random())
  a=Plan(y,s,thr).compute(w);b=reference_metric(y,s,w,thr)
  error=max(error,float(np.max(np.abs(a-b))));require(np.allclose(a,b,rtol=0,atol=2e-12),'METRIC_EQUIVALENCE_FAILED')
 return {'random_tied_zero_weight_cases':n,'max_absolute_error':error,'metrics':NAMES,'passed':True}
