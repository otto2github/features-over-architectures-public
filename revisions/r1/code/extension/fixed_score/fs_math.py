"""Fixed-score, integer-multiplicity statistics. No fitting or model inference.

Inputs must already be aligned in company_code/year ascending order. The weighted
ROC pruning, AP, threshold F1 and top-budget conventions reproduce V26's Plan.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

METRICS = ('roc_auc', 'ap', 'f1_val_threshold', 'p_at_5pct', 'r_at_10fpr', 'hits_at_budget')
SEEDS = (42, 123, 456, 789, 1024)

class ContractError(RuntimeError):
    def __init__(self, code: str, detail=None):
        self.code, self.detail = code, detail
        super().__init__(code)

def require(ok, code, detail=None):
    if not bool(ok):
        raise ContractError(code, detail)

class MetricPlan:
    def __init__(self, y, scores, threshold):
        y = np.asarray(y)
        scores = np.asarray(scores, dtype=np.float64)
        require(y.ndim == scores.ndim == 1 and y.size == scores.size and y.size > 0, 'METRIC_SHAPE')
        require(np.isin(y, [0, 1]).all() and np.isfinite(scores).all(), 'METRIC_DOMAIN')
        require(np.isfinite(threshold) and 0 <= threshold <= 1, 'THRESHOLD_DOMAIN')
        self.order = np.argsort(-scores, kind='mergesort')
        self.y = y[self.order].astype(np.float64)
        self.s = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(self.s) != 0), len(scores)-1]
        self.pred = self.s >= float(threshold)
        self.threshold = float(threshold)

    def compute(self, weights):
        """Exactly duplicated rows are represented by integer multiplicities."""
        raw = np.asarray(weights)
        require(raw.shape == self.y.shape and np.isfinite(raw).all() and (raw >= 0).all(), 'WEIGHT_DOMAIN')
        require(np.equal(raw, np.floor(raw)).all(), 'WEIGHT_NOT_INTEGER_MULTIPLICITY')
        w = raw[self.order].astype(np.float64)
        pos = np.cumsum(w*self.y); neg = np.cumsum(w*(1-self.y))
        P, N = float(pos[-1]), float(neg[-1])
        if P <= 0 or N <= 0:
            return np.full(len(METRICS), np.nan)
        tp, fp = pos[self.ends], neg[self.ends]
        live = np.diff(np.r_[0., tp+fp]) > 0
        tp, fp = tp[live], fp[live]
        xt, yt = np.r_[0., fp/N], np.r_[0., tp/P]
        auc = float(np.sum(np.diff(xt)*(yt[:-1]+yt[1:])*.5))
        ap = float(np.sum(np.diff(np.r_[0., tp])/P * tp/(tp+fp)))
        t = float(np.sum(w*self.y*self.pred)); f = float(np.sum(w*(1-self.y)*self.pred))
        f1 = 2*t/(P+t+f)
        k = max(1, int(math.ceil(.05*(P+N))))
        before = np.cumsum(w)-w
        take = np.clip(k-before, 0, w)
        hits = float(np.sum(take*self.y))
        p5 = hits/float(take.sum())
        # Preserve sklearn/V26 drop_intermediate after removing zero-weight ties.
        if len(fp) > 2:
            keep = np.r_[True, np.logical_or(np.diff(fp,2)!=0, np.diff(tp,2)!=0), True]
            fp, tp = fp[keep], tp[keep]
        allowed = fp/N <= .10 + 1e-12
        r10 = float(np.max(tp[allowed]/P)) if allowed.any() else 0.
        return np.array([auc, ap, f1, p5, r10, hits], dtype=float)

    def tie_report(self):
        n=len(self.y); k=max(1, math.ceil(.05*n)); cut=self.s[k-1]
        above=self.s>cut; equal=self.s==cut
        slots=k-int(above.sum()); pp=int(self.y[equal].sum()); nn=int(equal.sum())-pp
        base=int(self.y[above].sum())
        return dict(n=n, budget=k, cutoff_score=float(cut), cutoff_tie_rows=int(equal.sum()),
                    cutoff_tie_positive=pp, selected_from_cutoff_tie=slots,
                    tie_hits_min=base+max(0,slots-nn), tie_hits_max=base+min(slots,pp))

    def confusion(self):
        tp=int(np.sum(self.pred*self.y)); fp=int(np.sum(self.pred*(1-self.y)))
        fn=int(np.sum((~self.pred)*self.y)); tn=int(np.sum((~self.pred)*(1-self.y)))
        return dict(tp=tp,fp=fp,fn=fn,tn=tn,predicted_positive=tp+fp,
                    threshold=self.threshold,threshold_rule='score_ge_frozen_validation_threshold')

def connected_company_groups(company_codes, incidence_rows):
    """Union only usable TEST document keys; attach ALL rows of each company.

    incidence_rows is [(company_code, opaque_document_hash), ...]. No labels or
    scores are used to choose or split groups. Companies without links singleton.
    """
    companies=sorted(set(str(x) for x in company_codes))
    at={c:i for i,c in enumerate(companies)}; parent=np.arange(len(companies))
    def root(i):
        while parent[i] != i:
            parent[i]=parent[parent[i]]; i=int(parent[i])
        return i
    def join(a,b):
        a,b=root(a),root(b)
        if a != b:
            parent[max(a,b)]=min(a,b)
    by_doc={}
    for cc,doc in sorted(set((str(c),str(k)) for c,k in incidence_rows)):
        require(cc in at, 'GROUP_COMPANY_OUTSIDE_TEST')
        require(doc and doc.lower() not in ('nan','none','null'), 'EMPTY_DOCUMENT_KEY')
        if doc in by_doc: join(at[cc], by_doc[doc])
        else: by_doc[doc]=at[cc]
    representatives=[root(i) for i in range(len(companies))]
    groups=sorted(set(representatives)); group_ix={g:i for i,g in enumerate(groups)}
    company_to_group={cc:group_ix[root(i)] for cc,i in at.items()}
    row_groups=np.array([company_to_group[str(cc)] for cc in company_codes], dtype=np.int64)
    return row_groups, company_to_group

def bootstrap_draw(b, cluster_idx, y, cluster_seed, fit_seed, max_attempts=1000):
    cluster_idx=np.asarray(cluster_idx, dtype=np.int64); y=np.asarray(y,dtype=np.float64)
    n=int(cluster_idx.max())+1
    require(set(np.unique(cluster_idx))==set(range(n)), 'NONCONTIGUOUS_GROUP_IDS')
    rng=np.random.default_rng(int(cluster_seed)+10*int(b))
    rejected=0; w=None
    for _ in range(int(max_attempts)):
        counts=np.bincount(rng.integers(0,n,size=n), minlength=n)
        w=counts[cluster_idx]
        positive=float(np.dot(w,y)); negative=float(w.sum()-positive)
        if positive > 0 and negative > 0:
            break
        rejected+=1
    else:
        return None,None,rejected
    seed_draw=np.random.default_rng(int(fit_seed)+10*int(b)).integers(0,5,size=5)
    return w,seed_draw,rejected

def condition_name(run):
    base='|'.join((run['family'],run['model'],run['modality']))
    return base+(f"|{int(run['imputation_seed'])}" if run['imputation_seed'] is not None else '')

def contrast_catalog(conditions):
    """Protocol-only contrasts, explicitly fixed before reading scores."""
    catalog=[]
    def add(name,family,terms,world=None):
        terms={k:float(v) for k,v in terms.items() if v}
        require(set(terms)<=set(conditions),'CONTRAST_UNKNOWN_CONDITION',name)
        require(abs(sum(terms.values()))<1e-12,'CONTRAST_COEFFICIENT_SUM',name)
        catalog.append(dict(contrast_id=name,family=family,world=world,coefficients=terms))
    def a(m):return f'A|{m}|M11'
    def o(m):return f'V26|{m}|M11'
    models=['MLP','RandomForest_unweighted','GCN_reverse','GraphSAGE_reverse']
    for m in models:add(f'A_minus_V26__{m}','A',{a(m):1,o(m):-1})
    for l in models[:2]:
        for r in models[2:]:
            add(f'A_gap__{l}_minus_{r}','A',{a(l):1,a(r):-1})
            add(f'A_gap_change__{l}_minus_{r}','A',{a(l):1,a(r):-1,o(l):-1,o(r):1})
    worlds=[271707,271708,271709]
    for m in models[:2]:
        old5=f'ASOF|{m}|M5';old11=f'ASOF|{m}|M11'
        for w in worlds+['mean3']:
            ws=worlds if w=='mean3' else [w]
            coef=1/len(ws)
            inc={**{f'B|{m}|M11|{s}':coef for s in ws},**{f'B|{m}|M5|{s}':-coef for s in ws}}
            add(f'B_increment__{m}__{w}','B',inc,w)
            for mod,old in [('M5',old5),('M11',old11)]:
                add(f'B_minus_ASOF__{m}__{mod}__{w}','B',
                    {**{f'B|{m}|{mod}|{s}':coef for s in ws},old:-1},w)
            add(f'B_increment_change__{m}__{w}','B',{**inc,old11:-1,old5:1},w)
        add(f'B0_42_minus_17__{m}','B0',{f'B0|{m}|M11_no_FIN87':1,f'B0|{m}|M5_no_FIN87':-1})
    require(len(catalog)==46,'CONTRAST_COUNT')
    return catalog
