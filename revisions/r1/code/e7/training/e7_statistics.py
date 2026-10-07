"""Frozen populations, stable point metrics and paired company/seed bootstrap."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import itertools
import math
import multiprocessing as mp
import numpy as np
import pandas as pd
from e7_runtime_common import require
from e7_weighted_metrics import Plan,NAMES

METRICS = NAMES+['hits_at_budget']
SEEDS = [42,123,456,789,1024]
MODELS = ['GCN','SAGE']
POPS = ['full_test','D_frozen','O_frozen','S_frozen','J_frozen']
BOOT = {}


def normalize_code(series):
    value=series.astype('string').str.strip().str.replace(r"^'",'',regex=True)
    value=value.str.replace(r'\.0$','',regex=True).str.replace(r'\.(SZ|SH|BJ)$','',regex=True)
    value=value.str.replace(r'^C:','',regex=True).str.zfill(6)
    require(value.notna().all() and value.str.fullmatch(r'[0-9]{6}').all(),'INVALID_COMPANY_CODE')
    return value.astype(str)


def align_masks(pred,masks,contract):
    pred=pred.copy();masks=masks.copy()
    for table in [pred,masks]:
        table['company_code']=normalize_code(table.company_code)
        table['fiscal_year']=table.fiscal_year.astype(int)
        require(not table.duplicated(['company_code','fiscal_year']).any(),'DUPLICATE_EVALUATION_KEYS')
    required=set(POPS+['D_positive','O_positive'])
    require(required <= set(masks),'FROZEN_EVAL_MASK_COLUMNS_MISSING')
    masks=masks[['company_code','fiscal_year',*sorted(required)]]
    result=pred.merge(masks,on=['company_code','fiscal_year'],how='left',validate='one_to_one')
    for column in required:
        require(result[column].notna().all() and result[column].isin([True,False,0,1]).all(),
                'FROZEN_MASK_VALUE_OR_ALIGNMENT_INVALID:'+column)
        result[column]=result[column].astype(bool)
    for pop,expected in contract.items():
        selected=result[result[pop]]
        actual={'rows':len(selected),'positive':int(selected.y_true.sum()),
                'negative':len(selected)-int(selected.y_true.sum())}
        require(actual==expected,'FROZEN_POPULATION_COUNTS_MISMATCH:'+pop)
    positive=result.y_true.eq(1)
    require(result.full_test.all()
            and (result.D_positive & result.O_positive).sum()==0
            and ((result.D_positive | result.O_positive)==positive).all(), 'D_O_POSITIVE_PARTITION')
    require((result.D_frozen==(~positive | result.D_positive)).all()
            and (result.O_frozen==(~positive | result.O_positive)).all()
            and (result.J_frozen==(result.S_frozen & result.D_frozen)).all(), 'FROZEN_POPULATION_MEMBERSHIP')
    return result.sort_values(['company_code','fiscal_year'],kind='stable').reset_index(drop=True)


def point_metrics(frame,threshold):
    frame=frame.sort_values(['company_code','fiscal_year'],kind='stable').reset_index(drop=True)
    y=frame.y_true.to_numpy(int);s=frame.score.to_numpy(float)
    require(len(y)>0 and np.unique(y).size==2,'POINT_POPULATION_SINGLE_CLASS')
    plan=Plan(y,s,threshold)
    values=plan.compute(np.ones(len(y)))
    # Point checks against sklearn and the exact stable budget rule, not just another copy of Plan.
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,roc_curve
    order=np.argsort(-s,kind='mergesort');budget=max(1,math.ceil(.05*len(y)))
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=True)
    expected=np.array([roc_auc_score(y,s),average_precision_score(y,s),
        f1_score(y,s>=threshold,zero_division=0),float(y[order[:budget]].mean()),
        float(tpr[fpr<=.10+1e-12].max())])
    require(np.allclose(values,expected,rtol=0,atol=2e-12),'POINT_METRIC_REFERENCE_MISMATCH')
    result=dict(zip(NAMES,map(float,values)))
    result.update({'hits_at_budget':int(y[order[:budget]].sum()),'n':len(y),'n_pos':int(y.sum()),
                   'n_neg':len(y)-int(y.sum()),**plan.tie_report()})
    return result


def bootstrap_init(payload,cluster_index,n_companies,base_seed):
    BOOT.clear()
    BOOT.update({'plans':{},'clusters':np.asarray(cluster_index,int),'n_companies':n_companies,'seed':base_seed})
    for key,(y,s,threshold) in payload.items():
        BOOT['plans'][key]=Plan(y,s,threshold)


def bootstrap_one(replicate):
    rng=np.random.default_rng(BOOT['seed']+int(replicate)*10)
    n=BOOT['n_companies']
    counts=np.bincount(rng.integers(0,n,size=n),minlength=n)
    weights=counts[BOOT['clusters']].astype(float)
    seed_draw=rng.integers(0,len(SEEDS),size=len(SEEDS))
    exemplar=next(iter(BOOT['plans'].values()))
    yy=np.empty(len(exemplar.y),dtype=float);yy[exemplar.order]=exemplar.y
    if float(np.sum(weights*yy))<=0 or float(np.sum(weights*(1-yy)))<=0:
        return None
    values={}
    for model in MODELS:
        contrasts=[]
        for seed_index in set(seed_draw.tolist()):
            seed=SEEDS[seed_index]
            pair=[]
            budget=max(1,math.ceil(float(weights.sum())*.05))
            for condition in ['control','E7']:
                met=BOOT['plans'][(model,condition,seed)].compute(weights)
                pair.append(np.r_[met,met[3]*budget])
            values[(model,seed_index)]=pair[1]-pair[0]
        contrasts=[values[(model,int(i))] for i in seed_draw]
        values[model]=np.mean(contrasts,axis=0)
    return np.stack([values[m] for m in MODELS])


def summarize(per_seed,payload,keys,settings):
    groups=[]
    frame=pd.DataFrame(per_seed)
    for (model,condition,pop),selected in frame.groupby(['model','condition','population'],sort=True):
        require(sorted(selected.seed.tolist())==SEEDS,'FIVE_SEEDS_REQUIRED_PER_GROUP')
        item={'model':model,'condition':condition,'population':pop,'n':int(selected.n.iloc[0]),
              'n_pos':int(selected.n_pos.iloc[0])}
        for metric in METRICS:
            item[metric+'_mean']=float(selected[metric].mean())
            item[metric+'_sd']=float(selected[metric].std(ddof=1))
        groups.append(item)
    point=[]
    for model in MODELS:
        for pop in POPS:
            c=frame[(frame.model==model)&(frame.condition=='control')&(frame.population==pop)].set_index('seed')
            t=frame[(frame.model==model)&(frame.condition=='E7')&(frame.population==pop)].set_index('seed')
            for metric in METRICS:
                diffs=(t.loc[SEEDS,metric]-c.loc[SEEDS,metric]).to_numpy(float)
                item={'model':model,'population':pop,'metric':metric,'contrast':'E7 minus control',
                      'paired_seed_differences':diffs.tolist(),'paired_mean_difference':float(diffs.mean()),
                      'paired_seed_sd':float(diffs.std(ddof=1))}
                if pop=='full_test':
                    observed=abs(float(diffs.mean()))
                    permutation=[abs(float(np.mean(diffs*np.array(s)))) for s in itertools.product([-1,1],repeat=5)]
                    item['descriptive_exact_seed_signflip_p']=sum(v>=observed-1e-15 for v in permutation)/32
                point.append(item)
    company_index,companies=pd.factorize(keys.company_code,sort=True)
    n_companies=len(companies)
    require(n_companies>1 and len(keys)==8435,'FULL_BOOTSTRAP_KEY_POPULATION')
    print(f'PAIRED_BOOTSTRAP_START replicates={settings["replicates"]} companies={n_companies}',flush=True)
    with ProcessPoolExecutor(max_workers=settings['workers'],mp_context=mp.get_context('spawn'),
            initializer=bootstrap_init,initargs=(payload,company_index,n_companies,settings['rng_seed'])) as pool:
        draws=list(pool.map(bootstrap_one,range(settings['replicates']),chunksize=8))
    valid=[x for x in draws if x is not None]
    require(len(valid)>=.95*settings['replicates'],'TOO_FEW_VALID_FULL_BOOTSTRAP_REPLICATES')
    array=np.stack(valid)
    for item in point:
        if item['population']=='full_test':
            column=array[:,MODELS.index(item['model']),METRICS.index(item['metric'])]
            lower,upper=np.quantile(column,[.025,.975],method='linear')
            item.update({'paired_company_seed_bootstrap_ci95':[float(lower),float(upper)],
                         'bootstrap_valid_replicates':len(valid)})
    print(f'PAIRED_BOOTSTRAP_DONE valid={len(valid)}',flush=True)
    return {'group_means':groups,'paired_contrasts':point,
            'bootstrap':{**settings,'valid_replicates':len(valid),'invalid_single_class_replicates':len(draws)-len(valid),
                         'company_clusters':n_companies,'seed_draw_shared':True,'company_draw_shared':True}},array
