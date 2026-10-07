"""Postfreeze analyses from saved scores, without importing model code."""
from __future__ import annotations
import ast, hashlib, json, math, multiprocessing as mp, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fs_math import (MetricPlan, METRICS, SEEDS, ContractError, require,
                     bootstrap_draw, connected_company_groups, contrast_catalog)
from fs_io import (ROOT, KEYS, POPS, POP_COUNTS, EXPECTED, sha256,
                   read_json, strict_keys, binary_array, bool_array, one_to_one)

def finite_or_none(x):return float(x) if np.isfinite(x) else None

def condition_layout(fits):
    names=sorted(set(f.condition for f in fits))
    groups=[]
    for name in names:
        loc={f.seed:i for i,f in enumerate(fits) if f.condition==name}
        require(set(loc)==set(SEEDS),'CONDITION_SEED_SET',name)
        groups.append([loc[s] for s in SEEDS])
    return names,np.asarray(groups,dtype=np.int64)

def definition_matrix(catalog,names):
    c=np.zeros((len(catalog),len(names)))
    for j,spec in enumerate(catalog):
        for name,w in spec['coefficients'].items():c[j,names.index(name)]=w
    return c

def groups_and_provenance(bundle,out):
    cc=bundle.canonical.company_code.to_numpy(str)
    testinc=bundle.incidence[bundle.incidence.partition=='test']
    pairs=list(zip(testinc.company_code.astype(str),testinc.doc_key_hash.astype(str)))
    components,company_map=connected_company_groups(cc,pairs)
    company_names,company_idx=np.unique(cc,return_inverse=True)
    comp_sizes=np.bincount(components);company_sizes=np.bincount(company_idx)
    comp_pos=np.bincount(components,weights=bundle.y,minlength=len(comp_sizes)).astype(int)
    group_members=pd.DataFrame({'company_code':cc,'fiscal_year':bundle.canonical.fiscal_year,
                                'company_cluster':company_idx,'document_component':components})
    # Timestamped NOW; never represented as a previously produced artifact.
    dest=out.check(out.root/'PRIVATE/company_document_membership_materialized_now.csv')
    dest.parent.mkdir(parents=True,exist_ok=True);group_members.to_csv(dest,index=False)
    candidates=[];verified_prior=[];unresolved=[]
    # Limit inspection to files named in actual pre-fit binding manifests.
    for origin,records in [('final_binding.input_files',bundle.final_binding.get('input_files',{})),
                           ('data_stage.files',bundle.data_binding.get('files',{}))]:
        for rel,meta in records.items():
            name=Path(rel).name.lower()
            if any(s in name for s in ('component','cluster','group')) and name.endswith(('.parquet','.csv','.json')):
                item={'origin':origin,'member':rel,'sha256':meta.get('sha256')}
                if any(x.get('member')==rel for x in candidates):continue
                candidates.append(item)
                try:
                    p=Path(rel) if Path(rel).is_absolute() else ROOT/rel
                    bundle.ledger.verify(p,meta['sha256'],meta.get('size_bytes'),role='prefit_group_candidate')
                    if p.suffix=='.parquet':d=pd.read_parquet(p)
                    elif p.suffix=='.csv':d=pd.read_csv(p,dtype={'company_code':str},keep_default_na=False)
                    else:
                        j=read_json(p);d=pd.DataFrame(j if isinstance(j,list) else j.get('rows',[]))
                    d=strict_keys(d)
                    cols=[c for c in ('document_component','component_id','company_document_group','cluster_id','group_id') if c in d]
                    require(len(cols)==1,'GROUP_CANDIDATE_SCHEMA')
                    a=one_to_one(bundle.canonical,d,cols,'prefit_group_candidate')
                    require(a[cols[0]].notna().all(),'GROUP_CANDIDATE_NULL')
                    # Equivalence of partitions, not equality of arbitrary group numbering.
                    comp_pair=pd.DataFrame({'a':a[cols[0]].astype(str),'b':components})
                    equal=(comp_pair.groupby('a').b.nunique().max()==1 and comp_pair.groupby('b').a.nunique().max()==1)
                    require(equal,'PREBOUND_COMPONENT_MAP_DISAGREES')
                    verified_prior.append(item)
                except Exception as exc:
                    unresolved.append({'member':rel,'code':getattr(exc,'code',type(exc).__name__)})
    positive_test_keys=set(zip(bundle.canonical.loc[bundle.y==1,'company_code'],bundle.canonical.loc[bundle.y==1,'fiscal_year']))
    known_keys=set(zip(testinc.company_code,testinc.fiscal_year))
    status=('PRE_FIT_BOUND_MEMBERSHIP_SEMANTICALLY_MATCHED' if verified_prior and not unresolved else
            'HOLD_GROUP_CANDIDATE_REVIEW' if unresolved else 'POSTFIT_MATERIALIZATION_OF_PRESPECIFIED_RULE')
    summary=dict(
        definition='test companies connected by usable test-document keys; all eligible years/negatives attached',
        generated_at_utc=datetime.now(timezone.utc).isoformat(),materialized_now=True,
        source_incidence_sha256=EXPECTED['inputs/case/positive_document_key_hash_incidence_v27.parquet'],
        private_membership_sha256=sha256(dest),membership_timing_status=status,
        inspected_binding_fields=['final_binding.input_files','data_stage.files'],
        prefit_membership_candidates=candidates,prefit_bound_maps_verified=verified_prior,
        unresolved_candidates=unresolved,
        no_unregistered_or_unrelated_directory_scan=True,
        prior_artifact_absence_claim='No candidate in the inspected bound-manifest fields; not proof of absence everywhere.',
        file_mtime_not_accepted_as_preregistration_evidence=True,
        n_test_rows=len(cc),n_test_companies=len(company_names),n_document_components=len(comp_sizes),
        company_cluster_size_histogram=dict(Counter(int(v) for v in company_sizes)),
        component_row_size_histogram=dict(Counter(int(v) for v in comp_sizes)),
        component_positive_size_histogram=dict(Counter(int(v) for v in comp_pos)),
        positive_components=int((comp_pos>0).sum()),max_component_rows=int(comp_sizes.max()),
        test_document_keys=int(testinc.doc_key_hash.nunique()),
        identified_positive_firmyears=len(positive_test_keys & known_keys),
        unidentified_positive_firmyears=len(positive_test_keys-known_keys),
        grouping_uses_prediction_scores=False,secondary_computation_allowed=not unresolved,
        timing_deviation_requires_reporting=(status!='PRE_FIT_BOUND_MEMBERSHIP_SEMANTICALLY_MATCHED'))
    out.json('SHARE/GROUP_PROVENANCE.json',summary)
    return company_idx.astype(np.int64),components.astype(np.int64),summary

def execution_deviation(bundle,out):
    runner=ROOT/'code/sensitivity_runner.py';orch=ROOT/'code/sensitivity_orchestrator.py'
    source=runner.read_text();tree=ast.parse(source)
    defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    def call_name(n):
        if isinstance(n,ast.Name):return n.id
        if isinstance(n,ast.Attribute):return call_name(n.value)+'.'+n.attr
        return ''
    calls={name:sorted({call_name(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)}) for name,fn in defs.items()}
    def closure(start):
        seen=set();todo=[start]
        while todo:
            f=todo.pop()
            if f in seen:continue
            seen.add(f);todo.extend(c for c in calls.get(f,[]) if c in defs and c not in seen)
        return sorted(seen)
    e=closure('run_stage_e');t=closure('run_stage_t')
    enforced=any('torch.use_deterministic_algorithms' in calls.get(name,[]) for name in e)
    rows=[]
    for phase in ['stage_t','stage_e']:
        p=ROOT/f'logs/{phase}_105_20260927.log'
        if p.is_file():
            h=sha256(p);text=p.read_text(encoding='utf-8',errors='replace')
            rows.append({'phase':phase,'log_sha256':h,'bytes':p.stat().st_size,
                         'completion_tokens':text.count('STAGE_T_LOCKED' if phase=='stage_t' else 'STAGE_E_LOCKED'),
                         'traceback_tokens':text.count('Traceback (most recent call last)'),
                         'nonwritable_warning_tokens':text.count('The given NumPy array is not writable'),
                         'determinism_word_occurrences':text.lower().count('deterministic')})
    result=dict(component='STATIC_EXECUTION_DEVIATION_REVIEW',
        runner_sha256=sha256(runner),orchestrator_sha256=sha256(orch),
        stage_t_reachable_helpers=t,stage_e_reachable_helpers=e,
        stage_t_strict_seed_reachable='strict_seed' in t,
        stage_e_explicit_strict_algorithms_call_reachable=enforced,
        inspected_top_level_entrypoints=['run_stage_t','run_stage_e','main'],
        source_function_line_ranges={n:[defs[n].lineno,defs[n].end_lineno] for n in ['strict_seed','infer_test','run_stage_e','main']},
        logs_summary=rows,
        status='UNRESOLVED_EXECUTION_ENFORCEMENT_DEVIATION' if not enforced else 'STATIC_ENFORCEMENT_PRESENT_RUNTIME_NOT_REPLAYED',
        consequence='Absence of the Stage-E enforcement call is not proof that saved scores are wrong or changed.',
        fixed_score_results_not_replaced=True,checkpoint_replay_performed=False,
        model_training_performed=False,model_inference_performed=False,
        retrospective_runtime_determinism_established=False,
        action='Disclose the enforcement gap; any checkpoint replay needs separate explicit authorization and isolated provenance.')
    out.json('SHARE/EXECUTION_DEVIATION_REVIEW.json',result)
    return result

def observed_tables(bundle,out):
    names,indices=condition_layout(bundle.fits)
    observations={};longrows=[];perfit=[];old_diff=[];attr=[]
    for pop in POPS:
        mask=bundle.masks[pop];yp=bundle.y[mask]
        values=[]
        for fit in bundle.fits:
            plan=MetricPlan(yp,fit.score[mask],fit.threshold);v=plan.compute(np.ones(mask.sum(),dtype=np.int64));values.append(v)
            ties=plan.tie_report()
            row=dict(run_id=fit.run_id,condition=fit.condition,family=fit.family,model_seed=fit.seed,
                     donor_world=fit.world,population=pop,n_positive=int(yp.sum()),
                     validation_threshold=fit.threshold,**dict(zip(METRICS,v)),**ties)
            perfit.append(row)
            if fit.family!='REFERENCE':
                stored=fit.stored['metrics'][pop]
                for k,val in zip(METRICS,v):
                    require(k in stored and np.isfinite(float(stored[k])) and abs(float(stored[k])-val)<=2e-12,'NEW_STORED_METRIC_PARITY',{'run_id':fit.run_id,'population':pop,'metric':k})
            elif pop=='full_test':
                for k,val in zip(METRICS[:5],v[:5]):
                    if k in fit.stored.get('metrics',{}):
                        delta=val-float(fit.stored['metrics'][k])
                        old_diff.append(dict(run_id=fit.run_id,metric=k,stable_recomputed=float(val),archived=float(fit.stored['metrics'][k]),difference=float(delta)))
                        if k in METRICS[:3]:require(abs(delta)<=2e-12,'OLD_NONRANKING_METRIC_PARITY',{'run_id':fit.run_id,'metric':k})
        arr=np.asarray(values);bycondition=arr[indices]
        observations[pop]=bycondition
        for i,name in enumerate(names):
            for k,metric in enumerate(METRICS):
                x=bycondition[i,:,k]
                longrows.append(dict(condition=name,population=pop,metric=metric,n_model_seeds=5,
                    donor_worlds_averaged=1,estimate=float(x.mean()),seed_sd=float(x.std(ddof=1)),
                    seed_min=float(x.min()),seed_max=float(x.max()),uncertainty_type='seed_description_not_test_CI'))
        for model in ['MLP','RandomForest_unweighted']:
            for mod in ['M5','M11']:
                ix=[names.index(f'B|{model}|{mod}|{w}') for w in [271707,271708,271709]]
                cube=bycondition[ix];x=cube.mean(axis=0)
                for k,metric in enumerate(METRICS):
                    longrows.append(dict(condition=f'B_mean3|{model}|{mod}',population=pop,metric=metric,n_model_seeds=5,
                        donor_worlds_averaged=3,estimate=float(x[:,k].mean()),seed_sd=float(x[:,k].std(ddof=1)),
                        seed_min=float(x[:,k].min()),seed_max=float(x[:,k].max()),
                        world_mean_min=float(cube[:,:,k].mean(axis=1).min()),world_mean_max=float(cube[:,:,k].mean(axis=1).max()),
                        uncertainty_type='fixed_world_mean_then_matched_seed_description_not_independent_15_cohorts'))
    for fit in bundle.fits:
        order=np.argsort(-fit.score,kind='mergesort');top=order[:422]
        dh=int((bundle.masks['D_positive'][top]&bundle.y[top].astype(bool)).sum())
        oh=int((bundle.masks['O_positive'][top]&bundle.y[top].astype(bool)).sum())
        nh=int(bundle.y[top].sum());require(nh==dh+oh,'ATTRIBUTION_SUM')
        a=dict(run_id=fit.run_id,condition=fit.condition,model_seed=fit.seed,budget=422,
               full_hits=nh,D_hits_in_same_full_list=dh,O_hits_in_same_full_list=oh)
        attr.append(a)
        if fit.family!='REFERENCE':
            old=fit.stored['full_budget_attribution']
            require(old['full_hits']==nh and old['D_positive_hits_in_full_top422']==dh and old['O_positive_hits_in_full_top422']==oh,'FULL_ATTRIBUTION_PARITY')
    catalog=contrast_catalog(names);coef=definition_matrix(catalog,names)
    contrast_rows=[]
    for pop,obs in observations.items():
        deltas=np.einsum('dc,csm->dsm',coef,obs)
        for i,c in enumerate(catalog):
            for k,m in enumerate(METRICS):
                x=deltas[i,:,k]
                contrast_rows.append(dict(contrast_id=c['contrast_id'],family=c['family'],world=c['world'],population=pop,metric=m,
                    observed_mean_paired_delta=float(x.mean()),paired_seed_sd=float(x.std(ddof=1)),
                    seed_min=float(x.min()),seed_max=float(x.max()),positive_seed_deltas=int((x>0).sum()),n_seeds=5))
    out.csv('SHARE/01_FIXED_PER_FIT_METRICS.csv',perfit)
    out.csv('SHARE/02_CONDITION_DESCRIPTIVES.csv',longrows)
    out.csv('SHARE/03_CONTRAST_POINT_ESTIMATES.csv',contrast_rows)
    out.csv('SHARE/04_FULL_BUDGET_ATTRIBUTION.csv',attr)
    out.csv('SHARE/05_REFERENCE_STORED_VERSUS_STABLE_METRICS.csv',old_diff)
    out.json('SHARE/CONTRAST_DEFINITIONS.json',catalog)
    return names,indices,observations,catalog

def flag_and_threshold_diagnostics(bundle,out):
    flagrows=[]
    for name,score in bundle.flags.items():
        for pop in POPS:
            m=bundle.masks[pop];p=MetricPlan(bundle.y[m],score[m],.5)
            v=p.compute(np.ones(m.sum(),dtype=int));metric_dict=dict(zip(METRICS,v))
            metric_dict['f1_at_native_binary_rule']=metric_dict.pop('f1_val_threshold')
            cf=p.confusion();cf['threshold_rule']='native_binary_flag_value_1_no_validation_or_test_selection'
            flagrows.append(dict(flag=name,population=pop,score_direction='as_stored_1',
                           **metric_dict,**p.tie_report(),**cf))
    rfrows=[]
    for f in bundle.fits:
        if f.family=='A' and f.model=='RandomForest_unweighted':
            p=MetricPlan(bundle.y,f.score,f.threshold);cf=p.confusion()
            rfrows.append(dict(run_id=f.run_id,model_seed=f.seed,**cf,
                               f1_at_frozen_validation_threshold=float(p.compute(np.ones(len(bundle.y),dtype=int))[2])))
    # Test-only D identities stay inside this process; release counts, not IDs.
    gcn=[f for f in bundle.fits if f.family=='A' and f.model=='GCN_reverse']
    incidence=bundle.incidence[bundle.incidence.partition=='test']
    key_to_docs={}
    for row in incidence.itertuples(index=False):key_to_docs.setdefault((row.company_code,int(row.fiscal_year)),set()).add(str(row.doc_key_hash))
    summaries=[];pairs=[]
    for ranking in ['D_subset_reranked','membership_in_full_top422']:
        sets={}
        for f in gcn:
            ix=np.flatnonzero(bundle.masks['D_frozen']) if ranking=='D_subset_reranked' else np.arange(len(bundle.y))
            top=ix[np.argsort(-f.score[ix],kind='mergesort')[:422]]
            hits={int(i) for i in top if bundle.masks['D_positive'][i] and bundle.y[i]==1}
            sets[f.seed]=hits
        union=set().union(*sets.values());common=set.intersection(*sets.values()) if sets else set()
        companies={bundle.canonical.iloc[i].company_code for i in union}
        docs=set().union(*(key_to_docs.get((bundle.canonical.iloc[i].company_code,int(bundle.canonical.iloc[i].fiscal_year)),set()) for i in union)) if union else set()
        summaries.append(dict(ranking=ranking,n_seeds=len(sets),hits_per_seed={str(s):len(v) for s,v in sets.items()},
            distinct_D_positive_rows_hit=len(union),D_positive_rows_hit_in_all_seeds=len(common),
            distinct_companies_hit=len(companies),distinct_usable_document_keys_hit=len(docs)))
        seeds=sorted(sets)
        for i,a in enumerate(seeds):
            for b in seeds[i+1:]:pairs.append(dict(ranking=ranking,left_seed=a,right_seed=b,shared_positive_row_hits=len(sets[a]&sets[b])))
    out.csv('SHARE/06_SINGLE_FLAG_FIXED_SCORES.csv',flagrows)
    out.csv('SHARE/07_A_RF_FIXED_THRESHOLD_CONFUSION.csv',rfrows)
    out.json('SHARE/08_A_GCN_D_HIT_OVERLAP_AGGREGATE.json',{'summaries':summaries,'pairwise_intersection_counts':pairs,'company_codes_exported':False,'document_keys_exported':False})

def influence_tables(bundle,components,names,indices,catalog,out):
    coef=definition_matrix(catalog,names)
    dpos=np.flatnonzero(bundle.masks['D_positive'])
    p_groups=sorted(set(int(g) for g in components[bundle.y==1]))
    designs=[]
    designs.append(('D_leave_one_positive_out',bundle.masks['D_frozen'],[np.array([i]) for i in dpos]))
    designs.append(('full_leave_one_positive_component_out',bundle.masks['full_test'],
                    [np.flatnonzero((components==g)&(bundle.y==1)) for g in p_groups]))
    designs.append(('D_leave_one_positive_component_out',bundle.masks['D_frozen'],
                    [np.flatnonzero((components==g)&bundle.masks['D_positive']) for g in p_groups if ((components==g)&bundle.masks['D_positive']).any()]))
    condition_rows=[];fit_rows=[];contrast_rows=[]
    for design,mask,deletions in designs:
        values=np.full((len(deletions),len(bundle.fits),len(METRICS)),np.nan)
        plans=[MetricPlan(bundle.y,f.score,f.threshold) for f in bundle.fits]
        for i,deleted in enumerate(deletions):
            w=mask.astype(int).copy();w[deleted]=0
            require(int(w[bundle.y==0].sum())==int(mask[bundle.y==0].sum()),'INFLUENCE_NEGATIVES_CHANGED')
            for j,p in enumerate(plans):values[i,j]=p.compute(w)
        cube=values[:,indices,:] # deletion,condition,seed,metric
        avg=cube.mean(axis=2)
        for ci,name in enumerate(names):
            for mi,m in enumerate(METRICS):
                x=avg[:,ci,mi];valid=x[np.isfinite(x)]
                condition_rows.append(dict(design=design,condition=name,metric=m,n_deletions=len(deletions),estimable_deletions=len(valid),
                    influence_min=float(valid.min()) if len(valid) else None,influence_max=float(valid.max()) if len(valid) else None,
                    influence_mean=float(valid.mean()) if len(valid) else None,undefined_reason=None if len(valid)==len(x) else 'single_class_after_positive_deletion',
                    interpretation='fixed_model_positive_only_deletion_range_NOT_confidence_interval'))
        for fi,f in enumerate(bundle.fits):
            for mi,m in enumerate(METRICS):
                x=values[:,fi,mi];v=x[np.isfinite(x)]
                fit_rows.append(dict(design=design,run_id=f.run_id,condition=f.condition,model_seed=f.seed,metric=m,n_deletions=len(x),estimable_deletions=len(v),
                         influence_min=float(v.min()) if len(v) else None,influence_max=float(v.max()) if len(v) else None))
        # Do not multiply NaNs by unrelated zero coefficients.
        for c in catalog:
            subset=[(names.index(n),w) for n,w in c['coefficients'].items()]
            delta=sum(w*avg[:,ix,:] for ix,w in subset)
            for mi,m in enumerate(METRICS):
                x=delta[:,mi];v=x[np.isfinite(x)]
                contrast_rows.append(dict(design=design,contrast_id=c['contrast_id'],metric=m,n_deletions=len(x),estimable_deletions=len(v),
                     influence_min=float(v.min()) if len(v) else None,influence_max=float(v.max()) if len(v) else None,
                     interpretation='paired_fixed_score_influence_range_NOT_CI'))
    out.csv('SHARE/09_INFLUENCE_CONDITION_RANGES.csv',condition_rows)
    out.csv('SHARE/10_INFLUENCE_PER_FIT_RANGES.csv',fit_rows)
    out.csv('SHARE/11_INFLUENCE_CONTRAST_RANGES.csv',contrast_rows)
    return {name:len(deletions) for name,mask,deletions in designs}

_WORK={}
def _init_worker(plans,idx,y,groups,cluster_seed,fit_seed,max_attempts):
    global _WORK
    _WORK=dict(plans=plans,cluster_idx=idx,y=y,groups=groups,cluster_seed=cluster_seed,fit_seed=fit_seed,max_attempts=max_attempts)

def _replicate(b):
    g=_WORK;w,draw,rejected=bootstrap_draw(b,g['cluster_idx'],g['y'],g['cluster_seed'],g['fit_seed'],g['max_attempts'])
    ncond=len(g['groups'])
    if w is None:return b,np.full((ncond,len(METRICS)),np.nan),dict(replicate=b,accepted=False,rejected_draws=rejected)
    values=np.array([p.compute(w) for p in g['plans']])
    require(np.isfinite(values).all(),'BOOTSTRAP_NONFINITE_METRIC')
    estimate=values[g['groups']][:,draw,:].mean(axis=1)
    h=hashlib.sha256(np.asarray(w,dtype='<i8').tobytes()+np.asarray(draw,dtype='<i8').tobytes()).hexdigest()
    audit=dict(replicate=b,accepted=True,rejected_draws=rejected,n_rows=int(w.sum()),n_positive=int(np.dot(w,g['y'])),
               budget=max(1,math.ceil(.05*int(w.sum()))),draw_sha256=h,
               seed_multiplicity=[int(v) for v in np.bincount(draw,minlength=5)])
    return b,estimate,audit

def bootstrap_scheme(bundle,cluster_idx,names,indices,observations,catalog,out,scheme,workers=2,nboot=2000,max_attempts=1000,progress=print):
    seed_pair=(1807141,1807142) if scheme=='company_x_matched_seed' else (2707141,2707142)
    plans=[MetricPlan(bundle.y,f.score,f.threshold) for f in bundle.fits]
    args=(plans,cluster_idx,bundle.y,indices,*seed_pair,max_attempts)
    arr=np.full((nboot,len(names),len(METRICS)),np.nan);drawlogs=[None]*nboot
    started=time.monotonic()
    if workers==1:
        _init_worker(*args);iterator=map(_replicate,range(nboot));pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init_worker,initargs=args)
        iterator=pool.map(_replicate,range(nboot),chunksize=5)
    try:
        for count,(b,est,log) in enumerate(iterator,1):
            arr[b]=est;drawlogs[b]=log
            if count%100==0 or count==nboot:progress(f'{scheme}: {count}/{nboot} fixed-score resamples completed',flush=True)
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    accepted=np.array([d['accepted'] for d in drawlogs],bool);enough=bool(accepted.all())
    observed=observations['full_test'].mean(axis=1)
    condition_rows=[];contrast_rows=[]
    nclusters=int(cluster_idx.max())+1
    def interval(x):
        if not enough:return None,None
        a,b=np.quantile(x,[.025,.975],axis=0,method='linear');return float(a),float(b)
    for ci,c in enumerate(names):
        for mi,m in enumerate(METRICS):
            lo,hi=interval(arr[:,ci,mi])
            condition_rows.append(dict(scheme=scheme,condition=c,population='full_test',metric=m,
                estimate=float(observed[ci,mi]),ci_low=lo,ci_high=hi,valid_replicates=int(accepted.sum()),required_replicates=nboot,
                clusters=nclusters,estimand='mean_of_per_fit_metrics_over_five_matched_seed_labels',
                interval_status='ESTIMABLE' if enough else 'NON_ESTIMABLE_1000_ATTEMPT_LIMIT',
                inference='marginal_exploratory_percentile_no_pvalue'))
    # Mean across donor worlds is a mean of metrics conditional on all three worlds.
    for model in ['MLP','RandomForest_unweighted']:
        for mod in ['M5','M11']:
            ix=[names.index(f'B|{model}|{mod}|{w}') for w in [271707,271708,271709]]
            mean=arr[:,ix,:].mean(axis=1);point=observed[ix].mean(axis=0)
            for mi,m in enumerate(METRICS):
                lo,hi=interval(mean[:,mi])
                condition_rows.append(dict(scheme=scheme,condition=f'B_mean3|{model}|{mod}',population='full_test',metric=m,
                   estimate=float(point[mi]),ci_low=lo,ci_high=hi,valid_replicates=int(accepted.sum()),required_replicates=nboot,
                   clusters=nclusters,estimand='equal_mean_of_metrics_over_fixed_three_worlds_and_matched_five_seeds',
                   interval_status='ESTIMABLE' if enough else 'NON_ESTIMABLE_1000_ATTEMPT_LIMIT',
                   inference='conditional_on_three_completions_not_imputation_uncertainty'))
    deltas=[]
    for c in catalog:
        terms=[(names.index(k),v) for k,v in c['coefficients'].items()]
        d=sum(w*arr[:,ix,:] for ix,w in terms);p=sum(w*observed[ix,:] for ix,w in terms);deltas.append(d)
        for mi,m in enumerate(METRICS):
            lo,hi=interval(d[:,mi])
            contrast_rows.append(dict(scheme=scheme,contrast_id=c['contrast_id'],family=c['family'],world=c['world'],population='full_test',metric=m,
                observed_mean_paired_delta=float(p[mi]),ci_low=lo,ci_high=hi,valid_replicates=int(accepted.sum()),required_replicates=nboot,
                clusters=nclusters,interval_status='ESTIMABLE' if enough else 'NON_ESTIMABLE_1000_ATTEMPT_LIMIT',
                inference='marginal_exploratory_paired_percentile_not_equivalence_or_calibrated_pvalue'))
    out.csv(f'SHARE/{scheme}_CONDITION_INTERVALS.csv',condition_rows)
    out.csv(f'SHARE/{scheme}_CONTRAST_INTERVALS.csv',contrast_rows)
    out.json(f'SHARE/{scheme}_DRAW_AUDIT.json',drawlogs)
    rp=out.check(out.root/f'SHARE/{scheme}_AGGREGATE_REPLICATES.npz')
    np.savez_compressed(rp,condition_metrics=arr,contrast_metrics=np.stack(deltas,axis=1),
                        condition_names=np.asarray(names,dtype=str),contrast_names=np.asarray([c['contrast_id'] for c in catalog],dtype=str),
                        metric_names=np.asarray(METRICS,dtype=str),valid=accepted,replicate=np.arange(nboot))
    summary=dict(scheme=scheme,required_replicates=nboot,valid_replicates=int(accepted.sum()),failed_replicates=int((~accepted).sum()),
        rejected_draws=sum(d['rejected_draws'] for d in drawlogs),replicates_with_redraw=sum(d['rejected_draws']>0 for d in drawlogs),
        max_rejected_draws=max(d['rejected_draws'] for d in drawlogs),max_attempts_per_replicate=max_attempts,
        cluster_seed=seed_pair[0],matched_seed_rng_seed=seed_pair[1],replicate_index_start=0,replicate_rng_rule='base_seed+10*b',
        same_cluster_draw_across_all_fits_and_worlds=True,same_five_seed_draw_across_all_conditions=True,
        n_clusters=nclusters,conditions=len(names),contrasts=len(catalog),metrics=list(METRICS),workers=workers,
        intervals='linear percentile 2.5/97.5; marginal exploratory',resampled_budget='ceil(0.05*resampled_rows)',
        ensemble_mean_score_used=False,retraining_performed=False,seconds=time.monotonic()-started,
        result='PASS_FIXED_SCORE_BOOTSTRAP' if enough else 'COMPLETE_WITH_NON_ESTIMABLE_INTERVALS')
    out.json(f'SHARE/{scheme}_DESIGN_AND_STATUS.json',summary)
    return summary
