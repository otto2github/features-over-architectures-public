#!/usr/bin/env python3
"""Regenerate current main-table cells from frozen aggregate evidence.

No licensed rows, model fits, inference or network calls are used. The default
output directory is deliberately separate from all immutable source folders.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MODELS=['MLP','RandomForest_unweighted','GCN_reverse','GraphSAGE_reverse']
LABELS=['MLP','RF, unweighted','GCN, reverse','GraphSAGE, reverse']
CONTRASTS=['MLP_minus_GCN','MLP_minus_GraphSAGE','RF_minus_GCN','RF_minus_GraphSAGE']
CLABELS=['MLP − GCN','MLP − GraphSAGE','RF − GCN','RF − GraphSAGE']
def unique(df, **where):
    q=df
    for k,v in where.items():q=q[q[k].eq(v)]
    if len(q)!=1:raise ValueError(f'Expected one source row for {where}, found {len(q)}')
    return q.iloc[0]
def signed(x,unicode_minus=True):
    s=f'{float(x):+.4f}'
    return s.replace('-', '−') if unicode_minus else s
def ci(lo,hi,unicode_minus=True):return f'[{signed(lo,unicode_minus)}, {signed(hi,unicode_minus)}]'
def regenerate(root=ROOT):
    root=Path(root)
    d27=pd.read_csv(root/'aggregates/sensitivity_fixed_score/02_CONDITION_DESCRIPTIVES.csv')
    p27=pd.read_csv(root/'aggregates/sensitivity_fixed_score/01_FIXED_PER_FIT_METRICS.csv')
    c27=pd.read_csv(root/'aggregates/sensitivity_fixed_score/company_x_matched_seed_CONTRAST_INTERVALS.csv')
    d28=pd.read_csv(root/'aggregates/extension_fixed_score/02_MATCHED_DESCRIPTIVES.csv')
    c28=pd.read_csv(root/'aggregates/extension_fixed_score/company_x_matched_seed_CONTRAST_INTERVALS.csv')
    s28=pd.read_csv(root/'aggregates/extension_fixed_score/company_document_x_matched_seed_CONTRAST_INTERVALS.csv')
    flag=pd.read_csv(root/'aggregates/sensitivity_fixed_score/06_SINGLE_FLAG_FIXED_SCORES.csv')
    def old(cond,pop,metric):return float(unique(d27,condition=cond,population=pop,metric=metric)['estimate'])
    def new(model,pop,metric):return float(unique(d28,model=model,population=pop,metric=metric)['estimate'])
    # Counts are the fixed protocol contract, not estimates inferred from a model.
    proto=json.loads((root/'protocol/reader/extension_protocol.json').read_text())
    contract=json.loads((root/'documentation/TABLE1_POPULATION_CONTRACT.json').read_text())
    out={'Table1_panel1.csv':[['Analysis / partition','Years','Rows','Positive','Negative']]}
    for x in contract['rows']:
        if x['n']!=x['positive']+x['negative']:raise ValueError('Population arithmetic mismatch')
        out['Table1_panel1.csv'].append([x['label'],x['years'],f"{x['n']:,}",f"{x['positive']:,}",f"{x['negative']:,}"])
    rows=[['Condition / input count','Learner','Full AUC','D AUC','O AUC','Full hits / 422','D hits / 422']]
    for prefix,label,mods in [('V26','Primary, 129',MODELS),('A','A purge, 129',MODELS),('ASOF','As-of zero-fill, 129',MODELS[:2]),('B_mean3','Donor mean3, 129',MODELS[:2])]:
        for model in mods:
            cond=f'{prefix}|{model}|M11'
            rows.append([label,LABELS[MODELS.index(model)],*[f'{old(cond,p,"roc_auc"):.4f}' for p in ['full_test','D_frozen','O_frozen']],f'{old(cond,"full_test","hits_at_budget"):.1f}',f'{old(cond,"D_frozen","hits_at_budget"):.1f}'])
    for model in MODELS:
        rows.append(['FIN87 deleted, 42',LABELS[MODELS.index(model)],*[f'{new(model,p,"roc_auc"):.4f}' for p in ['full_test','D_frozen','O_frozen']],f'{new(model,"full_test","hits_at_budget"):.1f}',f'{new(model,"D_frozen","hits_at_budget"):.1f}'])
    cond='C|RandomForest_unweighted|flags3'
    rows.append(['Source flags, 3','RF, unweighted',*[f'{old(cond,p,"roc_auc"):.4f}' for p in ['full_test','D_frozen','O_frozen']],f'{old(cond,"full_test","hits_at_budget"):.1f}',f'{old(cond,"D_frozen","hits_at_budget"):.1f}'])
    out['Table2_panel1.csv']=rows
    rows=[['Paired comparison','Learner / input','Absolute AUC means','ΔAUC','95% interval']]
    def add_compare(title,label,a,b,contrast):
        r=unique(c27,contrast_id=contrast,metric='roc_auc',population='full_test')
        rows.append([title,label,f'{a:.4f} → {b:.4f}',signed(r.observed_mean_paired_delta),ci(r.ci_low,r.ci_high)])
    for model,mlabel in zip(MODELS[:2],['MLP','RF']):
        for modal in ['M5','M11']:
            add_compare('Donor − as-of zero-fill',f'{mlabel} {modal}',old(f'ASOF|{model}|{modal}','full_test','roc_auc'),old(f'B_mean3|{model}|{modal}','full_test','roc_auc'),f'B_minus_ASOF__{model}__{modal}__mean3')
    for model,mlabel in zip(MODELS[:2],['MLP','RF']):
        add_compare('Donor M11 − M5',f'{mlabel}, mean3',old(f'B_mean3|{model}|M5','full_test','roc_auc'),old(f'B_mean3|{model}|M11','full_test','roc_auc'),f'B_increment__{model}__mean3')
    for model,mlabel in zip(MODELS[:2],['MLP','RF']):
        add_compare('B0 42 − 17 columns',mlabel,old(f'B0|{model}|M5_no_FIN87','full_test','roc_auc'),old(f'B0|{model}|M11_no_FIN87','full_test','roc_auc'),f'B0_42_minus_17__{model}')
    out['Table3_panel1.csv']=rows
    rows=[['Learner','AUC ± seed SD','AP','F1 (val)','P@5%','R@10FPR','Hits / 422']]
    for model,label in zip(MODELS,LABELS):
        r=unique(d28,model=model,population='full_test',metric='roc_auc')
        rows.append([label,f'{r.estimate:.4f} ± {r.seed_sd:.4f}',f'{new(model,"full_test","ap"):.5f}',*[f'{new(model,"full_test",m):.4f}' for m in ['f1_val_threshold','p_at_5pct','r_at_10fpr']],f'{new(model,"full_test","hits_at_budget"):.1f}'])
    out['Table4_panel1.csv']=rows
    rows=[['Contrast','Mean ΔAUC','Company × seed 95% interval','Component × seed 95% interval']]
    for ident,label in zip(CONTRASTS,CLABELS):
        r=unique(c28,contrast=ident,metric='roc_auc');s=unique(s28,contrast=ident,metric='roc_auc')
        rows.append([label,signed(r.point_estimate,False),ci(r.lower,r.upper,False),ci(s.lower,s.upper,False)])
    out['Table4_panel2.csv']=rows
    rows=[['Source-status model / fixed score','Full AUC','D AUC','Full tie bounds; fixed hits / 422','D tie bounds; fixed hits / 422']]
    def tie_string(q):
        values=[]
        for c in ['tie_hits_min','tie_hits_max','hits_at_budget']:
            vals=q[c].drop_duplicates()
            if len(vals)!=1:raise ValueError('C-model tie summary varies by seed; revise table rule explicitly')
            values.append(int(vals.iloc[0]))
        return f'{values[0]}–{values[1]}; {values[2]}'
    c=p27[p27.condition.eq('C|RandomForest_unweighted|flags3')]
    rows.append(['Three-flag RF',f'{old(cond,"full_test","roc_auc"):.4f}',f'{old(cond,"D_frozen","roc_auc"):.4f}',tie_string(c[c.population.eq('full_test')]),tie_string(c[c.population.eq('D_frozen')])])
    for ident,label in [('asof_any_source_missing','Selected source absent'),('asof_any_final_financial_nonfinite','Final FIN87 nonfinite'),('fin_any_no_prior_candidate','No prior candidate')]:
        a=unique(flag,flag=ident,population='full_test');b=unique(flag,flag=ident,population='D_frozen')
        rows.append([label,f'{a.roc_auc:.4f}',f'{b.roc_auc:.4f}',f'{int(a.tie_hits_min)}–{int(a.tie_hits_max)}; {int(a.hits_at_budget)}',f'{int(b.tie_hits_min)}–{int(b.tie_hits_max)}; {int(b.hits_at_budget)}'])
    out['Table5_panel1.csv']=rows
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT.parent/'regenerated_tables');a=ap.parse_args()
    dest=a.output.resolve()
    # Output must be a separate directory, never an immutable source directory.
    for src in ['aggregates','code','config','documentation','protocol','provenance','run_records','tables','tools','tests','figures']:
        s=(ROOT/src).resolve()
        if dest==s or s in dest.parents:raise SystemExit(f'Output would enter immutable source folder: {dest}')
    if dest==ROOT.resolve():raise SystemExit('Choose a separate output directory')
    tables=regenerate();dest.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        with (dest/name).open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(rows)
    print(json.dumps({'status':'MAIN_TABLES_REGENERATED','panels':len(tables),'output':str(dest)},indent=2))
if __name__=='__main__':main()
