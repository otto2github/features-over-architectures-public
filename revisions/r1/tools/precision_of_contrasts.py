#!/usr/bin/env python3
"""Precision of the paired comparisons, from the distributed company x matched-seed bootstrap arrays (no fitting,
no new resampling). For each contrast: linear 2.5/97.5 percentile interval, bootstrap SE (SD of replicates) and the
approximate minimum detectable difference MDD80 = (z_0.975 + z_0.80) x SE, i.e. the true difference a two-sided 5%
test would detect with 80% probability under a normal approximation. Usage: python3 tools/precision_of_contrasts.py"""
from pathlib import Path
from statistics import NormalDist
import json, sys
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
K = NormalDist().inv_cdf(.975) + NormalDist().inv_cdf(.80)
PAIRS = [('MLP', 'GCN'), ('MLP', 'GraphSAGE'), ('RF', 'GCN'), ('RF', 'GraphSAGE')]


def row(group, label, rep, metric):
    rep = np.asarray(rep, float); lo, hi = np.quantile(rep, [.025, .975], method='linear'); se = float(rep.std(ddof=1))
    return {'group': group, 'comparison': label, 'metric': metric, 'ci95': [float(lo), float(hi)], 'se': se, 'mdd80': float(K * se)}


def main(root=ROOT):
    rows = []
    z = np.load(root / 'aggregates/sensitivity_fixed_score/company_x_matched_seed_AGGREGATE_REPLICATES.npz')
    cond = list(map(str, z['condition_names'])); met = list(map(str, z['metric_names']))
    C = z['condition_metrics'][z['valid'].astype(bool)]
    name = {'MLP': 'MLP', 'RF': 'RandomForest_unweighted', 'GCN': 'GCN_reverse', 'GraphSAGE': 'GraphSAGE_reverse'}
    for m in ['roc_auc', 'hits_at_budget']:
        for a, b in PAIRS:
            rep = C[:, cond.index(f'V26|{name[a]}|M11'), met.index(m)] - C[:, cond.index(f'V26|{name[b]}|M11'), met.index(m)]
            rows.append(row('primary window, 129 inputs', f'{a} − {b}', rep, m))
    e = np.load(root / 'aggregates/extension_fixed_score/company_x_matched_seed_AGGREGATE_REPLICATES.npz')
    em = list(map(str, e['metrics']))
    for m in ['roc_auc', 'hits_at_budget']:
        for i, (a, b) in enumerate(PAIRS):
            rows.append(row('primary window, 42 inputs', f'{a} − {b}', e['contrast_replicates'][:, i, em.index(m)], m))
    b7 = np.load(root / 'aggregates/e7/FULL_BOOTSTRAP_DIFFERENCES.npy')
    for i, mod in enumerate(['GCN', 'GraphSAGE']):
        rows.append(row('primary window, 129 inputs, E7 links added', f'{mod}: E7 − control', b7[:, i, 0], 'roc_auc'))
        rows.append(row('primary window, 129 inputs, E7 links added', f'{mod}: E7 − control', b7[:, i, 5], 'hits_at_budget'))
    w2 = root / 'aggregates/w2/BOOTSTRAP_DIFFERENCES.npy'
    if w2.exists():
        a2 = np.load(w2)
        for m_i, m in [(0, 'roc_auc'), (5, 'hits_at_budget')]:
            for i, (a, b) in enumerate(PAIRS):
                rows.append(row('earlier window W2, 129 inputs', f'{a} − {b}', a2[:, i, m_i], m))
    out = {'status': 'PRECISION_OF_CONTRASTS_COMPUTED', 'mdd_multiplier': K,
           'definition': 'MDD80 = (z_0.975 + z_0.80) x bootstrap SE; normal approximation; interval = linear 2.5/97.5 percentiles',
           'rows': rows, 'model_training_or_resampling_performed': False}
    return out


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, ensure_ascii=False))
