#!/usr/bin/env python3
"""Verify the distributed W2 (earlier window) aggregates: no training, inference or new resampling.
Recomputes group means/SDs and paired seed differences from per-fit metrics and all 24 Full interval endpoints pairs
(48 endpoint values) from the distributed bootstrap-difference array."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODELS = ['RF', 'MLP', 'GCN', 'SAGE']; SEEDS = [42, 123, 456, 789, 1024]
METRICS = ['roc_auc', 'ap', 'f1_val_threshold', 'p_at_5pct', 'r_at_10fpr', 'hits_at_budget']
CONTRASTS = [('MLP', 'GCN'), ('MLP', 'SAGE'), ('RF', 'GCN'), ('RF', 'SAGE')]


def require(ok, msg):
    if not ok: raise AssertionError(msg)


def verify(root=ROOT):
    root = Path(root); A = root / 'aggregates/w2'
    p = pd.read_csv(A / 'PER_SEED_METRICS.csv'); g = pd.read_csv(A / 'GROUP_MEANS.csv')
    pc = json.loads((A / 'PAIRED_CONTRASTS.json').read_text()); counts = json.loads((A / 'WINDOW_COUNTS.json').read_text())
    lock = json.loads((root / 'run_records/w2/ALL_20_LOCKED.json').read_text())
    require(len(p) == 20 and {(r.model, r.seed) for r in p.itertuples()} == {(m, s) for m in MODELS for s in SEEDS}, 'W2_FIT_SET')
    require((p.n == 7309).all() and (p.n_pos == 40).all() and (p.budget == 366).all(), 'W2_TEST_COUNTS')
    require(counts['test'] == {'rows': 7309, 'positive': 40} and counts['train'] == {'rows': 17448, 'positive': 158}
            and counts['validation'] == {'rows': 6658, 'positive': 96}, 'W2_WINDOW_COUNTS')
    require(lock['status'] == 'ALL_20_LOCKED' and lock['test_performance_evaluated_before_lock'] is False, 'W2_LOCK')
    err = 0.0
    for row in g.itertuples():
        q = p[p.model == row.model]
        for m in METRICS:
            err = max(err, abs(q[m].mean() - getattr(row, m + '_mean')), abs(q[m].std(ddof=1) - getattr(row, m + '_sd')))
    perr = 0.0
    arr = np.load(A / 'BOOTSTRAP_DIFFERENCES.npy')
    require(arr.shape == (2000, 4, 6) and np.isfinite(arr).all(), 'W2_ARRAY')
    ierr = 0.0; excluded = []
    for v in pc['values']:
        a = p[p.model == v['model_a']].set_index('seed').loc[SEEDS, v['metric']].to_numpy()
        b = p[p.model == v['model_b']].set_index('seed').loc[SEEDS, v['metric']].to_numpy()
        perr = max(perr, float(np.max(np.abs((a - b) - np.array(v['paired_seed_differences'])))), abs((a - b).mean() - v['paired_mean_difference']))
        col = arr[:, CONTRASTS.index((v['model_a'], v['model_b'])), METRICS.index(v['metric'])]
        lo, hi = np.quantile(col, [.025, .975], method='linear')
        ierr = max(ierr, abs(lo - v['company_seed_bootstrap_ci95'][0]), abs(hi - v['company_seed_bootstrap_ci95'][1]))
        if lo > 0 or hi < 0: excluded.append(f"{v['contrast']}|{v['metric']}")
    require(err < 1e-12 and perr < 1e-12 and ierr < 1e-12, 'W2_RECOMPUTE')
    require(len(pc['values']) == 24 and pc['bootstrap']['valid_replicates'] == 2000, 'W2_CONTRAST_COUNT')
    return {'status': 'W2_AGGREGATE_CHECK_PASS', 'fits': 20, 'group_mean_sd_max_error': err, 'paired_difference_max_error': perr,
            'intervals_independently_recomputed': 24, 'interval_endpoints_recomputed': 48, 'max_interval_endpoint_abs_error': ierr,
            'intervals_excluding_zero': excluded, 'training_or_inference_performed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
