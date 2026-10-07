"""W2 point metrics and company x matched-seed paired bootstrap for the four fixed contrasts."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import itertools
import math
import multiprocessing as mp
import numpy as np
from w2_runtime_common import require
from w2_metrics import Plan, NAMES

METRICS = NAMES + ['hits_at_budget']
SEEDS = [42, 123, 456, 789, 1024]
MODELS = ['RF', 'MLP', 'GCN', 'SAGE']
CONTRASTS = [('MLP', 'GCN'), ('MLP', 'SAGE'), ('RF', 'GCN'), ('RF', 'SAGE')]
BOOT = {}


def point_metrics(y, s, threshold):
    """Archived Plan metrics (unit weights), sklearn cross-check, all-threshold R@10FPR, budget hits and tie bounds."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    require(len(y) > 0 and np.unique(y).size == 2 and np.isfinite(s).all(), 'POINT_POPULATION')
    plan = Plan(y, s, threshold)
    v = plan.compute(np.ones(len(y)))
    order = np.argsort(-s, kind='mergesort'); budget = max(1, math.ceil(.05 * len(y)))
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=True)
    ref = np.array([roc_auc_score(y, s), average_precision_score(y, s), f1_score(y, s >= threshold, zero_division=0),
                    float(y[order[:budget]].mean()), float(tpr[fpr <= .10 + 1e-12].max())])
    require(np.allclose(v, ref, rtol=0, atol=2e-12), 'POINT_METRIC_REFERENCE_MISMATCH')
    fa, ta, _ = roc_curve(y, s, drop_intermediate=False)
    out = dict(zip(NAMES, map(float, v)))
    out.update({'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum()), 'hits_at_budget': int(y[order[:budget]].sum()), 'r_at_10fpr_all_thresholds': float(ta[fa <= .10 + 1e-12].max()),
                **plan.tie_report()})
    return out


def bootstrap_init(payload, cluster_index, n_companies, base_seed):
    BOOT.clear()
    BOOT.update({'plans': {k: Plan(y, s, t) for k, (y, s, t) in payload.items()},
                 'clusters': np.asarray(cluster_index, int), 'n': n_companies, 'seed': base_seed,
                 'y': next(iter(payload.values()))[0]})


def bootstrap_one(replicate):
    rng = np.random.default_rng(BOOT['seed'] + int(replicate) * 10)
    n = BOOT['n']
    counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
    w = counts[BOOT['clusters']].astype(float)
    seed_draw = rng.integers(0, len(SEEDS), size=len(SEEDS))
    y = BOOT['y']
    if float(np.sum(w * y)) <= 0 or float(np.sum(w * (1 - y))) <= 0:
        return None
    budget = max(1, math.ceil(float(w.sum()) * .05))
    cache = {}
    for model in MODELS:
        for i in set(seed_draw.tolist()):
            m = BOOT['plans'][(model, SEEDS[i])].compute(w)
            cache[(model, i)] = np.r_[m, m[3] * budget]
    out = []
    for a, b in CONTRASTS:
        out.append(np.mean([cache[(a, int(i))] - cache[(b, int(i))] for i in seed_draw], axis=0))
    return np.stack(out)


def summarize(rows, payload, company_codes, settings):
    import pandas as pd
    frame = pd.DataFrame(rows)
    groups = []
    for model, sel in frame.groupby('model', sort=False):
        require(sorted(sel.seed.tolist()) == SEEDS, 'FIVE_SEEDS_REQUIRED')
        g = {'model': model, 'n': int(sel.n.iloc[0]), 'n_pos': int(sel.n_pos.iloc[0])}
        for metric in METRICS + ['r_at_10fpr_all_thresholds']:
            g[metric + '_mean'] = float(sel[metric].mean()); g[metric + '_sd'] = float(sel[metric].std(ddof=1))
        groups.append(g)
    point = []
    for a, b in CONTRASTS:
        A = frame[frame.model == a].set_index('seed'); B = frame[frame.model == b].set_index('seed')
        for metric in METRICS:
            d = (A.loc[SEEDS, metric] - B.loc[SEEDS, metric]).to_numpy(float)
            observed = abs(float(d.mean()))
            flips = [abs(float(np.mean(d * np.array(x)))) for x in itertools.product([-1, 1], repeat=5)]
            point.append({'contrast': f'{a} minus {b}', 'model_a': a, 'model_b': b, 'metric': metric,
                          'paired_seed_differences': d.tolist(), 'paired_mean_difference': float(d.mean()),
                          'paired_seed_sd': float(d.std(ddof=1)),
                          'descriptive_exact_seed_signflip_p': sum(v >= observed - 1e-15 for v in flips) / 32})
    index, companies = pd.factorize(pd.Series(company_codes), sort=True)
    with ProcessPoolExecutor(max_workers=settings['workers'], mp_context=mp.get_context('spawn'),
                             initializer=bootstrap_init, initargs=(payload, index, len(companies), settings['rng_seed'])) as pool:
        draws = list(pool.map(bootstrap_one, range(settings['replicates']), chunksize=8))
    valid = [x for x in draws if x is not None]
    require(len(valid) >= settings['min_valid_fraction'] * settings['replicates'], 'TOO_FEW_VALID_BOOTSTRAP_REPLICATES')
    array = np.stack(valid)
    for item in point:
        col = array[:, CONTRASTS.index((item['model_a'], item['model_b'])), METRICS.index(item['metric'])]
        lo, hi = np.quantile(col, [.025, .975], method='linear')
        item.update({'company_seed_bootstrap_ci95': [float(lo), float(hi)], 'bootstrap_valid_replicates': len(valid)})
    meta = {**settings, 'valid_replicates': len(valid), 'invalid_single_class_replicates': len(draws) - len(valid),
            'company_clusters': int(len(companies)), 'array_shape': list(array.shape),
            'array_axes': ['replicate', 'contrast ' + str([f'{a}-{b}' for a, b in CONTRASTS]), 'metric ' + str(METRICS)]}
    return {'group_means': groups, 'paired_contrasts': point, 'bootstrap': meta}, array
