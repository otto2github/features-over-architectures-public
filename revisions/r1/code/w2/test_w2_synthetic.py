#!/usr/bin/env python3
"""Local synthetic checks for the W2 kit (no private data, no CUDA): metrics, window patching, RF stage-T path,
paired bootstrap arithmetic. Run: python3 -B test_w2_synthetic.py"""
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from w2_metrics import self_test
from w2_statistics import point_metrics, summarize, CONTRASTS, METRICS, MODELS, SEEDS
import w2_worker as W

def main():
    out = {'metric_equivalence': self_test(200)}
    rng = np.random.default_rng(1)
    # point metrics with ties and all-threshold R@10FPR
    y = (rng.random(3000) < .02).astype(int); y[:2] = [0, 1]
    s = np.round(rng.random(3000) * .6 + y * .3, 2)
    pm = point_metrics(y, s, .5)
    assert pm['n'] == 3000 and pm['n_pos'] == int(y.sum()) and pm['budget'] == 150
    out['point_metrics'] = 'PASS'
    # window patching of the verbatim core
    spec = importlib.util.spec_from_file_location('core', KIT / 'formal_rerun_core_REFERENCE.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    binding = json.loads((KIT / 'INPUT_BINDINGS.json').read_text())
    core.TRAIN_YEARS, core.VAL_YEARS, core.TEST_YEARS = W.W_TRAIN, W.W_VAL, W.W_TEST
    cols = json.loads((KIT / 'M11_COLUMN_ORDER.json').read_text())['columns']
    rows = []
    for year in range(2010, 2023):
        n = 120
        d = pd.DataFrame(rng.normal(size=(n, len(cols))).astype(np.float32), columns=cols)
        d['year'] = year; d['company_code'] = [f'{i:06d}' for i in range(n)]; d['node_id'] = d.company_code
        t = (rng.random(n) < .1).astype(float); t[rng.random(n) < .05] = np.nan; d['target'] = t
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    neg, pos = core.dynamic_pos_weight(df)[1:]
    sel = df[df.year.isin(W.W_TRAIN) & df.target.notna()]
    assert (neg, pos) == (int((sel.target == 0).sum()), int(sel.target.sum())), 'pos weight not on W2 training years'
    mu, sd = core.fit_scaler(df, cols)
    assert np.allclose(mu[0], sel[cols].fillna(0).to_numpy(np.float32).mean(0), atol=1e-5), 'scaler not on W2 training years'
    out['window_patch'] = 'PASS'
    # RF stage-T path on synthetic training/validation years only
    train = df[df.year <= 2018].copy()
    b = dict(binding); b['rf_settings'] = {**binding['rf_settings'], 'n_estimators': 20, 'n_jobs': 1}
    va_exp = (int((train.year.isin(W.W_VAL) & train.target.notna()).sum()), int(train[train.year.isin(W.W_VAL)].target.sum()))
    b['expected_supervision'] = {**binding['expected_supervision'], 'validation': {'rows': va_exp[0], 'positive': va_exp[1]}}
    with tempfile.TemporaryDirectory() as tmp:
        res, names = W.fit_one({'model': 'RF', 'seed': 42, 'run_id': 'RF__M11__W2__seed42'}, core, train, cols, b, {}, Path(tmp))
        assert names == ['model.joblib'] and (Path(tmp) / 'model.joblib').is_file() and 0 <= res['val_threshold'] <= 1
    out['rf_stage_t'] = 'PASS'
    # paired bootstrap arithmetic on synthetic scores
    n = 2000; comp = np.array([f'{i // 2:06d}' for i in range(n)]); yy = (rng.random(n) < .03).astype(int); yy[:2] = [0, 1]
    payload, prow = {}, []
    for m_i, m in enumerate(MODELS):
        for sd_ in SEEDS:
            sc = np.clip(rng.random(n) * .7 + yy * (.2 + .05 * m_i), 0, 1)
            payload[(m, sd_)] = (yy, sc, .5)
            prow.append({'model': m, 'seed': sd_, **point_metrics(yy, sc, .5)})
    summ, arr = summarize(prow, payload, comp, {'replicates': 60, 'rng_seed': 241707, 'workers': 2, 'min_valid_fraction': .95})
    assert arr.shape[1:] == (len(CONTRASTS), len(METRICS))
    for item in summ['paired_contrasts']:
        a = [r for r in prow if r['model'] == item['model_a']]; bb = [r for r in prow if r['model'] == item['model_b']]
        d = np.mean([x[item['metric']] for x in a]) - np.mean([x[item['metric']] for x in bb])
        assert abs(d - item['paired_mean_difference']) < 1e-12
        lo, hi = item['company_seed_bootstrap_ci95']; assert lo <= hi
    out['paired_bootstrap'] = 'PASS'
    out['status'] = 'W2_SYNTHETIC_PASS'
    out['not_tested_here'] = 'CUDA/PyG training and private data; the server preflight runs one GNN/MLP step and a small RF before any counted fit.'
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
