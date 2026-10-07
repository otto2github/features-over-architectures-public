#!/usr/bin/env python3
"""W2 earlier-window comparison: 20 prescribed fits (RF/MLP/GCN/SAGE x 5 seeds); global lock before test inference.

Adapted from the executed E7 worker. The original training core is called verbatim with its year constants set to
the W2 window. Usage: w2_worker.py [--preflight]
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
from w2_runtime_common import KIT, require, safe_path, sha, verify_package, verify_records, write_json, zip_exact

BOUNDARY = Path('<AUTHOR_SERVER_ROOT>')
W_TRAIN, W_VAL, W_TEST = list(range(2010, 2017)), [2017, 2018], [2019, 2020]
RETURN_FILES = ['EXECUTION_CHECKS.json', 'RESULTS.json', 'PER_SEED_METRICS.csv', 'GROUP_MEANS.csv', 'PAIRED_CONTRASTS.json',
                'BOOTSTRAP_DIFFERENCES.npy', 'WINDOW_COUNTS.json', 'ALL_20_LOCKED.json', 'INPUT_IDENTITIES.json',
                'ENVIRONMENT.json', 'RUN_MATRIX.json', 'RUNNER_IDENTITIES.json', 'README_RETURN.txt']


# ----------------------------------------------------------------------------- environment
def configure_paths():
    safe_path(KIT, BOUNDARY)
    os.environ['W2_WRITE_BOUNDARY'] = str(BOUNDARY)
    for name, relative in [('TMPDIR', 'runtime/tmp'), ('XDG_CACHE_HOME', 'runtime/cache'), ('CUDA_CACHE_PATH', 'runtime/cache/cuda'),
                           ('TORCH_HOME', 'runtime/cache/torch'), ('TORCH_EXTENSIONS_DIR', 'runtime/cache/torch_extensions'),
                           ('TRITON_CACHE_DIR', 'runtime/cache/triton')]:
        path = safe_path(KIT / relative, BOUNDARY); path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ[name] = str(path)
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def status(state, **extra):
    write_json(KIT / 'JOB_STATUS.json', {'status': state, 'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                         'worker_pid': os.getpid(), **extra})
    print('JOB_STATUS ' + state + ' ' + json.dumps(extra, ensure_ascii=False), flush=True)


def strict_seed(seed):
    import numpy as np
    import torch
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_WORKSPACE_CONFIG_MISMATCH')
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
            'STRICT_DETERMINISM_MISMATCH')


def verify_environment(reference, operator_reference):
    import torch
    import torch_geometric.backend
    import torch_geometric.typing
    actual = {}
    for name, expected in reference['versions'].items():
        try: value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: value = None
        require(value == expected, 'ENVIRONMENT_VERSION_CHANGED:' + name)
        actual[name] = value
    require(sys.version == reference['python'] and sys.executable == reference['executable'], 'PYTHON_RUNTIME_CHANGED')
    require(torch.cuda.is_available(), 'CUDA_UNAVAILABLE_NO_FALLBACK')
    require(torch.cuda.get_device_name(0) == reference['gpu_name'] and
            torch.cuda.get_device_properties(0).total_memory == reference['gpu_total_memory'], 'CUDA_DEVICE_CHANGED')
    require(torch.version.cuda == reference['torch_cuda'] and torch.backends.cudnn.version() == reference['cudnn_version'],
            'CUDA_OR_CUDNN_CHANGED')
    require(torch.backends.cuda.matmul.allow_tf32 == reference['cuda_matmul_allow_tf32'] and
            torch.backends.cudnn.allow_tf32 == reference['cudnn_allow_tf32'], 'TF32_FLAGS_CHANGED')
    flags = {k: bool(getattr(torch_geometric.typing, k, False)) for k in reference['optional_backend_flags']}
    require(flags == reference['optional_backend_flags'], 'OPTIONAL_PYG_BACKEND_CHANGED')
    torch_geometric.backend.use_segment_matmul = False
    from torch_geometric.nn import GCNConv, SAGEConv
    for cls in [GCNConv, SAGEConv]:
        require(sha(Path(inspect.getsourcefile(cls))) == operator_reference['operator_source_identities'][cls.__name__]['sha256'],
                'GRAPH_OPERATOR_SOURCE_CHANGED:' + cls.__name__)
    strict_seed(42)
    return {**reference, 'versions': actual, 'strict_determinism': True, 'warn_only': False,
            'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'segment_matmul_effective': False}


# ----------------------------------------------------------------------------- inputs and window
def load_core(binding):
    path = KIT / 'formal_rerun_core_REFERENCE.py'
    require(sha(path) == binding['baseline_inputs'][-1]['sha256'], 'PACKAGED_ORIGINAL_CORE_CHANGED')
    spec = importlib.util.spec_from_file_location('w2_original_core', path)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    # Only the year constants change; every function reads these module globals at call time.
    core.TRAIN_YEARS, core.VAL_YEARS, core.TEST_YEARS = list(W_TRAIN), list(W_VAL), list(W_TEST)
    core.set_seed = strict_seed
    return core


def load_train_validation_df(data, binding):
    """Same join as core.load_joined, physically reading fiscal years 2010-2018 only (no W2 test labels)."""
    import pyarrow.parquet as pq
    years = W_TRAIN + W_VAL
    nf = pq.read_table(data / 'node_features_v1_1.parquet', filters=[('year', 'in', years)]).to_pandas()
    labels = pq.read_table(data / 'fraud_labels_v1_0.parquet', columns=['company_code', 'fiscal_year', 'split_v1', binding['label_column']],
                           filters=[('fiscal_year', 'in', years)]).to_pandas()
    nf['company_code'] = nf.firm_id.astype(str).str.zfill(6)
    labels = labels.rename(columns={binding['label_column']: 'target'})
    joined = nf.merge(labels, left_on=['company_code', 'year'], right_on=['company_code', 'fiscal_year'], how='left', validate='one_to_one')
    require(joined.year.isin(years).all() and len(joined) > 0, 'TRAIN_VALIDATION_CONTEXT_ROWS')
    return joined


def supervision_counts(df, years):
    sel = df[df.year.isin(years) & df.target.notna()]
    require(sel.target.isin([0, 1]).all(), 'NON_BINARY_TARGET')
    return int(len(sel)), int(sel.target.sum())


def check_counts(df, binding, parts):
    exp = binding['expected_supervision']
    for name, years in parts:
        n, pos = supervision_counts(df, years)
        require((n, pos) == (exp[name]['rows'], exp[name]['positive']), f'W2_SUPERVISION_MISMATCH:{name}:{n}/{pos}')


def scaler_hashes(core, df, columns):
    import numpy as np
    require(core.select_cols(df, 'M11') == columns, 'M11_COLUMN_ORDER_MISMATCH')
    mu, sd = core.fit_scaler(df, columns)
    require(np.isfinite(mu).all() and np.isfinite(sd).all(), 'NONFINITE_SCALER')
    return mu, sd, hashlib.sha256(mu.tobytes()).hexdigest(), hashlib.sha256(sd.tobytes()).hexdigest()


def reverse_cache(original, binding, years):
    """Reverse-message reference graph (Eq. 2): stored annual records followed by one reversed copy, same label."""
    import torch
    selected_years = list(years)
    def prepare(df, modal, data_dir):
        require(modal == 'M11', 'ONLY_M11_ALLOWED')
        cache, n, d = original(df, modal, data_dir)
        require(n == binding['expected_n_original'] and d == 129, 'ORIGINAL_CACHE_SHAPE')
        out = {}
        for year in selected_years:
            c = dict(cache[year])
            ei, et = c['edge_index'], c['edge_type']
            require(ei.shape[1] == binding['base_annual_edges'][str(year)], 'BASE_YEAR_EDGE_COUNT:' + str(year))
            c['edge_index'] = torch.cat((ei, ei.flip(0)), 1)
            c['edge_type'] = torch.cat((et, et), 0)
            require(c['company_x'].shape[1] == 129 and c['company_idx'].numel() > 0, 'COMPANY_FEATURES')
            out[year] = c
        return out, n, d
    return prepare


class Guard:
    """Wraps core.eval_graph during stage T: validation years only; keeps the validation history."""
    def __init__(self, original, auc, expected):
        self.original, self.auc, self.expected = original, auc, expected
        self.history = []; self.last = None
    def __call__(self, model, cache, years, n, d, device):
        import numpy as np
        require(list(years) == W_VAL, 'TEST_EVALUATION_FORBIDDEN_DURING_STAGE_T')
        y, s, keys = self.original(model, cache, years, n, d, device)
        require((len(y), int(np.sum(y))) == self.expected and np.isfinite(s).all(), 'VALIDATION_COUNTS_OR_VALUES')
        self.history.append(float(self.auc(y, s))); self.last = (y.copy(), s.copy(), keys.copy())
        return y, s, keys


class Capture:
    """Wraps core.make_model to retain the fitted model object (best state restored by the original loop)."""
    def __init__(self, original): self.original, self.model = original, None
    def __call__(self, *a, **k):
        self.model = self.original(*a, **k); return self.model


def load_inputs():
    verify_package()
    binding = json.loads((KIT / 'INPUT_BINDINGS.json').read_text())
    require(binding['training_authorized'] is True, 'TRAINING_AUTHORIZATION_MISSING')
    require(binding['window'] == {'train_years': W_TRAIN, 'validation_years': W_VAL, 'test_years': W_TEST}, 'WINDOW_BINDING')
    matrix = json.loads((KIT / 'RUN_MATRIX.json').read_text())
    expected = {(m, s) for m in ['RF', 'MLP', 'GCN', 'SAGE'] for s in [42, 123, 456, 789, 1024]}
    require(len(matrix) == 20 and {(r['model'], r['seed']) for r in matrix} == expected and len({r['run_id'] for r in matrix}) == 20,
            'RUN_MATRIX_SCOPE')
    verify_records(binding['baseline_inputs'])
    environment = verify_environment(json.loads((KIT / 'ENVIRONMENT_REFERENCE.json').read_text()),
                                     json.loads((KIT / 'OPERATOR_CHECKS_REFERENCE.json').read_text()))
    core = load_core(binding)
    data = Path(binding['baseline_root']) / 'data'
    train = load_train_validation_df(data, binding)
    check_counts(train, binding, [('train', W_TRAIN), ('validation', W_VAL)])
    require(core.dynamic_pos_weight(train)[1:] == (binding['expected_supervision']['train']['negative'],
                                                    binding['expected_supervision']['train']['positive']), 'W2_POS_WEIGHT')
    columns = json.loads((KIT / 'M11_COLUMN_ORDER.json').read_text())['columns']
    _, _, mh, sh = scaler_hashes(core, train, columns)
    context = {'binding_file_sha256': sha(KIT / 'INPUT_BINDINGS.json'), 'protocol_sha256': sha(KIT / 'W2_PROTOCOL.md'),
               'matrix_sha256': sha(KIT / 'RUN_MATRIX.json'), 'package_manifest_sha256': sha(KIT / 'PACKAGE_SHA256.json'),
               'source_ledger': binding['baseline_inputs'], 'scaler_mean_sha256': mh, 'scaler_std_sha256': sh}
    frozen = KIT / 'FROZEN_RUNTIME_BINDING.json'
    if frozen.exists(): require(json.loads(frozen.read_text()) == context, 'RESUME_RUNTIME_BINDING_CHANGED')
    else: write_json(frozen, context)
    return binding, matrix, core, train, columns, context, environment


# ----------------------------------------------------------------------------- stage T
def file_ledger(folder, names):
    return {n: {'sha256': sha(folder / n), 'size_bytes': (folder / n).stat().st_size} for n in names}


def verify_completed(folder, context, stage):
    require(folder.is_dir() and not folder.is_symlink(), 'STAGE_OUTPUT_PATH_REJECTED')
    done = json.loads((folder / (stage + '_COMPLETE.json')).read_text())
    require(done['status'] == 'LOCKED' and done['context'] == context, 'COMPLETED_STAGE_BINDING_CHANGED')
    for name, rec in done['artifacts'].items():
        p = safe_path(folder / name, BOUNDARY)
        require(p.is_file() and p.stat().st_size == rec['size_bytes'] and sha(p) == rec['sha256'], 'LOCKED_ARTIFACT_CHANGED:' + name)
    return done


def rf_model(binding, seed):
    from sklearn.ensemble import RandomForestClassifier
    r = binding['rf_settings']
    return RandomForestClassifier(n_estimators=r['n_estimators'], criterion=r['criterion'], max_depth=r['max_depth'],
                                  min_samples_split=r['min_samples_split'], min_samples_leaf=r['min_samples_leaf'],
                                  max_features=r['max_features'], bootstrap=r['bootstrap'], class_weight=r['class_weight'],
                                  n_jobs=r['n_jobs'], random_state=seed)


def fit_one(row, core, train, columns, binding, context, tmp):
    import numpy as np
    exp_val = (binding['expected_supervision']['validation']['rows'], binding['expected_supervision']['validation']['positive'])
    data = Path(binding['baseline_root']) / 'data'
    ns = binding['neural_settings']
    args = SimpleNamespace(device='cuda', hidden_dim=ns['hidden_dim'], lr=ns['lr'], epochs=ns['epochs'],
                           patience=ns['patience'], no_test=True)
    if row['model'] == 'RF':
        import joblib
        from sklearn.metrics import roc_auc_score
        X = train[columns].fillna(0).to_numpy(np.float32)
        y = train.target.fillna(0).astype(int).to_numpy(); el = train.target.notna().to_numpy(); yr = train.year.to_numpy()
        tr = np.where(el & np.isin(yr, W_TRAIN))[0]; va = np.where(el & np.isin(yr, W_VAL))[0]
        require((len(va), int(y[va].sum())) == exp_val, 'RF_VALIDATION_COUNTS')
        random.seed(row['seed']); np.random.seed(row['seed'])
        model = rf_model(binding, row['seed']); model.fit(X[tr], y[tr])
        vs = model.predict_proba(X[va])[:, 1]
        thr = core.f1_val_threshold(y[va], vs)
        joblib.dump(model, tmp / 'model.joblib', compress=3)
        result = {'model': 'RF', 'seed': row['seed'], 'best_val_auc': float(roc_auc_score(y[va], vs)), 'val_threshold': float(thr),
                  'train_rows': int(len(tr)), 'train_pos': int(y[tr].sum()), 'test_evaluated': False, 'rf_settings': binding['rf_settings']}
        return result, ['model.joblib']
    import torch
    original_make, original_eval = core.make_model, core.eval_graph
    cap = Capture(original_make); guard = Guard(original_eval, core.roc_auc_score, exp_val)
    core.make_model = cap
    try:
        if row['model'] == 'MLP':
            result = core.run_mlp(train, 'M11', row['seed'], args, tmp)
            mu, sd = core.fit_scaler(train, columns)
            X = core.transformed(train, columns, mu, sd)
            y = train.target.fillna(0).astype(int).to_numpy(); el = train.target.notna().to_numpy(); yr = train.year.to_numpy()
            va = np.where(el & np.isin(yr, W_VAL))[0]
            cap.model.eval()
            with torch.no_grad(): vs = torch.sigmoid(cap.model(torch.from_numpy(X[va]).cuda())).cpu().numpy()
            require((len(va), int(y[va].sum())) == exp_val, 'MLP_VALIDATION_COUNTS')
            require(core.f1_val_threshold(y[va], vs) == result['val_threshold'], 'MLP_VALIDATION_THRESHOLD_REPLAY')
            require(abs(float(core.roc_auc_score(y[va], vs)) - result['best_val_auc']) <= 1e-9, 'MLP_BEST_CHECKPOINT_REPLAY')
        else:
            original_cache = core.prepare_graph_cache
            core.prepare_graph_cache = reverse_cache(original_cache, binding, W_TRAIN + W_VAL)
            core.eval_graph = guard
            try:
                result = core.run_gnn(train, row['model'], 'M11', row['seed'], args, tmp, data)
            finally:
                core.prepare_graph_cache = original_cache
            require(abs(guard.history[-1] - result['best_val_auc']) <= 1e-12, 'BEST_CHECKPOINT_VALIDATION_REPLAY')
            require(max(guard.history[:-1]) <= result['best_val_auc'] + 1e-8, 'CHECKPOINT_SELECTION_MISMATCH')
            y_v, s_v, _ = guard.last
            require(core.f1_val_threshold(y_v, s_v) == result['val_threshold'], 'VALIDATION_THRESHOLD_MISMATCH')
            write_json(tmp / 'validation_history.json', {'epoch_validation_auc': guard.history[:-1],
                                                         'restored_best_checkpoint_validation_auc': guard.history[-1]})
        require(result['test_evaluated'] is False, 'TRAINING_STAGE_TEST_READ')
        state = {k: v.detach().cpu().clone() for k, v in cap.model.state_dict().items()}
        torch.save(state, tmp / 'checkpoint.pt')
        names = ['checkpoint.pt'] + (['validation_history.json'] if row['model'] != 'MLP' else [])
        return result, names
    finally:
        core.make_model, core.eval_graph = original_make, original_eval
        cap.model = None
        torch.cuda.empty_cache()


def run_fit(row, core, train, columns, binding, context):
    out = safe_path(KIT / 'stage_t' / row['run_id'], BOUNDARY)
    if out.exists():
        verify_completed(out, context, 'STAGE_T'); print('REUSE_LOCKED_FIT ' + row['run_id'], flush=True); return
    fdir = KIT / 'stage_t_failures'
    require(not (fdir.exists() and list(fdir.glob(row['run_id'] + '__*'))), 'FAILED_FIT_REQUIRES_TECHNICAL_REVIEW:' + row['run_id'])
    require(not ((KIT / 'stage_t').exists() and list((KIT / 'stage_t').glob('.attempt_' + row['run_id'] + '_*'))),
            'INTERRUPTED_FIT_REQUIRES_TECHNICAL_REVIEW:' + row['run_id'])
    tmp = safe_path(out.with_name('.attempt_' + row['run_id'] + '_' + str(os.getpid())), BOUNDARY)
    tmp.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    try:
        strict_seed(row['seed'])
        result, names = fit_one(row, core, train, columns, binding, context, tmp)
        result.update({**row, 'strict_determinism': True, 'warn_only': False, 'runtime_seconds': time.monotonic() - started,
                       'original_training_loop_sha256': sha(KIT / 'formal_rerun_core_REFERENCE.py'),
                       'window': {'train': W_TRAIN, 'validation': W_VAL, 'test': W_TEST}})
        write_json(tmp / 'training_result.json', result)
        write_json(tmp / 'STAGE_T_COMPLETE.json', {'status': 'LOCKED', 'run_id': row['run_id'], 'context': context,
                                                   'artifacts': file_ledger(tmp, names + ['training_result.json']), 'test_evaluated': False})
        os.replace(tmp, out)
        print(f'FIT_LOCKED {row["run_id"]} val_auc={result["best_val_auc"]:.6f} seconds={result["runtime_seconds"]:.1f}', flush=True)
    except Exception as exc:
        write_json(tmp / 'FAILURE.json', {'run_id': row['run_id'], 'type': type(exc).__name__, 'message': str(exc), 'test_evaluated': False})
        failed = safe_path(KIT / 'stage_t_failures' / (row['run_id'] + '__' + str(int(time.time())) + '_' + str(os.getpid())), BOUNDARY)
        failed.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(tmp, failed)
        raise


def lock_all(matrix, context):
    locks = {}
    for row in matrix:
        folder = KIT / 'stage_t' / row['run_id']; verify_completed(folder, context, 'STAGE_T')
        locks[row['run_id']] = {'complete_sha256': sha(folder / 'STAGE_T_COMPLETE.json')}
    value = {'status': 'ALL_20_LOCKED', 'fit_count': 20, 'context': context, 'runs': locks, 'test_performance_evaluated_before_lock': False}
    path = KIT / 'ALL_20_LOCKED.json'
    if path.exists(): require(json.loads(path.read_text()) == value, 'GLOBAL_CHECKPOINT_LOCK_CHANGED')
    else: write_json(path, value)
    return value


def verify_all_lock(matrix, context):
    require((KIT / 'ALL_20_LOCKED.json').is_file(), 'ALL_20_LOCK_REQUIRED_BEFORE_TEST')
    lock = json.loads((KIT / 'ALL_20_LOCKED.json').read_text())
    require(lock['status'] == 'ALL_20_LOCKED' and lock['fit_count'] == 20 and lock['context'] == context, 'GLOBAL_LOCK_INVALID')
    require(set(lock['runs']) == {r['run_id'] for r in matrix}, 'GLOBAL_LOCK_RUN_SET_CHANGED')
    for row in matrix:
        p = KIT / 'stage_t' / row['run_id']; verify_completed(p, context, 'STAGE_T')
        require(sha(p / 'STAGE_T_COMPLETE.json') == lock['runs'][row['run_id']]['complete_sha256'], 'GLOBAL_LOCK_MEMBER_CHANGED')


# ----------------------------------------------------------------------------- stage E
def test_scores(row, core, full, columns, binding, test_cache):
    import numpy as np
    tout = KIT / 'stage_t' / row['run_id']
    el = full.target.notna().to_numpy(); yr = full.year.to_numpy()
    te = np.where(el & np.isin(yr, W_TEST))[0]
    keys = full.iloc[te][['company_code', 'year']].rename(columns={'year': 'fiscal_year'}).reset_index(drop=True)
    y = full.iloc[te].target.astype(int).to_numpy()
    if row['model'] == 'RF':
        import joblib
        model = joblib.load(tout / 'model.joblib')
        s = model.predict_proba(full[columns].fillna(0).to_numpy(np.float32)[te])[:, 1]
        return keys, y, s
    import torch
    strict_seed(row['seed'])
    state = torch.load(tout / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if row['model'] == 'MLP':
        model = core.make_model('MLP', 129, binding['neural_settings']['hidden_dim']); model.load_state_dict(state); model.cuda().eval()
        mu, sd = core.fit_scaler(full, columns)
        X = core.transformed(full, columns, mu, sd)
        with torch.no_grad(): s = torch.sigmoid(model(torch.from_numpy(X[te]).cuda())).cpu().numpy()
        return keys, y, s
    model = core.make_model(row['model'], 129, binding['neural_settings']['hidden_dim'], 5); model.load_state_dict(state); model.cuda().eval()
    cache, n, d = test_cache
    yy, s, k = core.eval_graph(model, cache, W_TEST, n, d, torch.device('cuda'))
    k = k.rename(columns={'year': 'fiscal_year'}).reset_index(drop=True)
    import pandas as pd
    a = pd.DataFrame({'company_code': k.company_code, 'fiscal_year': k.fiscal_year, 'y': yy, 's': s})
    b = keys.merge(a, on=['company_code', 'fiscal_year'], how='left', validate='one_to_one')
    require(b.s.notna().all() and (b.y.to_numpy() == y).all(), 'GNN_TEST_KEY_ALIGNMENT')
    return keys, y, b.s.to_numpy()


def run_evaluation(row, core, full, columns, binding, context, test_cache, matrix):
    verify_all_lock(matrix, context)
    import numpy as np
    from w2_statistics import point_metrics
    out = safe_path(KIT / 'stage_e' / row['run_id'], BOUNDARY)
    if out.exists(): verify_completed(out, context, 'STAGE_E'); return
    tmp = safe_path(out.with_name('.attempt_' + row['run_id'] + '_' + str(os.getpid())), BOUNDARY)
    require(not tmp.exists(), 'EVALUATION_ATTEMPT_EXISTS'); tmp.mkdir(parents=True, mode=0o700)
    training = json.loads((KIT / 'stage_t' / row['run_id'] / 'training_result.json').read_text())
    keys, y, s = test_scores(row, core, full, columns, binding, test_cache)
    exp = binding['expected_supervision']['test']
    require((len(y), int(y.sum())) == (exp['rows'], exp['positive']) and np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(),
            'TEST_SCORE_OR_LABEL_COUNTS_INVALID')
    pred = keys.copy(); pred['y_true'] = y; pred['score'] = s
    pred = pred.sort_values(['company_code', 'fiscal_year'], kind='stable').reset_index(drop=True)
    metrics = point_metrics(pred.y_true.to_numpy(), pred.score.to_numpy(), float(training['val_threshold']))
    pred.to_parquet(tmp / 'PRIVATE_test_predictions.parquet', index=False)
    write_json(tmp / 'evaluation_result.json', {**row, 'validation_threshold': float(training['val_threshold']), 'metrics': metrics,
                                                'test_evaluated_after_all_20_locked': True})
    write_json(tmp / 'STAGE_E_COMPLETE.json', {'status': 'LOCKED', 'context': context, 'run_id': row['run_id'],
                                               'artifacts': file_ledger(tmp, ['PRIVATE_test_predictions.parquet', 'evaluation_result.json'])})
    os.replace(tmp, out)
    print(f'TEST_EVALUATED {row["run_id"]} AUC={metrics["roc_auc"]:.6f} hits={metrics["hits_at_budget"]}', flush=True)


def analyze_and_return(matrix, binding, context, environment, counts, started):
    import numpy as np
    import pandas as pd
    from w2_statistics import summarize
    rows, payload, keys = [], {}, None
    for row in matrix:
        out = KIT / 'stage_e' / row['run_id']; verify_completed(out, context, 'STAGE_E')
        res = json.loads((out / 'evaluation_result.json').read_text())
        rows.append({**row, **res['metrics'], 'validation_threshold': res['validation_threshold']})
        pred = pd.read_parquet(out / 'PRIVATE_test_predictions.parquet')
        k = pred[['company_code', 'fiscal_year', 'y_true']]
        if keys is None: keys = k
        else: require(k.equals(keys), 'PAIRED_TEST_KEY_OR_LABEL_ALIGNMENT')
        payload[(row['model'], row['seed'])] = (pred.y_true.to_numpy(int), pred.score.to_numpy(float), res['validation_threshold'])
    summary, draws = summarize(rows, payload, keys.company_code.to_numpy(), binding['bootstrap'])
    ret = safe_path(KIT / 'return', BOUNDARY); ret.mkdir(exist_ok=True, mode=0o700)
    pd.DataFrame(rows).to_csv(ret / 'PER_SEED_METRICS.csv', index=False)
    pd.DataFrame(summary['group_means']).to_csv(ret / 'GROUP_MEANS.csv', index=False)
    write_json(ret / 'PAIRED_CONTRASTS.json', {'values': summary['paired_contrasts'], 'bootstrap': summary['bootstrap']})
    np.save(ret / 'BOOTSTRAP_DIFFERENCES.npy', draws)
    counts = {**counts, 'test_companies': summary['bootstrap']['company_clusters']}
    write_json(ret / 'WINDOW_COUNTS.json', counts)
    trained = [json.loads((KIT / 'stage_t' / r['run_id'] / 'training_result.json').read_text()) for r in matrix]
    for t in trained: t.pop('rf_settings', None)
    write_json(ret / 'RESULTS.json', {'status': 'W2_20_FITS_AND_PAIRED_ANALYSIS_COMPLETE', 'fit_count': 20, 'window': binding['window'],
                                      'group_means': summary['group_means'], 'paired_contrasts': summary['paired_contrasts'],
                                      'bootstrap': summary['bootstrap'], 'training_summaries': trained,
                                      'performance_used_to_select_models_or_seeds': False, 'runtime_seconds': time.monotonic() - started})
    write_json(ret / 'EXECUTION_CHECKS.json', {'status': 'PASS', 'all_fits': 20, 'validation_only_checkpoint_and_threshold_selection': True,
               'global_checkpoint_lock_before_any_test_performance': True, 'stage_t_read_fiscal_years': W_TRAIN + W_VAL,
               'strict_determinism': True, 'warn_only': False, 'M11_dimensions': 129, 'graph': 'reverse-message reference, original 303,775 nodes',
               'source_rechecked_after_runs': True, 'raw_keys_labels_scores_checkpoints_in_return': False, 'context': context})
    write_json(ret / 'ALL_20_LOCKED.json', json.loads((KIT / 'ALL_20_LOCKED.json').read_text()))
    write_json(ret / 'INPUT_IDENTITIES.json', context['source_ledger'])
    write_json(ret / 'ENVIRONMENT.json', environment)
    write_json(ret / 'RUN_MATRIX.json', matrix)
    manifest = json.loads((KIT / 'PACKAGE_SHA256.json').read_text())
    write_json(ret / 'RUNNER_IDENTITIES.json', {n: {'sha256': sha(KIT / n), 'size_bytes': (KIT / n).stat().st_size}
                                               for n in [*manifest['files'], 'PACKAGE_SHA256.json']})
    (ret / 'README_RETURN.txt').write_text(
        'Aggregate return of the prescribed 20 W2 fits (earlier window: train 2010-2016, validation 2017-2018, test 2019-2020).\n'
        'All checkpoints and validation thresholds were locked before any W2 test evaluation.\n'
        'The 2019-2020 rows were the validation partition of the primary analysis; this must be disclosed with any use of W2.\n'
        'No company keys, labels, scores or model weights are included.\n', encoding='utf-8')
    verify_package(); verify_records(context['source_ledger']); verify_all_lock(matrix, context)
    pending = safe_path(KIT / ('W2_TRAIN_RETURN.pending.' + str(os.getpid()) + '.zip'), BOUNDARY)
    zip_exact(pending, ret, RETURN_FILES)
    return pending


# ----------------------------------------------------------------------------- preflight
def preflight(binding, core, train, columns):
    """Before any counted fit: one optimizer step for MLP/GCN/SAGE and a 5-tree RF on W2 training data, in a scratch dir."""
    import numpy as np
    import torch
    import torch.nn as nn
    data = Path(binding['baseline_root']) / 'data'
    strict_seed(42)
    cache, n, d = reverse_cache(core.prepare_graph_cache, binding, [W_TRAIN[0], W_VAL[0]])(train, 'M11', data)
    posw = core.dynamic_pos_weight(train)[0]
    for name in ['GCN', 'SAGE']:
        m = core.make_model(name, 129, 64, 5).cuda(); opt = torch.optim.Adam(m.parameters(), lr=5e-4)
        c = cache[W_TRAIN[0]]
        x = torch.zeros((n, d), device='cuda'); x[c['company_idx'].cuda()] = c['company_x'].cuda()
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw, device='cuda'))(m(x, c['edge_index'].cuda(), c['edge_type'].cuda())[c['sup_idx'].cuda()], c['sup_y'].cuda())
        loss.backward(); opt.step()
        require(torch.isfinite(loss).item(), 'PREFLIGHT_GNN_LOSS:' + name)
    mu, sd = core.fit_scaler(train, columns); X = core.transformed(train, columns, mu, sd)
    m = core.make_model('MLP', 129, 64).cuda(); out = m(torch.from_numpy(X[:256]).cuda()); require(torch.isfinite(out).all().item(), 'PREFLIGHT_MLP')
    rf = rf_model(binding, 42); rf.set_params(n_estimators=5)
    el = train.target.notna().to_numpy(); tr = np.where(el & train.year.isin(W_TRAIN).to_numpy())[0]
    rf.fit(train[columns].fillna(0).to_numpy(np.float32)[tr], train.target.to_numpy()[tr].astype(int))
    torch.cuda.empty_cache()
    return {'status': 'PREFLIGHT_PASS', 'gnn_one_step': ['GCN', 'SAGE'], 'mlp_forward': True, 'rf_small_fit': True,
            'test_years_read': False, 'counted_fits_started': False}


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    preflight_only = parser.parse_args().preflight
    configure_paths()
    with safe_path(KIT / 'WORKER.lock', BOUNDARY).open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('ANOTHER_W2_WORKER_IS_RUNNING')
        started = time.monotonic()
        try:
            status('CHECKING_INPUT_BINDINGS', fits_locked=0, total_fits=20)
            binding, matrix, core, train, columns, context, environment = load_inputs()
            if (KIT / 'W2_TRAIN_RETURN.zip').exists():
                verify_all_lock(matrix, context)
                done = json.loads((KIT / 'RETURN_COMPLETE.json').read_text())
                require(done['context'] == context and done['sha256'] == sha(KIT / 'W2_TRAIN_RETURN.zip'), 'COMPLETED_RETURN_CHANGED')
                status('COMPLETE', fits_locked=20, return_zip=str(KIT / 'W2_TRAIN_RETURN.zip')); return
            if not (KIT / 'PREFLIGHT_PASS.json').exists():
                status('PREFLIGHT', fits_locked=0)
                write_json(KIT / 'PREFLIGHT_PASS.json', {**preflight(binding, core, train, columns), 'context': context})
            if preflight_only:
                status('PREFLIGHT_PASS', fits_locked=0); return
            for i, row in enumerate(matrix):
                status('TRAINING', current_run=row['run_id'], fits_locked=i, total_fits=20)
                run_fit(row, core, train, columns, binding, context)
            verify_records(context['source_ledger']); verify_package()
            lock_all(matrix, context)
            status('ALL_20_CHECKPOINTS_LOCKED', fits_locked=20, test_evaluated=False)
            verify_all_lock(matrix, context)
            # First read of W2 test labels in this run.
            full = core.load_joined(Path(binding['baseline_root']) / 'data', binding['label_column'])
            check_counts(full, binding, [('train', W_TRAIN), ('validation', W_VAL), ('test', W_TEST)])
            _, _, mh, sh = scaler_hashes(core, full, columns)
            require((mh, sh) == (context['scaler_mean_sha256'], context['scaler_std_sha256']), 'SCALER_CHANGED_AFTER_LOCK')
            test_cache = reverse_cache(core.prepare_graph_cache, binding, W_TEST)(full, 'M11', Path(binding['baseline_root']) / 'data')
            for row in matrix:
                status('TEST_EVALUATION', current_run=row['run_id'], fits_locked=20)
                run_evaluation(row, core, full, columns, binding, context, test_cache, matrix)
            exp = binding['expected_supervision']
            counts = {k: {'rows': v['rows'], 'positive': v['positive']} for k, v in exp.items()}
            status('PAIRED_BOOTSTRAP', fits_locked=20, test_fits_evaluated=20)
            pending = analyze_and_return(matrix, binding, context, environment, counts, started)
            verify_records(context['source_ledger']); verify_package(); verify_all_lock(matrix, context)
            target = safe_path(KIT / 'W2_TRAIN_RETURN.zip', BOUNDARY)
            require(not target.exists(), 'COMPLETED_RETURN_ZIP_ALREADY_EXISTS')
            os.replace(pending, target)
            write_json(KIT / 'RETURN_COMPLETE.json', {'status': 'COMPLETE', 'context': context, 'sha256': sha(target)})
            status('COMPLETE', fits_locked=20, test_fits_evaluated=20, return_zip=str(target))
        except Exception as exc:
            status('FAILED', exception_type=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    main()
