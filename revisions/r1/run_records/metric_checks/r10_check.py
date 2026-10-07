#!/usr/bin/env python3
"""R@10FPR check on frozen test scores (read-only; no training, no prediction, no resampling).

Compares the frozen retained-ROC-node convention (sklearn roc_curve drop_intermediate=True, FPR <= 0.10 + 1e-12)
with the maximum recall over ALL attainable thresholds (drop_intermediate=False).
Mathematical fact used: the two can differ only if some score value is shared by at least one positive AND at least
one negative ("mixed tie block"). A file with zero mixed tie blocks is identical for every subset (D/O/S/J) and
every bootstrap reweighting, because subsetting or duplicating rows cannot create a mixed tie.

Usage:  python3 r10_check.py ROOT [ROOT ...]  > R10_CHECK_RESULT.json
ROOT = directories holding frozen test predictions (e.g. primary/V27/V28 run roots and the E7 kit stage_e directory).
Recognized exact basenames: predictions.parquet, test_predictions.parquet,
and PRIVATE_test_predictions.parquet.
Output is aggregate only (no company codes, labels or scores)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

LABELS = ['y_true', 'label', 'y']
SCORES = ['score', 'y_score', 'prob', 'proba', 'pred_score']

def r10(y, s, drop):
    f, t, _ = roc_curve(y, s, drop_intermediate=drop)
    ok = f <= 0.10 + 1e-12
    return float(t[ok].max()) if ok.any() else 0.0

out, skipped = [], []
for root in sys.argv[1:]:
    for p in sorted(Path(root).expanduser().rglob('*.parquet')):
        # Primary frozen runs use predictions.parquet; V27/V28 use
        # test_predictions.parquet; E7 uses PRIVATE_test_predictions.parquet.
        # Restrict to those exact frozen-output basenames.
        if p.name.lower() not in {
            'predictions.parquet',
            'test_predictions.parquet',
            'private_test_predictions.parquet',
        }:
            continue
        try:
            cols = pd.read_parquet(p).columns
            lc = next((c for c in LABELS if c in cols), None); sc = next((c for c in SCORES if c in cols), None)
            if lc is None or sc is None:
                skipped.append({'file': str(p), 'reason': 'no label/score column', 'columns': list(map(str, cols))[:20]}); continue
            d = pd.read_parquet(p, columns=[lc, sc]).dropna()
            y = d[lc].to_numpy(int); s = d[sc].to_numpy(float)
            if len(np.unique(y)) != 2:
                skipped.append({'file': str(p), 'reason': 'single class'}); continue
            g = pd.DataFrame({'y': y, 's': s}).groupby('s')['y'].agg(['min', 'max', 'size'])
            mixed = g[(g['min'] == 0) & (g['max'] == 1)]
            conv, full = r10(y, s, True), r10(y, s, False)
            out.append({'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'n': int(len(y)),
                        'n_pos': int(y.sum()), 'mixed_tie_blocks': int(len(mixed)),
                        'rows_in_mixed_tie_blocks': int(mixed['size'].sum()),
                        'r10_retained_node_convention': conv, 'r10_all_thresholds': full,
                        'difference': full - conv})
        except Exception as e:
            skipped.append({'file': str(p), 'reason': repr(e)[:200]})
summary = {'status': 'R10_CHECK_DONE', 'files_checked': len(out),
           'files_with_mixed_ties': sum(r['mixed_tie_blocks'] > 0 for r in out),
           'files_with_full_test_difference': sum(abs(r['difference']) > 1e-12 for r in out),
           'max_abs_difference': max([abs(r['difference']) for r in out], default=None),
           'training_or_inference_performed': False}
print(json.dumps({'summary': summary, 'files': out, 'skipped': skipped}, indent=2, ensure_ascii=False))
