#!/usr/bin/env python3
"""W2 aggregate regression checks plus corruption rejection on isolated copies."""
from pathlib import Path
import json, shutil, sys, tempfile, unittest
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_w2 import verify


class W2Checks(unittest.TestCase):
    def test_actual_frozen_aggregates(self):
        r = verify(ROOT)
        self.assertEqual(r['fits'], 20); self.assertEqual(r['interval_endpoints_recomputed'], 48)

    def fixture(self):
        d = tempfile.TemporaryDirectory(prefix='w2_fixture_'); self.addCleanup(d.cleanup); t = Path(d.name)
        for rel in ['aggregates/w2', 'run_records/w2']: shutil.copytree(ROOT / rel, t / rel)
        return t

    def test_changed_group_mean_is_rejected(self):
        t = self.fixture(); p = t / 'aggregates/w2/GROUP_MEANS.csv'
        f = pd.read_csv(p); f.loc[0, 'roc_auc_mean'] += 0.01; f.to_csv(p, index=False)
        with self.assertRaisesRegex(AssertionError, 'W2_RECOMPUTE'): verify(t)

    def test_changed_bootstrap_array_is_rejected(self):
        t = self.fixture(); p = t / 'aggregates/w2/BOOTSTRAP_DIFFERENCES.npy'
        a = np.load(p); a[:, 0, 0] += 0.05; np.save(p, a)
        with self.assertRaisesRegex(AssertionError, 'W2_RECOMPUTE'): verify(t)

    def test_unlocked_run_is_rejected(self):
        t = self.fixture(); p = t / 'run_records/w2/ALL_20_LOCKED.json'
        d = json.loads(p.read_text()); d['test_performance_evaluated_before_lock'] = True; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(AssertionError, 'W2_LOCK'): verify(t)


if __name__ == '__main__':
    unittest.main(verbosity=2)
