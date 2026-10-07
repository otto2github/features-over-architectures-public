#!/usr/bin/env python3
"""Real aggregate regression checks plus corruption rejection on isolated copies."""
from pathlib import Path
import json, shutil, sys, tempfile, unittest
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify_e7 import verify

class E7Checks(unittest.TestCase):
    def test_actual_frozen_aggregates(self):
        self.assertEqual(verify(ROOT)['new_fits'],20)

    def fixture(self):
        directory=tempfile.TemporaryDirectory(prefix='e7_aggregate_fixture_')
        self.addCleanup(directory.cleanup);target=Path(directory.name)
        for relative in ['aggregates/e7','run_records/e7','protocol/matrices','config/e7','provenance']:
            shutil.copytree(ROOT/relative,target/relative)
        return target

    def test_executed_weighted_metric_function(self):
        sys.path.insert(0,str(ROOT/'code/e7/training'))
        from e7_weighted_metrics import self_test
        result=self_test(n=120)
        self.assertTrue(result['passed'])
        self.assertLess(result['max_absolute_error'],2e-12)

    def test_changed_group_mean_is_rejected(self):
        target=self.fixture();path=target/'aggregates/e7/GROUP_MEANS.csv'
        frame=pd.read_csv(path);frame.loc[0,'roc_auc_mean']+=0.01;frame.to_csv(path,index=False)
        with self.assertRaisesRegex(AssertionError,'E7_GROUP_MEAN'):verify(target)

    def test_missing_prescribed_fit_is_rejected(self):
        target=self.fixture();path=target/'protocol/matrices/e7_run_matrix.json'
        data=json.loads(path.read_text());data.pop();path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AssertionError,'E7_MATRIX'):verify(target)

    def test_test_before_global_lock_is_rejected(self):
        target=self.fixture();path=target/'run_records/e7/ALL_20_LOCKED.json'
        data=json.loads(path.read_text());data['test_performance_evaluated_before_lock']=True;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AssertionError,'E7_GLOBAL_LOCK'):verify(target)

    def test_reversed_interval_is_rejected(self):
        target=self.fixture();a=target/'aggregates/e7/PAIRED_CONTRASTS.json';b=target/'aggregates/e7/RESULTS.json'
        contrast=json.loads(a.read_text());contrast['values'][0]['paired_company_seed_bootstrap_ci95']=[0.2,-0.2]
        result=json.loads(b.read_text());result['paired_contrasts']=contrast['values']
        a.write_text(json.dumps(contrast));b.write_text(json.dumps(result))
        with self.assertRaisesRegex(AssertionError,'E7_RECORDED_CI_CONTRACT'):verify(target)

    def test_budget_perturbation_is_rejected(self):
        target=self.fixture();path=target/'aggregates/e7/PER_SEED_METRICS.csv'
        frame=pd.read_csv(path);frame.loc[0,'budget']=421;frame.to_csv(path,index=False)
        with self.assertRaisesRegex(AssertionError,'E7_COUNT_BUDGET_CONTRACT'):verify(target)

if __name__=='__main__':unittest.main(verbosity=2)
