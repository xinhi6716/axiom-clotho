"""Bounded benchmark controls and exact comparison, without fitting models."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/grouped_metric_positions.py'
spec = importlib.util.spec_from_file_location('grouped_metric_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GroupedMetricBenchmarkTests(unittest.TestCase):
    def test_invalid_case_and_repetitions_reject_before_fitting_or_planning(self):
        for case, repeats in [('missing', 1), ('few_groups', 0), ('few_groups', 22),
                              ('few_groups', True), ('few_groups', 1.5)]:
            with self.subTest(case=case, repeats=repeats), \
                 patch.object(benchmark, 'plan_prediction', side_effect=AssertionError('planning')), \
                 patch.object(benchmark, 'fit_predict', side_effect=AssertionError('fitting')):
                with self.assertRaises(ValueError):
                    benchmark.run_case(case, repeats)

    def test_exact_comparison_drops_only_creation_and_elapsed_fields(self):
        result = SimpleNamespace(
            manifest={'created_at_utc': 'one', 'elapsed_seconds': 1.,
                      'data_fingerprint': 'kept', 'package_source': {'digest': 'kept'}},
            predictions=pd.DataFrame({'row_position': [0], 'prediction': [-0.]}),
            estimator=SimpleNamespace(selection_results_=[{'score': 1.}], inner_audit_=[{'rows': 2}]))
        original = benchmark.stable_result(result)
        changed = deepcopy(result)
        changed.manifest.update(created_at_utc='two', elapsed_seconds=2.)
        self.assertEqual(benchmark.stable_result(changed), original)
        changed.manifest['data_fingerprint'] = 'different'
        self.assertNotEqual(benchmark.stable_result(changed), original)
        changed = deepcopy(result)
        changed.manifest['package_source']['digest'] = 'different'
        self.assertNotEqual(benchmark.stable_result(changed), original)
        changed = deepcopy(result)
        changed.predictions['prediction'] = [0.]
        self.assertNotEqual(benchmark.stable_result(changed), original)
        self.assertEqual(result.manifest['created_at_utc'], 'one')
        self.assertIn('data_fingerprint', json.loads(original)['manifest'])

    def test_mismatches_abort_instead_of_producing_timing_claims(self):
        for samples, calls in [([(0.1, 'first'), (0.2, 'changed')], 2),
                               ([(0.1, 'same'), (0.2, 'same'), (0.3, 'changed')], 3)]:
            with self.subTest(calls=calls), \
                 patch.object(benchmark, 'plan_prediction', return_value={'fit_budget': {'total_fits': 14}}), \
                 patch.object(benchmark, 'timed_call', side_effect=samples) as timed:
                with self.assertRaisesRegex(RuntimeError, 'results differ'):
                    benchmark.run_case('few_groups', 1)
                self.assertEqual(timed.call_count, calls)


if __name__ == '__main__':
    unittest.main()
