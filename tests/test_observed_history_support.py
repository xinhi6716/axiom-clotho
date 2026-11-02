"""Statistical scope metadata on original invented observations; no fits."""
import json
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench.observed_history import evaluate_observed_history


class ObservedHistorySupportTests(unittest.TestCase):
    def inputs(self, two=False, extra=False):
        traces = ['a'] + (['b'] if two else []) + (['unscored'] if extra else [])
        observations = pd.DataFrame({
            'trace_id': [trace for trace in traces for _ in range(4)],
            'timestamp': [0, 1, 4, 9] * len(traces),
            'value': [2., 4., 6., 8.] * len(traces),
        })
        selected = ['a'] + (['b'] if two else [])
        origins = pd.DataFrame({
            'trace_id': [trace for trace in selected for _ in range(2)],
            'target_index': [2, 3] * len(selected),
        })
        return observations, origins, origins.assign(prediction=[5., 9.] * len(selected))

    def evaluate(self, two=False, extra=False):
        obs, origins, pred = self.inputs(two, extra)
        return evaluate_observed_history(obs, origins=origins, saved_predictions=pred)

    def check_scope(self, result, count):
        support = result.summary['evaluation_support']
        self.assertEqual(support['claim_scope'], 'descriptive_observed_history')
        self.assertEqual(support['statistical_interpretation'], 'descriptive_only')
        self.assertEqual(support['n_scored_traces'], count)
        self.assertEqual(support['counting_unit'], 'distinct trace_id in the scored origin roster')
        self.assertIsNone(support['source_group_count'])
        self.assertEqual(support['source_group_status'], 'not_declared')
        self.assertEqual(support['physical_independence'], 'not_established')
        self.assertIn('source_groups_not_declared', support['support_reasons'])
        self.assertIn('physical_independence_not_established', support['support_reasons'])
        return support

    def test_single_trace_many_origins_preserves_baselines(self):
        result = self.evaluate()
        support = self.check_scope(result, 1)
        self.assertIn('no_between_trace_replication', support['support_reasons'])
        self.assertEqual(result.summary['n_origins'], 2)
        for name, mae in [('saved_model', 1.), ('persistence', 2.), ('index_continuation', 0.)]:
            metric = result.summary['models'][name]['overall']
            self.assertEqual(metric['mae'], mae)
            self.assertEqual(metric['n_groups'], 1)
        self.assertEqual(result.predictions.persistence.tolist(), [4., 6.])
        self.assertEqual(result.predictions.index_continuation.tolist(), [6., 8.])

    def test_multiple_traces_do_not_upgrade_statistical_scope(self):
        support = self.check_scope(self.evaluate(two=True), 2)
        self.assertNotIn('no_between_trace_replication', support['support_reasons'])

    def test_unscored_trace_does_not_inflate_support(self):
        plain, extra = self.evaluate(), self.evaluate(extra=True)
        self.check_scope(extra, 1)
        self.assertEqual(plain.summary, extra.summary)
        pd.testing.assert_frame_equal(plain.predictions, extra.predictions)

    def test_no_mutation_no_fit_and_json_roundtrip(self):
        inputs = self.inputs(two=True, extra=True)
        originals = [frame.copy(deep=True) for frame in inputs]
        with patch('groupbench.model.GroupBenchRegressor.fit', side_effect=AssertionError('no fit')) as fit:
            result = evaluate_observed_history(inputs[0], origins=inputs[1], saved_predictions=inputs[2])
        fit.assert_not_called()
        for actual, original in zip(inputs, originals):
            pd.testing.assert_frame_equal(actual, original)
        encoded = json.dumps(result.summary, allow_nan=False, sort_keys=True)
        self.assertEqual(json.loads(encoded), result.summary)
        self.assertIn('"source_group_count": null', encoded)
        self.assertEqual(result.summary['fit_calls'], 0)
        self.assertFalse(result.summary['selection_performed'])


if __name__ == '__main__':
    unittest.main()
