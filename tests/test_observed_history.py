"""Original synthetic checks only; no estimator is fitted."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.observed_history import evaluate_observed_history


class ObservedHistoryTests(unittest.TestCase):
    def fixture(self):
        obs = pd.DataFrame({'trace_id': ['a'] * 5 + ['b'] * 3,
                            'timestamp': [0, 1, 4, 9, 11, 2, 3, 8],
                            'value': [1., 3., 5., 7., 9., 20., 20., 20.]})
        origins = pd.DataFrame({'trace_id': ['a', 'a', 'b'], 'target_index': [2, 3, 2]})
        pred = origins.copy()
        pred['prediction'] = [4., 8., 20.]
        return obs, origins, pred

    def run_case(self, obs=None, origins=None, pred=None):
        a, b, c = self.fixture()
        return evaluate_observed_history(a if obs is None else obs,
            origins=b if origins is None else origins,
            saved_predictions=c if pred is None else pred)

    def test_hand_oracle_irregular_time(self):
        r = self.run_case()
        self.assertEqual(r.predictions.persistence.tolist(), [3., 5., 20.])
        self.assertEqual(r.predictions.index_continuation.tolist(), [5., 7., 20.])
        self.assertEqual(r.predictions.previous_index.tolist(), [1, 2, 1])
        self.assertEqual(r.summary['n_origins'], 3)
        m = r.summary['models']
        self.assertEqual(m['saved_model']['overall']['mae'], 2 / 3)
        self.assertAlmostEqual(m['saved_model']['overall']['rmse'], np.sqrt(2 / 3))
        self.assertEqual(m['persistence']['overall']['mae'], 4 / 3)
        self.assertEqual(m['index_continuation']['overall']['mae'], 0.)
        self.assertEqual(m['persistence']['overall']['group_mae'], 1.)
        for entry in m.values():
            self.assertEqual(entry['overall']['n_rows'], 3)
        self.assertEqual(r.summary['fit_calls'], 0)

    def test_target_future_mutation_does_not_change_same_origin_baselines(self):
        obs, origins, pred = self.fixture()
        before = self.run_case()
        obs.loc[2:4, 'value'] = [50., -7., 999.]
        after = self.run_case(obs)
        for name in ['persistence', 'index_continuation']:
            self.assertEqual(before.predictions.loc[0, name], after.predictions.loc[0, name])
        # The previously scored target is legitimately observed for the next origin.
        self.assertEqual(after.predictions.loc[1, 'persistence'], 50.)
        self.assertEqual(after.predictions.loc[2, 'persistence'], 20.)

    def test_exact_roster_failures(self):
        obs, origins, pred = self.fixture()
        variants = [pred.iloc[:-1], pred.iloc[::-1], pd.concat([pred, pred.iloc[:1]]),
                    pred.assign(trace_id=['a', 'a', 'unknown'])]
        for bad in variants:
            with self.subTest(bad=bad.shape), self.assertRaises(ValueError):
                self.run_case(pred=bad)

    def test_origin_boundaries(self):
        obs, origins, pred = self.fixture()
        for values in [[1, 3, 2], [2, 5, 2], [2, 2, 2], [3, 2, 2], [True, 3, 2], [2., 3., 2.]]:
            bad = origins.copy(); bad['target_index'] = pd.Series(values, dtype=object)
            p = bad.copy(); p['prediction'] = pred.prediction
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.run_case(origins=bad, pred=p)

    def test_nonmonotone_nonfinite_and_lossy_times(self):
        obs, _, _ = self.fixture()
        for times in [[0, 1, 1, 9, 11, 2, 3, 8], [0, 2, 1, 9, 11, 2, 3, 8],
                      [0, 1, np.nan, 9, 11, 2, 3, 8], [0, 1, np.inf, 9, 11, 2, 3, 8],
                      [0, 1, 4, 9, 2**53+1, 2, 3, 8]]:
            bad = obs.copy(); bad['timestamp'] = times
            with self.subTest(times=times), self.assertRaises(ValueError): self.run_case(bad)

    def test_wide_numpy_floats_rejected_before_scoring(self):
        if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
            self.skipTest('No wider NumPy floating dtype on this platform')
        obs, origins, pred = self.fixture()
        for container in ('native', 'object'):
            for source, column, message in (
                    ('obs', 'timestamp', 'timestamps'),
                    ('obs', 'value', 'observations'),
                    ('pred', 'prediction', 'saved predictions')):
                a, p = obs.copy(), pred.copy()
                table = a if source == 'obs' else p
                # Preserve the longdouble scalars even in object-typed columns.
                wide = [np.longdouble(v) for v in table[column]]
                table[column] = pd.Series(wide, dtype=np.longdouble if container == 'native' else object)
                with self.subTest(container=container, column=column):
                    with self.assertRaisesRegex(ValueError, message + '.*wider than float64'):
                        self.run_case(a, origins, p)

    def test_wide_values_cannot_be_silently_scored_as_equal(self):
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            self.skipTest('No additional longdouble mantissa bits on this platform')
        one = np.longdouble(1)
        distinct = one + np.longdouble(2) ** -60
        self.assertNotEqual(one, distinct)
        self.assertEqual(float(one), float(distinct))
        obs = pd.DataFrame({'trace_id': ['a'] * 3, 'timestamp': [0, 1, 2],
                            'value': pd.Series([one, one, distinct], dtype=np.longdouble)})
        origins = pd.DataFrame({'trace_id': ['a'], 'target_index': [2]})
        pred = origins.assign(prediction=[1.0])
        with self.assertRaisesRegex(ValueError, 'observations.*wider than float64'):
            self.run_case(obs, origins, pred)

    def test_integer_precision_guard_for_each_numeric_input(self):
        obs, origins, pred = self.fixture()
        for source, column, message in (
                ('obs', 'timestamp', 'timestamps'),
                ('obs', 'value', 'observations'),
                ('pred', 'prediction', 'saved predictions')):
            for integer in (2**53 + 1, -(2**53 + 1), np.int64(-(2**53 + 1)),
                            np.uint64(2**63 + 1)):
                a, p = obs.copy(), pred.copy()
                table = a if source == 'obs' else p
                values = list(table[column])
                values[2] = integer
                table[column] = pd.Series(values, dtype=object)
                with self.subTest(column=column, integer=integer):
                    with self.assertRaisesRegex(ValueError, message + '.*integer not exactly representable'):
                        self.run_case(a, origins, p)
        # Large integers that really are exactly representable are accepted;
        # don't impose a blanket 2**53 upper bound.
        from groupbench.observed_history import _numbers
        np.testing.assert_array_equal(_numbers([2**53 + 2, np.uint64(2**63)], 'values'),
                                      [float(2**53 + 2), float(2**63)])
        np.testing.assert_array_equal(_numbers([np.float16(1.5), np.float32(2.5),
                                                 np.float64(3.5)], 'values'),
                                      [1.5, 2.5, 3.5])

    def test_bad_schema_values_and_empty(self):
        obs, origins, pred = self.fixture()
        for value in [np.nan, np.inf, True, '7', 1j]:
            bad = obs.astype({'value': object}); bad.loc[2, 'value'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.run_case(bad)
        for bad in [obs.iloc[:0], obs.assign(extra=0), obs[['value','timestamp','trace_id']],
                    obs.assign(trace_id=''), obs.assign(trace_id='a\x00')]:
            with self.assertRaises(ValueError): self.run_case(bad)
        with self.assertRaises(TypeError): self.run_case(obs=obs.to_dict())
        for bad in [pred.assign(prediction=np.inf), pred.assign(prediction='7')]:
            with self.assertRaises(ValueError): self.run_case(pred=bad)

    def test_numerical_failures(self):
        obs, _, pred = self.fixture()
        bad = obs.copy(); bad.loc[1, 'value'] = 1e308
        with self.assertRaises(ValueError): self.run_case(bad)
        bad = obs.copy(); bad['value'] = 0.
        p = pred.copy(); p['prediction'] = 1e-200
        with self.assertRaises(ValueError): self.run_case(bad, pred=p)

    def test_constant_targets_null_r2_and_no_mutation(self):
        obs, origins, pred = self.fixture()
        obs['value'] = 4.; pred['prediction'] = 4.
        originals = [x.copy(deep=True) for x in [obs, origins, pred]]
        result = self.run_case(obs, origins, pred)
        for entry in result.summary['models'].values():
            self.assertEqual(entry['overall']['mae'], 0.)
            self.assertIsNone(entry['overall']['r2'])
        for actual, expected in zip([obs, origins, pred], originals):
            pd.testing.assert_frame_equal(actual, expected)

    def test_metrics_reused_without_fit(self):
        from groupbench.metrics import grouped_metrics
        with patch('groupbench.observed_history.grouped_metrics', wraps=grouped_metrics) as metric:
            with patch('groupbench.model.GroupBenchRegressor.fit', side_effect=AssertionError('no fit')):
                self.run_case()
        self.assertEqual(metric.call_count, 3)

    def test_interleaved_traces_and_nondefault_dataframe_index(self):
        obs, origins, pred = self.fixture()
        obs = obs.iloc[[0, 5, 1, 6, 2, 7, 3, 4]]
        r = self.run_case(obs, origins.set_axis([9, 8, 7]), pred.set_axis([5, 4, 3]))
        self.assertEqual(r.predictions.index_continuation.tolist(), [5., 7., 20.])


if __name__ == '__main__':
    unittest.main()
