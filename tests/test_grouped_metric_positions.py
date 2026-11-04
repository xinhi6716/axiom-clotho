"""No-fit parity with the prior mask-per-group algorithm, not timing gates."""
import json
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd

from groupbench.data import validate_groups
from groupbench.metrics import (
    _checked_arithmetic, _metric_inputs, grouped_metrics, regression_metrics,
)


def mask_reference(y_true, y_pred, groups):
    """Frozen pre-optimization grouping algorithm; keep arithmetic unchanged."""
    y, pred = _metric_inputs(y_true, y_pred)
    g = validate_groups(groups, len(y))
    per_group = []
    for group in pd.unique(g):
        mask = g == group
        per_group.append({"group": str(group), **regression_metrics(y[mask], pred[mask])})
    overall = regression_metrics(y, pred)
    with _checked_arithmetic():
        overall.update({
            "n_groups": len(per_group),
            "group_mae": float(np.mean([row["mae"] for row in per_group])),
            "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
            "worst_group_mae": float(max(row["mae"] for row in per_group)),
        })
    return overall, per_group


def outcome(scorer, *args):
    try:
        # Serialization also distinguishes signed zero and preserves list order.
        return "ok", json.dumps(scorer(*args), ensure_ascii=True, allow_nan=False)
    except Exception as error:
        return type(error), str(error)


class GroupedMetricPositionTests(unittest.TestCase):
    def assertParity(self, *args):
        self.assertEqual(outcome(grouped_metrics, *args), outcome(mask_reference, *args))

    def test_interleaved_unequal_singleton_and_single_group(self):
        rng = np.random.default_rng(17)
        for n in (1, 2, 3, 9, 31, 97):
            for count in sorted({1, max(1, n // 3), n}):
                y, pred = rng.normal(size=n), rng.normal(size=n)
                groups = np.array([f'g{i % count}' for i in rng.permutation(n)], dtype=object)
                for container in (lambda x: x, lambda x: x.tolist(), pd.Series,
                                  lambda x: x.astype(np.float32)):
                    with self.subTest(n=n, groups=count, container=container):
                        self.assertParity(container(y), container(pred), groups)
                before = (y.copy(), pred.copy(), groups.copy())
                grouped_metrics(y, pred, groups)
                for actual, expected in zip((y, pred, groups), before):
                    np.testing.assert_array_equal(actual, expected)

    def test_literal_numeric_identity_and_encounter_order(self):
        for groups in ([2**60, 2**60 + 1, .5], ['é', 'e\u0301', ' 1'],
                       [np.float32(.1), float(np.float32(.1)), .1], [-0., 0, 1]):
            self.assertParity([0., 1., 2.], [0., 2., 1.], groups)
        actual = grouped_metrics([0, 1, 2, 3, 4], [0, 2, 1, 2, 4],
                                 ['z', 'a', 'z', 'm', 'a'])[1]
        self.assertEqual([row['group'] for row in actual], ['z', 'a', 'm'])
        self.assertEqual([row['n_rows'] for row in actual], [2, 2, 1])

    def test_arithmetic_and_source_validation_parity(self):
        for y, pred in (([0., -0., 0.], [-0., 0., -0.]),
                        ([1e16, 1e16 + 2, 1e16], [1e16] * 3),
                        ([1e308], [1e308]), ([-1e308], [1e308]),
                        ([0.], [1e-200]), ([2**53 + 1], [0]),
                        ([True], [0]), ([1 + 1j], [0]),
                        (np.ma.array([1], mask=[True]), [0])):
            self.assertParity(y, pred, ['g'] * len(y))
        for groups in ([1, '1', 'b'], [None, 'a', 'b'], ['g\0', 'g', 'b'],
                       [' ', 'a', 'b'], [np.inf, 1, 2], ['a', 'b'],
                       np.ma.array(['a', 'b', 'c'], mask=[0, 1, 0])):
            self.assertParity([0, 0, 0], [0, 1, 0], groups)

    def test_per_group_then_overall_scoring_order_is_unchanged(self):
        calls = []
        def record(y, pred):
            calls.append((y.tolist(), pred.tolist()))
            return regression_metrics(y, pred)
        with patch('groupbench.metrics.regression_metrics', side_effect=record):
            grouped_metrics([1, 2, 3], [3, 2, 1], ['b', 'a', 'b'])
        self.assertEqual(calls, [([1., 3.], [3., 1.]), ([2.], [2.]),
                                 ([1., 2., 3.], [3., 2., 1.])])

    def test_memberships_do_not_materialize_labeled_group_indexes(self):
        groupby_type = type(pd.Series([0, 1]).groupby(['b', 'a'], sort=False))
        with patch.object(groupby_type, 'groups', new_callable=PropertyMock,
                          side_effect=AssertionError('Unnecessary labeled group indexes')) as labeled:
            self.assertParity([1., 2., 3.], [3., 2., 1.], ['b', 'a', 'b'])
            labeled.assert_not_called()

    def test_many_interleaved_groups_preserve_all_rows_and_exact_scores(self):
        rng = np.random.default_rng(619)
        for rows, count in ((2000, 500), (2000, 2000)):
            y, pred = rng.normal(size=rows), rng.normal(size=rows)
            groups = np.array([f'group-{i % count}' for i in rng.permutation(rows)], dtype=object)
            self.assertParity(y, pred, groups)


if __name__ == '__main__':
    unittest.main()
