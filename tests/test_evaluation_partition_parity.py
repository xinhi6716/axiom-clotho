"""Exact arithmetic parity against unchanged writer metrics, invented rows only."""
import unittest
import json
import numpy as np
from groupbench.metrics import grouped_metrics
from groupbench.evaluation_schema import _partitioned_metrics


def partitions(groups, folds, nf):
    group_rows, fold_rows, fold_groups = {}, [[] for _ in range(nf)], [[] for _ in range(nf)]
    for i, (g, f) in enumerate(zip(groups, folds)):
        if g not in group_rows:
            group_rows[g] = []; fold_groups[f].append(g)
        group_rows[g].append(i); fold_rows[f].append(i)
    return group_rows, fold_rows, fold_groups


class PartitionParityTests(unittest.TestCase):
    def check_case(self, y, pred, groups, folds, nf):
        expected_overall, expected_groups = grouped_metrics(y, pred, groups)
        expected_folds = []
        for f in range(nf):
            indexes = np.flatnonzero(np.asarray(folds) == f)
            metrics, _ = grouped_metrics(y[indexes], pred[indexes], np.asarray(groups)[indexes])
            expected_folds.append(dict(fold=f, **metrics))
        actual = _partitioned_metrics(y, pred, *partitions(groups, folds, nf))
        expected = dict(overall=expected_overall, per_group=expected_groups, per_fold=expected_folds)
        self.assertEqual(actual, expected)
        self.assertEqual(json.dumps(actual), json.dumps(expected))

    def test_exact_randomized_unbalanced_interleaved_parity(self):
        for seed in range(30):
            rng = np.random.default_rng(seed)
            groups = ['001', 'NA', 'nan', 'Inf', 'comma,group', 'line\ngroup']
            ids = np.concatenate([np.arange(6), rng.integers(0, 6, 91)])
            rng.shuffle(ids)
            y = rng.normal(size=len(ids)); pred = y + rng.normal(size=len(ids))
            self.check_case(y, pred, [groups[i] for i in ids], ids % 3, 3)

    def test_constant_singletons_large_offsets_and_signed_zero(self):
        groups = ['001', 'NA', 'nan', 'Inf', 'x', 'y']
        for y, pred in [(np.ones(6)*3, np.ones(6)*4),
                        (np.arange(6.)+1e14, np.arange(6.)+1e14+0.125),
                        (np.array([0., -0., 0., -0., 0., -0.]), np.zeros(6))]:
            self.check_case(y, pred, groups, np.arange(6)%3, 3)

    def test_overflow_underflow_fail_like_reference(self):
        groups=['a','b','a','b']; folds=[0,1,0,1]
        for y, pred in [(np.array([1e308]*4), np.array([-1e308]*4)),
                        (np.zeros(4), np.array([1e-200]*4))]:
            with self.assertRaises(ValueError): grouped_metrics(y,pred,groups)
            with self.assertRaises(ValueError): _partitioned_metrics(y,pred,*partitions(groups,folds,2))
