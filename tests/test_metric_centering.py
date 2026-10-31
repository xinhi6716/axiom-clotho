"""Exact-float rational oracles and invented writer bundles; no fitting."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.metrics import regression_metrics, grouped_metrics
from groupbench.evaluation import BenchmarkResult
from groupbench.evaluation_schema import validate_evaluation_bundle
from groupbench.verification import verify_bundle


def exact_r2(y, pred):
    values = [Fraction(float(v)) for v in y]
    predictions = [Fraction(float(v)) for v in pred]
    mean = sum(values) / len(values)
    total = sum((v - mean) ** 2 for v in values)
    return None if not total else 1 - sum((p - v) ** 2 for p, v in zip(predictions, values)) / total


def invented_result():
    offset = float(2**54)
    y = np.array([offset, offset + 4, offset, offset + 4])
    pred = np.full(4, offset)
    groups = np.array(['a', 'a', 'b', 'b'])
    folds = np.array([0, 0, 1, 1])
    overall, per_group = grouped_metrics(y, pred, groups)
    per_fold = [dict(fold=f, **grouped_metrics(y[folds == f], pred[folds == f], groups[folds == f])[0]) for f in range(2)]
    model = dict(overall=overall, per_group=per_group, per_fold=per_fold)
    audits = [dict(fold=f, train_rows=2, test_rows=2, train_group_count=1,
                   test_group_count=1, overlap_count=0,
                   train_groups=['b' if f == 0 else 'a'],
                   test_groups=['a' if f == 0 else 'b'],
                   selected_candidate='invented-no-fit') for f in range(2)]
    manifest = dict(schema_version='1.0', protocol='nested-group-kfold-v1',
                    n_rows=4, n_groups=2,
                    config=dict(target='y', group='g', baselines=['dummy'],
                                outer_splits=2, inner_splits=2, random_state=0,
                                selection_objective='group_mae'),
                    outer_folds=audits, limitations=['Invented; no fitting.'],
                    groupbench_version='test', created_at_utc='synthetic',
                    data_sha256='0'*64, run_key='0'*64, elapsed_seconds=0.)
    summary = dict(schema_version='1.0', models={'dummy': model, 'selected': model},
                   selection_objective='group_mae', primary_entry='selected')
    frame = pd.DataFrame(dict(row_position=range(4), group=groups, outer_fold=folds,
                              observed=y, prediction_dummy=pred, prediction_selected=pred))
    return BenchmarkResult(summary, manifest, frame)


class CenteringTests(unittest.TestCase):
    def test_minimal_exactly_representable_offset(self):
        base = regression_metrics([0., 2.], [0., 0.])
        moved = regression_metrics([1e16, 1e16 + 2.], [1e16, 1e16])
        self.assertEqual(base, moved)
        self.assertEqual(moved['r2'], -1.)
        self.assertEqual(exact_r2([1e16, 1e16 + 2.], [1e16, 1e16]), Fraction(-1))

    def test_offset_sign_scale_and_order_rational_oracle(self):
        for exponent in (40, 52, 54, 100, 400):
            for sign in (-1, 1):
                offset = sign * float(2**exponent)
                unit = float(2**(exponent - 50))
                y = np.array([0., 1., 3., 7.]) * unit
                pred = np.array([1., 1., 2., 6.]) * unit
                moved_y, moved_pred = y + offset, pred + offset
                self.assertTrue(np.array_equal(moved_y - offset, y))
                self.assertTrue(np.array_equal(moved_pred - offset, pred))
                for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
                    a, b = moved_y[order], moved_pred[order]
                    scores = regression_metrics(a, b)
                    self.assertAlmostEqual(scores['r2'], float(exact_r2(a, b)), delta=1e-14)
                    self.assertEqual(scores, regression_metrics(y[order], pred[order]))

    def test_constant_singleton_and_fail_closed_preserved(self):
        for y, pred in [([1e308], [1e308]), ([0.1]*3, [0.2]*3), ([1e16]*2, [1e16+2]*2)]:
            self.assertIsNone(regression_metrics(y, pred)['r2'])
        for y, pred in [([-1e308], [1e308]), ([-1e308, 1e308], [-1e308, 1e308]),
                        ([0., 1.], [1e200, 1e200]), ([0.], [1e-200])]:
            with self.assertRaises(ValueError):
                regression_metrics(y, pred)

    def test_writer_and_optional_validator_share_corrected_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('groupbench.model.GroupBenchRegressor._fit_prepared', side_effect=AssertionError('fit forbidden')):
                result = invented_result()
                result.write(directory)
                before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
                self.assertEqual(verify_bundle(directory).bundle_type, 'evaluation')
                self.assertEqual(validate_evaluation_bundle(directory).n_rows, 4)
                self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
            for row in [result.summary['models']['selected']['overall'],
                        *result.summary['models']['selected']['per_group'],
                        *result.summary['models']['selected']['per_fold']]:
                self.assertEqual(row['r2'], -1.)

    def test_legacy_wrong_r2_bundle_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            result = invented_result()
            # Reproduce the previous direct-mean outputs, not a new model run.
            for model in result.summary['models'].values():
                for row in [model['overall'], *model['per_group'], *model['per_fold']]:
                    row['r2'] = 0.
            result.write(directory)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            self.assertEqual(verify_bundle(directory).bundle_type, 'evaluation')
            with self.assertRaisesRegex(ValueError, 'saved metric differs'):
                validate_evaluation_bundle(directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
