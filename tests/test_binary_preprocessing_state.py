"""Synthetic binary learned-state failures, not predictive-performance evidence."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from groupbench import (GroupBenchClassifier, benchmark_binary, fit_predict_binary,
                        plan_binary_benchmark, plan_binary_prediction)
from groupbench.cli import main
import groupbench.model as regression_model


def frame(magnitude=1.0):
    return pd.DataFrame({'x': np.tile([-magnitude, magnitude], 8),
                         'lot': np.repeat([f'g{i}' for i in range(8)], 2),
                         'fail': np.tile([0, 1], 8)})


OPTIONS = dict(target='fail', group='lot', baselines=('dummy',), inner_splits=2, random_state=0)


class BinaryPreprocessingStateTests(unittest.TestCase):
    def estimator(self, families=('dummy',)):
        return GroupBenchClassifier(baselines=families, inner_splits=2, random_state=0)

    def assert_unfitted(self, model):
        self.assertFalse(any(key.endswith('_') for key in vars(model)))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_actual_inner_overflow_rejects_with_numpy_errors_ignored(self):
        data = frame(float(2**600))
        model = self.estimator()
        original = Pipeline.predict_proba
        calls = []
        def predict(pipeline, X, *args, **kwargs):
            calls.append(pipeline)
            return original(pipeline, X, *args, **kwargs)
        with np.errstate(all='ignore'), patch.object(Pipeline, 'predict_proba', predict):
            with self.assertRaisesRegex(ValueError, r'numeric state.*scale.var_.*Rescale'):
                model.fit(data[['x']], data.fail, data.lot)
            self.assertEqual(calls, [])
        self.assert_unfitted(model)

    def test_actual_final_only_overflow_preserves_prior_complete_state(self):
        normal, extreme = frame(), frame(float(2**510))
        model = self.estimator().fit(normal[['x']], normal.fail, normal.lot)
        old = dict(vars(model))
        before = model.predict_proba(normal[['x']])
        calls = []
        original = Pipeline.predict_proba
        def predict(pipeline, X, *args, **kwargs):
            values = original(pipeline, X, *args, **kwargs)
            calls.append(values.copy())
            return values
        with np.errstate(all='ignore'), patch.object(Pipeline, 'predict_proba', predict):
            with self.assertRaisesRegex(ValueError, r'numeric state.*scale.var_'):
                model.fit(extreme[['x']], extreme.fail, extreme.lot)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(np.isfinite(values).all() for values in calls))
        self.assertEqual(set(vars(model)), set(old))
        for name, value in old.items():
            self.assertIs(vars(model)[name], value)
        np.testing.assert_array_equal(model.predict_proba(normal[['x']]), before)

    def test_each_inner_and_family_fit_is_guarded_for_all_candidates(self):
        data = frame()
        original = regression_model._validate_fitted_numeric_state
        with patch.object(regression_model, '_validate_fitted_numeric_state', wraps=original) as guard:
            self.estimator(('dummy', 'logistic', 'extra_trees')).fit(data[['x']], data.fail, data.lot)
        # 1 + 3 + 2 candidates, two folds each, then three family refits.
        self.assertEqual(guard.call_count, 15)
        self.assertEqual(len({id(call.args[0]) for call in guard.call_args_list}), 15)

    def test_each_learned_numeric_array_fails_before_probabilities(self):
        data = frame()
        original = Pipeline.fit
        for step, field in (('impute', 'statistics_'), ('scale', 'mean_'),
                            ('scale', 'var_'), ('scale', 'scale_')):
            for invalid in (np.nan, np.inf, -np.inf):
                def fit(pipeline, X, y, *args, **kwargs):
                    result = original(pipeline, X, y, *args, **kwargs)
                    numeric = pipeline.named_steps['preprocess'].named_transformers_['numeric']
                    getattr(numeric.named_steps[step], field).flat[0] = invalid
                    return result
                model = self.estimator()
                with self.subTest(step=step, field=field, invalid=invalid), \
                     patch.object(Pipeline, 'fit', fit), patch.object(Pipeline, 'predict_proba') as predict:
                    with self.assertRaisesRegex(ValueError, f'numeric state.*{step}.{field}'):
                        model.fit(data[['x']], data.fail, data.lot)
                    predict.assert_not_called()
                    self.assert_unfitted(model)

    def test_fit_predict_and_nested_benchmark_fail_without_a_result(self):
        for magnitude in (float(2**600), float(2**510)):
            data = frame(magnitude)
            with self.subTest(magnitude=magnitude), np.errstate(all='ignore'):
                with self.assertRaisesRegex(ValueError, 'non-finite fitted numeric state'):
                    fit_predict_binary(data, pd.DataFrame({'x': [0., 1.]}), **OPTIONS)
                if magnitude == float(2**600):
                    with self.assertRaisesRegex(ValueError, 'non-finite fitted numeric state'):
                        benchmark_binary(data, outer_splits=2, **OPTIONS)
                else:
                    # Nested evaluation only refits eight training rows here,
                    # not the overflowing sixteen-row full-data refit.
                    result = benchmark_binary(data, outer_splits=2, **OPTIONS)
                    self.assertEqual(result.summary['models']['selected']['overall']['group_brier'], .25)

    def test_no_fit_plans_remain_structural_and_do_not_run_state_guard(self):
        data = frame(float(2**600))
        with patch.object(Pipeline, 'fit', side_effect=AssertionError('unexpected fit')), \
             patch.object(regression_model, '_validate_fitted_numeric_state', side_effect=AssertionError('unexpected state check')):
            prediction = plan_binary_prediction(data, pd.DataFrame({'x': [0.]}), **OPTIONS)
            evaluation = plan_binary_benchmark(data, outer_splits=2, **OPTIONS)
        self.assertEqual(prediction['fits_performed'], 0)
        self.assertEqual(evaluation['fits_performed'], 0)

    def test_cli_failure_creates_no_bundle_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, query = root/'source.csv', root/'query.csv'
            frame(float(2**600)).to_csv(source, index=False)
            pd.DataFrame({'x': [0., 1.]}).to_csv(query, index=False)
            for command in ('run', 'predict'):
                for existing in (False, True):
                    out = root/f'{command}-{existing}'
                    if existing:
                        out.mkdir()
                        (out/'keep.txt').write_text('untouched', encoding='utf-8')
                    arguments = ['binary', command, str(source)]
                    if command == 'predict':
                        arguments.append(str(query))
                    arguments += ['--target','fail','--group','lot','--baselines','dummy',
                                  '--inner-splits','2','--out',str(out)]
                    if command == 'run':
                        arguments += ['--outer-splits','2']
                    stderr, stdout = io.StringIO(), io.StringIO()
                    with self.subTest(command=command, existing=existing), np.errstate(all='ignore'), \
                         redirect_stderr(stderr), redirect_stdout(stdout):
                        code = main(arguments)
                    self.assertEqual(code, 2)
                    self.assertIn('non-finite fitted numeric state', stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), '')
                    if existing:
                        self.assertEqual({p.name for p in out.iterdir()}, {'keep.txt'})
                        self.assertEqual((out/'keep.txt').read_text(), 'untouched')
                    else:
                        self.assertFalse(out.exists())

    def test_finite_numeric_categorical_and_all_missing_models_remain_valid(self):
        data = frame()
        variants = [data[['x']], pd.DataFrame({'kind': ['a','b']*8}),
                    data[['x']].assign(empty=np.nan),
                    data[['x']].assign(kind=['a','b']*8, empty=np.nan)]
        for X in variants:
            with self.subTest(columns=list(X)):
                with np.errstate(all='ignore'):
                    model = self.estimator().fit(X, data.fail, data.lot)
                    probabilities = model.predict_proba(X)
                np.testing.assert_array_equal(probabilities, np.full((16,2), .5))
                self.assertEqual(model.best_candidate_, 'dummy_prior')
                self.assertEqual(model.selection_results_[0]['selection_score'], .25)

    def test_caller_numpy_error_mode_is_not_changed(self):
        for setting in ('ignore','warn','raise'):
            with self.subTest(setting=setting), np.errstate(over=setting, invalid=setting):
                before = np.geterr().copy()
                data = frame(float(2**600))
                with self.assertRaises(ValueError):
                    self.estimator().fit(data[['x']], data.fail, data.lot)
                self.assertEqual(np.geterr(), before)


if __name__ == '__main__':
    unittest.main()
