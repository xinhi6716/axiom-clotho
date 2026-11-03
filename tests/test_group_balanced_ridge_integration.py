"""Handwritten, tiny software fixtures; no scientific performance claims."""
from collections import Counter
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from groupbench import GroupBenchRegressor, benchmark, fit_predict, plan_benchmark, plan_prediction
from groupbench.baselines import GroupBalancedRidgeRegressor
from groupbench.model import Candidate, candidates_for, make_pipeline, selection_plan, _fit_pipeline
from groupbench.splitting import GroupKFold


def fixture():
    return pd.DataFrame({
        'x': [0., 2., 1., 4., 3., 8., 6., 5., 9., 7., 11., 10.],
        'kind': ['a', 'b', 'a', 'b', 'c', 'a', 'b', 'a', 'c', 'a', 'b', 'b'],
        'y': [1., 2., 4., 2., 6., 7., 3., 4., 8., 5., 10., 9.],
        'g': ['a', 'b', 'c', 'd', 'e', 'f', 'f', 'f', 'f', 'f', 'f', 'f'],
    })


class GroupBalancedRidgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.X = self.frame[['x', 'kind']]
        self.y = self.frame.y.to_numpy()
        self.groups = self.frame.g.to_numpy()
        self.options = dict(target='y', group='g', inner_splits=2,
                            baselines=('group_balanced_ridge',))

    def test_registry_and_defaults_unchanged(self):
        default = candidates_for(('dummy', 'ridge', 'extra_trees'))
        self.assertEqual([c.name for c in default], ['dummy_mean', 'ridge_alpha_0.1',
            'ridge_alpha_1', 'ridge_alpha_10', 'extra_trees_depth_6', 'extra_trees_depth_none'])
        result = candidates_for(('ridge', 'group_balanced_ridge', 'dummy'))
        self.assertEqual([c.name for c in result[3:6]], ['group_balanced_ridge_alpha_0.1',
            'group_balanced_ridge_alpha_1', 'group_balanced_ridge_alpha_10'])
        self.assertEqual([c.params for c in result[3:6]], [{'alpha': .1}, {'alpha': 1.}, {'alpha': 10.}])
        self.assertEqual(GroupBenchRegressor().baselines, ('dummy', 'ridge', 'extra_trees'))

    def test_current_slice_routing_and_local_weights(self):
        calls, weights = [], []
        original, ridge_fit = GroupBalancedRidgeRegressor.fit, Ridge.fit
        def spy(estimator, X, y, *, groups=None, sample_weight=None):
            calls.append((np.asarray(y).copy(), np.asarray(groups).copy()))
            return original(estimator, X, y, groups=groups, sample_weight=sample_weight)
        def weight_spy(estimator, X, y, sample_weight=None, **kwargs):
            weights.append(np.asarray(sample_weight).copy())
            return ridge_fit(estimator, X, y, sample_weight=sample_weight, **kwargs)
        plan = selection_plan(self.X, self.groups, baselines=self.options['baselines'],
                              inner_splits=2, random_state=0, objective='group_mae')
        with patch.object(GroupBalancedRidgeRegressor, 'fit', spy), patch.object(Ridge, 'fit', weight_spy):
            model = GroupBenchRegressor(baselines=self.options['baselines'], inner_splits=2).fit(self.X, self.y, self.groups)
        train_slices = [train for _ in range(3) for train, _ in plan.splits] + [np.arange(len(self.y))]
        self.assertEqual(len(calls), 7)
        for (actual_y, actual_groups), actual_weights, train in zip(calls, weights, train_slices):
            np.testing.assert_array_equal(actual_y, self.y[train])
            np.testing.assert_array_equal(actual_groups, self.groups[train])
            counts = Counter(self.groups[train]); n, G = len(train), len(counts)
            expected = [n / (G * counts[g]) for g in self.groups[train]]
            np.testing.assert_array_equal(actual_weights, expected)
        # No global weight vector sliced into smaller training partitions.
        self.assertNotEqual(weights[0].sum(), weights[-1][plan.splits[0][0]].sum())
        self.assertEqual(model.family_fit_details_['group_balanced_ridge']['training_rows'], 12)

    def test_preprocessing_identical_unweighted_and_independent_of_groups(self):
        X = self.X.copy(); X.loc[[1, 5], 'x'] = np.nan
        for encoding in ('legacy-v1', 'distinct-v2'):
            ordinary = make_pipeline(X, Candidate('r', 'ridge', {'alpha': 1.}), 0, encoding)
            balanced = make_pipeline(X, Candidate('b', 'group_balanced_ridge', {'alpha': 1.}), 0, encoding)
            ordinary.fit(X, self.y)
            balanced.fit(X, self.y, model__groups=self.groups)
            for name in ('statistics_',):
                np.testing.assert_array_equal(getattr(ordinary['preprocess'].named_transformers_['numeric']['impute'], name),
                                              getattr(balanced['preprocess'].named_transformers_['numeric']['impute'], name))
            for name in ('mean_', 'var_', 'scale_'):
                np.testing.assert_array_equal(getattr(ordinary['preprocess'].named_transformers_['numeric']['scale'], name),
                                              getattr(balanced['preprocess'].named_transformers_['numeric']['scale'], name))
            left, right = ordinary['preprocess'].transform(X), balanced['preprocess'].transform(X)
            np.testing.assert_array_equal(left.toarray() if sparse.issparse(left) else left,
                                          right.toarray() if sparse.issparse(right) else right)
            # Ordinary estimators/imputer/scaler reject group fit kwargs, so successful
            # mixed-family fitting also checks routing remains estimator-only.
            result = GroupBenchRegressor(baselines=('dummy', 'ridge', 'group_balanced_ridge'),
                inner_splits=2, categorical_encoding=encoding).fit(X, self.y, self.groups)
            self.assertTrue(np.isfinite(result.predict(X)).all())

    def test_pipeline_one_feature_analytic_oracle(self):
        X = pd.DataFrame({'x': [0., 1., 3., 8.]})
        y = np.array([1., 4., 2., 10.]); groups = ['a', 'a', 'a', 'b']
        pipe = make_pipeline(X, Candidate('b', 'group_balanced_ridge', {'alpha': 1.}), 0)
        pipe.fit(X, y, model__groups=groups)
        z = pipe['preprocess'].transform(X).ravel(); w = np.array([2/3, 2/3, 2/3, 2.])
        xb, yb = np.dot(w, z)/4, np.dot(w, y)/4
        beta = np.dot(w*(z-xb), y-yb)/(np.dot(w, (z-xb)**2)+1.)
        np.testing.assert_allclose(pipe.predict(X), yb + beta*(z-xb), rtol=1e-10, atol=1e-12)

    def test_prediction_uses_no_query_groups_and_both_objectives(self):
        for objective in ('group_mae', 'mae'):
            result = fit_predict(self.frame, self.X.iloc[:2].copy(), objective=objective, **self.options)
            self.assertEqual(result.manifest['schema_version'], '1.5')
            self.assertEqual(result.manifest['capabilities'], ['group_balanced_ridge-v1'])
            self.assertEqual(len(result.predictions), 2)
            self.assertTrue(all(row['objective'] == objective for row in result.manifest['inner_selection']))

    def test_exact_plan_budget_execution_and_no_fit(self):
        frame = pd.DataFrame({'x': list(range(20)), 'y': [0., 1., 3., 2., 4.] * 4,
                              'g': list('aabbccddeeffgghhiijj')})
        defaults = ('dummy', 'ridge', 'extra_trees')
        options = dict(target='y', group='g', inner_splits=3)
        with patch('groupbench.model.make_pipeline', side_effect=AssertionError('no fit preview')):
            before = plan_benchmark(frame, outer_splits=5, baselines=defaults, **options)
            after = plan_benchmark(frame, outer_splits=5, baselines=defaults+('group_balanced_ridge',), **options)
            prediction = plan_prediction(frame, frame[['x']].iloc[:2], baselines=defaults+('group_balanced_ridge',), **options)
        self.assertEqual(before['fit_budget']['total_fits'], 105)
        self.assertEqual(after['fit_budget']['total_fits'], 155)
        self.assertEqual(prediction['fit_budget']['total_fits'], 31)
        self.assertEqual(after['schema_version'], '1.4')
        with patch('groupbench.model._fit_pipeline', wraps=_fit_pipeline) as calls:
            benchmark(frame, outer_splits=5, baselines=defaults+('group_balanced_ridge',), **options)
        self.assertEqual(calls.call_count, 155)

    def test_held_out_labels_and_features_do_not_change_training(self):
        first = benchmark(self.frame, outer_splits=3, **self.options)
        _, test = next(GroupKFold(3, 0).split(self.X, groups=self.groups))
        for column in ('y', 'x'):
            changed = self.frame.copy(); changed.loc[test, column] += 20
            second = benchmark(changed, outer_splits=3, **self.options)
            self.assertEqual(first.manifest['outer_folds'][0], second.manifest['outer_folds'][0])
            if column == 'y':
                np.testing.assert_array_equal(first.predictions.loc[test, 'prediction_group_balanced_ridge'],
                                              second.predictions.loc[test, 'prediction_group_balanced_ridge'])

    def test_exact_ties_and_successful_old_only_reset(self):
        for order in [('dummy', 'group_balanced_ridge'), ('group_balanced_ridge', 'dummy')]:
            model = GroupBenchRegressor(baselines=order, inner_splits=2).fit(self.X, np.ones(12), self.groups)
            self.assertEqual(model.best_family_, order[0])
            self.assertEqual(model.family_candidates_['group_balanced_ridge'], 'group_balanced_ridge_alpha_0.1')
            model.set_params(baselines=('dummy',)).fit(self.X, self.y, self.groups)
            self.assertEqual(model.family_fit_details_, {})
            self.assertTrue(all('inner_fit_details' not in r for r in model.selection_results_))

    def test_atomic_failures_and_clean_recovery(self):
        model = GroupBenchRegressor(baselines=('group_balanced_ridge',), inner_splits=2).fit(self.X, self.y, self.groups)
        original = GroupBalancedRidgeRegressor.fit
        def final_failure(estimator, X, y, **kwargs):
            if len(y) == 12: raise ValueError('injected final failure')
            return original(estimator, X, y, **kwargs)
        failures = [patch.object(GroupBalancedRidgeRegressor, 'fit', side_effect=ValueError('injected inner failure')),
                    patch.object(GroupBalancedRidgeRegressor, 'fit', final_failure),
                    patch.object(GroupBalancedRidgeRegressor, 'predict', side_effect=ValueError('injected predict failure')),
                    patch('groupbench.model.grouped_metrics', side_effect=ValueError('injected metric failure'))]
        for failure in failures:
            old = {k: v for k, v in vars(model).items() if k.endswith('_')}
            before = model.predict(self.X)
            with failure, self.assertRaises(ValueError):
                model.fit(self.X, self.y + 1, self.groups)
            for key in old: self.assertIs(vars(model)[key], old[key])
            np.testing.assert_array_equal(model.predict(self.X), before)
        old = model.best_estimator_
        with self.assertRaisesRegex(ValueError, 'Rescale|rescale'):
            model.fit(self.X, np.full(12, 1e300), self.groups)
        self.assertIs(model.best_estimator_, old)
        model.fit(self.X, self.y + 1, self.groups)
        self.assertIsNot(model.best_estimator_, old)

    def test_details_are_copied_and_checked_against_training_slice(self):
        original = GroupBalancedRidgeRegressor.fit
        def wrong_count(estimator, X, y, **kwargs):
            original(estimator, X, y, **kwargs)
            estimator.fit_details_['training_rows'] += 1
            return estimator
        with patch.object(GroupBalancedRidgeRegressor, 'fit', wrong_count), self.assertRaisesRegex(ValueError, 'actual training slice'):
            GroupBenchRegressor(baselines=('group_balanced_ridge',), inner_splits=2).fit(self.X, self.y, self.groups)
        model = GroupBenchRegressor(baselines=('group_balanced_ridge',), inner_splits=2).fit(self.X, self.y, self.groups)
        saved = deepcopy(model.family_fit_details_)
        model.family_models_['group_balanced_ridge']['model'].fit_details_['alpha'] = 99
        self.assertEqual(model.family_fit_details_, saved)

    def test_real_bundle_roundtrips_with_both_encodings_and_family_orders(self):
        import tempfile
        from pathlib import Path
        from groupbench import validate_evaluation_bundle, validate_prediction_bundle_schema
        families = ('dummy', 'ridge', 'group_balanced_ridge', 'group_median', 'nystroem_ridge')
        for encoding, order in [('legacy-v1', families), ('distinct-v2', tuple(reversed(families)))]:
            args = dict(self.options, baselines=order, categorical_encoding=encoding)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evaluation = benchmark(self.frame, outer_splits=3, **args)
                prediction = fit_predict(self.frame, self.X.iloc[:2], **args)
                evaluation.write(root/'evaluation')
                prediction.write(root/'prediction')
                self.assertEqual(validate_evaluation_bundle(root/'evaluation').schema_version, '1.4')
                self.assertEqual(validate_prediction_bundle_schema(root/'prediction').schema_version, '1.5')

    def test_cli_plans_and_prediction_accept_explicit_family(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout, redirect_stderr
        from pathlib import Path
        from groupbench.cli import main
        from groupbench import validate_prediction_bundle_schema
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); train, query = root/'train.csv', root/'query.csv'
            self.frame.to_csv(train, index=False); self.X.iloc[:2].to_csv(query, index=False)
            arguments = [str(train), str(query), '--target', 'y', '--group', 'g',
                         '--inner-splits', '2', '--baselines', 'group_balanced_ridge']
            output = io.StringIO()
            with patch('groupbench.model.make_pipeline', side_effect=AssertionError('plan fitted')), redirect_stdout(output):
                self.assertEqual(main(['plan', 'predict'] + arguments), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['fit_budget']['total_fits'], 7)
            self.assertEqual(plan['schema_version'], '1.5')
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['query.csv', 'train.csv'])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['predict'] + arguments + ['--out', str(root/'out')]), 0)
            self.assertEqual(validate_prediction_bundle_schema(root/'out').schema_version, '1.5')

    def test_domain_binary_measurement_and_fet_reject_before_fit(self):
        frame = self.frame.copy(); frame['d'] = ['A' if g in 'abc' else 'B' for g in frame.g]
        args = dict(self.options, inner_strategy='leave_one_domain_out', inner_splits=None, domain='d')
        with patch('groupbench.model.make_pipeline', side_effect=AssertionError('fit occurred')):
            for operation in (lambda: plan_benchmark(frame, outer_splits=3, **args),
                              lambda: plan_prediction(frame, self.X, **args),
                              lambda: benchmark(frame, outer_splits=3, **args),
                              lambda: fit_predict(frame, self.X, **args)):
                with self.assertRaisesRegex(ValueError, 'group_kfold'): operation()
        from groupbench.binary.model import binary_candidates
        with self.assertRaises(ValueError): binary_candidates(('group_balanced_ridge',))
        from mlworkflow import contracts
        from mlworkflow.adapters import fet_transfer
        # These allowlists are deliberately not widened. Use exact existing
        # configurations, then alter only the baseline declaration.
        from pathlib import Path
        import json
        root = Path(__file__).resolve().parents[1]
        measurement = json.loads((root/'examples/measurement-regression/fixtures/config.json').read_text())
        measurement['baselines'] = ['dummy', 'group_balanced_ridge']
        with self.assertRaisesRegex(contracts.WorkflowError, 'supported'): contracts.validate_config(measurement)
        fet = json.loads((root/'examples/fet-device-regression/fixtures/config.json').read_text())
        fet['baselines'] = ['dummy', 'group_balanced_ridge']
        with self.assertRaisesRegex(contracts.WorkflowError, 'supported'): fet_transfer.validate_config(fet)


if __name__ == '__main__':
    unittest.main()
