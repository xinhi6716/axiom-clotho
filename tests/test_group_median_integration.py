"""Integration contract for optional group_median; synthetic software tests only."""
from collections import Counter
from fractions import Fraction
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupBenchRegressor, benchmark, fit_predict, plan_prediction, plan_benchmark
from groupbench.baselines import GroupBalancedMedianRegressor
from groupbench.model import Candidate, candidates_for, make_pipeline, selection_plan
from groupbench.splitting import GroupKFold


def fixture():
    groups = np.repeat(['a', 'b', 'c', 'd', 'e', 'f'], [1, 2, 3, 1, 2, 3])
    return pd.DataFrame({'x': np.arange(12, dtype=float),
                         'kind': ['even' if i % 2 == 0 else 'odd' for i in range(12)],
                         'target': np.arange(12, dtype=float), 'batch': groups})


def oracle(y, groups):
    counts = Counter(groups)
    losses = {float(c): sum(Fraction(abs(float(c) - float(v))) / counts[g]
                           for v, g in zip(y, groups)) for c in y}
    # Lower minimizer among observed targets; all fixtures use exact integers.
    return min(losses, key=lambda c: (losses[c], c))


class ExistingRoutingProofTests(unittest.TestCase):
    """These work on the published standalone estimator, before integration."""
    def test_explicit_pipeline_fit_routes_current_groups(self):
        frame = fixture(); X = frame[['x', 'kind']]
        pipe = make_pipeline(X, Candidate('dummy_mean', 'dummy', {'strategy': 'mean'}), 0)
        pipe.set_params(model=GroupBalancedMedianRegressor())
        pipe.fit(X, frame.target, model__groups=frame.batch.to_numpy())
        np.testing.assert_array_equal(pipe.predict(X), np.full(12, oracle(frame.target, frame.batch)))

    def test_unequal_group_loss_oracle_is_not_pooled_median(self):
        X = pd.DataFrame({'x': np.arange(4)})
        y = [0, 100, 101, 1]; groups = ['a', 'a', 'a', 'b']
        model = GroupBalancedMedianRegressor().fit(X, y, groups=groups)
        self.assertEqual(model.constant_, oracle(y, groups))
        self.assertEqual(model.constant_, 1)
        self.assertNotEqual(model.constant_, float(np.median(y)))

    def test_training_group_replication_preserves_constant(self):
        X = pd.DataFrame({'x': np.arange(4)})
        y = [0, 100, 101, 1]; groups = ['a', 'a', 'a', 'b']
        first = GroupBalancedMedianRegressor().fit(X, y, groups=groups)
        replicated = GroupBalancedMedianRegressor().fit(
            pd.DataFrame({'x': np.arange(7)}), y + y[:3], groups=groups + groups[:3])
        self.assertEqual(first.constant_, replicated.constant_)


class GroupMedianIntegrationTests(unittest.TestCase):
    """Run only once integration is implemented; no skip conceals missing support."""
    def setUp(self):
        self.frame = fixture(); self.X = self.frame[['x', 'kind']]
        self.y = self.frame.target.to_numpy(); self.groups = self.frame.batch.to_numpy()
        self.options = dict(target='target', group='batch', inner_splits=2, random_state=0,
                            baselines=('group_median',))

    def test_registry_one_candidate_and_explicit_order(self):
        candidates = candidates_for(('ridge', 'group_median', 'dummy'))
        self.assertEqual([c.name for c in candidates],
                         ['ridge_alpha_0.1', 'ridge_alpha_1', 'ridge_alpha_10',
                          'group_median_lower', 'dummy_mean'])
        median = candidates[3]
        self.assertEqual(median.family, 'group_median')
        self.assertEqual(median.params, {})

    def test_inner_and_full_refit_route_exact_training_groups(self):
        calls = []; original = GroupBalancedMedianRegressor.fit
        def spy(estimator, X, y, *, groups=None, sample_weight=None):
            calls.append((np.asarray(y).copy(), None if groups is None else np.asarray(groups).copy()))
            return original(estimator, X, y, groups=groups, sample_weight=sample_weight)
        plan = selection_plan(self.X, self.groups, baselines=('group_median',), inner_splits=2,
                              random_state=0, objective='group_mae')
        with patch.object(GroupBalancedMedianRegressor, 'fit', spy):
            model = GroupBenchRegressor(baselines=('group_median',), inner_splits=2)
            model.fit(self.X, self.y, self.groups)
        self.assertEqual(len(calls), 3)
        for (actual_y, actual_groups), train in zip(calls, [s[0] for s in plan.splits] + [np.arange(12)]):
            np.testing.assert_array_equal(actual_y, self.y[train])
            np.testing.assert_array_equal(actual_groups, self.groups[train])
        self.assertEqual(model.family_models_['group_median'].named_steps['model'].constant_, oracle(self.y, self.groups))

    def test_no_group_forwarding_to_ordinary_families(self):
        model = GroupBenchRegressor(baselines=('dummy', 'ridge', 'group_median'), inner_splits=2)
        # Dummy/Ridge reject unexpected groups, so successful fitting checks routing isolation.
        model.fit(self.X, self.y, self.groups)
        self.assertEqual(set(model.family_models_), {'dummy', 'ridge', 'group_median'})

    def test_prediction_needs_no_query_groups(self):
        result = fit_predict(self.frame, self.X.iloc[:2].copy(), **self.options)
        np.testing.assert_array_equal(result.predictions.prediction, [oracle(self.y, self.groups)] * 2)
        self.assertEqual(result.manifest['selected_family'], 'group_median')

    def test_no_fit_preview_matches_actual_median_fits(self):
        with patch.object(GroupBalancedMedianRegressor, 'fit', side_effect=AssertionError('preview must not fit')):
            prediction_plan = plan_prediction(self.frame, self.X.iloc[:2], **self.options)
            evaluation_plan = plan_benchmark(self.frame, outer_splits=3, **self.options)
        self.assertEqual(prediction_plan['fit_budget']['total_fits'], 3)
        self.assertEqual(evaluation_plan['fit_budget']['total_fits'], 9)
        calls = []; original = GroupBalancedMedianRegressor.fit
        def spy(estimator, X, y, **kwargs):
            calls.append(len(y)); return original(estimator, X, y, **kwargs)
        with patch.object(GroupBalancedMedianRegressor, 'fit', spy):
            benchmark(self.frame, outer_splits=3, **self.options)
        self.assertEqual(len(calls), evaluation_plan['fit_budget']['total_fits'])

    def test_outer_test_targets_cannot_change_their_predictions_or_selection(self):
        first = benchmark(self.frame, outer_splits=3, **self.options)
        _, test = next(GroupKFold(3, 0).split(self.X, groups=self.groups))
        altered = self.frame.copy(); altered.loc[test, 'target'] += 10000
        second = benchmark(altered, outer_splits=3, **self.options)
        np.testing.assert_array_equal(first.predictions.loc[test, 'prediction_group_median'],
                                      second.predictions.loc[test, 'prediction_group_median'])
        self.assertEqual(first.manifest['outer_folds'][0]['inner_selection'],
                         second.manifest['outer_folds'][0]['inner_selection'])

    def test_exact_ties_respect_requested_family_order(self):
        target = np.ones(len(self.y))
        for order in [('dummy', 'group_median'), ('group_median', 'dummy')]:
            model = GroupBenchRegressor(baselines=order, inner_splits=2).fit(self.X, target, self.groups)
            self.assertEqual(model.best_family_, order[0])
            self.assertEqual([r['selection_score'] for r in model.selection_results_], [0, 0])

    def test_final_fit_failure_preserves_prior_complete_state(self):
        model = GroupBenchRegressor(baselines=('group_median',), inner_splits=2).fit(self.X, self.y, self.groups)
        old = {k: v for k, v in vars(model).items() if k.endswith('_')}
        predictions = model.predict(self.X); original = GroupBalancedMedianRegressor.fit
        def fail_final(estimator, X, y, **kwargs):
            if len(y) == len(self.y): raise ValueError('invented final refit failure')
            return original(estimator, X, y, **kwargs)
        with patch.object(GroupBalancedMedianRegressor, 'fit', fail_final):
            with self.assertRaisesRegex(ValueError, 'invented final refit failure'):
                model.fit(self.X, self.y + 10, self.groups)
        current = {k: v for k, v in vars(model).items() if k.endswith('_')}
        self.assertEqual(set(current), set(old))
        for key in old: self.assertIs(current[key], old[key])
        np.testing.assert_array_equal(model.predict(self.X), predictions)

    def test_both_selection_objectives_accept_explicit_comparator(self):
        for objective in ['group_mae', 'mae']:
            model = GroupBenchRegressor(baselines=('group_median',), inner_splits=2, objective=objective)
            model.fit(self.X, self.y, self.groups)
            self.assertEqual(model.family_models_['group_median'].named_steps['model'].constant_, oracle(self.y, self.groups))
            self.assertEqual(model.selection_results_[0]['objective'], objective)


if __name__ == '__main__':
    unittest.main()
