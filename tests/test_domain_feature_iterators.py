"""One-shot feature-name collections retain domain-mode API selection semantics."""
from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import benchmark, fit_predict, plan_benchmark, plan_prediction
from groupbench.domain_selection import prepare_domain_data


def fixture():
    return pd.DataFrame({'g': list('abcdef'), 'd': ['A','A','B','B','C','C'],
                         'x': [0.,1.,2.,3.,4.,5.], 'z': [6.,4.,3.,2.,1.,0.],
                         'unused': [99.] * 6, 'y': [0.,2.,3.,4.,5.,6.]},
                        index=[7,7,3,3,1,1])


ARGS = dict(target='y', group='g', domain='d', inner_strategy='leave_one_domain_out',
            inner_splits=None, baselines=('dummy',), random_state=0)
ROUTES = (plan_benchmark, plan_prediction, benchmark, fit_predict)


def invoke(route, source, query, features):
    options = dict(ARGS, features=features)
    if route in (plan_prediction, fit_predict):
        return route(source, query, **options)
    return route(source, outer_splits=3, **options)


def stable_manifest(value):
    result = deepcopy(value)
    result.pop('created_at_utc', None)
    result.pop('elapsed_seconds', None)
    return result


class OneShotNames:
    def __init__(self, values):
        self.values = values
        self.iterations = 0
    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError('Original feature collection was consumed twice')
        return iter(self.values)


class DomainFeatureIteratorTests(unittest.TestCase):
    def setUp(self):
        self.source = fixture()
        self.query = pd.DataFrame({'x': [1.,4.], 'z': [4.,1.]}, index=[22,22])

    def assert_result_equal(self, first, second):
        if isinstance(first, dict):
            self.assertEqual(first, second)
        else:
            pd.testing.assert_frame_equal(first.predictions, second.predictions)
            self.assertEqual(stable_manifest(first.manifest), stable_manifest(second.manifest))
            if hasattr(first, 'summary'):
                self.assertEqual(first.summary, second.summary)
            if hasattr(first, 'estimator'):
                self.assertEqual(first.estimator.best_candidate_, second.estimator.best_candidate_)
                self.assertEqual(first.estimator.selection_results_, second.estimator.selection_results_)
                self.assertEqual(first.estimator.inner_audit_, second.estimator.inner_audit_)

    def test_all_public_routes_preserve_generator_and_iterator_results(self):
        before_source, before_query = self.source.copy(deep=True), self.query.copy(deep=True)
        for route in ROUTES:
            expected = invoke(route, self.source, self.query, ['z','x'])
            for factory in (iter, lambda names: (name for name in names), tuple, pd.Index):
                with self.subTest(route=route.__name__, factory=factory):
                    actual = invoke(route, self.source, self.query, factory(['z','x']))
                    self.assert_result_equal(expected, actual)
        pd.testing.assert_frame_equal(self.source, before_source)
        pd.testing.assert_frame_equal(self.query, before_query)

    def test_original_custom_iterable_is_consumed_only_once_per_route(self):
        for route in ROUTES:
            with self.subTest(route=route.__name__):
                names = OneShotNames(['z','x'])
                actual = invoke(route, self.source, self.query, names)
                self.assertEqual(names.iterations, 1)
                self.assert_result_equal(invoke(route, self.source, self.query, ['z','x']), actual)

    def test_direct_preparation_retains_order_targets_groups_and_domains(self):
        expected = prepare_domain_data(self.source, 'y','g',['z','x'],'d','leave_one_domain_out',None)
        actual = prepare_domain_data(self.source, 'y','g',iter(['z','x']),'d','leave_one_domain_out',None)
        pd.testing.assert_frame_equal(actual[0], expected[0])
        self.assertEqual(list(actual[0]), ['z','x'])
        for a,b in zip(actual[1:], expected[1:]):
            np.testing.assert_array_equal(a,b)

    def test_domain_target_group_and_invalid_names_still_reject_before_splits_or_fits(self):
        invalid = (['d','x'], ['x','d'], ['x','d','z'], ['y'], ['g'],
                   ['x','x'], ['absent'], [])
        for route in ROUTES:
            for names in invalid:
                for factory in (list, iter, lambda values: (name for name in values)):
                    with self.subTest(route=route.__name__, names=names, factory=factory), ExitStack() as stack:
                        blocked = [stack.enter_context(patch(name, side_effect=AssertionError('unexpected model work')))
                                   for name in ('groupbench.splitting.pd.factorize', 'groupbench.model.make_pipeline',
                                                'groupbench.model._fit_pipeline', 'groupbench.report.write_payloads')]
                        with self.assertRaises(ValueError):
                            invoke(route, self.source, self.query, factory(names))
                        for operation in blocked:
                            operation.assert_not_called()

    def test_bare_strings_and_bytes_reach_the_existing_common_rejection(self):
        for route in ROUTES:
            for names in ('x', 'd', '', b'x', b'd', b'', np.str_('x'), np.bytes_('x')):
                with self.subTest(route=route.__name__, kind=type(names).__name__), \
                     patch('groupbench.splitting.pd.factorize', side_effect=AssertionError('unexpected scheduling')):
                    with self.assertRaisesRegex(ValueError, 'not a bare string or bytes'):
                        invoke(route, self.source, self.query, names)

    def test_default_features_and_group_mode_are_unchanged(self):
        # Default includes all non-target/group/domain columns, not just our allowlist.
        default_query = self.source[['x','z','unused']].iloc[:2]
        for route in ROUTES:
            self.assert_result_equal(invoke(route, self.source, default_query, None),
                                     invoke(route, self.source, default_query, ['x','z','unused']))
        source = self.source.drop(columns='d')
        options = dict(target='y',group='g',inner_splits=2,baselines=('dummy',))
        self.assertEqual(plan_prediction(source,self.query,features=['z','x'],**options),
                         plan_prediction(source,self.query,features=iter(['z','x']),**options))

    def test_exhausted_iterator_is_rejected_not_reset_or_treated_as_default(self):
        names = iter(['z','x'])
        list(names)
        with patch('groupbench.splitting.pd.factorize', side_effect=AssertionError('unexpected scheduling')):
            with self.assertRaisesRegex(ValueError, 'features'):
                invoke(plan_prediction,self.source,self.query,names)

    def test_failing_iterator_propagates_before_scheduling_fitting_or_writing(self):
        def failed_names():
            yield 'd'
            raise RuntimeError('synthetic iterator failure')
        for route in ROUTES:
            with self.subTest(route=route.__name__), ExitStack() as stack:
                blocked = [stack.enter_context(patch(name, side_effect=AssertionError('unexpected work')))
                           for name in ('groupbench.splitting.pd.factorize', 'groupbench.model.make_pipeline',
                                        'groupbench.model._fit_pipeline', 'groupbench.report.write_payloads')]
                with self.assertRaisesRegex(RuntimeError, 'synthetic iterator failure'):
                    invoke(route, self.source, self.query, failed_names())
                for operation in blocked:
                    operation.assert_not_called()

    def test_failed_domain_declaration_does_not_consume_feature_iterable(self):
        names = OneShotNames(['z','x'])
        with self.assertRaisesRegex(ValueError, 'separate declared metadata'):
            plan_prediction(self.source,self.query,**(ARGS | {'domain':'missing','features':names}))
        self.assertEqual(names.iterations, 0)


if __name__ == '__main__':
    unittest.main()
