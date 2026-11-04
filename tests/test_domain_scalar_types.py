"""Original domain-label types must reach the existing exact-string validator."""
from copy import deepcopy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupBenchRegressor
from groupbench.domain_selection import domain_vector


def model():
    return GroupBenchRegressor(baselines=('dummy', 'ridge'), inner_splits=None,
                               inner_strategy='leave_one_domain_out', random_state=7)


def fixture():
    X = pd.DataFrame({'x': [0., 1., 2., 3., 4., 5.]}, index=[9, 9, 5, 5, 1, 1])
    return X, np.array([0., 2., 3., 4., 5., 6.]), list('abcdef')


class DomainScalarTypeTests(unittest.TestCase):
    def test_mixed_nonstring_labels_reject_before_splits_or_fitting(self):
        X, y, groups = fixture()
        for invalid in (2, 2.5, True, np.int64(2), np.float32(2.5),
                        np.bool_(True), b'B', np.bytes_('B')):
            values = ['A', 'A', invalid, invalid, 'C', 'C']
            factories = (list, tuple, lambda v: np.array(v, dtype=object),
                         lambda v: pd.Series(v, index=X.index, name='domain', dtype=object))
            for factory in factories:
                domains = factory(values)
                with self.subTest(scalar=type(invalid).__name__, container=type(domains).__name__), ExitStack() as stack:
                    blocked = [stack.enter_context(patch(path, side_effect=AssertionError('unexpected work')))
                               for path in ('groupbench.model.domain_splits', 'groupbench.model.make_pipeline',
                                            'groupbench.model._fit_pipeline')]
                    with self.assertRaisesRegex(ValueError, 'exact strings'):
                        model().fit(X, y, groups=groups, domains=domains)
                    for operation in blocked:
                        operation.assert_not_called()

    def test_numeric_and_text_domains_are_not_silently_merged(self):
        X, y, groups = fixture()
        values = [1, '1', 'A', 'A', 'C', 'C']
        with self.assertRaisesRegex(ValueError, 'exact strings'):
            domain_vector(values, groups, len(X))
        with patch('groupbench.model.make_pipeline', side_effect=AssertionError('unexpected fitting')):
            with self.assertRaisesRegex(ValueError, 'exact strings'):
                model().fit(X, y, groups=groups, domains=values)
        self.assertIs(type(values[0]), int)
        self.assertIs(type(values[1]), str)

    def test_valid_containers_preserve_selection_audits_and_predictions(self):
        X, y, groups = fixture()
        labels = ['001', '001', '01', '01', 'NA', 'NA']
        reference = model().fit(X, y, groups=groups, domains=labels)
        factories = (list, tuple, lambda v: np.array(v), lambda v: np.array(v, dtype=object),
                     pd.Index, lambda v: pd.Series(v, index=X.index, name='domain'),
                     lambda v: pd.Series(pd.Categorical(v, categories=['unused', 'NA', '01', '001']),
                                         index=X.index, name='domain'))
        source_before = X.copy(deep=True)
        for factory in factories:
            domains = factory(labels)
            before = list(domains)
            actual = model().fit(X, y, groups=groups, domains=domains)
            with self.subTest(container=type(domains).__name__):
                self.assertEqual(actual.inner_audit_, reference.inner_audit_)
                self.assertEqual(actual.selection_results_, reference.selection_results_)
                self.assertEqual(actual.best_candidate_, reference.best_candidate_)
                for family, prediction in reference.predict_baselines(X).items():
                    np.testing.assert_array_equal(actual.predict_baselines(X)[family], prediction)
                self.assertEqual(list(domains), before)
        pd.testing.assert_frame_equal(X, source_before)
        self.assertEqual([row['test_domains'] for row in reference.inner_audit_], [['001'], ['01'], ['NA']])

    def test_failed_refit_preserves_previous_model_and_selection_state(self):
        X, y, groups = fixture()
        estimator = model().fit(X, y, groups=groups, domains=['A', 'A', 'B', 'B', 'C', 'C'])
        before = deepcopy(estimator.selection_results_)
        audit, selected = deepcopy(estimator.inner_audit_), estimator.best_candidate_
        fitted = estimator.best_estimator_
        predictions = estimator.predict(X).copy()
        with patch('groupbench.model.make_pipeline', side_effect=AssertionError('unexpected fitting')):
            with self.assertRaisesRegex(ValueError, 'exact strings'):
                estimator.fit(X, y + 100., groups=groups, domains=['A', 'A', 2, 2, 'C', 'C'])
        self.assertIs(estimator.best_estimator_, fitted)
        self.assertEqual(estimator.best_candidate_, selected)
        self.assertEqual(estimator.selection_results_, before)
        self.assertEqual(estimator.inner_audit_, audit)
        np.testing.assert_array_equal(estimator.predict(X), predictions)

    def test_masks_series_alignment_and_predictor_exclusion_still_reject(self):
        X, y, groups = fixture()
        labels = ['A', 'A', 'B', 'B', 'C', 'C']
        invalid = (np.ma.array(labels, mask=False), np.ma.array(labels, mask=[0, 0, 1, 0, 0, 0]),
                   pd.Series(labels, index=X.index[::-1], name='domain'),
                   pd.Series(labels, index=X.index, name='x'), [labels], labels[:-1])
        for domains in invalid:
            with self.subTest(container=type(domains).__name__), \
                 patch('groupbench.model.make_pipeline', side_effect=AssertionError('unexpected fitting')):
                with self.assertRaises(ValueError):
                    model().fit(X, y, groups=groups, domains=domains)
        self.assertFalse(np.ma.getmaskarray(invalid[0]).any())
        np.testing.assert_array_equal(invalid[1].mask, [0, 0, 1, 0, 0, 0])

    def test_caller_string_conversion_cannot_recover_original_scalar_types(self):
        X, y, groups = fixture()
        promoted = np.array(['A', 'A', 2, 2, 'C', 'C'])
        explicit = ['A', 'A', '2', '2', 'C', 'C']
        first = model().fit(X, y, groups=groups, domains=promoted)
        second = model().fit(X, y, groups=groups, domains=explicit)
        self.assertEqual(first.inner_audit_, second.inner_audit_)
        self.assertEqual(first.selection_results_, second.selection_results_)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))


if __name__ == '__main__':
    unittest.main()
