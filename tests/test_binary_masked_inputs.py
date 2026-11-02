"""Synthetic no-fit checks: hidden labels/probabilities are not observations."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd

from groupbench.binary.data import validate_binary_target
from groupbench.binary.metrics import binary_metrics, checked_probabilities, grouped_binary_metrics
from groupbench.binary.model import GroupBenchClassifier, _probabilities


class BinaryMaskedInputsTests(unittest.TestCase):
    def test_active_target_masks_reject_for_training_and_single_class_scoring(self):
        for mask in ([True, False], [False, True], [True, True]):
            for require_both in (True, False):
                with self.subTest(mask=mask, require_both=require_both):
                    labels = np.ma.array([0, 1], mask=mask)
                    with self.assertRaisesRegex(ValueError, "masked"):
                        validate_binary_target(labels, 2, require_both=require_both)
        with self.assertRaisesRegex(ValueError, "masked"):
            validate_binary_target(np.ma.array([0, 0], mask=[False, True]), 2, require_both=False)

    def test_active_probability_masks_reject_in_row_and_group_scores(self):
        for mask in ([True, False], [False, True], [True, True]):
            p = np.ma.array([0., 1.], mask=mask)
            for scorer in (binary_metrics, grouped_binary_metrics):
                with self.subTest(mask=mask, scorer=scorer.__name__):
                    with self.assertRaisesRegex(ValueError, "masked"):
                        scorer([0, 1], p, *([['a', 'b']] if scorer is grouped_binary_metrics else []))

    def test_active_label_masks_reject_in_row_and_group_scores(self):
        labels = np.ma.array([0, 1], mask=[False, True])
        with self.assertRaisesRegex(ValueError, "masked"):
            binary_metrics(labels, [0., 1.])
        with self.assertRaisesRegex(ValueError, "masked"):
            grouped_binary_metrics(labels, [0., 1.], ['a', 'b'])

    def test_active_class_masks_reject_before_model_probabilities_are_used(self):
        probabilities = np.array([[1., 0.], [0., 1.]])
        for mask in ([True, False], [False, True], [True, True]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "masked"):
                checked_probabilities(probabilities, 2, np.ma.array([0, 1], mask=mask))

    def test_one_hidden_matrix_entry_rejects_the_entire_probability_matrix(self):
        for position in np.ndindex((2, 2)):
            mask = np.zeros((2, 2), dtype=bool)
            mask[position] = True
            p = np.ma.array([[1., 0.], [0., 1.]], mask=mask)
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "masked"):
                checked_probabilities(p, 2, [0, 1])

    def test_nomask_all_false_and_hard_unmasked_arrays_preserve_exact_outputs(self):
        y, p = np.array([0, 0, 0, 1]), np.array([0., 0.5, 0.25, 1.])
        groups = ['a', 'a', 'a', 'b']
        expected = binary_metrics(y, p)
        expected_grouped = grouped_binary_metrics(y, p, groups)
        matrix = np.column_stack([1-p, p])
        for mask in (np.ma.nomask, False, [False] * 4):
            for hard_mask in (False, True):
                labels = np.ma.array(y, mask=mask, hard_mask=hard_mask)
                probabilities = np.ma.array(p, mask=mask, hard_mask=hard_mask)
                with self.subTest(mask=repr(mask), hard_mask=hard_mask):
                    np.testing.assert_array_equal(validate_binary_target(labels, 4), y)
                    self.assertEqual(binary_metrics(labels, probabilities), expected)
                    self.assertEqual(grouped_binary_metrics(labels, probabilities, groups), expected_grouped)
                    np.testing.assert_array_equal(
                        checked_probabilities(np.ma.array(matrix, mask=False), 4, np.ma.array([0, 1], mask=False)), matrix)
                    np.testing.assert_array_equal(labels.data, y)
                    np.testing.assert_array_equal(probabilities.data, p)
                    self.assertFalse(np.ma.getmaskarray(labels).any())
                    self.assertFalse(np.ma.getmaskarray(probabilities).any())
        self.assertEqual(expected['n_rows'], 4)
        self.assertEqual(expected['brier'], 0.078125)
        self.assertEqual(expected['confusion'], {'tn': 2, 'fp': 1, 'fn': 0, 'tp': 1})
        self.assertEqual(expected_grouped[0]['group_brier'], (0.3125 / 3) / 2)

    def test_masked_views_fail_without_unmasking_or_disclosing_hidden_payload(self):
        for hidden in (0., 1., np.nan, np.inf, 987654321.):
            values = np.ma.array([0., hidden, 1.], mask=[False, True, False], hard_mask=True)
            view = values[::-1]
            before = values.copy()
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaisesRegex(ValueError, 'masked') as caught:
                    binary_metrics([0, 1, 1], view)
            self.assertNotIn('987654321', str(caught.exception))
            np.testing.assert_array_equal(values.data, before.data)
            np.testing.assert_array_equal(np.ma.getmaskarray(values), np.ma.getmaskarray(before))
            self.assertTrue(values.hardmask)

    def test_masked_scalar_and_unsupported_masked_dtypes_fail_cleanly(self):
        for value in (np.ma.masked, np.ma.array([0, 1], dtype=object, mask=True),
                      np.ma.array(['private-zero', 'private-one'], mask=True),
                      np.ma.array([(0.,), (1.,)], dtype=[('value', float)], mask=True)):
            with self.subTest(dtype=str(value.dtype)), warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaises(ValueError):
                    binary_metrics([0, 1], value)
                with self.assertRaises(ValueError):
                    validate_binary_target(value, 2)

    def test_classifier_fit_rejects_masked_labels_before_plan_or_estimator_construction(self):
        X = pd.DataFrame({'x': np.arange(8, dtype=float)})
        y = np.ma.array([0, 1] * 4, mask=[False] * 7 + [True])
        groups = np.repeat(['a', 'b', 'c', 'd'], 2)
        with patch('groupbench.binary.model.binary_selection_plan') as plan, \
             patch('groupbench.binary.model.binary_pipeline') as factory, \
             patch('groupbench.binary.model._fit_pipeline') as fit:
            with self.assertRaisesRegex(ValueError, 'masked'):
                GroupBenchClassifier(baselines=('dummy',), inner_splits=2).fit(X, y, groups)
        plan.assert_not_called()
        factory.assert_not_called()
        fit.assert_not_called()

    def test_estimator_probability_boundary_checks_masks_on_values_and_classes(self):
        X = pd.DataFrame({'x': [0., 1.]})
        matrix = np.array([[1., 0.], [0., 1.]])
        cases = ((np.ma.array(matrix, mask=[[False, False], [False, True]]), [0, 1]),
                 (matrix, np.ma.array([0, 1], mask=[False, True])))
        for p, labels in cases:
            pipeline = SimpleNamespace(predict_proba=Mock(return_value=p),
                                       named_steps={'model': SimpleNamespace(classes_=labels)})
            with self.assertRaisesRegex(ValueError, 'masked'):
                _probabilities(pipeline, X)
            pipeline.predict_proba.assert_called_once_with(X)


    def test_nested_list_and_tuple_rows_cannot_erase_active_masks(self):
        for outer in (list, tuple):
            for row_index in (0, 1):
                for column_index in (0, 1):
                    rows = [np.array([1., 0.]), np.array([0., 1.])]
                    row = np.ma.array(rows[row_index], mask=False)
                    row.mask[column_index] = True
                    rows[row_index] = row
                    before_data, before_mask = row.data.copy(), row.mask.copy()
                    with self.subTest(outer=outer.__name__, row=row_index, column=column_index):
                        with self.assertRaisesRegex(ValueError, 'masked'):
                            checked_probabilities(outer(rows), 2, [0, 1])
                        np.testing.assert_array_equal(row.data, before_data)
                        np.testing.assert_array_equal(row.mask, before_mask)

    def test_nested_masked_scalars_reject_before_coercion_warnings(self):
        for outer in (list, tuple):
            values = outer([0., np.ma.array(1., mask=True)])
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                for call in (lambda: binary_metrics([0, 1], values),
                             lambda: validate_binary_target(values, 2),
                             lambda: checked_probabilities([[1., 0.], [0., 1.]], 2, values)):
                    with self.assertRaisesRegex(ValueError, 'masked'):
                        call()
                with self.assertRaisesRegex(ValueError, 'masked'):
                    checked_probabilities([[1., 0.], outer([0., np.ma.array(1., mask=True)])], 2, [0, 1])

    def test_nested_active_object_payloads_reject_without_disclosure(self):
        for hidden in ('PRIVATE_MASKED_PAYLOAD', None, 987654321):
            row = np.ma.array([hidden, 0.], dtype=object, mask=[True, False])
            with self.assertRaisesRegex(ValueError, 'masked') as caught:
                checked_probabilities([row, [0., 1.]], 2, [0, 1])
            self.assertNotIn('PRIVATE', str(caught.exception))
            self.assertNotIn('987654321', str(caught.exception))
            self.assertEqual(row.mask.tolist(), [True, False])

    def test_object_array_payloads_remain_rejected_without_coercion_or_mutation(self):
        payload = np.empty(2, dtype=object)
        hidden = np.ma.array(1., mask=True)
        payload[:] = [0., hidden]
        for values in (payload, [payload, [0., 1.]]):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaises(ValueError):
                    checked_probabilities(values, 2, [0, 1])
        with self.assertRaises(ValueError):
            validate_binary_target(payload, 2)
        self.assertIs(payload[1], hidden)
        self.assertTrue(bool(hidden.mask))

    def test_nested_unmasked_rows_and_repeated_aliases_keep_exact_outputs(self):
        for outer in (list, tuple):
            for mask in (np.ma.nomask, False, [False, False]):
                row = np.ma.array([0.75, 0.25], mask=mask)
                matrix = outer([row, row])
                np.testing.assert_array_equal(checked_probabilities(matrix, 2, [0, 1]), [[0.75, 0.25]] * 2)
                self.assertFalse(np.ma.getmaskarray(row).any())
            expected = binary_metrics([0, 1], [0., 1.])
            self.assertEqual(binary_metrics([0, 1], outer([np.ma.array(0., mask=False), 1.])), expected)

    def test_cyclic_and_deep_containers_reject_without_recursion_errors(self):
        direct = [0.]
        direct.append(direct)
        indirect = [0.]
        indirect.append((indirect,))
        deep = [0.]
        for _ in range(80):
            deep = [deep]
        for values in (direct, indirect, deep):
            with self.assertRaisesRegex(ValueError, 'cyclic|depth'):
                checked_probabilities(values, 2, [0, 1])
            with self.assertRaisesRegex(ValueError, 'cyclic|depth'):
                validate_binary_target(values, 2)

    def test_generator_not_consumed_and_subclass_iterator_not_called_by_preflight(self):
        consumed = []
        def entries():
            consumed.append(True)
            yield np.ma.array(1., mask=True)
        source = entries()
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], source)
        self.assertEqual(consumed, [])
        class Rows(list):
            def __iter__(self):
                raise AssertionError('Do not call an overridden iterator')
        values = Rows([np.ma.array([1., 0.], mask=[True, False]), [0., 1.]])
        with self.assertRaisesRegex(ValueError, 'masked'):
            checked_probabilities(values, 2, [0, 1])


if __name__ == '__main__':
    unittest.main()
