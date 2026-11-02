"""No-fit source-type checks for explicit numeric binary probabilities."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import warnings

import numpy as np
import pandas as pd

from groupbench.binary.data import validate_binary_target
from groupbench.binary.metrics import binary_metrics, checked_probabilities, grouped_binary_metrics
from groupbench.binary.model import _probabilities


class BinaryProbabilityTypesTests(unittest.TestCase):
    def test_boolean_scalars_in_mixed_vectors_reject_before_numeric_promotion(self):
        for container in (list, tuple):
            for boolean in (False, True, np.bool_(False), np.bool_(True)):
                for position in (0, 1):
                    values = [0., 1.]
                    values[position] = boolean
                    with self.subTest(container=container.__name__, boolean=repr(boolean), position=position):
                        for scorer in (binary_metrics, grouped_binary_metrics):
                            with self.assertRaisesRegex(ValueError, 'Boolean'):
                                scorer([0, 1], container(values), *([['a', 'b']] if scorer is grouped_binary_metrics else []))

    def test_boolean_scalars_in_either_matrix_column_reject(self):
        for outer in (list, tuple):
            for inner in (list, tuple):
                for row, column in np.ndindex((2, 2)):
                    matrix = [[1., 0.], [0., 1.]]
                    matrix[row][column] = bool(matrix[row][column])
                    with self.assertRaisesRegex(ValueError, 'Boolean'):
                        checked_probabilities(outer(inner(r) for r in matrix), 2, [0, 1])

    def test_typed_boolean_row_containers_cannot_be_promoted_by_other_rows(self):
        rows = (np.array([True, False]), pd.Series([True, False]), pd.Index([True, False]),
                pd.array([True, False], dtype='boolean'), np.ma.array([True, False], mask=False),
                memoryview(np.array([True, False])))
        for row in rows:
            for outer in (list, tuple):
                with self.subTest(type=type(row).__name__, outer=outer.__name__):
                    with self.assertRaisesRegex(ValueError, 'Boolean'):
                        checked_probabilities(outer([row, [0., 1.]]), 2, [0, 1])
                    with self.assertRaisesRegex(ValueError, 'Boolean'):
                        binary_metrics([0, 1], row)
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            binary_metrics([0, 1], [np.array(False), 1.])

    def test_homogeneous_boolean_inputs_still_reject(self):
        for values in ([False, True], (False, True), np.array([False, True]),
                       pd.Series([False, True], dtype='boolean')):
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                binary_metrics([0, 1], values)

    def test_numeric_endpoint_vectors_and_matrices_keep_exact_outputs(self):
        y, values, groups = [0, 0, 0, 1], [0, 0.5, 0.25, 1], ['a', 'a', 'a', 'b']
        expected = binary_metrics(y, values)
        grouped = grouped_binary_metrics(y, values, groups)
        factories = (list, tuple, np.array, pd.Series, pd.Index,
                     lambda x: np.array(x, dtype=np.float16),
                     lambda x: np.array(x, dtype=np.float32),
                     lambda x: pd.array(x, dtype='Float64'))
        for factory in factories:
            p = factory(values)
            with self.subTest(factory=repr(factory)):
                self.assertEqual(binary_metrics(y, p), expected)
                self.assertEqual(grouped_binary_metrics(y, p, groups), grouped)
        for dtype in (np.int8, np.int64, np.uint64, np.float16, np.float32, np.float64):
            matrix = np.array([[1, 0], [0, 1]], dtype=dtype)
            matrix.flags.writeable = False
            np.testing.assert_array_equal(checked_probabilities(matrix, 2, [0, 1]), [[1, 0], [0, 1]])
            np.testing.assert_array_equal(matrix, [[1, 0], [0, 1]])
        self.assertEqual(expected['brier'], 0.078125)
        self.assertEqual(expected['confusion'], {'tn': 2, 'fp': 1, 'fn': 0, 'tp': 1})

    def test_already_promoted_numeric_inputs_are_not_reinterpreted_as_booleans(self):
        promoted = np.asarray([False, 1.])
        self.assertEqual(promoted.dtype.kind, 'f')
        self.assertEqual(binary_metrics([0, 1], promoted), binary_metrics([0, 1], [0., 1.]))
        explicit = np.array([False, True]).astype(np.int64)
        self.assertEqual(binary_metrics([0, 1], explicit)['brier'], 0.)

    def test_mask_contract_and_unmasked_aliases_are_preserved(self):
        row = np.ma.array([1., 0.], mask=False)
        np.testing.assert_array_equal(checked_probabilities([row, row], 2, [0, 1]), [[1., 0.]] * 2)
        self.assertFalse(row.mask.any())
        for masked in (np.ma.array([True, False], mask=[True, False]),
                       np.ma.array([1., 0.], mask=[True, False])):
            with self.assertRaisesRegex(ValueError, 'masked'):
                checked_probabilities([masked, [0., 1.]], 2, [0, 1])
            np.testing.assert_array_equal(masked.mask, [True, False])

    def test_probability_return_boundary_rejects_source_booleans_without_fitting(self):
        X = pd.DataFrame({'x': [0., 1.]})
        for probabilities in ([[True, 0.], [0., 1.]], [np.array([True, False]), [0., 1.]]):
            pipeline = SimpleNamespace(predict_proba=Mock(return_value=probabilities),
                                       named_steps={'model': SimpleNamespace(classes_=np.array([0, 1]))})
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                _probabilities(pipeline, X)
            pipeline.predict_proba.assert_called_once_with(X)

    def test_binary_label_contract_is_not_changed(self):
        for labels in ([0, 1], [0., 1.], np.array([0, 1]), pd.Series([0, 1], dtype='Int64')):
            np.testing.assert_array_equal(validate_binary_target(labels, 2), [0, 1])
        for labels in ([False, 1], [0, True], np.array([False, True])):
            with self.assertRaises(ValueError):
                validate_binary_target(labels, 2)

    def test_invalid_object_complex_string_inputs_keep_rejection(self):
        for values in (np.array([0., True], dtype=object), ['0', '1'], [0j, 1.],
                       [None, 1.], [np.nan, 1.], [0., np.inf]):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaises(ValueError):
                    binary_metrics([0, 1], values)

    def test_depth_ragged_and_shared_numeric_rows_keep_existing_contract(self):
        deep = [0.]
        for _ in range(65):
            deep = [deep]
        with self.assertRaisesRegex(ValueError, 'depth'):
            binary_metrics([0, 1], deep)
        with self.assertRaises(ValueError):
            checked_probabilities([[1., 0.], [1.]], 2, [0, 1])
        row = [0.25, 0.75]
        np.testing.assert_array_equal(checked_probabilities([row, row], 2, [0, 1]),
                                      [[0.25, 0.75], [0.25, 0.75]])
        self.assertEqual(row, [0.25, 0.75])

    def test_cycles_generators_and_private_payloads_fail_without_side_effects(self):
        cycle = [0.]
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            binary_metrics([0, 1], cycle)
        consumed = []
        def values():
            consumed.append(True)
            yield False
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], values())
        self.assertEqual(consumed, [])
        private = [False, 'PRIVATE_PROBABILITY_PAYLOAD']
        with self.assertRaises(ValueError) as caught:
            binary_metrics([0, 1], private)
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.assertEqual(private, [False, 'PRIVATE_PROBABILITY_PAYLOAD'])


if __name__ == '__main__':
    unittest.main()
