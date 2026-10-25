"""Original no-fit scoring oracles for exact independent-unit identity."""

import copy
import itertools
import unittest

import numpy as np
import pandas as pd

from groupbench.data import validate_groups
from groupbench.metrics import grouped_metrics
from groupbench.binary.metrics import grouped_binary_metrics


SCORERS = (grouped_metrics, grouped_binary_metrics)


class MetricGroupIdentityTests(unittest.TestCase):
    def test_regression_distinct_large_ids_do_not_reverse_model_comparison(self):
        groups = [2**60, 2**60 + 1, 0.5]
        a, per_group = grouped_metrics([0, 0, 0], [0, 10, 0], groups)
        b, _ = grouped_metrics([0, 0, 0], [0, 0, 9], groups)
        self.assertEqual(a["n_groups"], 3)
        self.assertEqual(a["group_mae"], 10 / 3)
        self.assertEqual(a["worst_group_mae"], 10)
        self.assertEqual(b["group_mae"], 3)
        self.assertLess(b["group_mae"], a["group_mae"])
        self.assertEqual([row["group"] for row in per_group],
                         [str(2**60), str(2**60 + 1), "0.5"])

    def test_binary_distinct_large_ids_do_not_reverse_model_comparison(self):
        groups = [2**60, 2**60 + 1, 0.5]
        a, per_group = grouped_binary_metrics([0, 0, 0], [0, 1, 0], groups)
        b, _ = grouped_binary_metrics([0, 0, 0], [0, 0, 0.9], groups)
        self.assertEqual(a["n_groups"], 3)
        self.assertEqual(a["group_brier"], 1 / 3)
        self.assertEqual(a["worst_group_brier"], 1)
        self.assertAlmostEqual(b["group_brier"], 0.27)
        self.assertLess(b["group_brier"], a["group_brier"])
        self.assertEqual([row["group"] for row in per_group],
                         [str(2**60), str(2**60 + 1), "0.5"])

    def test_equal_group_weighting_is_not_row_weighting(self):
        groups = [2**60, 2**60, 2**60 + 1, 0.5]
        reg, reg_rows = grouped_metrics([0] * 4, [0, 2, 9, 2], groups)
        binary, binary_rows = grouped_binary_metrics([0] * 4, [0, 1, 1, 1], groups)
        self.assertEqual([row["n_rows"] for row in reg_rows], [2, 1, 1])
        self.assertEqual([row["n_rows"] for row in binary_rows], [2, 1, 1])
        self.assertEqual(reg["mae"], 13 / 4)
        self.assertEqual(reg["group_mae"], (1 + 9 + 2) / 3)
        self.assertAlmostEqual(reg["group_rmse"], (np.sqrt(2) + 9 + 2) / 3)
        self.assertEqual(binary["brier"], 0.75)
        self.assertEqual(binary["group_brier"], (0.5 + 1 + 1) / 3)
        # Duplicating every row of one unit must not increase that unit's weight.
        for scorer, pred in ((grouped_metrics, [0, 2, 9, 2]),
                             (grouped_binary_metrics, [0, 1, 1, 1])):
            expected, _ = scorer([0] * 4, pred, groups)
            actual, _ = scorer([0] * 6, pred + pred[:2], groups + groups[:2])
            for key in ("group_mae", "group_rmse", "group_brier", "group_log_loss_clipped"):
                if key in expected:
                    self.assertEqual(actual[key], expected[key])

    def test_lossless_list_object_array_and_series_agree_without_mutation(self):
        values = [2**60, 2**60 + 1, 0.5]
        for scorer in SCORERS:
            expected = scorer([0, 0, 0], [0, 1, 0], values)
            for groups in (values, tuple(values), np.array(values, dtype=object),
                           pd.Series(values, dtype=object)):
                with self.subTest(scorer=scorer.__name__, container=type(groups).__name__):
                    before = copy.deepcopy(groups)
                    self.assertEqual(scorer([0, 0, 0], [0, 1, 0], groups), expected)
                    self.assertEqual(list(groups), list(before))

    def test_numeric_equivalence_and_distinction_follow_exact_stored_values(self):
        widened = float(np.float32(0.1))
        groups = [np.float32(0.1), widened, 0.1, 1, 1.0, -0.0, 0]
        for scorer in SCORERS:
            with self.subTest(scorer=scorer.__name__):
                overall, rows = scorer([0] * 7, [0] * 7, groups)
                self.assertEqual(overall["n_groups"], 4)
                self.assertEqual([row["group"] for row in rows],
                                 [str(widened), "0.1", "1", "0"])
                self.assertEqual([row["n_rows"] for row in rows], [2, 1, 2, 2])

    def test_literal_text_ids_are_not_numeric_or_unicode_normalized(self):
        groups = ["1", "01", "1.0", " 1", "1 ", "é", "e\u0301", "批次", "g\\0"]
        for scorer in SCORERS:
            with self.subTest(scorer=scorer.__name__):
                overall, rows = scorer([0] * len(groups), [0] * len(groups), groups)
                self.assertEqual(overall["n_groups"], len(groups))
                self.assertEqual([row["group"] for row in rows], groups)

    def test_row_permutation_preserves_scores_and_group_membership(self):
        groups, y, prediction = [2**60, 2**60 + 1, 0.5], [0, 0, 0], [0, 1, 0]
        for scorer in SCORERS:
            expected, rows = scorer(y, prediction, groups)
            by_group = {row["group"]: row for row in rows}
            for order in itertools.permutations(range(3)):
                with self.subTest(scorer=scorer.__name__, order=order):
                    actual, actual_rows = scorer([y[i] for i in order],
                                                [prediction[i] for i in order],
                                                [groups[i] for i in order])
                    self.assertEqual(actual, expected)
                    self.assertEqual({row["group"]: row for row in actual_rows}, by_group)

    def test_invalid_id_vectors_fail_closed_in_both_scorers(self):
        invalid = ([1, "1", "b"], [True, 1, 2], [np.bool_(False), 1, 2],
                   [None, "a", "b"], [np.nan, 1, 2], [pd.NA, "a", "b"],
                   [np.inf, 1, 2], ["", "a", "b"], [" \t", "a", "b"],
                   ["g\0", "g", "b"], ["g\0x", "g\0y", "b"],
                   ["a", "b"], [["a"], ["b"], ["c"]], "abc")
        for scorer in SCORERS:
            for groups in invalid:
                with self.subTest(scorer=scorer.__name__, groups=repr(groups)):
                    with self.assertRaises(ValueError):
                        scorer([0, 0, 0], [0, 1, 0], groups)

    def test_group_validation_error_does_not_disclose_private_label(self):
        for scorer in SCORERS:
            with self.subTest(scorer=scorer.__name__):
                with self.assertRaises(ValueError) as error:
                    scorer([0, 0, 0], [0, 1, 0], ["PRIVATE\0SECRET", "a", "b"])
                self.assertNotIn("PRIVATE", str(error.exception))
                self.assertNotIn("SECRET", str(error.exception))

    def test_prior_lossy_caller_conversion_cannot_be_recovered(self):
        # Conversion occurred outside the scorer, so only two stored IDs remain.
        groups = np.array([2**60, 2**60 + 1, 0.5], dtype=float)
        self.assertEqual(groups[0], groups[1])
        regression, _ = grouped_metrics([0, 0, 0], [0, 10, 0], groups)
        binary, _ = grouped_binary_metrics([0, 0, 0], [0, 1, 0], groups)
        self.assertEqual(regression["n_groups"], 2)
        self.assertEqual(regression["group_mae"], 2.5)
        self.assertEqual(binary["n_groups"], 2)
        self.assertEqual(binary["group_brier"], 0.25)

    def test_prevalidated_workflow_groups_keep_scores_and_labels(self):
        groups = [2**60, 2**60, 2**60 + 1, 0.5]
        normalized = validate_groups(groups, len(groups))
        for scorer in SCORERS:
            with self.subTest(scorer=scorer.__name__):
                self.assertEqual(scorer([0] * 4, [0, 1, 1, 0], groups),
                                 scorer([0] * 4, [0, 1, 1, 0], normalized))
                # The shared validator is idempotent on high-level API output.
                np.testing.assert_array_equal(validate_groups(normalized, 4), normalized)


if __name__ == "__main__":
    unittest.main()
