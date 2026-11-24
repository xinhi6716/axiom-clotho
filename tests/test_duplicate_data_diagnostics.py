"""Diagnose normalized duplicate records without rebuilding their signatures."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.data import duplicate_diagnostics, prepare_data


class DuplicateDataDiagnosticsTests(unittest.TestCase):
    def test_same_record_membership_for_all_duplicate_counts(self):
        source = pd.DataFrame({
            "quality": pd.Series([1, 1, 1, 2, 2, pd.NA, pd.NA, 1], dtype="Int64"),
            "flow": [0., -0., 0., np.nan, np.nan, 0., -0., 0.],
            "kind": pd.Categorical(["a", "a", "a", None, None, "a", "a", "a"]),
            "target": [1., 1., 1., 1., 1., 1., 1., 2.],
            "parent": ["p1", "p1", "p2", "p1", "p1", "p3", "p4", "p2"],
        })
        x, y, groups = prepare_data(source, "target", "parent")
        before = x.copy(deep=True)
        expected = {"identical_feature_target_rows": 7,
                    "identical_feature_target_rows_across_groups": 5,
                    "cross_group_duplicate_patterns": 2,
                    "action": "diagnostic_only_no_rows_removed"}
        self.assertEqual(duplicate_diagnostics(x, y, groups), expected)
        with patch.object(pd.DataFrame, "duplicated", side_effect=AssertionError("second record pass")):
            self.assertEqual(duplicate_diagnostics(x, y, groups), expected)
        pd.testing.assert_frame_equal(x, before)

    def test_all_unique_records(self):
        x = pd.DataFrame({"x": [1., 2., 3.]})
        got = duplicate_diagnostics(x, np.array([0., 0., 0.]), np.array(["a", "a", "b"]))
        self.assertEqual(got["identical_feature_target_rows"], 0)
        self.assertEqual(got["identical_feature_target_rows_across_groups"], 0)
        self.assertEqual(got["cross_group_duplicate_patterns"], 0)
