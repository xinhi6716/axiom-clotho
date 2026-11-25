"""Durations and datetimes never become physical group IDs implicitly."""
import unittest

import numpy as np
import pandas as pd

from groupbench.data import validate_groups
from groupbench.splitting import GroupKFold


class TemporalGroupIdentityTests(unittest.TestCase):
    def test_temporal_labels_never_alias_numeric_groups(self):
        for values in (
            [np.timedelta64(1, "ns"), 1], [np.timedelta64(1, "s"), 1],
            np.array([1, 2], dtype="timedelta64[ns]"),
            np.array([1, 2], dtype="datetime64[ns]"),
            pd.Series([np.timedelta64(1, "ns"), 1], dtype=object),
            pd.Series(pd.to_datetime([1, 2], utc=True)),
        ):
            with self.subTest(container=type(values).__name__):
                with self.assertRaisesRegex(ValueError, "Datetime and timedelta"):
                    validate_groups(values, 2)

    def test_splitter_does_not_merge_duration_with_numeric_identity(self):
        groups = [np.timedelta64(1, "ns"), 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Datetime and timedelta"):
            list(GroupKFold(n_splits=2).split(np.zeros((4, 1)), groups=groups))

    def test_explicit_caller_ids_preserve_distinct_groups(self):
        converted = [f"elapsed_ns:{int(v)}" for v in np.array([1, 2], dtype="timedelta64[ns]").astype("int64")]
        np.testing.assert_array_equal(validate_groups(converted, 2), ["elapsed_ns:1", "elapsed_ns:2"])
        np.testing.assert_array_equal(validate_groups(np.array([1, 2]), 2), ["1", "2"])
