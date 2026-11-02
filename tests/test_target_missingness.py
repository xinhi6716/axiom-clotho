"""Missing outcome metadata must survive validation until it is rejected."""

from contextlib import ExitStack
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (GroupBenchRegressor, benchmark, fit_predict,
                        plan_benchmark, plan_prediction)
from groupbench.data import prepare_data, validate_target


def temporal_vectors():
    """Original synthetic vectors; no observed dates or measurements."""
    for unit in ("D", "s", "ms", "us", "ns"):
        for kind in ("datetime64", "timedelta64"):
            yield np.array([1, "NaT"], dtype=f"{kind}[{unit}]")
            yield np.array([1, 2], dtype=f"{kind}[{unit}]")
    yield np.array([np.datetime64("2000-01-01"), np.datetime64("NaT")], dtype=object)
    yield np.array([np.timedelta64(1, "D"), np.timedelta64("NaT")], dtype=object)
    yield [date(2000, 1, 1), date(2000, 1, 2)]
    yield [timedelta(days=1), timedelta(days=2)]
    yield pd.Series(pd.to_datetime(["2000-01-01", None], utc=True))


class TargetMissingnessTests(unittest.TestCase):
    def test_temporal_targets_never_become_implicit_numeric_units(self):
        for values in temporal_vectors():
            with self.subTest(dtype=str(getattr(values, "dtype", type(values)))):
                with self.assertRaisesRegex(ValueError, "explicit numeric units"):
                    validate_target(values, 2)

    def test_masked_finite_labels_are_missing_not_observed_payloads(self):
        for hidden in (0.0, 999.0, np.nan, np.inf):
            values = np.ma.array([1.0, hidden], mask=[False, True])
            original_data, original_mask = values.data.copy(), values.mask.copy()
            with self.subTest(hidden=hidden), self.assertRaisesRegex(ValueError, "missing targets"):
                validate_target(values, 2)
            np.testing.assert_array_equal(values.data, original_data)
            np.testing.assert_array_equal(values.mask, original_mask)
        with self.assertRaisesRegex(ValueError, "missing targets"):
            validate_target(np.ma.array([1.0, 2.0], mask=True), 2)

    def test_unmasked_arrays_and_existing_numeric_inputs_remain_accepted(self):
        values = ([1, 2], ["1", "2"], np.array([1, 2], dtype=np.int64),
                  pd.Series([1, 2], dtype="Int64"), pd.Series([1, 2], dtype="Float64"),
                  np.ma.array([1, 2], mask=False), np.ma.array([1, 2], mask=np.ma.nomask))
        for y in values:
            with self.subTest(kind=type(y).__name__):
                result = validate_target(y, 2)
                np.testing.assert_array_equal(result, [1.0, 2.0])
                self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(validate_target([False, True], 2), [0.0, 1.0])

    def test_explicit_numeric_conversion_must_preserve_missingness(self):
        values = pd.Series(pd.to_timedelta([1, None], unit="D")).dt.total_seconds()
        with self.assertRaisesRegex(ValueError, "missing targets"):
            validate_target(values, 2)
        observed = pd.Series(pd.to_timedelta([1, 2], unit="D")).dt.total_seconds()
        np.testing.assert_array_equal(validate_target(observed, 2), [86400.0, 172800.0])

    def test_dataframe_workflows_reject_nat_before_splits_fit_or_writes(self):
        for dtype in ("datetime64[ns]", "timedelta64[ns]"):
            source = pd.DataFrame({"lot": np.repeat(list("abcdef"), 2),
                                   "x": np.arange(12, dtype=float),
                                   "target": np.arange(12).astype(dtype)})
            source.loc[3, "target"] = pd.NaT
            before = source.copy(deep=True)
            workflows = ((benchmark, False), (plan_benchmark, False),
                         (fit_predict, True), (plan_prediction, True))
            for workflow, prediction in workflows:
                with self.subTest(dtype=dtype, workflow=workflow.__name__), ExitStack() as stack:
                    blocked = [stack.enter_context(patch(name, side_effect=AssertionError("unexpected work")))
                               for name in ("groupbench.model.make_pipeline",
                                            "groupbench.splitting.pd.factorize",
                                            "groupbench.report.write_payloads")]
                    options = dict(target="target", group="lot", inner_splits=2, baselines=("dummy",))
                    with self.assertRaisesRegex(ValueError, "explicit numeric units"):
                        if prediction:
                            workflow(source, pd.DataFrame({"x": [1.0]}), **options)
                        else:
                            workflow(source, outer_splits=3, **options)
                    for operation in blocked:
                        operation.assert_not_called()
            with self.assertRaisesRegex(ValueError, "explicit numeric units"):
                prepare_data(source, "target", "lot")
            pd.testing.assert_frame_equal(source, before)

    def test_estimator_rejects_masked_target_before_splitting_or_fitting(self):
        X = pd.DataFrame({"x": np.arange(12, dtype=float)})
        y = np.ma.array(np.arange(12, dtype=float), mask=False)
        y.mask[3] = True
        with ExitStack() as stack:
            blocked = [stack.enter_context(patch(name, side_effect=AssertionError("unexpected work")))
                       for name in ("groupbench.model.make_pipeline", "groupbench.splitting.pd.factorize")]
            with self.assertRaisesRegex(ValueError, "missing targets"):
                GroupBenchRegressor(inner_splits=2, baselines=("dummy",)).fit(
                    X, y, groups=np.repeat(list("abcdef"), 2))
            for operation in blocked:
                operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
