"""Fail closed on NUL group IDs; original synthetic validation-only probes."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (
    GroupBenchClassifier, GroupBenchRegressor, GroupKFold,
    benchmark, benchmark_binary, fit_predict, fit_predict_binary,
    plan_benchmark, plan_binary_benchmark, plan_prediction, plan_binary_prediction,
)
from groupbench.data import prepare_data, validate_groups


INVALID_IDS = ("g\0", "g\0\0", "g\0a", "\0g", "\0", "\0\0", "批次\0甲")


def fixture():
    return pd.DataFrame({"g": np.repeat(["a", "b", "c", "d", "e", "f"], 4),
                         "x": np.arange(24, dtype=float), "y": [0, 1] * 12})


class TextGroupIdentityTests(unittest.TestCase):
    def test_embedded_trailing_repeated_and_nul_only_ids_rejected(self):
        for value in INVALID_IDS:
            for container in (list, lambda v: np.array(v, dtype=object), pd.Series,
                              lambda v: pd.Series(v, dtype="string")):
                with self.subTest(value=repr(value), container=container):
                    groups = container(["ordinary", value, "another"])
                    with self.assertRaisesRegex(ValueError, "NUL \\(U\\+0000\\)"):
                        validate_groups(groups, 3)

    def test_diagnostic_does_not_disclose_identifier(self):
        with self.assertRaises(ValueError) as caught:
            validate_groups(["PRIVATE_IDENTIFIER\0secret"], 1)
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_literal_escape_visible_nul_symbol_and_unicode_are_unchanged(self):
        values = ["g", "g\\0", "g\\x00", "g␀", "批次甲", "é", "e\u0301",
                  " 001 ", "001", "1", "line\nfeed", "tab\tname", "🙂"]
        actual = validate_groups(values, len(values))
        self.assertEqual(actual.dtype, object)
        self.assertEqual(actual.tolist(), values)
        self.assertEqual(len(set(actual)), len(values))

    def test_blank_and_missing_rules_remain_in_force(self):
        for value in ("", " ", "\t\r\n", None, np.nan, pd.NA):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_groups(["g", value], 2)

    def test_numeric_id_contract_is_unchanged(self):
        same = np.float32(0.1)
        values = [0, -0., 1, 1., same, float(same), 2**60, 2**60 + 1]
        self.assertEqual(validate_groups(values, len(values)).tolist(),
                         ["0", "0", "1", "1", str(float(same)), str(float(same)),
                          str(2**60), str(2**60 + 1)])

    def test_splitter_rejects_before_factorization_without_mutating_input(self):
        for value in INVALID_IDS:
            groups = np.array(["a", "a", "b", "b", value, value], dtype=object)
            before = groups.copy()
            with self.subTest(value=repr(value)), patch(
                "groupbench.splitting.pd.factorize", side_effect=AssertionError("Must not factorize")
            ) as factorize:
                with self.assertRaisesRegex(ValueError, "NUL"):
                    list(GroupKFold(2).split(groups=groups))
                factorize.assert_not_called()
            self.assertEqual(groups.tolist(), before.tolist())

    def test_all_high_level_plans_and_execution_reject_before_work(self):
        workflows = (
            (plan_benchmark, False), (benchmark, False),
            (plan_prediction, True), (fit_predict, True),
            (plan_binary_benchmark, False), (benchmark_binary, False),
            (plan_binary_prediction, True), (fit_predict_binary, True),
        )
        blocked = ("groupbench.model.make_pipeline", "groupbench.binary.model.binary_pipeline",
                   "groupbench.splitting.pd.factorize", "groupbench.data.data_fingerprint",
                   "groupbench.prediction.data_fingerprint", "groupbench.evaluation.data_fingerprint",
                   "groupbench.binary.workflow.data_fingerprint", "groupbench.report.write_payloads")
        for bad in INVALID_IDS:
            frame = fixture()
            frame.loc[8:11, "g"] = bad
            before = frame.copy(deep=True)
            new = pd.DataFrame({"x": [1., 2.]})
            for encoding in ("legacy-v1", "distinct-v2"):
                for workflow, prediction in workflows:
                    with self.subTest(id=repr(bad), workflow=workflow.__name__, encoding=encoding), ExitStack() as stack:
                        mocks = [stack.enter_context(patch(path, side_effect=AssertionError(path))) for path in blocked]
                        options = dict(target="y", group="g", inner_splits=2, baselines=("dummy",),
                                       categorical_encoding=encoding)
                        if not prediction:
                            options["outer_splits"] = 3
                        with self.assertRaisesRegex(ValueError, "NUL"):
                            workflow(frame, new, **options) if prediction else workflow(frame, **options)
                        for mocked in mocks:
                            mocked.assert_not_called()
            pd.testing.assert_frame_equal(frame, before)

    def test_estimator_refit_rejection_preserves_existing_state(self):
        # A sentinel avoids model work: validate_groups must fail before any
        # estimator state can be replaced or a selection plan can be created.
        frame = fixture()
        for estimator_type in (GroupBenchRegressor, GroupBenchClassifier):
            estimator = estimator_type(baselines=("dummy",), inner_splits=2)
            sentinel = object()
            estimator.best_estimator_ = sentinel
            groups = frame.g.to_numpy(dtype=object)
            groups[0] = "a\0"
            with self.subTest(estimator=estimator_type.__name__), patch.object(
                estimator, "_fit_prepared", side_effect=AssertionError("Must not fit")
            ) as fitting:
                with self.assertRaisesRegex(ValueError, "NUL"):
                    estimator.fit(frame[["x"]], frame.y, groups=groups)
                fitting.assert_not_called()
                self.assertIs(estimator.best_estimator_, sentinel)

    def test_group_validation_does_not_inspect_non_group_columns(self):
        frame = pd.DataFrame({"g": ["a", "b"], "x": ["feature\0a", "feature\0b"], "y": [0, 1]})
        before = frame.copy(deep=True)
        # The separate feature validator owns categorical-value rejection.
        groups = validate_groups(frame.g, len(frame))
        self.assertEqual(groups.tolist(), ["a", "b"])
        pd.testing.assert_frame_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
