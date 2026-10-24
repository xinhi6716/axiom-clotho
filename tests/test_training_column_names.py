"""Validate named target/group selectors before work and preserve string identity."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (
    benchmark, benchmark_binary, fit_predict, fit_predict_binary,
    plan_benchmark, plan_binary_benchmark, plan_prediction, plan_binary_prediction,
)
from groupbench.binary.data import prepare_binary_data
from groupbench.data import prepare_data
from groupbench.verification import _SAFE_OPEN_SUPPORTED, verify_bundle


WORKFLOWS = (
    (plan_benchmark, False), (benchmark, False),
    (plan_prediction, True), (fit_predict, True),
    (plan_binary_benchmark, False), (benchmark_binary, False),
    (plan_binary_prediction, True), (fit_predict_binary, True),
)
INVALID_NAMES = (0, np.int64(0), 1.5, False, ("part", "name"), None)


def fixture():
    return pd.DataFrame({"g": np.repeat(list("abcdef"), 4),
                         "x": np.arange(24, dtype=float), "y": [0, 1] * 12})


class TrainingColumnNameTests(unittest.TestCase):
    def test_non_string_existing_names_rejected_before_planning_or_fitting(self):
        blocked = ("groupbench.model.make_pipeline", "groupbench.binary.model.binary_pipeline",
                   "groupbench.splitting.pd.factorize", "groupbench.data.data_fingerprint",
                   "groupbench.prediction.data_fingerprint", "groupbench.evaluation.data_fingerprint",
                   "groupbench.binary.workflow.data_fingerprint", "groupbench.report.write_payloads")
        for role in ("target", "group"):
            original_name = "y" if role == "target" else "g"
            for name in INVALID_NAMES:
                frame = fixture().rename(columns={original_name: name})
                before = frame.copy(deep=True)
                for encoding in ("legacy-v1", "distinct-v2"):
                    for workflow, prediction in WORKFLOWS:
                        options = dict(target="y", group="g", inner_splits=2,
                                       baselines=("dummy",), categorical_encoding=encoding)
                        options[role] = name
                        if not prediction:
                            options["outer_splits"] = 3
                        with self.subTest(role=role, name=repr(name), workflow=workflow.__name__,
                                          encoding=encoding), ExitStack() as stack:
                            mocks = [stack.enter_context(patch(path, side_effect=AssertionError(path)))
                                     for path in blocked]
                            with self.assertRaisesRegex(ValueError, "must be string column names"):
                                if prediction:
                                    workflow(frame, pd.DataFrame({"x": [1., 2.]}), **options)
                                else:
                                    workflow(frame, **options)
                            for mocked in mocks:
                                mocked.assert_not_called()
                pd.testing.assert_frame_equal(frame, before)

    def test_unsupported_unhashable_or_missing_selectors_have_actionable_error(self):
        for prepare in (prepare_data, prepare_binary_data):
            for value in (["y"], {"name": "y"}, pd.NA, np.nan):
                with self.subTest(prepare=prepare.__name__, value=repr(value)):
                    with self.assertRaisesRegex(ValueError, "explicitly rename non-string columns"):
                        prepare(fixture(), value, "g")
                    with self.assertRaisesRegex(ValueError, "explicitly rename non-string columns"):
                        prepare(fixture(), "y", value)

    def test_binary_check_precedes_reading_target_values(self):
        frame = fixture().rename(columns={"g": np.int64(1)})
        with patch("groupbench.binary.data.validate_binary_target", side_effect=AssertionError("Must not inspect labels")) as labels:
            with self.assertRaisesRegex(ValueError, "must be string column names"):
                prepare_binary_data(frame, "y", np.int64(1))
            labels.assert_not_called()

    def test_error_does_not_echo_selector_contents(self):
        frame = fixture().rename(columns={"y": ("PRIVATE", "target")})
        with self.assertRaises(ValueError) as caught:
            prepare_data(frame, ("PRIVATE", "target"), "g")
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_non_string_unused_columns_remain_allowed_with_feature_allowlist(self):
        frame = fixture()
        frame[np.int64(10)] = "unused metadata"
        for prepare in (prepare_data, prepare_binary_data):
            actual = prepare(frame, "y", "g", features=["x"])
            expected = prepare(fixture(), "y", "g", features=["x"])
            pd.testing.assert_frame_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[2], expected[2])

    def test_existing_name_distinctness_and_membership_checks_remain(self):
        for prepare in (prepare_data, prepare_binary_data):
            for target, group in (("y", "y"), ("absent", "g"), ("y", "absent")):
                with self.subTest(prepare=prepare.__name__, target=target, group=group):
                    with self.assertRaisesRegex(ValueError, "two distinct existing columns"):
                        prepare(fixture(), target, group)

    def test_valid_string_selectors_round_trip_without_coercion(self):
        # Strings remain literal, including whitespace, newlines, Unicode and
        # NUL in column names. This is not a new CSV-header validation rule.
        for target, group in (("回應🙂", "批次\n甲"), (" y ", " g "),
                              ("y\0label", "g\0label"), (np.str_("y"), np.str_("g"))):
            frame = fixture().rename(columns={"y": target, "g": group})
            for encoding in ("legacy-v1", "distinct-v2"):
                for workflow, prediction in WORKFLOWS:
                    options = dict(target=target, group=group, inner_splits=2,
                                   baselines=("dummy",), categorical_encoding=encoding)
                    if not prediction:
                        options["outer_splits"] = 3
                    with self.subTest(target=repr(target), group=repr(group),
                                      workflow=workflow.__name__, encoding=encoding):
                        result = workflow(frame, pd.DataFrame({"x": [1., 2.]}), **options) if prediction else workflow(frame, **options)
                        record = result if isinstance(result, dict) else result.manifest
                        serialized = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
                        self.assertEqual(serialized["config"]["target"], target)
                        self.assertEqual(serialized["config"]["group"], group)
                        if not isinstance(result, dict):
                            with tempfile.TemporaryDirectory() as temp:
                                out = Path(temp, "bundle")
                                result.write(out)
                                saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                                self.assertEqual(saved["config"]["target"], target)
                                self.assertEqual(saved["config"]["group"], group)
                                if _SAFE_OPEN_SUPPORTED:
                                    verified = verify_bundle(out)
                                    self.assertEqual(verified.bundle_type, "prediction" if prediction else "evaluation")


if __name__ == "__main__":
    unittest.main()
