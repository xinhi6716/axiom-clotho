"""A bare API string must not be silently split into character predictors."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (benchmark, benchmark_binary, fit_predict, fit_predict_binary,
                        plan_benchmark, plan_binary_benchmark, plan_prediction,
                        plan_binary_prediction)
from groupbench.data import prepare_data
from groupbench.binary.data import prepare_binary_data


def frame():
    return pd.DataFrame({"lot": np.repeat(list("abcdef"), 4),
                         "x": np.arange(24), "y": np.arange(24) * 2,
                         "xy": np.arange(24) * 3, "outcome": [0, 1] * 12})


class APIFeatureAllowlistTests(unittest.TestCase):
    def test_bare_strings_cannot_expand_to_character_columns(self):
        source = frame()
        before = source.copy(deep=True)
        for prepare in (prepare_data, prepare_binary_data):
            for features in ("xy", np.str_("xy"), "x", "", b"xy", b""):
                with self.subTest(prepare=prepare.__name__, features=repr(features)):
                    with self.assertRaisesRegex(ValueError, "not a bare string or bytes"):
                        prepare(source, "outcome", "lot", features=features)
        pd.testing.assert_frame_equal(source, before)

    def test_public_workflows_reject_before_scheduling_or_model_work(self):
        workflows = ((benchmark, False), (fit_predict, True), (plan_benchmark, False),
                     (plan_prediction, True), (benchmark_binary, False),
                     (fit_predict_binary, True), (plan_binary_benchmark, False),
                     (plan_binary_prediction, True))
        source = frame()
        for workflow, prediction in workflows:
            with self.subTest(workflow=workflow.__name__), ExitStack() as stack:
                blocked = [stack.enter_context(patch(name, side_effect=AssertionError("unexpected model work")))
                           for name in ("groupbench.model.make_pipeline", "groupbench.binary.model.binary_pipeline",
                                        "groupbench.splitting.pd.factorize", "groupbench.report.write_payloads")]
                options = dict(target="outcome", group="lot", features="xy", inner_splits=2,
                               baselines=("dummy",))
                with self.assertRaisesRegex(ValueError, "one-item list"):
                    if prediction:
                        workflow(source, pd.DataFrame({"xy": [3.]}), **options)
                    else:
                        workflow(source, outer_splits=3, **options)
                for operation in blocked:
                    operation.assert_not_called()

    def test_real_name_collections_preserve_selection_order_and_defaults(self):
        source = frame()
        for prepare in (prepare_data, prepare_binary_data):
            for collection in (["xy"], ("xy",), pd.Index(["xy"]), iter(["xy"]), ["y", "x"]):
                expected = ["y", "x"] if isinstance(collection, list) and collection == ["y", "x"] else ["xy"]
                with self.subTest(prepare=prepare.__name__, collection=type(collection).__name__):
                    X, _, _ = prepare(source, "outcome", "lot", features=collection)
                    self.assertEqual(list(X), expected)
            X, _, _ = prepare(source, "outcome", "lot")
            self.assertEqual(list(X), ["x", "y", "xy"])


if __name__ == "__main__":
    unittest.main()
