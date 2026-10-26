"""Explicit empty allowlists must never silently select every predictor."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.cli import main


class FeatureAllowlistCLITests(unittest.TestCase):
    def test_empty_allowlists_fail_before_planning_fitting_or_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, incoming, out = root / "train.csv", root / "new.csv", root / "out"
            train.write_text("lot,x,post_outcome,y\na,1,99,0\nb,2,98,1\n", encoding="utf-8")
            incoming.write_text("x\n3\n", encoding="utf-8")
            actions = (
                ([], "run", "groupbench.cli.benchmark"),
                ([], "predict", "groupbench.cli.fit_predict"),
                (["plan"], "run", "groupbench.cli.plan_benchmark"),
                (["plan"], "predict", "groupbench.cli.plan_prediction"),
                (["binary"], "run", "groupbench.binary.cli.benchmark_binary"),
                (["binary"], "predict", "groupbench.binary.cli.fit_predict_binary"),
                (["binary", "plan"], "run", "groupbench.binary.cli.plan_binary_benchmark"),
                (["binary", "plan"], "predict", "groupbench.binary.cli.plan_binary_prediction"),
            )
            for prefix, mode, action in actions:
                for value in ("", " ", ",", "x,", ",x", "x, ,post_outcome"):
                    with self.subTest(prefix=prefix, mode=mode, value=value):
                        command = [*prefix, mode, str(train)]
                        if mode == "predict":
                            command.append(str(incoming))
                        command += ["--target", "y", "--group", "lot", "--features", value]
                        if "plan" not in prefix:
                            command += ["--out", str(out)]
                        with patch(action) as operation, redirect_stderr(io.StringIO()) as error, redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(main(command), 2)
                        operation.assert_not_called()
                        self.assertIn("--features requires non-empty", error.getvalue())
                        self.assertNotIn("Traceback", error.getvalue())
                        self.assertEqual(output.getvalue(), "")
                        self.assertFalse(out.exists())

    def test_omitted_and_valid_allowlists_keep_their_original_meanings(self):
        from argparse import Namespace
        from groupbench.cli import _read_training
        from groupbench.data import prepare_data

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "train.csv")
            source.write_text("lot,x,post_outcome,y\na,1,99,0\nb,2,98,1\n", encoding="utf-8")
            for value, expected in ((None, ["x", "post_outcome"]), ("x", ["x"]), (" post_outcome , x ", ["post_outcome", "x"])):
                with self.subTest(value=value):
                    frame, features, _ = _read_training(Namespace(csv_path=source, group="lot", features_file=None, features=value))
                    X, _, _ = prepare_data(frame, "y", "lot", features)
                    self.assertEqual(list(X), expected)


if __name__ == "__main__":
    unittest.main()
