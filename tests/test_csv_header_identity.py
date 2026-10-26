"""The type-inference parser must preserve the already validated raw schema."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.cli import _read_csv_with_hash, main


class CSVHeaderIdentityTests(unittest.TestCase):
    def test_second_bom_cannot_rename_or_merge_raw_column_names(self):
        for header in ("\ufeff\ufeffg,x,y", "\ufeff\ufeffg,g,y", "\ufeff\ufeffg,g.1,g,y"):
            with self.subTest(header=repr(header)), tempfile.TemporaryDirectory() as temp:
                path = Path(temp, "input.csv")
                width = len(next(csv.reader(io.StringIO(header))))
                path.write_text(header + "\n" + ",".join(["PRIVATE"] * width) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "column names changed during parsing") as error:
                    _read_csv_with_hash(path, group="g")
                self.assertNotIn("PRIVATE", str(error.exception))

    def test_ordinary_bom_unicode_and_quoted_literal_bom_headers_are_preserved(self):
        for header in ("g,x,y", "\ufeffg,x,y", '"\ufeffg",g,y', '"批次\r\n甲",g,y', "\ufeffg,g.1,y"):
            with self.subTest(header=repr(header)), tempfile.TemporaryDirectory() as temp:
                path = Path(temp, "input.csv")
                raw = (header + "\r\n001,002,0\r\n").encode("utf-8")
                path.write_bytes(raw)
                expected = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
                frame, digest = _read_csv_with_hash(path, group="g")
                self.assertEqual(list(frame.columns), expected)
                self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
                self.assertEqual(frame.g.iloc[0], "001" if expected[0] == "g" else "002")

    def test_all_training_commands_reject_schema_changes_before_work(self):
        actions = ("groupbench.cli.benchmark", "groupbench.cli.fit_predict",
                   "groupbench.cli.plan_benchmark", "groupbench.cli.plan_prediction",
                   "groupbench.binary.cli.benchmark_binary", "groupbench.binary.cli.fit_predict_binary",
                   "groupbench.binary.cli.plan_binary_benchmark", "groupbench.binary.cli.plan_binary_prediction")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, incoming, out = root / "train.csv", root / "new.csv", root / "out"
            # Raw second column is g, but pandas would promote the first one
            # to g and rename the intended group column to g.1.
            train.write_text("\ufeff\ufeffg,g,y\nwrong-group,intended-group,0\n", encoding="utf-8")
            incoming.write_text("x\n1\n", encoding="utf-8")
            for binary in (False, True):
                for planning in (False, True):
                    for mode in ("run", "predict"):
                        with self.subTest(binary=binary, planning=planning, mode=mode), ExitStack() as stack:
                            operations = [stack.enter_context(patch(action)) for action in actions]
                            command = (["binary"] if binary else []) + (["plan"] if planning else []) + [mode, str(train)]
                            if mode == "predict":
                                command.append(str(incoming))
                            command += ["--target", "y", "--group", "g"]
                            if not planning:
                                command += ["--out", str(out)]
                            with redirect_stderr(io.StringIO()) as error, redirect_stdout(io.StringIO()) as output:
                                self.assertEqual(main(command), 2)
                            self.assertIn("column names changed during parsing", error.getvalue())
                            self.assertEqual(output.getvalue(), "")
                            self.assertFalse(out.exists())
                            for operation in operations:
                                operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
