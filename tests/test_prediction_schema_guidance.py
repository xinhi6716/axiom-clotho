"""Repair guidance names schema differences, never cells, before any fitting."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.cli import main
from groupbench.data import validate_prediction_features


class PredictionSchemaGuidanceTests(unittest.TestCase):
    def test_missing_and_unexpected_names_are_ordered_and_exact(self):
        frame = pd.DataFrame({"pressure_MPa": [2.0], "notes": ["CELL_NOT_FOR_ERROR"],
                              "temperature_c": [510.0]})
        before = frame.copy(deep=True)
        with self.assertRaises(ValueError) as raised:
            validate_prediction_features(frame, ["pressure_mpa", "temperature_c", "duration_min"],
                                         ["float64"] * 3)
        message = str(raised.exception)
        self.assertIn("Missing columns (2): 'pressure_mpa', 'duration_min'.", message)
        self.assertIn("Unexpected columns (2): 'pressure_MPa', 'notes'.", message)
        self.assertIn("including case and spaces", message)
        self.assertIn("No columns were renamed or dropped", message)
        self.assertNotIn("CELL_NOT_FOR_ERROR", message)
        pd.testing.assert_frame_equal(frame, before)

    def test_names_are_bounded_and_terminal_controls_are_escaped(self):
        hostile = "line\n\t\x1b[31m\r" + "x" * 3000
        columns = [hostile] + [f"extra_{i}" for i in range(9)]
        frame = pd.DataFrame([["CELL_NOT_FOR_ERROR"] * len(columns)], columns=columns)
        expected = [f"wanted_{i}" for i in range(10)]
        with self.assertRaises(ValueError) as raised:
            validate_prediction_features(frame, expected, ["object"] * len(expected))
        message = str(raised.exception)
        self.assertIn("Missing columns (10)", message)
        self.assertIn("Unexpected columns (10)", message)
        self.assertEqual(message.count("(+5 more)"), 2)
        self.assertIn("\\n\\t\\x1b[31m\\r", message)
        for raw in ("\n", "\t", "\x1b", "\r", "CELL_NOT_FOR_ERROR", "extra_4", "wanted_5"):
            self.assertNotIn(raw, message)
        self.assertLess(len(message), 650)

    def test_unicode_spaces_and_one_sided_differences(self):
        cases = [
            ({"溫度": [1.0]}, ["溫度", "pressure_mpa"], "Missing columns (1): 'pressure_mpa'.", "Unexpected columns (0): none."),
            ({"溫度": [1.0], " notes ": [2.0]}, ["溫度"], "Missing columns (0): none.", "Unexpected columns (1): ' notes '."),
        ]
        for values, expected, missing, unexpected in cases:
            with self.subTest(expected=expected), self.assertRaises(ValueError) as raised:
                validate_prediction_features(pd.DataFrame(values), expected, ["float64"] * len(expected))
            self.assertIn(missing, str(raised.exception))
            self.assertIn(unexpected, str(raised.exception))

    def test_valid_reordering_and_all_missing_categorical_still_work(self):
        frame = pd.DataFrame({"material": [np.nan], "pressure_mpa": [2]})
        normalized = validate_prediction_features(frame, ["pressure_mpa", "material"], ["float64", "object"])
        self.assertEqual(list(normalized.columns), ["pressure_mpa", "material"])
        self.assertEqual([str(t) for t in normalized.dtypes], ["float64", "object"])
        self.assertEqual(list(frame.columns), ["material", "pressure_mpa"])


class PredictionSchemaCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, self.new = self.root / "train.csv", self.root / "new.csv"
        self.train.write_text("g,pressure_mpa,temperature_c,y\n"
                              "a,1,400,0\na,2,500,1\nb,3,600,0\nb,4,700,1\nc,5,800,0\nc,6,900,1\n")

    def arguments(self, binary, planning):
        command = (["binary"] if binary else []) + (["plan"] if planning else []) + ["predict"]
        command += [str(self.train), str(self.new), "--target", "y", "--group", "g",
                    "--inner-splits", "2", "--baselines", "dummy"]
        if not planning:
            command += ["--out", str(self.root / "output")]
        return command

    def invoke(self, binary, planning, *, no_fit=True):
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            if no_fit:
                for path in ("groupbench.model.GroupBenchRegressor._fit_prepared",
                             "groupbench.binary.model.GroupBenchClassifier._fit_prepared"):
                    stack.enter_context(patch(path, side_effect=AssertionError("unexpected fit")))
            code = main(self.arguments(binary, planning))
        return code, out.getvalue(), err.getvalue()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_each_route_rejects_repeated_mismatches_without_fitting_or_writes(self):
        cases = [
            ("temperature_c\n510\n", "Missing columns (1): 'pressure_mpa'.", "Unexpected columns (0): none."),
            ("pressure_mpa,temperature_c,notes\n2,510,CELL_NOT_FOR_ERROR\n", "Missing columns (0): none.", "Unexpected columns (1): 'notes'."),
            ("pressure_MPa,temperature_c,notes\n2,510,CELL_NOT_FOR_ERROR\n", "Missing columns (1): 'pressure_mpa'.", "Unexpected columns (2): 'pressure_MPa', 'notes'."),
        ]
        for content, missing, unexpected in cases:
            self.new.write_text(content)
            before = self.snapshot()
            for binary in (False, True):
                for planning in (False, True):
                    with self.subTest(binary=binary, planning=planning, content=content):
                        first = self.invoke(binary, planning)
                        self.assertEqual(first, self.invoke(binary, planning))
                        code, out, err = first
                        self.assertEqual((code, out), (2, ""))
                        self.assertIn(missing, err)
                        self.assertIn(unexpected, err)
                        for absent in ("CELL_NOT_FOR_ERROR", str(self.root), "Traceback"):
                            self.assertNotIn(absent, err)
                        self.assertEqual(before, self.snapshot())
                        self.assertFalse((self.root / "output").exists())

    def test_duplicate_headers_still_fail_before_schema_matching(self):
        self.new.write_text("pressure_mpa,pressure_mpa,temperature_c\n2,2,510\n")
        before = self.snapshot()
        for binary in (False, True):
            for planning in (False, True):
                with self.subTest(binary=binary, planning=planning):
                    code, out, err = self.invoke(binary, planning)
                    self.assertEqual((code, out), (2, ""))
                    self.assertIn("unique column names", err)
                    self.assertNotIn("Missing columns", err)
                    self.assertEqual(before, self.snapshot())
                    self.assertFalse((self.root / "output").exists())

    def test_corrected_input_passes_no_fit_plan_then_prediction_for_both_tasks(self):
        for binary in (False, True):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as result_temp:
                self.new.write_text("pressure_MPa,temperature_c,notes\n2,510,CELL_NOT_FOR_ERROR\n")
                self.assertEqual(self.invoke(binary, True)[0], 2)
                # A deliberate export repair, not automatic renaming or dropping.
                self.new.write_text("temperature_c,pressure_mpa\n510,2\n")
                before = self.snapshot()
                code, out, err = self.invoke(binary, True)
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(json.loads(out)["fits_performed"], 0)
                self.assertEqual([c["name"] for c in json.loads(out)["feature_schema"]],
                                 ["pressure_mpa", "temperature_c"])
                self.assertEqual(before, self.snapshot())
                command = self.arguments(binary, False)
                command[-1] = str(Path(result_temp) / "prediction")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                    code = main(command)
                self.assertEqual(code, 0, err.getvalue())
                saved = Path(command[-1])
                self.assertEqual({p.name for p in saved.iterdir()}, {"manifest.json", "predictions.csv", "checksums.json"})
                self.assertEqual(len(pd.read_csv(saved / "predictions.csv")), 1)


if __name__ == "__main__":
    unittest.main()
