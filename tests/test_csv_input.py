"""Strict, bounded CSV structure checks before inference or model fitting."""

from contextlib import redirect_stderr
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench.cli import _read_csv, main


class CSVInputTests(unittest.TestCase):
    def read_text(self, text, *, group="batch"):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "input.csv")
            path.write_text(text, encoding="utf-8", newline="")
            return _read_csv(path, group=group)

    def test_consistently_overfull_records_cannot_shift_columns(self):
        # pandas alone promotes a/b to the index and labels 1/4 as the batch.
        with self.assertRaisesRegex(ValueError, "physical line 2 has 4 fields; the header defines 3"):
            self.read_text("batch,x,y\na,1,2,3\nb,4,5,6\n")

    def test_short_and_later_overfull_records_are_rejected(self):
        for text, line, fields in (
            ("batch,x,y\na,1\n", 2, 2),
            ("batch,x,y\na,1,2\nb,3,4,5\n", 3, 4),
            ('batch,x,y\n"two\nlines",1\n', 3, 2),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(
                ValueError, f"physical line {line} has {fields} fields; the header defines 3"
            ):
                self.read_text(text)

    def test_explicit_empty_fields_remain_missing_not_short_records(self):
        frame = self.read_text("batch,x,y\n001,,2\n1,3,\nNA,4,5\n")
        self.assertEqual(frame.batch.tolist(), ["001", "1", "NA"])
        self.assertTrue(pd.isna(frame.loc[0, "x"]))
        self.assertTrue(pd.isna(frame.loc[1, "y"]))
        self.assertEqual(frame.index.tolist(), [0, 1, 2])

    def test_quotes_commas_newlines_bom_and_identifier_text_are_preserved(self):
        text = '\ufeffbatch,x,y\r\n"001,alpha",1,2\r\n"line1\r\n\r\nline2",3,4\r\n"say ""hi""",5,6\r\n" 001 ",7,8\r\n'
        frame = self.read_text(text)
        self.assertEqual(frame.columns.tolist(), ["batch", "x", "y"])
        self.assertEqual(frame.batch.tolist(), ["001,alpha", "line1\r\n\r\nline2", 'say "hi"', " 001 "])
        self.assertEqual(frame.x.tolist(), [1, 3, 5, 7])

    def test_blank_lines_are_ignored_but_quoted_empty_records_survive(self):
        frame = self.read_text("batch,x,y\na,1,2\n\n  \t\n\nb,3,4\n\n \n")
        self.assertEqual(frame.batch.tolist(), ["a", "b"])
        one_column = self.read_text('batch\n""\n" "\n\n \n')
        self.assertEqual(one_column.batch.tolist(), ["", " "])

    def test_non_blank_unicode_and_control_whitespace_is_not_dropped(self):
        # pandas ignores ASCII space/tab-only lines, not every str.isspace().
        frame = self.read_text("x\nleft\n\u00a0\n\u2003\n\v\n\f\nright\n", group=None)
        self.assertEqual(frame.x.tolist(), ["left", "\u00a0", "\u2003", "\v", "\f", "right"])

    def test_empty_duplicate_and_blank_headers_are_rejected(self):
        for text in ("", "\n", "batch,x,x\na,1,2\n", "batch,,y\na,1,2\n", "batch,  ,y\na,1,2\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "unique column names"):
                self.read_text(text)

    def test_header_only_csv_remains_empty(self):
        frame = self.read_text("batch,x,y\n")
        self.assertEqual(frame.columns.tolist(), ["batch", "x", "y"])
        self.assertEqual(len(frame), 0)

    def test_malformed_quotes_have_clean_diagnostics_without_cell_values(self):
        for text in ('batch,x,y\n"PRIVATE CELL,1,2\n', 'batch,x,y\n"PRIVATE CELL"bad,1,2\n'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "could not be parsed near physical line") as error:
                self.read_text(text)
            self.assertNotIn("PRIVATE", str(error.exception))

    def test_field_parser_limit_is_reported_and_not_mutated(self):
        limit = csv.field_size_limit()
        with self.assertRaisesRegex(ValueError, f"field-size limit \\({limit:,} characters\\)"):
            self.read_text("batch,x,y\na," + "x" * (limit + 1) + ",2\n")
        self.assertEqual(csv.field_size_limit(), limit)
        frame = self.read_text("batch,x,y\na," + "x" * limit + ",2\n")
        self.assertEqual(len(frame.loc[0, "x"]), limit)

    def test_row_limit_counts_records_not_physical_lines(self):
        with patch("groupbench.cli.MAX_ROWS", 2):
            frame = self.read_text('batch,x,y\n"a\n\na",1,2\n\n\nb,3,4\n\n')
            self.assertEqual(len(frame), 2)
            with self.assertRaisesRegex(ValueError, "exceeds the MVP limit of 2 rows"):
                # A later malformed record must not trigger an unbounded scan.
                self.read_text('batch,x,y\na,1,2\nb,3,4\nc,5,6\n"unclosed')

    def test_invalid_utf8_reports_required_encoding_without_input_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "private-source.csv")
            path.write_bytes(b"batch,x,y\nprivate,\xff,2\n")
            with self.assertRaisesRegex(ValueError, "must use UTF-8 encoding") as error:
                _read_csv(path, group="batch")
            self.assertNotIn("private", str(error.exception))

    def test_invalid_training_and_prediction_csv_fail_before_fitting_or_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, new, out = root / "train.csv", root / "new.csv", root / "out"
            train.write_text("batch,x,y\na,1,2,3\n")
            options = ["--target", "y", "--group", "batch", "--out", str(out)]
            with patch("groupbench.cli.benchmark") as fitting, redirect_stderr(io.StringIO()) as err:
                code = main(["run", str(train), *options])
                fitting.assert_not_called()
            self.assertEqual(code, 2)
            self.assertIn("physical line 2", err.getvalue())
            self.assertFalse(out.exists())
            train.write_text("batch,x,y\na,1,2\nb,3,4\n")
            new.write_text("x\n1,2\n")
            with patch("groupbench.cli.fit_predict") as fitting, redirect_stderr(io.StringIO()) as err:
                code = main(["predict", str(train), str(new), *options])
                fitting.assert_not_called()
            self.assertEqual(code, 2)
            self.assertIn("physical line 2", err.getvalue())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
