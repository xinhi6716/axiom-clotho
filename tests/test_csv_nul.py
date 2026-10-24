"""Reject CSV NUL corruption before inference, selection, fitting or output."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench.cli import _read_csv, _read_csv_with_hash, main
from groupbench.data import prepare_data


class CSVNULTests(unittest.TestCase):
    def assert_rejected(self, content, physical_line, *, chunk_size=None):
        class Source(io.BytesIO):
            def readinto(self, buffer):
                if chunk_size is not None:
                    buffer = memoryview(buffer)[:chunk_size]
                return super().readinto(buffer)
        source = Source(content)
        with patch.object(Path, "open", return_value=source) as opening, \
                patch("groupbench.cli.pd.read_csv") as parsing:
            with self.assertRaises(ValueError) as error:
                _read_csv_with_hash(Path("PRIVATE_FILENAME.csv"), group="batch")
        self.assertEqual(str(error.exception),
                         f"CSV contains an unsupported NUL character at physical line {physical_line}. "
                         "Re-export the file as a clean UTF-8 CSV.")
        self.assertNotIn("PRIVATE", str(error.exception))
        self.assertTrue(source.closed)
        opening.assert_called_once_with("rb")
        parsing.assert_not_called()

    def test_nul_is_rejected_in_headers_and_every_field_role(self):
        cases = [
            (b"batch,x\0PRIVATE,y\ng,1,0\n", 1),
            (b'"batch\0PRIVATE",x,y\ng,1,0\n', 1),
            (b"batch,x,y\ng\0PRIVATE,1,0\n", 2),
            (b"batch,x,y\ng,1\0PRIVATE,0\n", 2),
            (b"batch,x,y\ng,1,0\0PRIVATE\n", 2),
            (b'batch,x,y\ng,"A\0PRIVATE",0\n', 2),
            (b"batch,x,y,excluded\ng,1,0,\0PRIVATE\n", 2),
            (b"\0", 1),
            (b"batch,x,y\ng,1,0\n \t\0PRIVATE\n", 3),
            (b"batch,x,y\ng,1,0\0", 2),
        ]
        for content, line in cases:
            with self.subTest(content=content):
                self.assert_rejected(content, line)

    def test_physical_lines_include_quoted_newlines_and_blank_lines(self):
        for ending in (b"\n", b"\r\n", b"\r"):
            for bom in (b"", b"\xef\xbb\xbf"):
                with self.subTest(ending=ending, bom=bom):
                    content = bom + ending.join((b"batch,x,y", b"", b'"group', b'\0PRIVATE",1,0'))
                    self.assert_rejected(content, 4)
                    header = bom + ending.join((b'"batch', b'\0PRIVATE",x,y', b"g,1,0"))
                    self.assert_rejected(header, 2)

    def test_fragmented_utf8_stream_rejects_nul_and_closes_source(self):
        content = '\ufeffbatch,x,y\r\n"測試\r\n組\x00PRIVATE",1,0\r\n'.encode("utf-8")
        for chunk_size in (1, 2, 7):
            with self.subTest(chunk_size=chunk_size):
                self.assert_rejected(content, 3, chunk_size=chunk_size)

    def test_read_csv_wrapper_also_rejects_nul(self):
        with patch.object(Path, "open", return_value=io.BytesIO(b"batch,x,y\ng,1,0\0PRIVATE\n")), \
                patch("groupbench.cli.pd.read_csv") as parsing:
            with self.assertRaisesRegex(ValueError, "NUL character at physical line 2"):
                _read_csv(Path("synthetic.csv"), group="batch")
            parsing.assert_not_called()

    def test_valid_literal_escapes_unicode_and_hash_are_unchanged(self):
        content = '\ufeffbatch,x,y\r\n001,"literal \\0",0\r\nNA,"literal \\x00",1\r\n"測試\r\n組","␀",0\r\ng,"\v\f",1\r\n\r\n \t\r\n'.encode("utf-8")
        with patch.object(Path, "open", return_value=io.BytesIO(content)):
            frame, digest = _read_csv_with_hash(Path("synthetic.csv"), group="batch")
        self.assertEqual(frame.batch.tolist(), ["001", "NA", "測試\r\n組", "g"])
        self.assertEqual(frame.x.tolist(), ["literal \\0", "literal \\x00", "␀", "\v\f"])
        self.assertEqual(frame.y.tolist(), [0, 1, 0, 1])
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())

    def test_dataframe_api_is_not_silently_sanitized(self):
        # This fix guards CSV decoding, not arbitrary in-memory string content.
        frame = pd.DataFrame({"batch": ["g\0a", "g\0b"], "x": ["A\0B", "A\0C"], "y": [0, 1]})
        X, _, groups = prepare_data(frame, "y", "batch")
        self.assertEqual(X.x.tolist(), ["A\0B", "A\0C"])
        self.assertEqual(groups.tolist(), ["g\0a", "g\0b"])


class CSVNULCLITests(unittest.TestCase):
    def test_all_cli_training_and_new_row_paths_fail_without_work_or_writes(self):
        valid_training = b"batch,x,y\n" + b"".join(
            f"g{group},{label + 1},{label}\n".encode() for group in range(6) for label in (0, 1))
        cases = []
        for binary in (False, True):
            for planning in (False, True):
                for mode in ("run", "predict"):
                    for invalid in (("training", "new") if mode == "predict" else ("training",)):
                        cases.append((binary, planning, mode, invalid))
        actions = ("groupbench.cli.benchmark", "groupbench.cli.fit_predict",
                   "groupbench.cli.plan_benchmark", "groupbench.cli.plan_prediction",
                   "groupbench.binary.cli.benchmark_binary", "groupbench.binary.cli.fit_predict_binary",
                   "groupbench.binary.cli.plan_binary_benchmark", "groupbench.binary.cli.plan_binary_prediction",
                   "groupbench.report.write_payloads", "groupbench.model.Pipeline.fit")
        real_read_csv = pd.read_csv
        for binary, planning, mode, invalid in cases:
            with self.subTest(binary=binary, planning=planning, mode=mode, invalid=invalid), \
                    tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                train, new, out = root / "train.csv", root / "new.csv", root / "output"
                train.write_bytes(valid_training if invalid == "new" else
                                  b"batch,x,y\ng,1,0\0PRIVATE\n")
                new.write_bytes(b"x\n1\0PRIVATE\n" if invalid == "new" else b"x\n1\n")
                (root / "manifest.json").write_bytes(b"PRESERVE EXISTING FILE")
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                args = (["binary"] if binary else []) + (["plan"] if planning else []) + [mode, str(train)]
                if mode == "predict":
                    args.append(str(new))
                args += ["--target", "y", "--group", "batch", "--features", "x", "--inner-splits", "2"]
                if not planning:
                    args += ["--out", str(out)]
                stdout, stderr = io.StringIO(), io.StringIO()
                with ExitStack() as stack, redirect_stdout(stdout), redirect_stderr(stderr):
                    for action in actions:
                        stack.enter_context(patch(action, side_effect=AssertionError(f"Unexpected work: {action}")))
                    parsing = stack.enter_context(patch("groupbench.cli.pd.read_csv", wraps=real_read_csv))
                    status = main(args)
                self.assertEqual(status, 2, stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("NUL character at physical line 2", stderr.getvalue())
                self.assertNotIn("PRIVATE", stderr.getvalue())
                # A valid training CSV is read first for bad prediction rows;
                # the offending stream must never reach pandas in either case.
                self.assertEqual(parsing.call_count, 1 if invalid == "new" else 0)
                self.assertFalse(out.exists())
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
