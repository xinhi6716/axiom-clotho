"""Exercise source-decimal rounding at the public CSV CLI boundary."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from groupbench.cli import _read_csv_with_hash, main


class CSVFloatPrecisionTests(unittest.TestCase):
    def test_cli_benchmark_does_not_make_adjacent_targets_constant(self):
        tokens = ("0.12345678901234566", "0.12345678901234568")
        expected = [float(token).hex() for token in tokens]
        self.assertNotEqual(*expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.csv", root / "evaluation"
            raw = ("g,x,y\n" + "".join(f"g{i//2},{i},{tokens[i%2]}\n" for i in range(12))).encode()
            source.write_bytes(raw)
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["run", str(source), "--target", "y", "--group", "g",
                             "--outer-splits", "3", "--inner-splits", "2",
                             "--baselines", "dummy", "--out", str(output)])
            self.assertEqual(code, 0, stderr.getvalue())
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as file:
                observed = [float(row["observed"]).hex() for row in csv.DictReader(file)]
            self.assertEqual(observed, expected * 6)
            scores = json.loads((output / "summary.json").read_bytes())["models"]["selected"]["overall"]
            self.assertGreater(scores["mae"], 0.0)
            self.assertGreater(scores["rmse"], 0.0)
            self.assertIsNotNone(scores["r2"])
            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(manifest["source_file_sha256"], hashlib.sha256(raw).hexdigest())

    def test_csv_reader_preserves_missing_text_integer_and_binary64_boundaries(self):
        tokens = ("-0.0", "5e-324", "1.0000000000000002")
        raw = ("\ufeffid,kind,flag,integer,x\r\n" +
               "\r\n".join(f"00{i},label-{i},{i%2==0},{i},{token}" for i, token in enumerate(tokens))
               + "\r\nNA,label-X,False,6,\r\n").encode()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            source.write_bytes(raw)
            frame, digest = _read_csv_with_hash(source, group="id")
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(frame.id.tolist(), ["000", "001", "002", "NA"])
        self.assertEqual(frame.kind.tolist(), ["label-0", "label-1", "label-2", "label-X"])
        self.assertEqual(frame.flag.dtype.kind, "b")
        self.assertEqual(frame.flag.tolist(), [True, False, True, False])
        self.assertEqual(frame.integer.dtype.kind, "i")
        self.assertEqual(frame.integer.tolist(), [0, 1, 2, 6])
        self.assertTrue(frame.x.isna().iloc[-1])
        self.assertEqual([v.hex() for v in frame.x.iloc[:-1]], [float(t).hex() for t in tokens])
