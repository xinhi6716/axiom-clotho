"""Bind CLI source provenance to the bytes actually validated and parsed."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import make_materials_fixture
from groupbench.cli import _read_csv, _read_csv_with_hash, main
from groupbench.data import data_fingerprint, prepare_data, validate_features


class TrackedBytes(io.BytesIO):
    """Forbid whole-file reads and record incremental decoder requests."""

    def __init__(self, content, *, chunk_size=None):
        super().__init__(content)
        self.requests = []
        self.bytes_read = 0
        self.chunk_size = chunk_size

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("Unbounded reads are not allowed.")
        return super().read(size)

    def readinto(self, buffer):
        self.requests.append(len(buffer))
        if self.chunk_size is not None:
            buffer = memoryview(buffer)[:self.chunk_size]
        count = super().readinto(buffer)
        self.bytes_read += count
        return count


class CSVProvenanceTests(unittest.TestCase):
    def test_hash_includes_exact_bom_line_endings_quotes_and_ignored_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "input.csv")
            hashes = set()
            for ending, bom, trailing in (
                (b"\n", b"", b""),
                (b"\r\n", b"\xef\xbb\xbf", b"\r\n \t\r\n"),
                (b"\r", b"", b"\r"),
            ):
                content = bom + ending.join((b"batch,x,y", b'"001,alpha",1,2', b'"NA",3,4')) + trailing
                with self.subTest(ending=ending):
                    path.write_bytes(content)
                    frame, digest = _read_csv_with_hash(path, group="batch")
                    self.assertEqual(digest, hashlib.sha256(content).hexdigest())
                    self.assertEqual(frame.batch.tolist(), ["001,alpha", "NA"])
                    self.assertEqual(frame.x.tolist(), [1, 3])
                    pd.testing.assert_frame_equal(frame, _read_csv(path, group="batch"))
                    hashes.add(digest)
            self.assertEqual(len(hashes), 3)

    def test_header_only_input_hash_includes_trailing_blank_lines(self):
        content = b"\xef\xbb\xbfbatch,x,y\r\n\r\n \r\n"
        source = TrackedBytes(content)
        with patch("groupbench.cli._open_csv_source", return_value=source) as opening:
            frame, digest = _read_csv_with_hash(Path("synthetic.csv"), group="batch")
        self.assertEqual(len(frame), 0)
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        opening.assert_called_once_with(Path("synthetic.csv"))

    def test_incremental_hash_handles_split_bom_and_multibyte_characters(self):
        content = '\ufeffbatch,x,y\r\n"測試,é\r\n組",1,2\r\n\r\n'.encode("utf-8")
        source = TrackedBytes(content, chunk_size=2)
        with patch("groupbench.cli._open_csv_source", return_value=source) as opening:
            frame, digest = _read_csv_with_hash(Path("synthetic.csv"), group="batch")
        self.assertEqual(frame.batch.tolist(), ["測試,é\r\n組"])
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertEqual(source.bytes_read, len(content))
        self.assertGreater(len(source.requests), 2)
        self.assertTrue(all(0 < request <= 64 * 1024 for request in source.requests))
        self.assertTrue(source.closed)
        opening.assert_called_once_with(Path("synthetic.csv"))

    def test_oversized_csv_is_not_fully_read_or_hashed(self):
        content = b"batch,x,y\n" + b"001,1,2\n" * 100_000
        source = TrackedBytes(content)
        with patch("groupbench.cli._open_csv_source", return_value=source) as opening, patch("groupbench.cli.MAX_ROWS", 2):
            with self.assertRaisesRegex(ValueError, "exceeds the MVP limit of 2 rows"):
                _read_csv_with_hash(Path("synthetic.csv"), group="batch")
        self.assertLess(source.bytes_read, len(content))
        self.assertTrue(all(0 < request <= 64 * 1024 for request in source.requests))
        self.assertTrue(source.closed)
        opening.assert_called_once_with(Path("synthetic.csv"))

    def test_midstream_rewrite_hashes_observed_bytes_not_a_claimed_atomic_version(self):
        original = b"batch,x,y\n001,1,2\n002,3,4\n"
        replacement = b"batch,x,y\n111,7,8\n222,9,6\n"
        boundary = len(b"batch,x,y\n001,1,2\n")
        delivered = bytearray()

        class MutatingBytes(TrackedBytes):
            def readinto(self, buffer):
                count = super().readinto(buffer)
                delivered.extend(memoryview(buffer)[:count])
                if len(self.requests) == 1:
                    # Deterministic in-place mutation after the first chunk,
                    # without threads or timing-dependent filesystem races.
                    position = self.tell()
                    self.seek(0)
                    self.write(replacement)
                    self.seek(position)
                return count

        source = MutatingBytes(original, chunk_size=boundary)
        with patch("groupbench.cli._open_csv_source", return_value=source):
            frame, digest = _read_csv_with_hash(Path("synthetic.csv"), group="batch")
        self.assertEqual(bytes(delivered), original[:boundary] + replacement[boundary:])
        self.assertEqual(frame.batch.tolist(), ["001", "222"])
        self.assertEqual(frame.y.tolist(), [2, 6])
        self.assertEqual(digest, hashlib.sha256(delivered).hexdigest())
        self.assertNotEqual(digest, hashlib.sha256(original).hexdigest())
        self.assertNotEqual(digest, hashlib.sha256(replacement).hexdigest())

    def test_rewrite_replace_or_delete_after_parsing_cannot_change_bound_hash(self):
        original = b"\xef\xbb\xbfbatch,x,y\r\n001,1,2\r\nNA,3,4\r\n\r\n"
        replacement = b"batch,x,y\nchanged,900,901\n"
        real_read_csv = pd.read_csv
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "input.csv")
            for action in ("rewrite", "replace", "delete"):
                with self.subTest(action=action):
                    path.write_bytes(original)
                    def mutate_after_parsing(*args, **kwargs):
                        frame = real_read_csv(*args, **kwargs)
                        if action == "rewrite":
                            path.write_bytes(replacement)
                        elif action == "replace":
                            other = Path(temp, "replacement.csv")
                            other.write_bytes(replacement)
                            other.replace(path)
                        else:
                            path.unlink()
                        return frame
                    with patch("groupbench.cli.pd.read_csv", side_effect=mutate_after_parsing):
                        frame, digest = _read_csv_with_hash(path, group="batch")
                    self.assertEqual(frame.batch.tolist(), ["001", "NA"])
                    self.assertEqual(frame.y.tolist(), [2, 4])
                    self.assertEqual(digest, hashlib.sha256(original).hexdigest())
                    self.assertNotEqual(digest, hashlib.sha256(replacement).hexdigest())

    def test_run_manifest_binds_replaced_source_to_consumed_training_data(self):
        training = make_materials_fixture(6, 2, 7)
        original = b"\xef\xbb\xbf" + training.to_csv(index=False, lineterminator="\r\n").encode("utf-8") + b"\r\n"
        replacement = make_materials_fixture(6, 2, 19).to_csv(index=False).encode("utf-8")
        parsed = pd.read_csv(io.BytesIO(original), converters={"batch": str})
        expected_hash = data_fingerprint(*prepare_data(parsed, "response", "batch"))
        real_read_csv = pd.read_csv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, other, out = root / "train.csv", root / "replacement.csv", root / "out"
            source.write_bytes(original)
            other.write_bytes(replacement)
            def swap_after_parsing(*args, **kwargs):
                frame = real_read_csv(*args, **kwargs)
                other.replace(source)
                return frame
            with patch("groupbench.cli.pd.read_csv", side_effect=swap_after_parsing), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = main(["run", str(source), "--target", "response", "--group", "batch", "--out", str(out), "--outer-splits", "3", "--inner-splits", "2", "--baselines", "dummy"])
            self.assertEqual(code, 0, err.getvalue())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["source_file_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(manifest["data_sha256"], expected_hash)
            self.assertEqual(source.read_bytes(), replacement)
            observations = pd.read_csv(out / "predictions.csv").observed
            np.testing.assert_allclose(observations, parsed.response)

    def test_predict_manifest_binds_both_changed_csv_sources(self):
        training = make_materials_fixture(6, 2, 7)
        new_rows = make_materials_fixture(2, 2, 19).drop(columns=["response", "batch"])
        original_train = training.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        original_new = b"\xef\xbb\xbf" + new_rows.to_csv(index=False, lineterminator="\r\n").encode("utf-8") + b"\r\n"
        parsed_train = pd.read_csv(io.BytesIO(original_train), converters={"batch": str})
        parsed_new = validate_features(pd.read_csv(io.BytesIO(original_new)))
        expected_train_hash = data_fingerprint(*prepare_data(parsed_train, "response", "batch"))
        expected_new_hash = hashlib.sha256(pd.util.hash_pandas_object(parsed_new, index=False).values.tobytes()).hexdigest()
        real_read_csv = pd.read_csv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, new, other, out = root / "train.csv", root / "new.csv", root / "replacement.csv", root / "out"
            train.write_bytes(original_train)
            new.write_bytes(original_new)
            other.write_bytes(b"replacement data is deliberately not a CSV\n")
            reads = []
            def mutate_each_source_after_parsing(*args, **kwargs):
                frame = real_read_csv(*args, **kwargs)
                reads.append(frame)
                if len(reads) == 1:
                    train.write_bytes(b"changed training input\n")
                else:
                    other.replace(new)
                return frame
            with patch("groupbench.cli.pd.read_csv", side_effect=mutate_each_source_after_parsing), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = main(["predict", str(train), str(new), "--target", "response", "--group", "batch", "--out", str(out), "--inner-splits", "2", "--baselines", "dummy"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(len(reads), 2)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["source_file_sha256"], hashlib.sha256(original_train).hexdigest())
            self.assertEqual(manifest["prediction_source_file_sha256"], hashlib.sha256(original_new).hexdigest())
            self.assertEqual(manifest["training_data_sha256"], expected_train_hash)
            self.assertEqual(manifest["new_features_sha256"], expected_new_hash)
            predictions = pd.read_csv(out / "predictions.csv")
            self.assertEqual(len(predictions), len(new_rows))
            np.testing.assert_allclose(predictions.prediction, parsed_train.response.mean())


if __name__ == "__main__":
    unittest.main()
