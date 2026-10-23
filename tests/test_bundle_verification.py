"""Synthetic checksum bundles: no fitting, network, or serialized model loads."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench.cli import main
from groupbench.report import write_payloads
import groupbench.verification as verification


@unittest.skipUnless(verification._SAFE_OPEN_SUPPORTED, "Requires POSIX no-follow descriptor access")
class BundleVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Use a resolved temporary root: /var or /tmp may themselves be links
        # on a test host. The verifier deliberately does not follow those.
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / "run"

    def make_bundle(self, kind="prediction"):
        manifest = {"schema_version": "1.0", "groupbench_version": "0.1.0",
                    "config": {"target": "response", "group": "batch"}}
        if kind == "prediction":
            manifest.update(mode="fit_predict", n_training_rows=12, n_predictions=2)
            csv = "row_position,prediction\n0,1.25\n1,2.5\n"
        else:
            manifest.update(protocol="nested_group_cv", n_rows=2, n_groups=2)
            csv = "row_position,group,observed,outer_fold,prediction_selected\n0,001,1.0,0,1.25\n1,材料,3.0,1,2.5\n"
        payloads = {"manifest.json": json.dumps(manifest, ensure_ascii=False) + "\n",
                    "predictions.csv": csv}
        if kind == "evaluation":
            payloads.update({"summary.json": '{"models":{"selected":{"overall":{"mae":0.375}}}}\n',
                             "report.html": "<!doctype html><title>Synthetic byte fixture</title>"})
        write_payloads(payloads, self.out)
        return payloads

    def checksums(self):
        return json.loads((self.out / "checksums.json").read_text())

    def save_checksums(self, checksums):
        (self.out / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")

    def cli(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(["verify", str(self.out)])
        return status, out.getvalue(), err.getvalue()

    def test_prediction_bundle_read_only(self):
        payloads = self.make_bundle()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.iterdir()}
        result = verification.verify_bundle(self.out)
        self.assertEqual(result.bundle_type, "prediction")
        self.assertEqual(result.files, tuple(sorted(payloads)))
        self.assertEqual(result.total_bytes, sum(len(p.encode("utf-8")) for p in payloads.values()))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.iterdir()})

    def test_evaluation_bundle_and_relative_path(self):
        self.make_bundle("evaluation")
        old_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            result = verification.verify_bundle("run")
            self.assertEqual(result.bundle_type, "evaluation")
            self.assertEqual(set(result.files), verification.EVALUATION_FILES)
            os.chdir(self.out)
            self.assertEqual(verification.verify_bundle(".").bundle_type, "evaluation")
        finally:
            os.chdir(old_cwd)

    def test_cli_success_is_integrity_only_and_never_fits(self):
        self.make_bundle()
        with patch("groupbench.cli.benchmark") as benchmark, patch("groupbench.cli.fit_predict") as fit_predict, \
                patch("groupbench.cli._read_csv_with_hash") as read_csv:
            status, out, err = self.cli()
        self.assertEqual((status, err), (0, ""))
        for expected in ("Verified prediction bundle: 2 files", "Byte integrity only", "not scientific/model correctness",
                         "authenticity", "Unlisted files are not checked", "not an atomic directory snapshot"):
            self.assertIn(expected, out)
        benchmark.assert_not_called()
        fit_predict.assert_not_called()
        read_csv.assert_not_called()

    def test_cli_missing_or_tampered_file_fails_without_success_output(self):
        self.make_bundle()
        for name in ("checksums.json", "manifest.json", "predictions.csv"):
            with self.subTest(name=name):
                path = self.out / name
                original = path.read_bytes()
                path.unlink()
                status, out, err = self.cli()
                self.assertEqual(status, 2)
                self.assertEqual(out, "")
                self.assertIn(name, err)
                path.write_bytes(original)
        (self.out / "predictions.csv").write_text("TAMPERED")
        status, out, err = self.cli()
        self.assertEqual((status, out), (2, ""))
        self.assertIn("SHA-256 mismatch for predictions.csv", err)

    def test_missing_run_directory_is_not_created(self):
        self.assertFalse(self.out.exists())
        self.assertEqual(self.cli()[0], 2)
        self.assertFalse(self.out.exists())

    def test_each_evaluation_payload_is_checked(self):
        self.make_bundle("evaluation")
        for name in verification.EVALUATION_FILES:
            path = self.out / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verification.verify_bundle(self.out)
            path.write_bytes(original)

    def test_malformed_schema_rejected_before_payload_reads(self):
        self.make_bundle()
        records = self.checksums()["files"]
        malformed = [None, [], {}, {"algorithm": "sha256"},
                     {"algorithm": "sha256", "files": records, "extra": True},
                     {"algorithm": "SHA256", "files": records},
                     {"algorithm": ["sha256"], "files": records},
                     {"algorithm": "sha256", "files": []},
                     {"algorithm": "sha256", "files": {}},
                     {"algorithm": "sha256", "files": {"manifest.json": "a" * 64}}]
        for record in malformed:
            with self.subTest(record=record):
                self.save_checksums(record)
                with patch.object(verification, "_read_file", wraps=verification._read_file) as reading:
                    self.assertEqual(self.cli()[0], 2)
                self.assertEqual([call.args[1] for call in reading.call_args_list], ["checksums.json"])

    def test_duplicate_json_keys_rejected_at_all_schema_levels(self):
        self.make_bundle()
        pairs = ','.join(json.dumps(k) + ':' + json.dumps(v) for k, v in self.checksums()["files"].items())
        samples = ['{"algorithm":"sha256","algorithm":"sha256","files":{' + pairs + '}}',
                   '{"algorithm":"sha256","files":{' + pairs + '},"files":{' + pairs + '}}',
                   '{"algorithm":"sha256","files":{' + pairs + ',"manifest.json":"' + '0' * 64 + '"}}',
                   '{"algorithm":"sha256","files":{"manifest.json":"' + '0' * 64 + '","manifest\\u002ejson":"' + '0' * 64 + '"}}']
        for raw in samples:
            with self.subTest(raw=raw):
                (self.out / "checksums.json").write_text(raw)
                status, _, err = self.cli()
                self.assertEqual(status, 2)
                self.assertIn("duplicate object keys", err)

    def test_invalid_json_utf8_constants_and_deep_nesting_fail_cleanly(self):
        self.make_bundle()
        for raw in (b"{", b"{} trailing", b"\xff", b"\xef\xbb\xbf{}", b'{"algorithm":NaN,"files":{}}',
                    b'{"algorithm":Infinity,"files":{}}', b"[" * 2000 + b"]" * 2000):
            with self.subTest(raw=raw[:40]):
                (self.out / "checksums.json").write_bytes(raw)
                status, out, err = self.cli()
                self.assertEqual((status, out), (2, ""))
                self.assertNotIn("Traceback", err)

    def test_invalid_digest_types_lengths_and_case(self):
        self.make_bundle()
        checks = self.checksums()
        for digest in (None, 12, True, [], {}, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"):
            with self.subTest(digest=digest):
                checks["files"]["manifest.json"] = digest
                self.save_checksums(checks)
                status, _, err = self.cli()
                self.assertEqual(status, 2)
                self.assertIn("64 lowercase hexadecimal", err)

    def test_unknown_and_unsafe_names_cannot_read_outside_bundle(self):
        self.make_bundle()
        checks = self.checksums()
        outside = self.root / "outside"
        outside.write_text("PRIVATE")
        for name in ("../outside", str(outside), "sub/file", "sub\\file", "C:\\outside", "./manifest.json",
                     "", ".", "..", "checksums.json", "unknown.txt", "manifest.json\0", "https://example.test/data"):
            with self.subTest(name=name):
                malformed = {"algorithm": "sha256", "files": {**checks["files"], name: "0" * 64}}
                self.save_checksums(malformed)
                with patch.object(verification, "_read_file", wraps=verification._read_file) as reading:
                    self.assertEqual(self.cli()[0], 2)
                self.assertEqual([call.args[1] for call in reading.call_args_list], ["checksums.json"])
        self.assertEqual(outside.read_text(), "PRIVATE")

    def test_unlisted_files_and_source_paths_are_not_opened(self):
        self.make_bundle()
        outside = self.root / "private-source.csv"
        outside.write_text("PRIVATE")
        payload = json.dumps({"source_path": str(outside), "url": "https://example.test/source"}).encode()
        (self.out / "manifest.json").write_bytes(payload)
        checks = self.checksums()
        checks["files"]["manifest.json"] = hashlib.sha256(payload).hexdigest()
        self.save_checksums(checks)
        (self.out / "unlisted-link").symlink_to(outside)
        with patch.object(verification, "_read_file", wraps=verification._read_file) as reading:
            verification.verify_bundle(self.out)
        self.assertEqual([call.args[1] for call in reading.call_args_list],
                         ["checksums.json", "manifest.json", "predictions.csv"])

    def test_payloads_are_opaque_even_when_not_valid_json_or_html(self):
        self.make_bundle("evaluation")
        checks = self.checksums()
        for name in checks["files"]:
            payload = b"\x80\x04UNTRUSTED DATA NOT A MODEL\xff"
            (self.out / name).write_bytes(payload)
            checks["files"][name] = hashlib.sha256(payload).hexdigest()
        self.save_checksums(checks)
        with patch("pickle.loads") as load_pickle, patch("pickle.load") as load_pickle_file:
            self.assertEqual(verification.verify_bundle(self.out).bundle_type, "evaluation")
        load_pickle.assert_not_called()
        load_pickle_file.assert_not_called()

    def test_prediction_rejects_stale_evaluation_files_including_broken_links(self):
        self.make_bundle()
        for name in ("report.html", "summary.json"):
            for kind in ("file", "broken-link", "directory"):
                with self.subTest(name=name, kind=kind):
                    path = self.out / name
                    if kind == "file":
                        path.write_text("OLD")
                    elif kind == "broken-link":
                        path.symlink_to(self.root / "missing")
                    else:
                        path.mkdir()
                    with self.assertRaisesRegex(ValueError, "unverified evaluation artifacts"):
                        verification.verify_bundle(self.out)
                    path.rmdir() if kind == "directory" else path.unlink()

    def test_checksum_and_payload_symlinks_are_not_followed(self):
        self.make_bundle()
        for name in ("checksums.json", "manifest.json", "predictions.csv"):
            with self.subTest(name=name):
                path = self.out / name
                original = path.read_bytes()
                outside = self.root / name
                outside.write_bytes(original)
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                    verification.verify_bundle(self.out)
                self.assertEqual(outside.read_bytes(), original)
                path.unlink()
                path.write_bytes(original)

    def test_directory_components_and_parent_traversal_are_rejected(self):
        self.make_bundle()
        alias = self.root / "alias"
        alias.symlink_to(self.out, target_is_directory=True)
        ancestor_alias = self.root / "ancestor-alias"
        ancestor_alias.symlink_to(self.root, target_is_directory=True)
        for path in (alias, ancestor_alias / "run", self.root / ".." / self.root.name / "run"):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                verification.verify_bundle(path)

    def test_directories_and_fifos_are_rejected_before_open(self):
        self.make_bundle()
        path = self.out / "predictions.csv"
        original = path.read_bytes()
        for kind in ("directory", "fifo"):
            path.unlink()
            path.mkdir() if kind == "directory" else os.mkfifo(path)
            with patch.object(verification.os, "open", wraps=os.open) as opening:
                with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                    verification.verify_bundle(self.out)
            self.assertNotIn("predictions.csv", [call.args[0] for call in opening.call_args_list])
            path.rmdir() if kind == "directory" else path.unlink()
            path.write_bytes(original)

    def test_fixed_size_guards_fail_before_reading_oversized_files(self):
        self.make_bundle()
        checksum = self.out / "checksums.json"
        raw = checksum.read_bytes()
        with patch.object(verification, "MAX_CHECKSUM_BYTES", len(raw) - 1), patch.object(verification.os, "read", wraps=os.read) as reading:
            with self.assertRaisesRegex(ValueError, "verification limit"):
                verification.verify_bundle(self.out)
        reading.assert_not_called()
        with patch.object(verification, "MAX_CHECKSUM_BYTES", len(raw)):
            verification.verify_bundle(self.out)
        with patch.object(verification, "MAX_PAYLOAD_BYTES", 8):
            self.assertEqual(self.cli()[0], 2)

    def test_stream_reads_are_bounded_and_growth_is_rejected(self):
        self.make_bundle()
        real_read = os.read
        requests = []
        def read(fd, size):
            requests.append(size)
            return real_read(fd, size)
        with patch.object(verification, "READ_CHUNK_BYTES", 7), patch.object(verification.os, "read", side_effect=read):
            verification.verify_bundle(self.out)
        self.assertTrue(requests)
        self.assertLessEqual(max(requests), 7)
        # A continuously growing source cannot make the read loop unbounded.
        with patch.object(verification, "MAX_CHECKSUM_BYTES", 1024), patch.object(verification.os, "read", side_effect=lambda _fd, n: b"x" * n) as reading:
            with self.assertRaisesRegex(ValueError, "verification limit"):
                verification.verify_bundle(self.out)
        self.assertEqual(sum(call.args[1] for call in reading.call_args_list), 1025)

    def test_symlink_swap_between_stat_and_open_does_not_escape(self):
        self.make_bundle()
        path = self.out / "predictions.csv"
        outside = self.root / "outside"
        outside.write_bytes(path.read_bytes())
        real_open = os.open
        def swap(name, flags, *args, **kwargs):
            if name == "predictions.csv":
                path.unlink()
                path.symlink_to(outside)
            return real_open(name, flags, *args, **kwargs)
        with patch.object(verification.os, "open", side_effect=swap):
            with self.assertRaises(OSError):
                verification.verify_bundle(self.out)

    def test_same_size_edit_during_read_is_rejected(self):
        self.make_bundle()
        path = self.out / "predictions.csv"
        real_read = os.read
        expected_inode = path.stat().st_ino
        changed = False
        def mutate(fd, size):
            nonlocal changed
            chunk = real_read(fd, size)
            if not changed and os.fstat(fd).st_ino == expected_inode:
                changed = True
                data = path.read_bytes()
                path.write_bytes(data.replace(b"1.25", b"9.25"))
            return chunk
        with patch.object(verification.os, "read", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                verification.verify_bundle(self.out)

    def test_payload_replaced_after_read_is_rejected(self):
        self.make_bundle()
        path = self.out / "manifest.json"
        real_read_file = verification._read_file
        def replace(directory_fd, name, *args, **kwargs):
            result = real_read_file(directory_fd, name, *args, **kwargs)
            if name == "manifest.json":
                replacement = self.out / "replacement"
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            return result
        with patch.object(verification, "_read_file", side_effect=replace):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                verification.verify_bundle(self.out)

    def test_unsupported_platform_fails_closed_without_opening(self):
        self.make_bundle()
        with patch.object(verification, "_SAFE_OPEN_SUPPORTED", False), patch.object(verification.os, "open") as opening:
            status, _, err = self.cli()
        self.assertEqual(status, 2)
        self.assertIn("platform is unsupported", err)
        opening.assert_not_called()

    def test_cli_help_and_subprocess_exit_status(self):
        self.make_bundle()
        for args, expected in ((["verify", "--help"], 0), (["verify", str(self.out)], 0),
                               (["verify", str(self.root / "missing")], 2), (["verify"], 2)):
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, "-m", "groupbench", *args], capture_output=True, text=True,
                                         timeout=20, check=False)
                self.assertEqual(process.returncode, expected, process.stderr)
                self.assertNotIn("Traceback", process.stderr)
                if "--help" in args:
                    self.assertIn("run_dir", process.stdout)
                    self.assertIn("no symlink components", " ".join(process.stdout.split()))
                    self.assertIn("Exit status: 0 verified, 2 failed", " ".join(process.stdout.split()))

    def test_cli_evaluation_success(self):
        self.make_bundle("evaluation")
        status, out, err = self.cli()
        self.assertEqual((status, err), (0, ""))
        self.assertIn("Verified evaluation bundle: 4 files", out)

    def test_regular_file_swap_between_stat_and_open_is_rejected(self):
        self.make_bundle()
        path = self.out / "predictions.csv"
        replacement = self.root / "replacement"
        replacement.write_bytes(path.read_bytes())
        real_open = os.open
        def swap(name, flags, *args, **kwargs):
            if name == "predictions.csv":
                replacement.replace(path)
            return real_open(name, flags, *args, **kwargs)
        with patch.object(verification.os, "open", side_effect=swap):
            with self.assertRaisesRegex(ValueError, "changed before verification"):
                verification.verify_bundle(self.out)

    def test_fifo_swap_between_stat_and_open_never_blocks(self):
        self.make_bundle()
        path = self.out / "predictions.csv"
        real_open = os.open
        def swap(name, flags, *args, **kwargs):
            if name == "predictions.csv":
                self.assertTrue(flags & os.O_NONBLOCK)
                path.unlink()
                os.mkfifo(path)
            return real_open(name, flags, *args, **kwargs)
        with patch.object(verification.os, "open", side_effect=swap):
            with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                verification.verify_bundle(self.out)

    def test_checksum_replacement_after_read_is_rejected(self):
        self.make_bundle()
        path = self.out / "checksums.json"
        real_read_file = verification._read_file
        def replace(directory_fd, name, *args, **kwargs):
            result = real_read_file(directory_fd, name, *args, **kwargs)
            if name == "checksums.json":
                replacement = self.out / "replacement"
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            return result
        with patch.object(verification, "_read_file", side_effect=replace):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                verification.verify_bundle(self.out)


if __name__ == "__main__":
    unittest.main()
