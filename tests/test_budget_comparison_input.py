"""Regular-file boundary for saved comparisons; invented inputs only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench.budget_comparison import _read_input, load_document


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/saved-budget-comparison/invented.json"


class SavedComparisonInputTests(unittest.TestCase):
    def run_cli(self, source):
        # The timeout is a test guard, not a claimed CLI storage deadline.
        return subprocess.run(
            [sys.executable, "-m", "groupbench.budget_comparison", str(source)],
            capture_output=True, text=True, timeout=10,
        )

    def assert_rejected_cli(self, source, message=None):
        result = self.run_cli(source)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("budget comparison rejected:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        if message:
            self.assertIn(message, result.stderr)

    def test_regular_utf8_and_newline_decoding_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "saved.json")
            raw = '{"unit_id": "試片-é"}\r\n'
            source.write_bytes(raw.encode("utf-8"))
            self.assertEqual(_read_input(source), raw.replace("\r\n", "\n"))
            self.assertEqual(load_document(_read_input(source)), {"unit_id": "試片-é"})
            self.assertEqual(source.read_bytes(), raw.encode("utf-8"))

    def test_character_read_limit_not_changed_to_byte_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "saved.json")
            # Multi-byte text is not valid JSON, but exercises the reader's
            # existing character, not byte, bound before JSON validation.
            source.write_text("é" * 8_000_002, encoding="utf-8")
            text = _read_input(source)
            self.assertEqual(len(text), 8_000_001)
            self.assertEqual(text[-1], "é")
            with self.assertRaisesRegex(ValueError, "character limit"):
                load_document(text)

    def test_known_nonregular_paths_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root]
            if hasattr(os, "mkfifo"):
                fifo = root / "saved.json"
                os.mkfifo(fifo)
                sources.append(fifo)
            if os.name == "posix":
                sources.append(Path(os.devnull))
                link = root / "linked-directory.json"
                link.symlink_to(root, target_is_directory=True)
                sources.append(link)
            for source in sources:
                with self.subTest(source=source), patch("groupbench.budget_comparison.os.open") as opening:
                    with self.assertRaisesRegex(ValueError, "regular local file"):
                        _read_input(source)
                    opening.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX symlinks")
    def test_regular_symlink_and_direct_cli_reports_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            source, link = Path(directory, "saved.json"), Path(directory, "linked.json")
            raw = FIXTURE.read_bytes()
            source.write_bytes(raw)
            link.symlink_to(source)
            direct, linked = self.run_cli(source), self.run_cli(link)
            for result in (direct, linked):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertEqual(json.loads(result.stdout)["status"], "complete_descriptive_comparison")
            self.assertEqual(direct.stdout, linked.stdout)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(sorted(p.name for p in source.parent.iterdir()), ["linked.json", "saved.json"])

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes")
    def test_no_writer_fifo_cli_rejected_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory, "saved.json")
            os.mkfifo(fifo)
            self.assert_rejected_cli(fifo, "regular local file")
            if os.name == "posix":
                link = Path(directory, "linked.json")
                link.symlink_to(fifo)
                self.assert_rejected_cli(link, "regular local file")

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"), "nonblocking named pipes")
    def test_fifo_substitution_checked_on_descriptor_and_closed_without_read(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "saved.json")
            source.write_text("{}", encoding="utf-8")
            actual_open, actual_close = os.open, os.close
            opened = []

            def replace_before_open(path, flags):
                # Fail before the OS open if the nonblocking protection is lost,
                # so a regression cannot hang the test process itself.
                self.assertTrue(flags & os.O_NONBLOCK)
                source.unlink()
                os.mkfifo(source)
                fd = actual_open(path, flags)
                opened.append(fd)
                return fd

            with patch("groupbench.budget_comparison.os.open", side_effect=replace_before_open), \
                    patch("groupbench.budget_comparison.os.close", wraps=actual_close) as closing, \
                    patch("groupbench.budget_comparison.os.fdopen") as reading:
                with self.assertRaisesRegex(ValueError, "regular local file"):
                    _read_input(source)
                reading.assert_not_called()
                closing.assert_called_once_with(opened[0])
            with self.assertRaises(OSError):
                os.fstat(opened[0])

    def test_decode_failure_still_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "saved.json")
            source.write_bytes(b"\xff")
            actual_close = os.close
            with patch("groupbench.budget_comparison.os.close", wraps=actual_close) as closing:
                with self.assertRaises(UnicodeDecodeError):
                    _read_input(source)
                closing.assert_called_once()
            with self.assertRaises(OSError):
                os.fstat(closing.call_args.args[0])

    def test_cli_missing_directory_and_invalid_utf8_fail_without_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_bytes(b"\xff")
            for source in (root / "absent.json", root, invalid):
                with self.subTest(source=source):
                    self.assert_rejected_cli(source)
            self.assertEqual(invalid.read_bytes(), b"\xff")
            self.assertEqual(list(root.iterdir()), [invalid])


if __name__ == "__main__":
    unittest.main()
