"""Known impossible output paths must fail before any model work or writes."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.cli import _check_output_directory, main


class OutputDirectoryPreflightTests(unittest.TestCase):
    def test_regular_file_or_parent_file_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "existing-file"
            existing.write_text("KEEP", encoding="utf-8")
            for out in (existing, existing / "child", existing / "child" / "grandchild"):
                self.assert_commands_reject(out)
            self.assertEqual(existing.read_text(), "KEEP")
            self.assertEqual(list(root.iterdir()), [existing])

    def assert_commands_reject(self, out):
        for mode in ("demo", "run", "predict"):
            for overwrite in (False, True):
                with self.subTest(mode=mode, out=out.name, overwrite=overwrite):
                    command = [mode]
                    if mode != "demo":
                        command += ["not-read-training.csv"]
                        if mode == "predict":
                            command += ["not-read-prediction.csv"]
                        command += ["--target", "y", "--group", "lot"]
                    command += ["--out", str(out)]
                    if overwrite:
                        command.append("--overwrite")
                    with patch("groupbench.cli.benchmark") as benchmark, patch("groupbench.cli.fit_predict") as predict, patch("groupbench.cli._read_training") as read, patch("groupbench.cli.make_materials_fixture") as fixture, redirect_stderr(io.StringIO()) as error, redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(main(command), 2)
                    for operation in (benchmark, predict, read, fixture):
                        operation.assert_not_called()
                    self.assertIn("--out must name a directory", error.getvalue())
                    self.assertEqual(output.getvalue(), "")

    def test_dangling_and_regular_file_links_rejected_before_model_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "file"
            existing.write_text("KEEP", encoding="utf-8")
            for name, target in (("dangling", root / "absent"), ("file-link", existing)):
                link = root / name
                try:
                    link.symlink_to(target)
                except (OSError, NotImplementedError):
                    self.skipTest("symlink creation unavailable")
                self.assert_commands_reject(link)
                self.assert_commands_reject(link / "child")
                self.assertTrue(link.is_symlink())
            self.assertEqual(existing.read_text(), "KEEP")
            self.assertFalse((root / "absent").exists())

    def test_existing_new_and_symlink_directories_allowed_without_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in (root, root / "new", root / "new" / "nested"):
                _check_output_directory(path)
            self.assertEqual(list(root.iterdir()), [])
            link = root / "dir-link"
            try:
                link.symlink_to(root, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            _check_output_directory(link)
            _check_output_directory(link / "new")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
