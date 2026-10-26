"""Known impossible binary output paths must fail before reading data or fitting."""
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.cli import main
from groupbench.binary.report import ensure_fresh_binary_output


class BinaryOutputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'existing-file'
        self.file.write_text('preserve these bytes', encoding='utf-8')

    def reject_cli(self, out):
        for mode in ('run', 'predict'):
            with self.subTest(mode=mode, output=str(out)):
                args = ['binary', mode, str(self.root / 'not-read.csv')]
                if mode == 'predict':
                    args.append(str(self.root / 'not-read-new.csv'))
                args += ['--target', 'y', '--group', 'g', '--out', str(out)]
                errors = io.StringIO()
                with patch('groupbench.cli._read_training', side_effect=AssertionError('read data before rejection')) as read, \
                     patch('groupbench.binary.cli.benchmark_binary', side_effect=AssertionError('fit before rejection')) as evaluate, \
                     patch('groupbench.binary.cli.fit_predict_binary', side_effect=AssertionError('fit before rejection')) as predict, \
                     redirect_stdout(io.StringIO()), redirect_stderr(errors):
                    status = main(args)
                self.assertEqual(status, 2)
                self.assertIn('Binary output', errors.getvalue())
                read.assert_not_called()
                evaluate.assert_not_called()
                predict.assert_not_called()
        self.assertEqual(self.file.read_text(encoding='utf-8'), 'preserve these bytes')

    def test_regular_file_destination(self):
        self.reject_cli(self.file)

    def test_regular_file_parent(self):
        self.reject_cli(self.file / 'report')

    def test_regular_file_remote_ancestor(self):
        self.reject_cli(self.file / 'a' / 'b' / 'report')

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_dangling_link_destination_and_parent(self):
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'missing-target')
        self.reject_cli(link)
        self.reject_cli(link / 'report')
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root / 'missing-target').exists())

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_link_to_file_parent(self):
        link = self.root / 'file-link'
        link.symlink_to(self.file)
        self.reject_cli(link / 'report')
        self.assertTrue(link.is_symlink())

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFOs unavailable')
    def test_fifo_destination_and_parent(self):
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        self.reject_cli(fifo)
        self.reject_cli(fifo / 'report')

    def test_missing_directory_chain_remains_valid_and_uncreated(self):
        out = self.root / 'new' / 'nested' / 'report'
        ensure_fresh_binary_output(out)
        self.assertFalse((self.root / 'new').exists())

    def test_existing_empty_directory_remains_valid(self):
        out = self.root / 'empty'
        out.mkdir()
        ensure_fresh_binary_output(out)
        self.assertEqual(list(out.iterdir()), [])

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_link_to_directory_is_not_rejected_by_preflight(self):
        directory = self.root / 'directory'
        directory.mkdir()
        link = self.root / 'directory-link'
        link.symlink_to(directory, target_is_directory=True)
        ensure_fresh_binary_output(link / 'new')
        self.assertFalse((directory / 'new').exists())

    def test_known_payload_collision_still_rejected(self):
        out = self.root / 'out'
        out.mkdir()
        (out / 'manifest.json').write_text('keep')
        with self.assertRaises(FileExistsError):
            ensure_fresh_binary_output(out)
        self.assertEqual((out / 'manifest.json').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
