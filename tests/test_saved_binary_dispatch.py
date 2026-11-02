"""Unified saved-binary navigation delegates without fitting or writing outputs."""
from contextlib import redirect_stdout, redirect_stderr
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
from groupbench.binary.saved_evaluation import main as canonical_main
from test_saved_binary_evaluation import document


class SavedBinaryDispatchTests(unittest.TestCase):
    def invoke(self, args, command=main):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), \
             patch('groupbench.cli.benchmark', side_effect=AssertionError('fit')), \
             patch('groupbench.cli.fit_predict', side_effect=AssertionError('fit')), \
             patch('groupbench.cli.plan_benchmark', side_effect=AssertionError('plan')):
            try:
                code = command(args)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_dispatch_forwards_unchanged_args_and_return_status(self):
        args = ['evaluate-binary', '--', 'a path.json']
        with patch('groupbench.binary.saved_evaluation.main', return_value=7) as delegate:
            self.assertEqual(main(args), 7)
        delegate.assert_called_once_with(['--', 'a path.json'])
        self.assertEqual(args, ['evaluate-binary', '--', 'a path.json'])

    def test_sys_argv_dispatch(self):
        with patch.object(sys, 'argv', ['mlcalc', 'evaluate-binary', 'input.json']), \
             patch('groupbench.binary.saved_evaluation.main', return_value=2) as delegate:
            self.assertEqual(main(), 2)
        delegate.assert_called_once_with(['input.json'])

    def test_help_and_argument_errors_are_exact_canonical_parser(self):
        for args in (['--help'], [], ['--out', 'PRIVATE_ARGUMENT'], ['a', 'PRIVATE_ARGUMENT']):
            with self.subTest(args=args):
                result = self.invoke(['evaluate-binary', *args])
                self.assertEqual(result, self.invoke(args, canonical_main))
                self.assertNotIn('PRIVATE_ARGUMENT', result[1] + result[2])

    def test_root_help_keeps_saved_regression_intervals_and_optional_families(self):
        code, out, err = self.invoke(['--help'])
        self.assertEqual((code, err), (0, ''))
        for command in ('evaluate-binary', 'score-predictions', 'evaluate-intervals', 'binary'):
            self.assertIn(command, out)
        code, out, err = self.invoke(['predict', '--help'])
        self.assertEqual((code, err), (0, ''))
        for family in ('nystroem_ridge', 'group_median'):
            self.assertIn(family, out)

    def test_other_early_dispatchers_stay_independent(self):
        for command, module in [('score-predictions', 'saved_scoring'),
                                ('evaluate-intervals', 'interval_evaluation')]:
            with patch('groupbench.' + module + '.main', return_value=13) as delegate, \
                 patch('groupbench.binary.saved_evaluation.main', side_effect=AssertionError('wrong evaluator')):
                self.assertEqual(main([command, '--help']), 13)
                delegate.assert_called_once_with(['--help'])

    def test_valid_invalid_and_missing_files_match_canonical_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / 'input.json'; valid.write_text(json.dumps(document()))
            invalid = root / 'invalid.json'; invalid.write_text('{"PRIVATE_VALUE":')
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            for path, expected in ((valid, 0), (invalid, 2), (root / 'PRIVATE_MISSING.json', 2)):
                result = self.invoke(['evaluate-binary', str(path)])
                self.assertEqual(result, self.invoke([str(path)], canonical_main))
                self.assertEqual(result[0], expected)
                if expected:
                    self.assertEqual(result[1], '')
                    self.assertNotIn('PRIVATE_', result[2])
                    self.assertNotIn(str(root), result[2])
                else:
                    self.assertIsInstance(json.loads(result[1]), dict)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_real_process_module_and_unified_aliases_have_identical_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'; path.write_text(json.dumps(document()))
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1')
            commands = [
                ['-m', 'groupbench.binary.saved_evaluation', str(path)],
                ['-m', 'groupbench', 'evaluate-binary', str(path)],
                ['-m', 'mlworkflow', 'tabular', 'evaluate-binary', str(path)],
                ['-c', 'from groupbench.cli import main; raise SystemExit(main())', 'evaluate-binary', str(path)],
            ]
            results = [subprocess.run([sys.executable, *cmd], env=env, capture_output=True, timeout=30) for cmd in commands]
            self.assertEqual(results[0].returncode, 0, results[0].stderr)
            for result in results:
                self.assertEqual((result.returncode, result.stdout, result.stderr),
                                 (results[0].returncode, results[0].stdout, results[0].stderr))


if __name__ == '__main__':
    unittest.main()
