"""Strict no-fit unified CLI for the existing paired saved prediction API."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from groupbench import cli, saved_scoring as scoring
import test_saved_prediction_comparison as comparison_fixtures
from test_saved_prediction_scoring import snapshot


class SavedComparisonCliTests(unittest.TestCase):
    def setUp(self):
        self.fixture = comparison_fixtures.SavedPredictionComparisonTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.contract = f.root / 'contract.json'
        self.contract.write_text(json.dumps(f.contract))
        self.args = ['--run-a', str(f.a), '--run-b', str(f.b), '--query', str(f.query),
                     '--outcomes', str(f.outcomes), '--contract', str(self.contract)]

    def invoke(self, args=None, direct=False):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), \
             patch('groupbench.model.GroupBenchRegressor.fit', side_effect=AssertionError('fit')), \
             patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('fit')):
            try:
                arguments = self.args if args is None else args
                code = scoring.compare_main(arguments) if direct else cli.main(['compare-predictions', *arguments])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_cli_is_exact_api_output_without_mutation(self):
        before = snapshot(self.fixture.root)
        code, out, err = self.invoke()
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), self.fixture.score())
        self.assertEqual(before, snapshot(self.fixture.root))
        self.assertEqual(self.invoke(), self.invoke(direct=True))

    def test_dispatch_forwards_args_return_code_and_sys_argv(self):
        with patch.object(scoring, 'compare_main', return_value=9) as delegate:
            self.assertEqual(cli.main(['compare-predictions', *self.args]), 9)
            delegate.assert_called_once_with(self.args)
        with patch.object(sys, 'argv', ['mlcalc', 'compare-predictions', '--help']), \
             patch.object(scoring, 'compare_main', return_value=0) as delegate:
            self.assertEqual(cli.main(), 0)
            delegate.assert_called_once_with(['--help'])

    def test_help_and_invalid_args_redact_supplied_values(self):
        code, out, err = self.invoke(['--help'])
        self.assertEqual((code, err), (0, ''))
        self.assertIn('mlcalc compare-predictions', out)
        for bad in ([], self.args + ['--out', 'PRIVATE_PATH'],
                    self.args + ['--ignore-source-hash', 'PRIVATE_VALUE'],
                    ['--run', 'PRIVATE_PATH'], self.args + ['PRIVATE_VALUE']):
            code, out, err = self.invoke(bad)
            self.assertEqual((code, out), (2, ''))
            self.assertNotIn('PRIVATE_', err)
            self.assertNotIn(str(self.fixture.root), err)

    def test_duplicate_unknown_numeric_nonfinite_and_nested_contracts_reject_before_api(self):
        original = self.contract.read_text()
        for text in (original.replace('"method_a":', '"method_a":"duplicate", "method_a":', 1),
                     original[:-1] + ',"unknown":"PRIVATE_VALUE"}',
                     original.replace('"reference"', '1'),
                     original.replace('"reference"', 'NaN'),
                     original.replace('"reference"', 'true'),
                     original.replace('"reference"', '{"nested":"PRIVATE_VALUE"}'),
                     '[' * 1100 + ']' * 1100, '{PRIVATE_VALUE', '[]'):
            with self.subTest(text=text[:30]):
                self.contract.write_text(text)
                with patch.object(scoring, 'compare_saved_predictions', side_effect=AssertionError('API must not run')):
                    code, out, err = self.invoke()
                self.assertEqual((code, out), (2, ''))
                self.assertNotIn('PRIVATE_VALUE', err)

    def test_contract_utf8_size_and_missing_input_rejections(self):
        for raw in (b'\xff', b' ' * (scoring.MAX_COMPARISON_CONTRACT_BYTES + 1),
                    '界'.encode() * (scoring.MAX_COMPARISON_CONTRACT_BYTES // 3 + 1)):
            self.contract.write_bytes(raw)
            self.assertEqual(self.invoke()[:2], (2, ''))
        self.contract.unlink()
        code, out, err = self.invoke()
        self.assertEqual((code, out), (2, '')); self.assertNotIn(str(self.contract), err)

    def test_contract_regular_file_policy_and_explicit_alias(self):
        original = self.contract.read_bytes(); target = self.fixture.root / 'target.json'
        target.write_bytes(original); self.contract.unlink(); self.contract.symlink_to(target)
        self.assertEqual(self.invoke()[0], 0)
        self.contract.unlink(); self.contract.mkdir()
        self.assertEqual(self.invoke()[:2], (2, ''))
        self.contract.rmdir()
        if hasattr(os, 'mkfifo'):
            os.mkfifo(self.contract)
            self.assertEqual(self.invoke()[:2], (2, ''))

    def test_contract_uses_existing_reader_once_and_retained_values(self):
        original_reader = scoring._read_outcomes
        calls = []
        def reader(path):
            calls.append(Path(path))
            result = original_reader(path)
            if Path(path) == self.contract:
                self.contract.write_text('INVALID_PRIVATE_REPLACEMENT')
            return result
        expected = self.fixture.score()
        with patch.object(scoring, '_read_outcomes', side_effect=reader):
            code, out, err = self.invoke()
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), expected)
        self.assertEqual(calls.count(self.contract), 1)
        self.assertEqual(calls.count(self.fixture.outcomes), 1)

    def test_output_is_complete_or_empty_on_serialization_and_size_errors(self):
        size = len(self.invoke()[1].encode('utf-8'))
        with patch.object(scoring, 'MAX_COMPARISON_OUTPUT_BYTES', size):
            self.assertEqual(self.invoke()[0], 0)
        with patch.object(scoring, 'MAX_COMPARISON_OUTPUT_BYTES', size - 1):
            self.assertEqual(self.invoke()[:2], (2, ''))
        with patch.object(scoring, 'MAX_COMPARISON_OUTPUT_BYTES', 1):
            self.assertEqual(self.invoke()[:2], (2, ''))
        with patch.object(scoring.json, 'dumps', side_effect=ValueError('PRIVATE_SERIALIZATION')):
            code, out, err = self.invoke()
        self.assertEqual((code, out), (2, '')); self.assertNotIn('PRIVATE_SERIALIZATION', err)

    def test_identity_gates_not_bypassed_and_errors_do_not_echo_paths(self):
        self.fixture.query.write_text('x,z,response\n3,6,99\n2,5,99\n1,4,99\n')
        code, out, err = self.invoke()
        self.assertEqual((code, out), (2, '')); self.assertNotIn(str(self.fixture.root), err)
        self.assertIn('mismatched query', err)

    def test_real_groupbench_workflow_and_console_target_agree(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        commands = [ ['-m', 'groupbench', 'compare-predictions'],
                     ['-m', 'mlworkflow', 'tabular', 'compare-predictions'],
                     ['-c', 'from groupbench.cli import main; raise SystemExit(main())', 'compare-predictions'] ]
        expected = self.invoke()
        for command in commands:
            result = subprocess.run([sys.executable, *command, *self.args], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual((result.returncode, result.stdout, result.stderr), expected)

    def test_root_help_and_existing_dispatch_preserved(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as raised: cli.main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        for name in ('compare-predictions', 'score-predictions', 'evaluate-binary', 'evaluate-intervals'):
            self.assertIn(name, out.getvalue())
        for command, module in [('score-predictions', 'saved_scoring'), ('evaluate-intervals', 'interval_evaluation'),
                                ('evaluate-binary', 'binary.saved_evaluation')]:
            with patch('groupbench.' + module + '.main', return_value=7) as delegate:
                self.assertEqual(cli.main([command, '--help']), 7)
                delegate.assert_called_once_with(['--help'])

    def test_shipped_contract_example_validates_and_declares_self_comparison(self):
        path = Path(__file__).resolve().parents[1] / 'examples/optional-model-delayed-scoring/comparison_contract.json'
        contract = scoring._load_comparison_contract(path)
        self.assertEqual(contract['target_column'], 'response')
        self.assertIn('same-bundle', contract['method_a'])
        self.assertIn('same-bundle', contract['method_b'])


if __name__ == '__main__':
    unittest.main()
