"""Root help is navigation only; it must not execute, write, or hide commands."""
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

from groupbench.cli import main


class OnboardingHelpTests(unittest.TestCase):
    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with patch('groupbench.cli.benchmark', side_effect=AssertionError('help fitted')):
                with patch('groupbench.cli.plan_benchmark', side_effect=AssertionError('help planned')):
                    with self.assertRaises(SystemExit) as raised:
                        main(args)
        return raised.exception.code, out.getvalue(), err.getvalue()

    def test_root_help_has_actionable_synthetic_start_and_separate_binary(self):
        code, out, err = self.invoke(['--help'])
        self.assertEqual((code, err), (0, ''))
        for text in ('mlcalc demo --out runs/first-demo', 'fits models and writes files',
                     'new output directory', 'runs/first-demo/report.html',
                     'Synthetic results do not establish', 'explicit binary namespace',
                     'mlcalc plan run --help', 'groupbench command is an alias'):
            self.assertIn(text, out)
        for command in ('demo', 'run', 'predict', 'plan', 'verify',
                        'validate-prediction-bundle', 'fixture', 'binary'):
            self.assertIn(command, out)

    def test_no_arguments_is_still_an_error_not_a_demo(self):
        code, out, err = self.invoke([])
        self.assertEqual((code, out), (2, ''))
        self.assertIn('required', err)

    def test_all_help_destinations_remain_read_only(self):
        commands = [['demo'], ['run'], ['predict'], ['plan'], ['plan', 'run'],
                    ['plan', 'predict'], ['verify'], ['validate-prediction-bundle'],
                    ['fixture'], ['binary'], ['binary', 'run'], ['binary', 'predict'],
                    ['binary', 'plan'], ['binary', 'plan', 'run'], ['binary', 'plan', 'predict']]
        for command in commands:
            with self.subTest(command=command):
                code, out, err = self.invoke(command + ['--help'])
                self.assertEqual((code, err), (0, ''))
                self.assertIn('usage:', out)

if __name__ == '__main__':
    unittest.main()
