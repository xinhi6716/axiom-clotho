"""Bounded synthetic CLI domain selection checks."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from groupbench.cli import main, _read_csv_with_hash
from groupbench.model import make_pipeline
from groupbench.evaluation_schema import validate_evaluation_bundle
from groupbench.prediction_schema import validate_prediction_bundle_schema
from mlworkflow.cli import main as clotho
from test_planning import no_fitting

FLAGS = ['--domain', 'domain', '--inner-strategy', 'leave_one_domain_out']

class DomainCLITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train, self.query = self.root / 'train.csv', self.root / 'query.csv'
        self.train.write_text('g,x,y,domain\n' + ''.join(
            f'g{i},{i},{i * 2},{["001", "01", "NA"][i // 2]}\n' for i in range(6)))
        self.query.write_text('x\n2.5\n5.5\n')

    def args(self, mode, plan=False):
        return (['plan'] if plan else []) + [mode, str(self.train)] + (
            [str(self.query)] if mode == 'predict' else []) + [
            '--target', 'y', '--group', 'g', '--baselines', 'dummy,ridge'] + (
            ['--outer-splits', '3'] if mode == 'run' else [])

    def invoke(self, args, entry=main):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = entry(args)
        return code, out.getvalue(), err.getvalue()

    def test_exact_labels_no_fit_plan_and_actual_budget(self):
        frame, _ = _read_csv_with_hash(self.train, group='g', domain='domain')
        self.assertEqual(set(frame.domain), {'001', '01', 'NA'})
        for mode in ('run', 'predict'):
            before = sorted(self.root.iterdir())
            with ExitStack() as stack:
                no_fitting(stack)
                code, output, error = self.invoke(['tabular', *self.args(mode, True), *FLAGS], clotho)
            self.assertEqual((code, error), (0, ''))
            self.assertEqual(sorted(self.root.iterdir()), before)
            plan = json.loads(output)
            self.assertEqual(plan['fits_performed'], 0)
            self.assertEqual(plan['config']['features'], ['x'])
            if mode == 'run':
                self.assertEqual(plan['config']['inner_splits_per_outer'], [3, 3, 2])
                self.assertEqual(plan['fit_budget']['total_fits'], 38)
            for secret in ('g0', '001', '"NA"'):
                self.assertNotIn(secret, output)
            calls = []
            def factory(*args, **kwargs):
                pipeline = make_pipeline(*args, **kwargs)
                original = pipeline.fit
                def fit(*a, **kw):
                    calls.append(1)
                    return original(*a, **kw)
                pipeline.fit = fit
                return pipeline
            destination = self.root / mode
            with patch('groupbench.model.make_pipeline', side_effect=factory):
                code, _, error = self.invoke(self.args(mode) + FLAGS + ['--out', str(destination)])
            self.assertEqual((code, error), (0, ''))
            self.assertEqual(len(calls), plan['fit_budget']['total_fits'])
            manifest = json.loads((destination / 'manifest.json').read_text())
            self.assertEqual(manifest['config']['features'], ['x'])
            self.assertEqual(set(manifest['config']['group_domains'].values()), {'001', '01', 'NA'})
            validator = validate_evaluation_bundle if mode == 'run' else validate_prediction_bundle_schema
            validator(destination)

    def test_incompatible_arguments_fail_before_read_fit_or_write(self):
        invalid = [['--domain', 'domain'], ['--inner-strategy', 'leave_one_domain_out']]
        invalid += [FLAGS + ['--inner-splits', n] for n in ('2', '3', '0')]
        for mode in ('run', 'predict'):
            for plan in (False, True):
                for flags in invalid:
                    with self.subTest(mode=mode, plan=plan, flags=flags):
                        args = self.args(mode, plan) + flags
                        if not plan:
                            args += ['--out', str(self.root / 'absent')]
                        with patch('groupbench.cli._read_training', side_effect=AssertionError('read')):
                            code, output, error = self.invoke(args)
                        self.assertEqual(code, 2)
                        self.assertEqual(output, '')
                        self.assertNotIn('Traceback', error)
                        self.assertFalse((self.root / 'absent').exists())

    def test_invalid_metadata_and_domain_predictor(self):
        for extra in (['--features', 'x,domain'], ['--domain', 'y'], ['--domain', 'missing']):
            code, _, error = self.invoke(self.args('predict', True) + FLAGS + extra)
            self.assertEqual(code, 2)
            self.assertNotIn('Traceback', error)
        original = self.train.read_text()
        for bad in (original.replace('g2,2,4,01', 'g0,2,4,01'), original.replace(',001\n', ',\n')):
            self.train.write_text(bad)
            code, _, error = self.invoke(self.args('predict', True) + FLAGS)
            self.assertEqual(code, 2)
            self.assertNotIn('Traceback', error)

    def test_feature_manifest_cannot_promote_domain(self):
        schema = self.root / 'feature_schema.json'
        schema.write_text(json.dumps(dict(schema_version=1, row_key=['g'],
            model_feature_allowlist=['x', 'domain'], metadata_columns=['g'],
            target_columns=['y'], feature_units={'x': 'unit', 'domain': 'label'}, warnings=[])))
        for mode in ('run', 'predict'):
            with patch('groupbench.model.make_pipeline', side_effect=AssertionError('fit')):
                code, _, error = self.invoke(self.args(mode, True) + FLAGS + ['--features-file', str(schema)])
            self.assertEqual(code, 2)
            self.assertIn('Domain metadata cannot be a predictor', error)

    def test_group_default_equals_explicit_three(self):
        for mode in ('run', 'predict'):
            a = self.invoke(self.args(mode, True) + ['--features', 'x'])
            b = self.invoke(self.args(mode, True) + ['--features', 'x', '--inner-splits', '3', '--inner-strategy', 'group_kfold'])
            self.assertEqual(a, b)
            self.assertEqual(a[0], 0)
            self.assertNotIn('inner_strategy', json.loads(a[1])['config'])

    def test_help_regression_only(self):
        for mode in ('run', 'predict'):
            for prefix in ([], ['plan']):
                with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                    main(prefix + [mode, '--help'])
                self.assertIn('--domain', out.getvalue())
                self.assertIn('leave_one_domain_out', out.getvalue())
        for args in (['demo'], ['binary', 'run']):
            with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                main(args + ['--help'])
            self.assertNotIn('--domain', out.getvalue())

if __name__ == '__main__':
    unittest.main()
