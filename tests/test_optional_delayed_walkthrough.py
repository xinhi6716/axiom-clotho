"""Execute bilingual documented commands on tiny invented fixtures, never research data."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlworkflow.cli import main

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path('examples/optional-model-delayed-scoring')


def commands(path):
    blocks = re.findall(r'```sh\n(.*?)```', path.read_text(), flags=re.S)
    return [shlex.split(line) for block in blocks
            for line in block.replace('\\\n', '').splitlines()
            if line.startswith('clotho ')]


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


class OptionalDelayedWalkthroughTests(unittest.TestCase):
    def test_documented_bilingual_workflow_and_zero_refit_score(self):
        steps = commands(ROOT / RELATIVE / 'README.md')
        self.assertEqual(steps, commands(ROOT / RELATIVE / 'README.zh-TW.md'))
        self.assertEqual(len(steps), 6)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / RELATIVE).mkdir(parents=True)
            for name in ('training.csv', 'new_features.csv', 'later_outcomes.csv'):
                shutil.copyfile(ROOT / RELATIVE / name, work / RELATIVE / name)
            inputs = hashes(work / RELATIVE)
            for index, args in enumerate(steps[:-1]):
                result = subprocess.run([sys.executable, '-m', 'mlworkflow', *args[1:]],
                                        cwd=work, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                if index in (0, 2):
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan['fits_performed'], 0)
                    self.assertEqual(plan['fit_budget']['total_fits'], 30 if index == 0 else 10)
            self.assertEqual(inputs, hashes(work / RELATIVE))
            prediction = work / 'runs/optional-later-prediction'
            manifest = json.loads((prediction / 'manifest.json').read_text())
            self.assertEqual(manifest['schema_version'], '1.3')
            self.assertEqual(manifest['capabilities'], ['group_median-v1', 'nystroem_ridge-v1'])
            # Scoring must need neither source training data nor model fitting.
            (work / RELATIVE / 'training.csv').unlink()
            (work / RELATIVE / 'new_features.csv').unlink()
            before = hashes(work)
            out, err = io.StringIO(), io.StringIO()
            args = steps[-1][1:]
            args = [str(work / value) if value.startswith(('runs/', 'examples/')) else value for value in args]
            with redirect_stdout(out), redirect_stderr(err), \
                 patch('groupbench.model.GroupBenchRegressor.fit', side_effect=AssertionError('refit')), \
                 patch('groupbench.prediction.fit_predict', side_effect=AssertionError('refit')), \
                 patch('groupbench.evaluation.benchmark', side_effect=AssertionError('refit')), \
                 patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('refit')):
                self.assertEqual(main(args), 0)
            self.assertEqual(err.getvalue(), '')
            score = json.loads(out.getvalue())
            self.assertEqual((score['fits_performed'], score['n_rows'], score['n_groups']), (0, 3, 2))
            self.assertEqual([row['row_position'] for row in score['scored_rows']], [0, 1, 2])
            self.assertEqual([row['observed'] for row in score['scored_rows']], [.225, 9.025, 40.])
            self.assertEqual(score['per_group'][0]['group'], 'queryA')
            self.assertIsNone(score['per_group'][0]['r2'])
            self.assertEqual(before, hashes(work))


if __name__ == '__main__':
    unittest.main()
