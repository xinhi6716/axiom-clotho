"""Bounded installed-wheel smoke, not a full-suite or scientific validation.

Uses only the repository's invented 18-device measurement fixture. Invoke with
the installed environment's Python -I and a fresh work directory outside source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from urllib.parse import urlsplit


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}



def checked_build_identity(source, wheel, expected_checkout, distribution):
    """Link a tested install to bytes + checkout, without attesting their derivation."""
    if wheel is None and expected_checkout is None:
        return None
    require(wheel is not None and isinstance(expected_checkout, str)
            and re.fullmatch(r'[0-9a-f]{40}', expected_checkout) is not None,
            'Supply both --wheel and the full lowercase --expected-source-commit')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args],
                                       stderr=subprocess.PIPE, text=True,
                                       encoding='utf-8', timeout=30).strip()

    try:
        root = Path(git('rev-parse', '--show-toplevel')).resolve()
        checkout = git('rev-parse', 'HEAD')
        changes = git('status', '--porcelain', '--untracked-files=normal')
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Cannot verify the source checkout identity') from exc
    require(root == source and checkout == expected_checkout and not changes,
            'Source must be the expected clean Git checkout root')

    require(wheel.suffix == '.whl' and wheel.is_file() and wheel.stat().st_size <= 64 * 1024**2,
            'Expected a bounded local wheel file')
    digest = hashlib.sha256()
    with wheel.open('rb') as stream:
        count = 0
        while block := stream.read(1024 * 1024):
            count += len(block)
            require(count <= 64 * 1024**2, 'Wheel exceeds the byte limit')
            digest.update(block)
    sha256 = digest.hexdigest()
    try:
        raw = distribution.read_text('direct_url.json')
        direct = json.loads(raw) if raw is not None and len(raw) <= 8192 else {}
        url = urlsplit(direct['url'])
        hashes = direct['archive_info']['hashes']
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError('No usable installed local wheel provenance') from exc
    require(url.scheme == 'file' and type(hashes) is dict and hashes.get('sha256') == sha256,
            'Installed local archive digest differs from the selected wheel')
    return {
        'source_checkout_sha': checkout,
        'installed_local_wheel_sha256': sha256,
        'scope': 'clean checkout and installed local wheel matched separately; no build attestation',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--wheel', type=Path, help='Local wheel installed by pip')
    parser.add_argument('--expected-source-commit', help='Full Git checkout SHA for the current-source smoke')
    args = parser.parse_args()
    source, work = args.source.resolve(), args.work_dir.resolve()
    require(not work.is_relative_to(source) and not source.is_relative_to(work),
            'Work directory and source must be separate trees')
    require(not Path.cwd().resolve().is_relative_to(source), 'Run outside the checkout')
    require(sys.flags.isolated, 'Invoke this check with Python -I')
    require(sys.platform in ('linux', 'win32'), 'This smoke covers Linux and Windows only')
    purelib = Path(sysconfig.get_path('purelib')).resolve()
    require(not purelib.is_relative_to(source), 'Do not import the source checkout')
    require(metadata.version('ml-super-calculator') == args.expected_version,
            'Installed distribution version differs from expected source version')
    for name in ('groupbench', 'mlworkflow'):
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None
                and Path(spec.origin).resolve().is_relative_to(purelib),
                name + ' must come from this environment, not editable/source imports')
    build_identity = checked_build_identity(
        source, args.wheel, args.expected_source_commit, metadata.distribution('ml-super-calculator'))
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    scripts = Path(sysconfig.get_path('scripts'))
    commands = []

    def command(label, argv, expected=0):
        result = subprocess.run(list(map(str, argv)), cwd=work, env=env,
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=90)
        commands.append({'label': label, 'exit_code': result.returncode})
        if result.returncode != expected:
            raise RuntimeError(f'{label}: expected exit {expected}, got {result.returncode}\n'
                               + result.stdout + result.stderr)
        print(f'{label}: expected exit {expected}', flush=True)
        return result

    def cli(name, *arguments):
        executable = scripts / (name + ('.exe' if os.name == 'nt' else ''))
        require(executable.is_file(), f'Missing installed entry point: {name}')
        return [executable, *arguments]

    for name in ('clotho', 'mlcalc', 'groupbench', 'mlcalc-workflow'):
        output = command(name + '-version', cli(name, '--version')).stdout.strip()
        require(args.expected_version in output.replace(';', ' ').split(),
                name + ' version mismatch')
    command('vendored-component-integrity', cli('clotho', 'components'))
    fixture = source / 'examples/measurement-regression/fixtures'
    inputs = work / 'invented inputs'
    # Later outcomes are absent from the active input tree until after prediction.
    outcomes_bytes = (fixture / 'later_outcomes.csv').read_bytes()
    shutil.copytree(fixture, inputs, ignore=shutil.ignore_patterns('later_outcomes.csv'))
    config = json.loads((inputs / 'config.json').read_bytes())
    require(config['baselines'] == ['dummy', 'ridge'], 'Unexpected fixture candidates')
    config['max_fits'] = 40  # The already-tested fixture needs exactly 40 planned fits.
    (inputs / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    prepared, run_dir, score_dir = (work / n for n in ('prepared bundle', 'run bundle', 'score bundle'))
    command('prepare', cli('clotho', 'prepare', '--input-dir', inputs / 'raw',
                          '--partitions', inputs / 'partitions.csv',
                          '--training-targets', inputs / 'training_targets.csv',
                          '--config', inputs / 'config.json', '--output-dir', prepared))
    before_plan = snapshot(work)
    plan = json.loads(command('plan', cli('clotho', 'plan', '--prepared', prepared)).stdout)
    require(plan['fits_performed'] == 0 and plan['total_planned_fits'] == 40,
            'Unexpected planning fit count')
    require(snapshot(work) == before_plan, 'Plan wrote or changed files')
    with (prepared / 'future.csv').open(encoding='utf-8', newline='') as stream:
        require('target' not in next(csv.reader(stream)), 'Future input contains targets')
    command('run', cli('clotho', 'run', '--prepared', prepared, '--output-dir', run_dir))
    run_info = json.loads((run_dir / 'run.json').read_bytes())
    require(run_info['training_rows'] == 12 and len(run_info['future_cohort']) == 6,
            'Unexpected fixture cohort size')
    require(run_info['caller_prediction_outcomes_opened'] is False, 'Outcomes used early')
    for kind in ('comparison', 'prediction'):
        manifest = json.loads((run_dir / kind / 'manifest.json').read_bytes())
        observation = manifest['package_source']
        if os.name == 'nt':
            require(observation['status'] == 'unavailable'
                    and observation['reason'] == 'unsupported_source_access',
                    'Windows must not claim POSIX-safe source observation')
        else:
            require(observation['status'] == 'available' and len(observation['sha256']) == 64,
                    'Expected a Linux package-source byte observation')
    outcomes = inputs / 'later_outcomes.csv'
    require(not outcomes.exists(), 'Outcomes should still be withheld')
    outcomes.write_bytes(outcomes_bytes)
    before_score = snapshot(run_dir)
    command('score', cli('clotho', 'score', '--run-dir', run_dir,
                        '--outcomes', outcomes, '--output-dir', score_dir))
    scores = json.loads((score_dir / 'scores.json').read_bytes())
    require(scores['fits_performed'] == 0 and scores['cohort_rows'] == 6,
            'Scoring must report zero new fits for six devices')
    require(snapshot(run_dir) == before_score, 'Scoring changed the saved run')

    # Repeat the installed score CLI under -S with ML imports forbidden. Add only
    # this venv's installed package location, never source or user site paths.
    guarded_score = r'''
import importlib.abc, sys
sys.path.insert(0, sys.argv.pop(1))
blocked = {'numpy', 'pandas', 'sklearn', 'groupbench'}
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise RuntimeError('Scoring attempted an ML import: ' + fullname)
sys.meta_path.insert(0, NoML())
from mlworkflow.cli import main
raise SystemExit(main())
'''
    guarded_dir = work / 'score without ML'
    command('score-no-ML-imports', [sys.executable, '-I', '-S', '-c', guarded_score,
                                   purelib, 'score', '--run-dir', run_dir,
                                   '--outcomes', outcomes, '--output-dir', guarded_dir])
    require(snapshot(guarded_dir) == snapshot(score_dir), 'Guarded scoring changed results')
    require(snapshot(run_dir) == before_score, 'Guarded scoring changed the saved run')

    from mlworkflow.contracts import INVENTORIES, load_bundle
    for path, kind in ((prepared, 'prepared'), (run_dir, 'run'), (score_dir, 'score')):
        payloads, digest = load_bundle(path, kind)
        require(set(payloads) == INVENTORIES[kind] and len(digest) == 64,
                kind + ' portable receipt check failed')
    # Keep the existing security boundary: Windows must fail closed, not claim
    # that POSIX no-follow/directory-relative verification is supported.
    for kind in ('comparison', 'prediction'):
        if os.name == 'nt':
            result = command('verify-' + kind + '-unsupported-on-Windows',
                             cli('mlcalc', 'verify', run_dir / kind), expected=2)
            require('Safe verification requires POSIX no-follow' in result.stderr,
                    'Expected the documented unsupported-platform diagnostic')
        else:
            command('verify-' + kind, cli('mlcalc', 'verify', run_dir / kind))
    receipt = {
        'status': 'PASS', 'scope': 'installed-wheel synthetic smoke only',
        'release_asset_verification': 'not performed; no release asset download',
        'version': args.expected_version, 'platform': sys.platform,
        'python': platform.python_version(), 'architecture': platform.machine(),
        'dependencies': {name: metadata.version(name) for name in
                         ('numpy', 'pandas', 'scikit-learn', 'threadpoolctl')},
        'planned_fits': 40, 'scored_devices': 6, 'score_fits_performed': 0,
        'score_without_ML_imports': True,
        'portable_measurement_receipts': 3,
        'tabular_verifier': 'expected unsupported exit 2' if os.name == 'nt' else 'passed',
        'package_source_observation': 'unsupported_source_access' if os.name == 'nt' else 'available',
        'commands': commands,
        'not_claimed': ['full test suite', 'all supported dependency versions',
                        'real-data accuracy', 'macOS', 'Windows POSIX verifier support'],
    }
    if build_identity is not None:
        receipt['build_identity'] = build_identity
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
