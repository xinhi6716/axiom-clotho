"""A same-version package is not an artifact identity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/check_installed_smoke.py'
spec = importlib.util.spec_from_file_location('installed_smoke_identity', path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class Receipt:
    def __init__(self, text=None):
        self.text = text

    def read_text(self, name):
        assert name == 'direct_url.json'
        return self.text


@unittest.skipUnless(shutil.which('git'), 'Git is required for checkout identity')
class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.wheel = self.root / 'same-version artifact.whl'
        self.wheel.write_bytes(b'first artifact')
        self.git('init', '-q')
        (self.source / 'fixture.txt').write_text('synthetic', encoding='utf-8')
        self.git('add', 'fixture.txt')
        self.git('-c', 'user.name=Synthetic', '-c', 'user.email=test@invalid.invalid',
                 'commit', '-qm', 'Invented checkout')
        self.checkout = self.git('rev-parse', 'HEAD').strip()
        self.sha = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.direct_url = json.dumps({'url': self.wheel.as_uri(),
                                      'archive_info': {'hashes': {'sha256': self.sha}}})

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.source), *args], text=True)

    def test_original_optional_behavior_and_authenticated_local_archive(self):
        self.assertIsNone(smoke.checked_build_identity(self.source, None, None, Receipt()))
        output = smoke.checked_build_identity(self.source, self.wheel, self.checkout, Receipt(self.direct_url))
        self.assertEqual(output['source_checkout_sha'], self.checkout)
        self.assertEqual(output['installed_local_wheel_sha256'], self.sha)
        self.assertIn('no build attestation', output['scope'])

    def test_mismatches_and_missing_paths_do_not_pass(self):
        good = Receipt(self.direct_url)
        for w, sha, record in ((self.wheel, None, good), (None, self.checkout, good),
                               (self.wheel, '0'*40, good), (self.wheel, self.checkout, Receipt()),
                               (self.wheel, self.checkout, Receipt('null')),
                               (self.wheel, self.checkout, Receipt(self.direct_url.replace('file:', 'https:')))):
            with self.assertRaises(RuntimeError):
                smoke.checked_build_identity(self.source, w, sha, record)
        self.wheel.write_bytes(b'changed bytes same file name and version')
        with self.assertRaisesRegex(RuntimeError, 'digest differs'):
            smoke.checked_build_identity(self.source, self.wheel, self.checkout, good)

    def test_dirty_checkout_not_given_clean_source_identity(self):
        (self.source / 'untracked.txt').write_text('untracked', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'clean Git checkout root'):
            smoke.checked_build_identity(self.source, self.wheel, self.checkout, Receipt(self.direct_url))
