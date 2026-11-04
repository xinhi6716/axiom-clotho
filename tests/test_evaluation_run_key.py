"""Historical evaluation identity, not execution authenticity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench.evaluation import _evaluation_run_key, benchmark
from groupbench.evaluation_schema import validate_evaluation_bundle
from groupbench.verification import verify_bundle


# Frozen from db2a0f839b808ea9f3af379d00685fdc96a834e4, using this invented
# 12-row fixture. They are compatibility identities, not performance evidence.
HISTORICAL_KEYS = {
    ('group_kfold', 'dummy', 'legacy-v1'): '97c0ec8e27eff6300c531b3b023e31902dd1b2ceb443e4b011517c2d1f837da5',
    ('group_kfold', 'dummy', 'distinct-v2'): 'a8350226dd18a928ba6dedefa057254d0429ddcee21818180b20fa4ec7bd36fc',
    ('group_kfold', 'group_median', 'legacy-v1'): '2d5ae07bd0af2058ffb8de2c211939ab86602e3cc74885c8aff69a2ad3872f7b',
    ('group_kfold', 'group_median', 'distinct-v2'): '5af183631f9f548c36cfadee6418b837061b6a0ee00e0ab3e1d00b59c69cd19f',
    ('group_kfold', 'nystroem_ridge', 'legacy-v1'): '9b50024509dddf17178e35122066c2ad6d9fe0c84aa86174f32d8903a7739680',
    ('group_kfold', 'nystroem_ridge', 'distinct-v2'): '15c55956640e0b96f93ba25671107a95b24c74aa6b7157aafdd9f3a4ac1ff470',
    ('group_kfold', 'group_balanced_ridge', 'legacy-v1'): 'e252c2a7ab82b2c3caabc9ba0533c6a84a3d94ce6327fed2d07d871c1bc18248',
    ('group_kfold', 'group_balanced_ridge', 'distinct-v2'): '6778234498dfed0f08ff9a6d3807990b082b2681eb4dfbcb13ff12a55381b4be',
    ('leave_one_domain_out', 'dummy', 'legacy-v1'): 'a44dc042ee35ef06530c06b23befc5ad9041182a941aa4c77503cf3cbea3a977',
    ('leave_one_domain_out', 'dummy', 'distinct-v2'): 'e9ba2c4255f4a00c016504a8913b81689dd3c60b7685f5f10a71605fcbd0d94e',
}


def reconstructed_key(manifest):
    # Independent public recipe, without invoking the producer helper.
    config = dict(manifest['config'])
    if config.get('categorical_encoding') == 'legacy-v1':
        config.pop('categorical_encoding')
    return hashlib.sha256((manifest['data_sha256'] +
                           json.dumps(config, sort_keys=True)).encode('utf-8')).hexdigest()


def write_saved_result(directory, result, *, key=None):
    manifest = deepcopy(result.manifest)
    if key is not None:
        manifest['run_key'] = key
    payloads = {
        'manifest.json': json.dumps(manifest).encode(),
        'summary.json': json.dumps(result.summary).encode(),
        'predictions.csv': result.predictions.to_csv(index=False).encode(),
        'report.html': b'<html><body>Invented checksum payload.</body></html>',
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    (directory / 'checksums.json').write_text(json.dumps({
        'algorithm': 'sha256',
        'files': {name: hashlib.sha256(payload).hexdigest()
                  for name, payload in payloads.items()},
    }))


class EvaluationRunKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = pd.DataFrame({
            'x': range(12), 'y': range(12),
            'group': [f'g{i // 2}' for i in range(12)],
            'domain': [f'd{i // 4}' for i in range(12)],
        })
        cls.results, cls.hashed_configs = {}, {}
        for identity in HISTORICAL_KEYS:
            strategy, family, encoding = identity
            options = dict(target='y', group='group', features=['x'],
                           outer_splits=2, inner_splits=2,
                           baselines=(family,), categorical_encoding=encoding,
                           inner_strategy=strategy)
            if strategy == 'leave_one_domain_out':
                options.update(inner_splits=None, domain='domain')
            captured = []

            def capture(fingerprint, config):
                captured.append(deepcopy(config))
                return _evaluation_run_key(fingerprint, config)

            with patch('groupbench.evaluation._evaluation_run_key', side_effect=capture):
                cls.results[identity] = benchmark(frame, **options)
            cls.hashed_configs[identity] = captured

    def test_historical_keys_and_documented_recipe(self):
        for identity, result in self.results.items():
            with self.subTest(identity=identity):
                self.assertEqual(result.manifest['run_key'], HISTORICAL_KEYS[identity])
                self.assertEqual(reconstructed_key(result.manifest), HISTORICAL_KEYS[identity])

    def test_saved_config_is_final_before_hashing(self):
        for identity, result in self.results.items():
            with self.subTest(identity=identity):
                self.assertEqual(self.hashed_configs[identity], [result.manifest['config']])

    def test_canonicalization_is_narrow_and_nonmutating(self):
        config = {'features': ['電壓'], 'categorical_encoding': 'legacy-v1',
                  'nested': {'categorical_encoding': 'legacy-v1'}, 'threads': 1}
        original = deepcopy(config)
        expected = dict(config)
        expected.pop('categorical_encoding')
        expected_key = hashlib.sha256(('a' * 64 + json.dumps(expected, sort_keys=True)).encode()).hexdigest()
        self.assertEqual(_evaluation_run_key('a' * 64, config), expected_key)
        self.assertEqual(config, original)
        distinct = dict(config, categorical_encoding='distinct-v2')
        self.assertNotEqual(_evaluation_run_key('a' * 64, distinct), expected_key)

    def test_saved_reader_contracts_remain_unchanged(self):
        for identity, result in self.results.items():
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                write_saved_result(directory, result)
                verify_bundle(directory)
                checked = validate_evaluation_bundle(directory)
                self.assertEqual(checked.schema_version, result.manifest['schema_version'])

    def test_readers_do_not_claim_run_key_validation(self):
        result = self.results[('group_kfold', 'group_median', 'legacy-v1')]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_saved_result(directory, result, key='0' * 64)
            verify_bundle(directory)
            validate_evaluation_bundle(directory)
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertNotEqual(reconstructed_key(manifest), manifest['run_key'])


if __name__ == '__main__':
    unittest.main()
