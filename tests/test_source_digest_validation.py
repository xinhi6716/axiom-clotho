"""Caller-declared digest syntax must fail before scheduling or fitting."""

import json
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench import evaluation, planning, prediction
from groupbench.binary import workflow
from groupbench.data import validate_source_sha256


APIS = [
    (evaluation, 'benchmark', '_prepare_benchmark', False),
    (prediction, 'fit_predict', '_prepare_prediction', True),
    (planning, 'plan_benchmark', '_prepare_benchmark', False),
    (planning, 'plan_prediction', '_prepare_prediction', True),
    (workflow, 'benchmark_binary', '_prepare_evaluation', False),
    (workflow, 'fit_predict_binary', '_prepare_prediction', True),
    (workflow, 'plan_binary_benchmark', '_prepare_evaluation', False),
    (workflow, 'plan_binary_prediction', '_prepare_prediction', True),
]
INVALID = ['', 'a' * 63, 'a' * 65, 'g' * 64, 'a' * 64 + '\n',
           ' ' + 'a' * 64, '０' * 64, 'é' * 64, True, 123, b'a' * 64,
           ['a' * 64], {'digest': 'a' * 64}, object()]


class SourceDigestValidationTests(unittest.TestCase):
    def test_invalid_digests_rejected_before_any_data_preparation(self):
        for module, name, prepare, predicts in APIS:
            for parameter in ['source_sha256'] + (['prediction_source_sha256'] if predicts else []):
                for value in INVALID:
                    with self.subTest(api=name, parameter=parameter, value=value):
                        args = [None, None] if predicts else [None]
                        with patch.object(module, prepare, side_effect=AssertionError('preparation reached')) as mocked:
                            with self.assertRaisesRegex(ValueError, parameter):
                                getattr(module, name)(*args, target='y', group='group', **{parameter: value})
                        mocked.assert_not_called()

    def test_validator_preserves_optional_and_hex_case(self):
        for value in (None, '0' * 64, 'abcdef0123456789' * 4, 'ABCDEF0123456789' * 4):
            self.assertEqual(validate_source_sha256(value), value)

    def test_valid_and_absent_digests_preserved_in_all_reports(self):
        frame = pd.DataFrame({'group': [f'g{i}' for i in range(8) for _ in range(2)],
                              'x': list(range(16)), 'y': [0, 1] * 8})
        incoming = pd.DataFrame({'x': [2, 7]})
        for module, name, _, predicts in APIS:
            for source, new_source in [(None, None), ('a' * 64, 'B' * 64)]:
                with self.subTest(api=name, source=source):
                    args = [frame, incoming] if predicts else [frame]
                    kwargs = dict(target='y', group='group', inner_splits=2,
                                  baselines=('dummy',), source_sha256=source)
                    if predicts:
                        kwargs['prediction_source_sha256'] = new_source
                    else:
                        kwargs['outer_splits'] = 2
                    result = getattr(module, name)(*args, **kwargs)
                    report = result if isinstance(result, dict) else result.manifest
                    self.assertEqual(report['source_file_sha256'], source)
                    if predicts:
                        self.assertEqual(report['prediction_source_file_sha256'], new_source)
                    json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
