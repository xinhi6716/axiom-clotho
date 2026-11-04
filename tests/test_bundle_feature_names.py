"""Workflow feature names must survive the existing saved-prediction reader."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (GroupBenchRegressor, benchmark, benchmark_binary,
                        fit_predict, fit_predict_binary, plan_benchmark,
                        plan_binary_benchmark, plan_binary_prediction,
                        plan_prediction, validate_prediction_bundle_schema)
from groupbench import cli
from groupbench.binary.data import prepare_binary_data
from groupbench.data import prepare_data, validate_features
from groupbench.input_support import prediction_input_support
from groupbench.saved_scoring import score_saved_predictions
from groupbench.verification import verify_bundle


WORKFLOWS = ((plan_benchmark, False), (benchmark, False),
             (plan_prediction, True), (fit_predict, True),
             (plan_binary_benchmark, False), (benchmark_binary, False),
             (plan_binary_prediction, True), (fit_predict_binary, True))
INVALID = ('', '\0', 'PRIVATE\0field')
OPTIONS = dict(target='y', group='g', inner_splits=2, baselines=('dummy',))


def fixture(name='x'):
    return pd.DataFrame({name: np.arange(24, dtype=float),
                         'g': np.repeat(list('abcdef'), 4), 'y': [0, 1] * 12},
                        index=[7, 7, 3, 3] * 6)


class BundleFeatureNameTests(unittest.TestCase):
    def test_selected_names_reject_without_mutation_or_name_echo(self):
        for name in INVALID:
            frame = fixture(name)
            before = frame.copy(deep=True)
            for prepare in (prepare_data, prepare_binary_data):
                for features in (None, [name], iter([name])):
                    with self.subTest(name=repr(name), prepare=prepare.__name__):
                        with self.assertRaisesRegex(ValueError, 'nonempty strings without NUL') as caught:
                            prepare(frame, 'y', 'g', features)
                        self.assertNotIn('PRIVATE', str(caught.exception))
            pd.testing.assert_frame_equal(frame, before)

    def test_all_frame_workflows_reject_before_scheduling_fitting_or_writing(self):
        for name in INVALID:
            frame, query = fixture(name), pd.DataFrame({name: [1., 2.]})
            for encoding in ('legacy-v1', 'distinct-v2'):
                for route, predicts in WORKFLOWS:
                    with self.subTest(name=repr(name), route=route.__name__, encoding=encoding), ExitStack() as stack:
                        blocked = [stack.enter_context(patch(path, side_effect=AssertionError('unexpected work')))
                                   for path in ('groupbench.splitting.pd.factorize', 'groupbench.model.make_pipeline',
                                                'groupbench.binary.model.binary_pipeline', 'groupbench.report.write_payloads')]
                        options = dict(OPTIONS, categorical_encoding=encoding)
                        with self.assertRaisesRegex(ValueError, 'nonempty strings without NUL'):
                            if predicts:
                                route(frame, query, **options)
                            else:
                                route(frame, outer_splits=3, **options)
                        for operation in blocked:
                            operation.assert_not_called()

    def test_domain_routes_reject_selected_names_after_collecting_iterator(self):
        for name in INVALID:
            source = fixture(name)
            source['domain'] = np.repeat(['A', 'B', 'C'], 8)
            for route, predicts in WORKFLOWS[:4]:
                with self.subTest(name=repr(name), route=route.__name__), \
                     patch('groupbench.splitting.pd.factorize', side_effect=AssertionError('unexpected scheduling')):
                    options = dict(OPTIONS, features=iter([name]), inner_splits=None,
                                   inner_strategy='leave_one_domain_out', domain='domain')
                    with self.assertRaisesRegex(ValueError, 'nonempty strings without NUL'):
                        if predicts:
                            route(source, pd.DataFrame({name: [1., 2.]}), **options)
                        else:
                            route(source, outer_splits=3, **options)

    def test_excluded_metadata_and_target_group_names_keep_their_identity(self):
        for name in INVALID:
            source = fixture()
            source[name] = 'unused metadata'
            for prepare in (prepare_data, prepare_binary_data):
                actual = prepare(source, 'y', 'g', ['x'])
                expected = prepare(fixture(), 'y', 'g', ['x'])
                pd.testing.assert_frame_equal(actual[0], expected[0])
                for a, b in zip(actual[1:], expected[1:]):
                    np.testing.assert_array_equal(a, b)
            for role in ('target', 'group'):
                renamed = fixture().rename(columns={OPTIONS[role]: name})
                options = dict(OPTIONS, **{role: name})
                plan = plan_prediction(renamed, pd.DataFrame({'x': [1.]}), **options)
                self.assertEqual(plan['config'][role], name)

    def test_valid_literal_names_round_trip_through_schema_cli_and_saved_scoring(self):
        for name in ('x', ' ', ' x ', 'line\nbreak', '製程🙂', '\tfeature'):
            for encoding in ('legacy-v1', 'distinct-v2'):
                with self.subTest(name=repr(name), encoding=encoding), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    result = fit_predict(fixture(name), pd.DataFrame({name: [1., 2.]}),
                                         **OPTIONS, categorical_encoding=encoding)
                    result.write(root / 'bundle')
                    self.assertEqual(verify_bundle(root / 'bundle').bundle_type, 'prediction')
                    self.assertEqual(validate_prediction_bundle_schema(root / 'bundle').n_predictions, 2)
                    saved = json.loads((root / 'bundle' / 'manifest.json').read_text())
                    self.assertEqual(saved['feature_schema'][0]['name'], name)
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(['validate-prediction-bundle', str(root / 'bundle')]), 0)
                    (root / 'outcomes.csv').write_text('row_position,group,observed\n0,a,0\n1,b,1\n')
                    scores = score_saved_predictions(root / 'bundle', root / 'outcomes.csv')
                    self.assertEqual(scores['overall']['n_rows'], 2)

    def test_standalone_diagnostic_and_direct_estimator_names_are_unchanged(self):
        for name in INVALID:
            source = fixture(name)
            X = validate_features(source[[name]])
            query = pd.DataFrame({name: [1., 2.]})
            report = prediction_input_support(X, query)
            self.assertEqual(report['features'][0]['name'], name)
        # A trailing NUL has separate upstream indexing behavior; preserve
        # supported direct-estimator examples without promising all names fit.
        for name in ('', 'inside\0name'):
            source = fixture(name)
            X, query = source[[name]], pd.DataFrame({name: [1., 2.]})
            estimator = GroupBenchRegressor(baselines=('dummy',), inner_splits=2)
            estimator.fit(X, source['y'], groups=source['g'])
            np.testing.assert_array_equal(estimator.predict(query), [0.5, 0.5])

    def test_existing_duplicate_nonstring_and_missing_name_checks_remain(self):
        source = fixture()
        source[42] = 1.
        for names in (['x', 'x'], ['missing'], [42]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                prepare_data(source, 'y', 'g', names)


if __name__ == '__main__':
    unittest.main()
