"""Anonymous small protocol tests, not empirical predictive-performance evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from groupbench import benchmark, fit_predict, validate_evaluation_bundle, validate_prediction_bundle_schema
from groupbench.model import GroupBenchRegressor, Candidate, candidates_for, make_pipeline
from groupbench.kernel import NystroemRidgeRegressor
from groupbench.planning import plan_benchmark, plan_prediction
from groupbench.verification import verify_bundle


def frame():
    return pd.DataFrame({'x': np.arange(18, dtype=float), 'kind': ['a', 'b', None]*6,
                         'g': np.repeat([f'g{i}' for i in range(6)], 3),
                         'y': np.arange(18, dtype=float)**2 / 10})


class KernelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.data = frame()
        self.X = self.data[['x', 'kind']]
        self.y = self.data.y.to_numpy()
        self.g = self.data.g.to_numpy()

    def test_explicit_registry_order_and_unknown_candidate_refusal(self):
        self.assertEqual([c.name for c in candidates_for(('nystroem_ridge', 'dummy'))],
                         ['nystroem_ridge_alpha_0.1', 'nystroem_ridge_alpha_1',
                          'nystroem_ridge_alpha_10', 'dummy_mean'])
        with self.assertRaisesRegex(ValueError, 'Unknown model family'):
            make_pipeline(self.X, Candidate('bad', 'not_known', {}), 0)
        self.assertEqual(GroupBenchRegressor().baselines, ('dummy', 'ridge', 'extra_trees'))

    def test_planned_and_executed_fit_counts_and_conditional_detail_records(self):
        families = ('dummy', 'nystroem_ridge')
        plan = plan_prediction(self.data, self.X.iloc[:2], target='y', group='g',
                               baselines=families, inner_splits=3)
        calls = []
        original = Pipeline.fit
        def tracked(model, X, y, **kwargs):
            calls.append((len(X), type(model.named_steps['model']).__name__, kwargs))
            return original(model, X, y, **kwargs)
        with patch.object(Pipeline, 'fit', tracked):
            result = fit_predict(self.data, self.X.iloc[:2], target='y', group='g',
                                 baselines=families, inner_splits=3)
        # Only full top-level pipeline fits have model, nested imputer pipelines
        # use fit_transform rather than the patched method here.
        self.assertEqual(len(calls), plan['fit_budget']['total_fits'])
        self.assertEqual(len(calls), 14)
        self.assertTrue(all(not kwargs for _, _, kwargs in calls))
        self.assertNotIn('inner_fit_details', result.manifest['inner_selection'][0])
        for row in result.manifest['inner_selection'][1:]:
            self.assertEqual(len(row['inner_fit_details']), 3)
            for item in row['inner_fit_details']:
                self.assertEqual(item['details']['training_rows'], item['train_rows'])
        kernel_records = result.manifest['inner_selection'][1:]
        for fold in range(3):
            digests = [record['inner_fit_details'][fold]['details']['basis_indices_sha256']
                       for record in kernel_records]
            self.assertEqual(len(set(digests)), 1)
        saved_rows = result.estimator.selection_results_[1]['inner_fit_details'][0]['details']['training_rows']
        result.manifest['inner_selection'][1]['inner_fit_details'][0]['details']['training_rows'] = 999
        self.assertEqual(result.estimator.selection_results_[1]['inner_fit_details'][0]['details']['training_rows'], saved_rows)
        details = result.manifest['family_fit_details']['nystroem_ridge']
        self.assertEqual(details['training_rows'], 18)
        result.manifest['family_fit_details']['nystroem_ridge']['training_rows'] = 1
        self.assertEqual(result.estimator.family_fit_details_['nystroem_ridge']['training_rows'], 18)
        self.assertEqual(plan['schema_version'], '1.3')

    def test_kernel_state_is_atomic_and_successful_old_only_refit_clears_details(self):
        model = GroupBenchRegressor(baselines=('nystroem_ridge',), inner_splits=2).fit(self.X, self.y, self.g)
        before = model.predict(self.X)
        detail = model.family_fit_details_
        original = NystroemRidgeRegressor.fit
        def fail_final(estimator, X, y):
            if X.shape[0] == 18:
                raise ValueError('injected final fit')
            return original(estimator, X, y)
        with patch.object(NystroemRidgeRegressor, 'fit', fail_final), self.assertRaisesRegex(ValueError, 'injected'):
            model.fit(self.X, self.y + 1, self.g)
        self.assertIs(model.family_fit_details_, detail)
        np.testing.assert_array_equal(model.predict(self.X), before)
        model.set_params(baselines=('dummy',)).fit(self.X, self.y, self.g)
        self.assertEqual(model.family_fit_details_, {})
        self.assertNotIn('inner_fit_details', model.selection_results_[0])

    def test_outer_labels_and_features_cannot_change_training_landmarks_or_selection(self):
        first = benchmark(self.data, target='y', group='g', baselines=('dummy', 'nystroem_ridge'),
                          outer_splits=3, inner_splits=2)
        for audit in first.manifest['outer_folds']:
            mask = self.data.g.isin(audit['test_groups'])
            changed = self.data.copy()
            changed.loc[mask, 'y'] += 10000
            labels = benchmark(changed, target='y', group='g', baselines=('dummy', 'nystroem_ridge'),
                               outer_splits=3, inner_splits=2)
            counterpart = labels.manifest['outer_folds'][audit['fold']]
            self.assertEqual(audit['inner_selection'], counterpart['inner_selection'])
            self.assertEqual(audit['family_fit_details'], counterpart['family_fit_details'])
            np.testing.assert_array_equal(first.predictions.loc[mask, 'prediction_selected'],
                                          labels.predictions.loc[mask, 'prediction_selected'])
            changed.loc[mask, 'x'] = -9999
            changed.loc[mask, 'kind'] = 'unseen'
            features = benchmark(changed, target='y', group='g', baselines=('dummy', 'nystroem_ridge'),
                                 outer_splits=3, inner_splits=2)
            counterpart = features.manifest['outer_folds'][audit['fold']]
            self.assertEqual(audit['inner_selection'], counterpart['inner_selection'])
            self.assertEqual(audit['family_fit_details'], counterpart['family_fit_details'])

    def test_optional_schema_roundtrips_and_preview_match_both_encodings(self):
        for encoding in ('legacy-v1', 'distinct-v2'):
            with self.subTest(encoding=encoding):
                kwargs = dict(target='y', group='g', baselines=('nystroem_ridge',),
                              inner_splits=2, categorical_encoding=encoding)
                evaluation = benchmark(self.data, outer_splits=3, **kwargs)
                prediction = fit_predict(self.data, self.X.iloc[:2], **kwargs)
                preview = plan_benchmark(self.data, outer_splits=3, **kwargs)
                self.assertEqual(preview['fit_budget']['total_fits'], 21)
                self.assertEqual(preview['protocol'], evaluation.manifest['protocol'])
                self.assertEqual(preview['capabilities'], evaluation.manifest['capabilities'])
                for result, validator, version in ((evaluation, validate_evaluation_bundle, '1.2'),
                                                   (prediction, validate_prediction_bundle_schema, '1.3')):
                    with tempfile.TemporaryDirectory() as temp:
                        result.write(temp)
                        verify_bundle(temp)
                        self.assertEqual(validator(temp).schema_version, version)

    def test_rehashed_extended_metadata_corruption_is_rejected(self):
        evaluation = benchmark(self.data, target='y', group='g', baselines=('nystroem_ridge',),
                               outer_splits=3, inner_splits=2)
        prediction = fit_predict(self.data, self.X.iloc[:2], target='y', group='g',
                                 baselines=('nystroem_ridge',), inner_splits=2)
        for result, validator in ((evaluation, validate_evaluation_bundle),
                                  (prediction, validate_prediction_bundle_schema)):
            def fit_record(d):
                return d['outer_folds'][0] if 'outer_folds' in d else d
            mutations = [lambda d: d.pop('capabilities'),
                         lambda d: d.update(capabilities=['unrecognized']),
                         lambda d: d['config'].update(categorical_encoding='wrong'),
                         lambda d: d.update(schema_version='1.0'),
                         lambda d: d['config'].update(baselines=['nystroem_ridge', 'nystroem_ridge']),
                         lambda d: fit_record(d)['family_fit_details']['nystroem_ridge'].update(training_rows=True),
                         lambda d: fit_record(d)['family_fit_details']['nystroem_ridge'].update(gamma=100),
                         lambda d: fit_record(d)['inner_selection'][0]['inner_fit_details'].pop(),
                         lambda d: fit_record(d)['inner_selection'][0]['inner_fit_details'][0]['details'].update(rank_cap=63)]
            for mutate in mutations:
                with self.subTest(validator=validator.__name__, mutate=mutate), tempfile.TemporaryDirectory() as temp:
                    altered = deepcopy(result)
                    mutate(altered.manifest)
                    altered.write(temp)
                    verify_bundle(temp)
                    with self.assertRaises(ValueError):
                        validator(temp)

    def test_old_only_manifests_do_not_gain_extension_fields(self):
        for encoding in ('legacy-v1', 'distinct-v2'):
            kwargs = dict(target='y', group='g', baselines=('dummy',), inner_splits=2,
                          categorical_encoding=encoding)
            evaluation = benchmark(self.data, outer_splits=3, **kwargs)
            prediction = fit_predict(self.data, self.X.iloc[:2], **kwargs)
            self.assertEqual(evaluation.manifest['schema_version'], '1.0' if encoding=='legacy-v1' else '1.1')
            self.assertEqual(prediction.manifest['schema_version'], '1.1' if encoding=='legacy-v1' else '1.2')
            for result in (evaluation, prediction):
                self.assertNotIn('capabilities', result.manifest)
                self.assertNotIn('family_fit_details', result.manifest)
            for audit in evaluation.manifest['outer_folds']:
                self.assertNotIn('family_fit_details', audit)

    def test_exact_capability_subsets_roundtrip_and_metadata_is_family_specific(self):
        for families in [('group_median',), ('nystroem_ridge',), ('nystroem_ridge', 'group_median')]:
            for encoding in ('legacy-v1', 'distinct-v2'):
                kwargs = dict(target='y', group='g', baselines=families, inner_splits=2,
                              categorical_encoding=encoding)
                prediction = fit_predict(self.data, self.X.iloc[:2], **kwargs)
                evaluation = benchmark(self.data, outer_splits=3, **kwargs)
                expected = sorted(f + '-v1' for f in families)
                for result, validator in [(prediction, validate_prediction_bundle_schema),
                                           (evaluation, validate_evaluation_bundle)]:
                    self.assertEqual(result.manifest['capabilities'], expected)
                    with tempfile.TemporaryDirectory() as temp:
                        result.write(temp)
                        validator(temp)
                self.assertEqual(set(prediction.manifest['family_fit_details']), set(families))
                for record in prediction.manifest['inner_selection']:
                    for item in record['inner_fit_details']:
                        if record['family'] == 'group_median':
                            self.assertEqual(item['details']['implementation'], 'group-balanced-lower-median-v1')
                            self.assertNotIn('gamma', item['details'])
                for capability in ([], ['unknown-v1'], expected + [expected[0]],
                                   ['group_median-v1', 'nystroem_ridge-v1'] if len(expected)==1 else expected[::-1]):
                    altered = deepcopy(prediction)
                    altered.manifest['capabilities'] = capability
                    with tempfile.TemporaryDirectory() as temp:
                        altered.write(temp)
                        with self.assertRaises(ValueError):
                            validate_prediction_bundle_schema(temp)

    def test_median_saved_counts_constant_and_stale_metadata_are_checked(self):
        result = fit_predict(self.data, self.X.iloc[:2], target='y', group='g',
                             baselines=('group_median',), inner_splits=2)
        for key, bad in [('training_groups', 1), ('training_rows', True),
                         ('transformed_features', 0), ('weight_rule', 'row_weighted')]:
            altered = deepcopy(result)
            altered.manifest['family_fit_details']['group_median'][key] = bad
            with tempfile.TemporaryDirectory() as temp:
                altered.write(temp)
                with self.assertRaises(ValueError):
                    validate_prediction_bundle_schema(temp)
        result.estimator.set_params(baselines=('dummy',)).fit(self.X, self.y, self.g)
        self.assertEqual(result.estimator.family_fit_details_, {})

    def test_loader_returns_exact_consumed_bytes_hashes_with_one_read(self):
        from groupbench import prediction_schema as schema
        result = fit_predict(self.data, self.X.iloc[:2], target='y', group='g',
                             baselines=('group_median',), inner_splits=2)
        with tempfile.TemporaryDirectory() as temp:
            result.write(temp)
            calls = []
            original = schema._read_file
            def track(fd, name, limit, **kwargs):
                calls.append(name)
                return original(fd, name, limit, **kwargs)
            with patch.object(schema, '_read_file', track):
                checked, manifest, csv, hashes = schema._load_prediction_bundle(temp)
            self.assertEqual(calls, ['checksums.json', 'manifest.json', 'predictions.csv'])
            self.assertEqual(checked.schema_version, '1.3')
            self.assertEqual(manifest, result.manifest)
            self.assertEqual(csv, (Path(temp)/'predictions.csv').read_bytes())
            self.assertEqual(hashes, {name: hashlib.sha256((Path(temp)/name).read_bytes()).hexdigest()
                                     for name in calls})


if __name__ == '__main__':
    unittest.main()
