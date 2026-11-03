"""Invented no-fit common-query comparisons; no real study inputs."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench import cli, saved_scoring as scoring
from groupbench.metrics import paired_absolute_error_difference, contrast_float
from test_saved_prediction_scoring import make_bundle, snapshot


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def amend(bundle, change):
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text()); change(manifest)
    path.write_text(json.dumps(manifest))
    hashes = {name: digest((bundle / name).read_bytes()) for name in ('manifest.json', 'predictions.csv')}
    (bundle / 'checksums.json').write_text(json.dumps({'algorithm': 'sha256', 'files': hashes}))


class SavedPredictionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / 'a', self.root / 'b'
        self.query, self.outcomes = self.root / 'query.csv', self.root / 'outcomes.csv'
        self.query.write_text('x,z,response\n1,4,99\n2,5,99\n3,6,99\n')
        self.outcomes.write_text('row_position,group,observed\n2,B,0\n0,A,0\n1,B,0\n')
        self.contract = dict(method_a='reference', method_b='candidate', target_column='response',
                             target_definition='invented response', target_unit='arbitrary units',
                             cohort_definition='three invented query rows',
                             row_mapping_definition='unchanged original query positions, no filtered or reordered rows',
                             group_definition='invented supplied categories, not independent units')
        self.build(self.a, (0, 0, 0)); self.build(self.b, (0, 2, 4))

    def build(self, bundle, predictions, version='1.0'):
        make_bundle(bundle, predictions, version)
        amend(bundle, lambda m: (m.update(prediction_source_file_sha256=digest(self.query.read_bytes())),
                                 m['config'].update(target='response')))

    def score(self, **changes):
        args = dict(run_a=self.a, run_b=self.b, query_csv=self.query,
                    outcomes_csv=self.outcomes, contract=self.contract)
        args.update(changes)
        return scoring.compare_saved_predictions(**args)

    def test_complete_reordered_cohort_weighting_and_provenance(self):
        before = snapshot(self.root); contract = copy.deepcopy(self.contract)
        result = self.score()
        self.assertEqual((result['n_rows'], result['n_groups'], result['omitted_rows']), (3, 2, 0))
        self.assertEqual(result['row_mean_absolute_error_b_minus_a'], 2)
        self.assertEqual(result['equal_group_mean_absolute_error_b_minus_a'], 1.5)
        self.assertEqual([r['absolute_error_b_minus_a'] for r in result['paired_rows']], [0, 2, 4])
        self.assertEqual([r['observed'] for r in result['paired_rows']], [0, 0, 0])  # not query.response
        self.assertFalse(result['physical_identity_verified']); self.assertFalse(result['provenance_verified'])
        self.assertTrue(result['query_bytes_match_both_manifest_declarations'])
        self.assertFalse(result['identical_bundle_hashes'])
        self.assertEqual(result['query_sha256'], digest(self.query.read_bytes()))
        self.assertEqual(result['outcomes_sha256'], digest(self.outcomes.read_bytes()))
        self.assertEqual(result['model_a'], scoring.score_saved_predictions(self.a, self.outcomes))
        self.assertEqual(result['model_b'], scoring.score_saved_predictions(self.b, self.outcomes))
        self.assertEqual(before, snapshot(self.root)); self.assertEqual(self.contract, contract)

    def test_single_shared_reads_no_fits_or_training_sources(self):
        with patch.object(scoring, '_load_prediction_bundle', wraps=scoring._load_prediction_bundle) as bundles, \
             patch.object(scoring, '_read_outcomes', wraps=scoring._read_outcomes) as outcomes, \
             patch.object(scoring, '_outcomes', wraps=scoring._outcomes) as parse, \
             patch.object(cli, '_read_csv_with_hash', wraps=cli._read_csv_with_hash) as query, \
             patch('groupbench.model.GroupBenchRegressor.fit', side_effect=AssertionError('fit')), \
             patch('sklearn.pipeline.Pipeline.fit', side_effect=AssertionError('fit')), \
             patch('groupbench.prediction.fit_predict', side_effect=AssertionError('fit')):
            self.assertEqual(self.score()['fits_performed'], 0)
        self.assertEqual((bundles.call_count, outcomes.call_count, parse.call_count, query.call_count), (2, 1, 1, 1))

    def test_same_bundle_and_method_reversal(self):
        same = self.score(run_b=self.a)
        self.assertTrue(same['identical_bundle_hashes'])
        self.assertEqual(same['row_mean_absolute_error_b_minus_a'], 0)
        reverse = self.score(run_a=self.b, run_b=self.a)
        self.assertEqual(reverse['row_mean_absolute_error_b_minus_a'], -2)
        self.assertEqual(reverse['equal_group_mean_absolute_error_b_minus_a'], -1.5)

    def test_query_hash_required_on_both_bundles_and_no_override(self):
        for value in (None, '', '0' * 64, True, 'not-a-hash'):
            with self.subTest(value=value):
                amend(self.b, lambda m: m.update(prediction_source_file_sha256=value))
                with self.assertRaises(scoring.SavedScoringError): self.score()
        self.build(self.b, (0, 2, 4))
        self.query.write_text('x,z,response\n3,6,99\n2,5,99\n1,4,99\n')
        with self.assertRaises(scoring.SavedScoringError): self.score()
        with self.assertRaises(TypeError): self.score(ignore_source_hash=True)

    def test_query_roster_headers_and_missing_features(self):
        for text in ('x,z,response\n1,4,99\n', 'x,x,response\n1,4,99\n2,5,99\n3,6,99\n',
                     'q,z,response\n1,4,99\n2,5,99\n3,6,99\n'):
            with self.subTest(text=text):
                self.query.write_text(text)
                for bundle in (self.a, self.b):
                    amend(bundle, lambda m: m.update(prediction_source_file_sha256=digest(self.query.read_bytes())))
                with self.assertRaises(scoring.SavedScoringError): self.score()

    def test_all_versions_and_different_feature_subsets(self):
        for version in ('1.0', '1.1', '1.2', '1.3'):
            with self.subTest(version=version):
                self.build(self.b, (0, 2, 4), version)
                self.assertEqual(self.score()['row_mean_absolute_error_b_minus_a'], 2)
        self.build(self.b, (0, 2, 4))
        amend(self.b, lambda m: (m['config'].update(features=['z']),
                                 m.update(feature_schema=[{'name': 'z', 'dtype': 'float64'}], source_file_sha256='f' * 64)))
        self.assertEqual(self.score()['row_mean_absolute_error_b_minus_a'], 2)

    def test_target_and_contract_mismatches(self):
        for key in self.contract:
            bad = dict(self.contract); bad[key] = ''
            with self.subTest(key=key), self.assertRaises(scoring.SavedScoringError): self.score(contract=bad)
        for bad in (dict(self.contract, extra='x'), dict(self.contract, method_b='reference'),
                    dict(self.contract, target_column='different'), dict(self.contract, target_unit=1),
                    dict(self.contract, group_definition='\0'), dict(self.contract, method_a='a' * 257)):
            with self.assertRaises(scoring.SavedScoringError): self.score(contract=bad)
        amend(self.b, lambda m: m['config'].update(target='other'))
        with self.assertRaises(scoring.SavedScoringError): self.score()

    def test_incomplete_invalid_or_ambiguous_outcomes_reject(self):
        for rows in ('0,A,0\n1,B,0\n', '0,A,0\n0,A,0\n2,B,0\n',
                     '00,A,0\n1,B,0\n2,B,0\n', '0,A,0\n1,B,0\n3,B,0\n',
                     '0, ,0\n1,B,0\n2,B,0\n', '0,A,1e-999\n1,B,0\n2,B,0\n',
                     '0,A,9007199254740993\n1,B,0\n2,B,0\n'):
            with self.subTest(rows=rows):
                self.outcomes.write_text('row_position,group,observed\n' + rows)
                with self.assertRaises(scoring.SavedScoringError): self.score()

    def test_prediction_integer_loss_and_underflow_rejected_on_either_side(self):
        for bundle in (self.a, self.b):
            for token in ('9007199254740993', '1e-999'):
                with self.subTest(bundle=bundle, token=token):
                    self.build(bundle, (token, 0, 0))
                    with self.assertRaises(scoring.SavedScoringError): self.score()
            self.build(bundle, (0, 0, 0))

    def test_large_common_target_retains_exact_paired_contrast(self):
        self.build(self.a, (1, 1, 1)); self.build(self.b, (2, 2, 2))
        self.outcomes.write_text('row_position,group,observed\n0,A,1e20\n1,B,1e20\n2,B,1e20\n')
        result = self.score()
        self.assertEqual(result['model_a']['overall']['mae'], result['model_b']['overall']['mae'])
        self.assertEqual(result['row_mean_absolute_error_b_minus_a'], -1)
        self.assertEqual(result['equal_group_mean_absolute_error_b_minus_a'], -1)

    def test_literal_groups_and_outcome_reordering(self):
        self.outcomes.write_text('row_position,group,observed\n2,1,0\n0,01,0\n1,1,0\n')
        first = self.score()
        self.outcomes.write_text('row_position,group,observed\n1,1,0\n0,01,0\n2,1,0\n')
        second = self.score()
        self.assertEqual(first['paired_rows'], second['paired_rows'])
        self.assertEqual(first['per_group'], second['per_group'])
        self.assertEqual([r['group'] for r in first['per_group']], ['01', '1'])

    def test_retained_prediction_bytes_not_reopened_after_validation(self):
        original = scoring._load_prediction_bundle
        def replace_after_loading(path):
            loaded = original(path)
            (Path(path) / 'predictions.csv').write_text('PRIVATE_INVALID_REPLACEMENT')
            return loaded
        with patch.object(scoring, '_load_prediction_bundle', side_effect=replace_after_loading):
            self.assertEqual(self.score()['row_mean_absolute_error_b_minus_a'], 2)

    def test_shared_arithmetic_exact_oracle_and_conversion_bounds(self):
        self.assertEqual(paired_absolute_error_difference(1, 2, 1e20), Fraction(-1))
        for value in (Fraction(1, 10**1000), Fraction(10**1000)):
            with self.assertRaises(ValueError): contrast_float(value)
        self.assertEqual(contrast_float(Fraction()), 0)


if __name__ == '__main__':
    unittest.main()
