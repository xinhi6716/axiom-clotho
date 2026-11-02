"""No-fit invented cohorts test exact descriptive reference differences."""
import copy
from decimal import Decimal
from fractions import Fraction
import json
import math
import socket
import unittest
from unittest.mock import patch

from groupbench.interval_evaluation import (
    diagnose_saved_interval_reference, evaluate_saved_intervals, load_document,
)
from test_interval_tail_misses import document


def fixture():
    doc = document()
    doc['alpha'] = Decimal('.2')
    return doc


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_fraction_hand_oracle_unequal_groups(self):
        result = diagnose_saved_interval_reference(fixture())
        self.assertEqual(result['declared_reference_exact'], {
            'coverage': '4/5', 'below_lower': '1/10', 'above_upper': '1/10'})
        self.assertEqual(result['row_weighted']['observed_rate_exact'], {
            'coverage': '1/2', 'below_lower': '1/4', 'above_upper': '1/4'})
        self.assertEqual(result['row_weighted']['observed_minus_declared_exact'], {
            'coverage': '-3/10', 'below_lower': '3/20', 'above_upper': '3/20'})
        self.assertEqual(result['equal_group']['observed_rate_exact'], {
            'coverage': '1/3', 'below_lower': '1/6', 'above_upper': '1/2'})
        self.assertEqual(result['equal_group']['observed_minus_declared_exact'], {
            'coverage': '-7/15', 'below_lower': '1/15', 'above_upper': '2/5'})

    def test_counts_support_and_no_independent_sample_claim(self):
        result = diagnose_saved_interval_reference(fixture())
        self.assertEqual(result['row_weighted']['counts'], {
            'covered': 2, 'below_lower': 1, 'above_upper': 1})
        self.assertEqual([row['n_rows'] for row in result['group_counts']], [3, 1])
        self.assertEqual(result['support']['n_scored_rows'], 4)
        self.assertEqual(result['support']['n_declared_groups'], 2)
        self.assertEqual(result['support']['independence'], 'not_established')
        self.assertNotIn('n_rows', result['equal_group'])
        self.assertFalse(result['provenance_verified'])
        self.assertEqual(result['claim_scope'], 'descriptive_only')

    def test_exact_rate_and_difference_conservation(self):
        for alpha in [Decimal('.2'), Decimal('.99999999999999999999'), None]:
            doc = fixture(); doc['alpha'] = alpha
            result = diagnose_saved_interval_reference(doc)
            for weighting in ['row_weighted', 'equal_group']:
                rates = result[weighting]['observed_rate_exact']
                self.assertEqual(sum(map(Fraction, rates.values())), 1)
                differences = result[weighting]['observed_minus_declared_exact']
                if alpha is not None:
                    self.assertEqual(sum(map(Fraction, differences.values())), 0)
                else:
                    self.assertIsNone(differences)

    def test_null_alpha_keeps_observed_rates(self):
        doc = fixture(); old = diagnose_saved_interval_reference(doc)
        doc['alpha'] = None; new = diagnose_saved_interval_reference(doc)
        self.assertIsNone(new['declared_reference_exact'])
        self.assertIsNone(new['alpha_exact'])
        for weighting in ['row_weighted', 'equal_group']:
            self.assertEqual(new[weighting]['observed_rate_exact'], old[weighting]['observed_rate_exact'])
            self.assertIsNone(new[weighting]['observed_minus_declared_exact'])

    def test_permutation_and_input_immutability(self):
        doc = fixture(); before = copy.deepcopy(doc)
        result = diagnose_saved_interval_reference(doc)
        self.assertEqual(doc, before)
        doc['rows'].reverse(); doc['expected_row_ids'].reverse()
        self.assertEqual(diagnose_saved_interval_reference(doc), result)

    def test_group_replication_preserves_equal_group_rates(self):
        doc = fixture(); old = diagnose_saved_interval_reference(doc)
        for row in copy.deepcopy(doc['rows'][:3]):
            row['row_id'] += '-copy'
            doc['rows'].append(row); doc['expected_row_ids'].append(row['row_id'])
        new = diagnose_saved_interval_reference(doc)
        self.assertEqual(new['equal_group'], old['equal_group'])
        self.assertNotEqual(new['row_weighted']['observed_rate_exact'], old['row_weighted']['observed_rate_exact'])

    def test_positive_scale_and_translation_preserve_rates(self):
        doc = fixture(); old = diagnose_saved_interval_reference(doc)
        for row in doc['rows']:
            for key in ['observed', 'lower', 'upper']:
                row[key] = row[key] * 7 + 100
        new = diagnose_saved_interval_reference(doc)
        for weighting in ['row_weighted', 'equal_group']:
            self.assertEqual(new[weighting], old[weighting])

    def test_one_group_matches_row_weighting(self):
        doc = fixture()
        for row in doc['rows']: row['group'] = 'one'
        result = diagnose_saved_interval_reference(doc)
        for key in ['observed_rate_exact', 'observed_minus_declared_exact']:
            self.assertEqual(result['row_weighted'][key], result['equal_group'][key])

    def test_singleton_point_is_covered_not_failure(self):
        doc = fixture(); doc['rows'] = doc['rows'][:1]; doc['expected_row_ids'] = ['a']
        doc['rows'][0].update(observed=2, lower=2, upper=2)
        result = diagnose_saved_interval_reference(doc)
        self.assertEqual(result['row_weighted']['observed_minus_declared_exact'], {
            'coverage': '1/5', 'below_lower': '-1/10', 'above_upper': '-1/10'})
        self.assertNotIn('pass', result)
        self.assertEqual(result['status'], 'complete_descriptive_reference_diagnostic')

    def test_tiny_alpha_half_remains_exact_without_underflow(self):
        doc = fixture(); doc['rows'] = doc['rows'][:1]; doc['expected_row_ids'] = ['a']
        doc['rows'][0].update(observed=0, lower=0, upper=0)
        doc['alpha'] = math.nextafter(0.0, 1.0)
        alpha = Fraction.from_float(doc['alpha'])
        result = diagnose_saved_interval_reference(doc)
        self.assertEqual(Fraction(result['declared_reference_exact']['below_lower']), alpha / 2)
        self.assertEqual(Fraction(result['row_weighted']['observed_minus_declared_exact']['coverage']), alpha)
        self.assertEqual(evaluate_saved_intervals(doc)['nominal_coverage_approx'], 1)
        json.dumps(result, allow_nan=False)

    def test_exact_decimal_endpoints_and_large_integer(self):
        for observed, endpoint, expected in [
            (Decimal('.100000000000000000001'), Decimal('.1'), 'above_upper'),
            (Decimal('.099999999999999999999'), Decimal('.1'), 'below_lower'),
            (Decimal('.1'), Decimal('.1'), 'coverage'),
            (2**60 + 1, 2**60, 'above_upper')]:
            with self.subTest(expected=expected, observed=observed):
                doc = fixture(); doc['rows'] = doc['rows'][:1]; doc['expected_row_ids'] = ['a']
                doc['rows'][0].update(observed=observed, lower=endpoint, upper=endpoint)
                rates = diagnose_saved_interval_reference(doc)['row_weighted']['observed_rate_exact']
                self.assertEqual(rates[expected], '1')
                self.assertEqual(sum(map(Fraction, rates.values())), 1)

    def test_exact_zero_gaps_are_canonical_zero(self):
        doc = fixture(); doc['alpha'] = Decimal('.5')
        gaps = diagnose_saved_interval_reference(doc)['row_weighted']['observed_minus_declared_exact']
        self.assertEqual(set(gaps.values()), {'0'})

    def test_default_report_unchanged_and_separate(self):
        doc = fixture(); before = evaluate_saved_intervals(doc)
        sidecar = diagnose_saved_interval_reference(doc)
        self.assertEqual(evaluate_saved_intervals(doc), before)
        self.assertNotIn('coverage_reference_diagnostics', before)
        sidecar['metadata']['value_unit'] = 'changed'
        sidecar['cohort']['scored_rows'] = -1
        self.assertEqual(evaluate_saved_intervals(doc), before)

    def test_json_round_trip_is_deterministic(self):
        doc = fixture(); doc['alpha'] = None
        parsed = load_document(json.dumps(doc))
        first = json.dumps(diagnose_saved_interval_reference(parsed), sort_keys=True, allow_nan=False)
        second = json.dumps(diagnose_saved_interval_reference(parsed), sort_keys=True, allow_nan=False)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)['schema_id'], 'mlcalc.saved_interval_reference_diagnostic')

    def test_invalid_rows_reject_whole_cohort(self):
        cases = []
        doc = fixture(); doc['rows'][-1]['status'] = 'failed'; cases.append(doc)
        doc = fixture(); doc['rows'].pop(); cases.append(doc)
        doc = fixture(); doc['rows'][-1]['row_id'] = 'a'; cases.append(doc)
        doc = fixture(); doc['expected_row_ids'].append('a'); cases.append(doc)
        doc = fixture(); doc['rows'][0]['observed'] = None; cases.append(doc)
        doc = fixture(); doc['rows'][0]['lower'] = 2; cases.append(doc)
        doc = fixture(); doc['rows'][0]['observed'] = float('nan'); cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                diagnose_saved_interval_reference(doc)

    def test_invalid_alpha_is_not_estimated(self):
        for alpha in [0, 1, -1, True, '0.2', float('inf')]:
            doc = fixture(); doc['alpha'] = alpha
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                diagnose_saved_interval_reference(doc)

    def test_no_fit_predict_network_or_file_io(self):
        from sklearn.linear_model import Ridge
        from sklearn.dummy import DummyRegressor
        doc = fixture()
        with patch.object(Ridge, 'fit', side_effect=AssertionError('fit forbidden')), \
             patch.object(Ridge, 'predict', side_effect=AssertionError('predict forbidden')), \
             patch.object(DummyRegressor, 'fit', side_effect=AssertionError('fit forbidden')), \
             patch.object(DummyRegressor, 'predict', side_effect=AssertionError('predict forbidden')), \
             patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden')), \
             patch('builtins.open', side_effect=AssertionError('file IO forbidden')):
            self.assertEqual(diagnose_saved_interval_reference(doc)['cohort']['scored_rows'], 4)


if __name__ == '__main__':
    unittest.main()
