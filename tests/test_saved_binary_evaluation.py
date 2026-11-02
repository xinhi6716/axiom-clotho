"""Invented no-fit cases for complete saved binary probability cohorts."""
import copy
from decimal import Decimal
import json
import socket
import unittest
from unittest.mock import patch

import numpy as np

from groupbench.binary.metrics import grouped_binary_metrics
from groupbench.binary.saved_evaluation import evaluate_saved_binary_probabilities, MAX_ROWS


def document():
    return {
        'schema_version': 1,
        'metadata': {'target_definition': 'invented pass/fail',
                     'positive_class_definition': '1 means fail',
                     'negative_class_definition': '0 means pass',
                     'prediction_definition': 'frozen invented probabilities',
                     'group_definition': 'declared batch',
                     'cohort_definition': 'all four invented observations'},
        'expected_row_ids': ['a', 'b', 'c', 'd'],
        'rows': [dict(row_id=key, group=group, observed_label=y, probability_1=p, status='ok')
                 for key, group, y, p in [('a', 'A', 0, 0.), ('b', 'A', 0, .5),
                                           ('c', 'A', 0, .25), ('d', 'B', 1, 1.)]],
    }


class SavedBinaryEvaluationTests(unittest.TestCase):
    def test_existing_metrics_reused_and_hand_brier_oracle(self):
        result = evaluate_saved_binary_probabilities(document())
        expected, rows = grouped_binary_metrics([0, 0, 0, 1], [0., .5, .25, 1.], ['A', 'A', 'A', 'B'])
        self.assertEqual(result['metrics'], expected)
        self.assertEqual(result['groups'], rows)
        self.assertEqual(result['metrics']['brier'], .078125)
        self.assertAlmostEqual(result['metrics']['group_brier'], .052083333333333336)
        self.assertEqual(result['metrics']['confusion'], {'tn': 2, 'fp': 1, 'fn': 0, 'tp': 1})

    def test_canonical_row_order_and_correspondence(self):
        doc = document(); expected = evaluate_saved_binary_probabilities(doc)
        doc['rows'].reverse(); doc['expected_row_ids'].reverse()
        self.assertEqual(evaluate_saved_binary_probabilities(doc), expected)
        self.assertEqual([(r['row_id'], r['probability_1']) for r in expected['observations']],
                         [('a', 0.), ('b', .5), ('c', .25), ('d', 1.)])

    def test_input_and_output_metadata_ownership(self):
        doc = document(); before = copy.deepcopy(doc)
        out = evaluate_saved_binary_probabilities(doc)
        self.assertEqual(doc, before)
        out['metadata']['target_definition'] = 'changed'
        out['observations'][0]['observed_label'] = 1
        out['groups'][0]['confusion']['tn'] = -99
        self.assertEqual(doc, before)
        self.assertEqual(evaluate_saved_binary_probabilities(doc)['metrics']['confusion']['tn'], 2)

    def test_schema_provenance_and_complete_cohort(self):
        result = evaluate_saved_binary_probabilities(document())
        self.assertEqual(result['schema_id'], 'mlcalc.saved_binary_probability_evaluation')
        self.assertEqual(result['schema_version'], '1.0')
        self.assertEqual(result['input_schema_version'], 1)
        self.assertEqual(result['cohort'], {'expected_rows': 4, 'input_rows': 4, 'scored_rows': 4, 'omitted_rows': 0})
        self.assertFalse(result['provenance_verified'])
        self.assertEqual(result['claim_scope'], 'descriptive_only')
        self.assertEqual(result['metric_contract']['threshold'], .5)
        self.assertNotIn('auc', result['metrics'])
        json.dumps(result, allow_nan=False)

    def test_unequal_group_and_single_class_support(self):
        result = evaluate_saved_binary_probabilities(document())
        self.assertEqual(result['support'], {'n_declared_groups': 2, 'independence': 'not_established',
                         'n_groups_with_both_classes': 0, 'n_groups_only_class_0': 1, 'n_groups_only_class_1': 1})
        self.assertIsNone(result['groups'][0]['recall_1'])
        self.assertIsNone(result['groups'][1]['specificity_0'])
        for label in [0, 1]:
            doc = document()
            for row in doc['rows']: row['observed_label'] = label
            out = evaluate_saved_binary_probabilities(doc)
            self.assertIsNone(out['metrics']['balanced_accuracy'])
            self.assertEqual(out['cohort']['scored_rows'], 4)

    def test_singleton_and_numeric_endpoints_supported(self):
        doc = document(); doc['rows'] = doc['rows'][:1]; doc['expected_row_ids'] = ['a']
        doc['rows'][0]['probability_1'] = 0
        result = evaluate_saved_binary_probabilities(doc)
        self.assertEqual(result['metrics']['brier'], 0.)
        self.assertEqual(result['observations'][0]['predicted_label'], 0)

    def test_duplicate_missing_extra_failed_and_unknown_status_reject_whole(self):
        cases = []
        doc = document(); doc['rows'].pop(); cases.append(doc)
        doc = document(); doc['expected_row_ids'].pop(); cases.append(doc)
        doc = document(); doc['rows'][1]['row_id'] = 'a'; cases.append(doc)
        doc = document(); doc['expected_row_ids'][1] = 'a'; cases.append(doc)
        doc = document(); doc['rows'][0]['row_id'] = 'extra'; cases.append(doc)
        for status in ['failed', 'missing', 'OK', None, 1, ['ok']]:
            doc = document(); doc['rows'][0]['status'] = status; cases.append(doc)
        for doc in cases:
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)

    def test_schema_and_unknown_keys_reject(self):
        cases = []
        for schema in [True, 1., '1', 0, 2, None]:
            doc = document(); doc['schema_version'] = schema; cases.append(doc)
        for layer in ['document', 'metadata', 'row']:
            doc = document(); target = doc if layer == 'document' else doc['metadata'] if layer == 'metadata' else doc['rows'][0]
            target['extra'] = 'value'; cases.append(doc)
        for key in document()['rows'][0]:
            doc = document(); del doc['rows'][0][key]; cases.append(doc)
        for doc in cases:
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)

    def test_metadata_and_identifier_bounds(self):
        for value in ['', ' leading', 'trailing ', 'control\n', 'x' * 2049, None, 7]:
            doc = document(); doc['metadata']['target_definition'] = value
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)
        for value in ['', ' spaced', '\x00', 'x' * 257, None, 7]:
            doc = document(); doc['rows'][0]['row_id'] = value
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)
        doc = document(); del doc['metadata']['positive_class_definition']
        with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)

    def test_probability_rejection_without_coercion(self):
        for value in [True, np.bool_(False), None, '0.5', Decimal('.5'), np.array(.5),
                      np.ma.masked, np.float64(.5), .5 + 0j, float('nan'), float('inf'),
                      -0.1, 1.1, 10**100]:
            doc = document(); doc['rows'][0]['probability_1'] = value
            with self.subTest(type=type(value).__name__), self.assertRaises(ValueError):
                evaluate_saved_binary_probabilities(doc)

    def test_label_rejection_without_remapping(self):
        for value in [True, None, '0', Decimal('0'), np.int64(0), -1, 2, .5, float('nan')]:
            doc = document(); doc['rows'][0]['observed_label'] = value
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)
        doc = document(); doc['rows'][0]['observed_label'] = 0.
        self.assertEqual(evaluate_saved_binary_probabilities(doc)['observations'][0]['observed_label'], 0)

    def test_numeric_group_identity_not_promoted(self):
        doc = document()
        for row, group in zip(doc['rows'], [2**60, 2**60 + 1, 2**60, 2**60 + 1]): row['group'] = group
        out = evaluate_saved_binary_probabilities(doc)
        self.assertEqual(out['metrics']['n_groups'], 2)
        self.assertEqual({r['group'] for r in out['observations']}, {str(2**60), str(2**60 + 1)})
        for row, group in zip(doc['rows'], [1, 1., 2, 2.]): row['group'] = group
        self.assertEqual(evaluate_saved_binary_probabilities(doc)['metrics']['n_groups'], 2)

    def test_invalid_groups_and_mixed_types_reject(self):
        for value in [True, None, [], '', '\x00', float('nan'), float('inf'), 'x' * 1025, 10**100]:
            doc = document(); doc['rows'][0]['group'] = value
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)
        doc = document(); doc['rows'][0]['group'] = 1
        with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)

    def test_row_count_bounds_and_plain_containers(self):
        for key in ['rows', 'expected_row_ids']:
            for value in [[], (), None, [None] * (MAX_ROWS + 1)]:
                doc = document(); doc[key] = value
                with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(doc)
        class Custom(dict): pass
        with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(Custom(document()))

    def test_underflow_guard_and_caller_settings_preserved(self):
        doc = document(); doc['rows'][0]['probability_1'] = 1e-300
        with self.assertRaisesRegex(ValueError, 'scoring resolution'):
            evaluate_saved_binary_probabilities(doc)
        tiny = np.nextafter(0., 1.)
        doc = document(); doc['rows'] = doc['rows'][:2]; doc['expected_row_ids'] = ['a', 'b']
        for row, value in zip(doc['rows'], [float(3 * tiny), 0.]):
            row['observed_label'] = 1; row['probability_1'] = value
        with np.errstate(all='raise'):
            before = np.geterr().copy()
            result = evaluate_saved_binary_probabilities(doc)
            self.assertEqual(np.geterr(), before)
        self.assertGreater(result['metrics']['mean_probability_1'], 0.)

    def test_no_training_prediction_io_or_network(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.dummy import DummyClassifier
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('no fit')), \
             patch.object(LogisticRegression, 'predict_proba', side_effect=AssertionError('no predict')), \
             patch.object(DummyClassifier, 'fit', side_effect=AssertionError('no fit')), \
             patch('builtins.open', side_effect=AssertionError('no IO')), \
             patch.object(socket, 'create_connection', side_effect=AssertionError('no network')):
            self.assertEqual(evaluate_saved_binary_probabilities(document())['cohort']['scored_rows'], 4)

    def test_rejection_does_not_echo_private_payload(self):
        doc = document(); doc['rows'][0]['probability_1'] = 'PRIVATE-PAYLOAD'
        with self.assertRaises(ValueError) as caught:
            evaluate_saved_binary_probabilities(doc)
        self.assertNotIn('PRIVATE', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
