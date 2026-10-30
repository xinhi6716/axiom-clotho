"""Invented intervals: exact tail direction, weighting and boundary invariants."""
import copy
from decimal import Decimal
import unittest
from groupbench.interval_evaluation import evaluate_saved_intervals


def document():
    return {"schema_version": 1, "metadata": {
        "target_definition": "invented voltage", "value_unit": "V",
        "group_definition": "declared synthetic device", "cohort_definition": "four invented rows"},
        "expected_row_ids": ["a", "b", "c", "d"], "alpha": None,
        "rows": [{"row_id": name, "group": group, "observed": value,
                  "lower": 0, "upper": 1, "status": "ok"}
                 for name, group, value in [("a", "A", -1), ("b", "A", 0),
                                            ("c", "A", 1), ("d", "B", 2)]]}


class TailMissTests(unittest.TestCase):
    def test_unequal_groups_hand_oracle(self):
        report = evaluate_saved_intervals(document())
        tails = report["tail_misses"]
        self.assertEqual(tails["row_weighted"], {"n_rows": 4, "below_lower": 1,
            "above_upper": 1, "below_lower_rate_approx": .25, "above_upper_rate_approx": .25})
        self.assertEqual(tails["equal_group"], {"n_groups": 2,
            "below_lower_rate_approx": 1/6, "above_upper_rate_approx": .5})
        self.assertEqual([r["below_lower"] for r in tails["groups"]], [1, 0])
        self.assertEqual([r["above_upper"] for r in tails["groups"]], [0, 1])
        for row in report["observations"]:
            self.assertEqual(sum(row[key] for key in ("covered", "below_lower", "above_upper")), 1)

    def test_exact_point_endpoints_not_rounded_to_coverage(self):
        for observed, point, low, high in [(2**60+1, 2**60, False, True),
                (Decimal('0.099999999999999999999'), Decimal('.1'), True, False),
                (Decimal('.1'), Decimal('.1'), False, False)]:
            doc = document(); doc['rows'] = doc['rows'][:1]; doc['expected_row_ids'] = ['a']
            doc['rows'][0].update(observed=observed, lower=point, upper=point)
            row = evaluate_saved_intervals(doc)['observations'][0]
            self.assertEqual((row['below_lower'], row['above_upper']), (low, high))

    def test_reordering_no_mutation_and_group_replication(self):
        doc = document(); before = copy.deepcopy(doc)
        result = evaluate_saved_intervals(doc)
        self.assertEqual(doc, before)
        doc['rows'].reverse(); doc['expected_row_ids'].reverse()
        self.assertEqual(evaluate_saved_intervals(doc), result)
        for row in copy.deepcopy(before['rows'][:3]):
            row['row_id'] += '-copy'; doc['rows'].append(row); doc['expected_row_ids'].append(row['row_id'])
        newer = evaluate_saved_intervals(doc)['tail_misses']
        self.assertEqual(newer['equal_group'], result['tail_misses']['equal_group'])
        self.assertNotEqual(newer['row_weighted'], result['tail_misses']['row_weighted'])

    def test_alpha_does_not_change_tail_diagnostics(self):
        doc = document(); original = evaluate_saved_intervals(doc)['tail_misses']
        doc['alpha'] = Decimal('.2')
        self.assertEqual(evaluate_saved_intervals(doc)['tail_misses'], original)

    def test_reject_dont_drop_failed_row(self):
        doc = document(); doc['rows'][-1]['status'] = 'failed'
        with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
