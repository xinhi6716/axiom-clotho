"""Invented paired central intervals; no fit or calibration claim."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from itertools import combinations_with_replacement, product
import unittest

import numpy as np

from groupbench.interval_evaluation import (
    bound_paired_interval_scores,
    compare_saved_interval_scores,
    diagnose_saved_interval_whole_group_coverage,
)


def document(last, scale=1, offset=0):
    ids = [str(i) for i in range(10)]
    rows = [{"row_id": row_id, "group": "early" if i < 9 else "late", "status": "ok",
             "observed": offset + scale * int(i == 9),
             "lower": offset + scale * (last if i == 9 else 0),
             "upper": offset + scale * (last + 1 if i == 9 else 1)}
            for i, row_id in enumerate(ids)]
    return {"schema_version": 1, "alpha": Decimal("0.1"), "expected_row_ids": ids, "rows": rows,
            "metadata": {"target_definition": "toy", "value_unit": "toy unit",
                         "group_definition": "declared toy grouping", "cohort_definition": "frozen toy"}}


class PairedIntervalScoreTests(unittest.TestCase):
    def test_direct_python_status_must_be_literal_text(self):
        good = document(-1)
        for status in (np.array("ok"), np.array(["ok"]), np.array(["failed", "ok"])):
            bad = deepcopy(good)
            bad["rows"][0]["status"] = status
            with self.subTest(shape=status.shape):
                with self.assertRaisesRegex(ValueError, "non-ok row"):
                    diagnose_saved_interval_whole_group_coverage(bad)
                with self.assertRaisesRegex(ValueError, "non-ok row"):
                    compare_saved_interval_scores(good, bad)
        self.assertEqual(compare_saved_interval_scores(good, good)["n_rows"], 10)

    def test_same_coverage_and_width_unequal_miss_distance_and_score(self):
        a, b = document(-1), document(-10)
        before = deepcopy((a, b))
        results = compare_saved_interval_scores(a, b)
        self.assertEqual((a, b), before)
        score = results["row_weighted"]
        for arm, total, distance in (("a", "3", 0.1), ("b", "21", 1)):
            self.assertEqual(score[arm]["coverage_approx"], 0.9)
            self.assertEqual(score[arm]["width_approx"], 1)
            self.assertEqual(score[arm]["lower_miss_distance_approx"], 0)
            self.assertEqual(score[arm]["upper_miss_distance_approx"], distance)
            self.assertEqual(score[arm]["interval_score_exact"], total)
        self.assertEqual(score["b_minus_a"]["interval_score_exact"], "18")
        self.assertEqual(results["equal_group"]["b_minus_a"]["interval_score_exact"], "90")
        b["rows"].reverse()
        b["expected_row_ids"].reverse()
        self.assertEqual(compare_saved_interval_scores(a, b), results)

    def test_shared_positive_unit_transform(self):
        results = compare_saved_interval_scores(document(-1, 1000, 10), document(-10, 1000, 10))
        row = results["row_weighted"]
        self.assertEqual(row["a"]["coverage_approx"], 0.9)
        self.assertEqual(row["a"]["width_approx"], 1000)
        self.assertEqual(row["a"]["interval_score_exact"], "3000")
        self.assertEqual(row["b"]["interval_score_exact"], "21000")

    def test_aligned_contract_rejects_mismatch(self):
        a = document(-1)
        for case, error in (
            ("alpha", "nonnull exact alpha"), ("null", "nonnull exact alpha"),
            ("unit", "metadata"), ("target", "metadata"), ("cohort", "metadata"),
            ("group", "bindings"), ("observed", "exact observations"),
            ("roster", "roster"), ("missing", "Missing or extra"), ("duplicate", "Duplicate observed"),
            ("reversed", "Reversed interval"), ("failed", "non-ok"),
        ):
            b = document(-10)
            if case == "alpha":  # float and Decimal 0.1 differ exactly.
                b["alpha"] = 0.1
            elif case == "null":
                b["alpha"] = None
            elif case in ("unit", "target", "cohort"):
                b["metadata"][{"unit": "value_unit", "target": "target_definition",
                               "cohort": "cohort_definition"}[case]] = "other"
            elif case in ("group", "observed"):
                b["rows"][0][case] = "late" if case == "group" else 1
            elif case == "roster":
                b["expected_row_ids"][0] = b["rows"][0]["row_id"] = "other"
            elif case == "missing":
                b["rows"].pop()
            elif case == "duplicate":
                b["rows"][0]["row_id"] = b["rows"][1]["row_id"]
            elif case == "reversed":
                b["rows"][0]["lower"] = 2
            elif case == "failed":
                b["rows"][0]["status"] = "failed"
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, error):
                compare_saved_interval_scores(a, b)


    def test_same_display_group_labels_do_not_merge_text_and_numeric_namespaces(self):
        a, b = document(-1), document(-10)
        for i, (one, two) in enumerate(zip(a["rows"], b["rows"])):
            one["group"] = i
            two["group"] = str(i)
        with self.assertRaisesRegex(ValueError, "row-to-group bindings"):
            compare_saved_interval_scores(a, b)


class BoundedPairedIntervalScoreTests(unittest.TestCase):
    def test_internal_knot_changes_the_conclusion_and_group_weights(self):
        rows = [('ambiguous', 0, 2, 0, 2, 1, 1), ('known-1', 0, 0, 0, 2, 0, 1),
                ('known-2', 0, 0, 0, 2, 0, 1)]
        roster = [('ambiguous', 'large'), ('known-1', 'large'), ('known-2', 'small')]
        result = bound_paired_interval_scores(rows, expected_roster=roster, alpha=Decimal('.2'))
        self.assertEqual(result['row_weighted_b_minus_a_bounds'], (Fraction(-4, 3), Fraction(2)))
        self.assertEqual(result['equal_group_b_minus_a_bounds'], (Fraction(-5, 4), Fraction(5, 4)))
        self.assertEqual((result['n_rows'], result['n_declared_groups']), (3, 2))
        self.assertEqual(result['alpha_exact'], Fraction(1, 5))
        swapped = [row[:3] + row[5:] + row[3:5] for row in rows[::-1]]
        reverse = bound_paired_interval_scores(swapped, expected_roster=roster[::-1], alpha=Decimal('.2'))
        for key in ('row_weighted_b_minus_a_bounds', 'equal_group_b_minus_a_bounds'):
            low, high = result[key]
            self.assertEqual(reverse[key], (-high, -low))

    def test_exhaustive_small_exact_grid_against_direct_score(self):
        intervals = list(combinations_with_replacement(range(-1, 3), 2))
        for truth, a, b in product(intervals, repeat=3):
            alternatives = []
            for tick in range(2 * truth[0], 2 * truth[1] + 1):
                y = Fraction(tick, 2)
                scores = []
                for lower, upper in (a, b):
                    scores.append(upper - lower + 8 * (abs(y - min(max(y, lower), upper))))
                alternatives.append(scores[1] - scores[0])
            result = bound_paired_interval_scores(
                [('row', *truth, *a, *b)], expected_roster=[('row', 'g')], alpha=.25)
            self.assertEqual(result['row_weighted_b_minus_a_bounds'],
                             (min(alternatives), max(alternatives)))

    def test_bounds_collapse_to_the_complete_paired_interval_score(self):
        a, b = document(-1), document(-10)
        exact = compare_saved_interval_scores(a, b)
        rows = [(one['row_id'], one['observed'], one['observed'],
                 one['lower'], one['upper'], two['lower'], two['upper'])
                for one, two in zip(a['rows'], b['rows'])]
        roster = [(r['row_id'], r['group']) for r in a['rows']]
        result = bound_paired_interval_scores(rows, expected_roster=roster, alpha=a['alpha'])
        for key in ('row_weighted', 'equal_group'):
            complete = Fraction(exact[key]['b_minus_a']['interval_score_exact'])
            self.assertEqual(result[key + '_b_minus_a_bounds'], (complete, complete))

    def test_invalid_design_is_never_subset_scored(self):
        row = ('a', 0, 2, 0, 2, 1, 1)
        roster = [('a', 'g'), ('b', 'g')]
        for rows in ([row], [row, row], [row, ('extra', *row[1:])],
                     [row, ('b', 2, 0, 0, 2, 1, 1)],
                     [row, ('b', 0, 2, 2, 0, 1, 1)], [row, ('b', True, 2, 0, 2, 1, 1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                bound_paired_interval_scores(rows, expected_roster=roster, alpha=.25)
        for alpha in (0., 1., True, None, '0.25'):
            with self.assertRaises(ValueError):
                bound_paired_interval_scores([row], expected_roster=[('a', 'g')], alpha=alpha)
        with self.assertRaisesRegex(ValueError, 'unique'):
            bound_paired_interval_scores([row], expected_roster=[('a', 'g'), ('a', 'h')], alpha=.25)
