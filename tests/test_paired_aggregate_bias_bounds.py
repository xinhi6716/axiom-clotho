from decimal import Decimal
from fractions import Fraction
from itertools import product
import unittest

from groupbench.interval_evaluation import bound_paired_aggregate_bias


class PairedAggregateBiasBoundsTests(unittest.TestCase):
    def test_common_truth_eliminates_incompatible_extrema(self):
        rows = [("x", 0, 8, 1, 2)]
        out = bound_paired_aggregate_bias(rows, expected_roster=[("x", "g")])["row_weighted"]
        self.assertEqual(out["a_signed_bias_bounds"], (-7, 1))
        self.assertEqual(out["b_signed_bias_bounds"], (-6, 2))
        self.assertEqual(out["signed_bias_b_minus_a"], 1)
        self.assertEqual(out["absolute_bias_b_minus_a_bounds"], (-1, 1))
        identical = bound_paired_aggregate_bias([("x", 0, 8, 2, 2)],
                                               expected_roster=[("x", "g")])["equal_group"]
        self.assertEqual(identical["absolute_bias_b_minus_a_bounds"], (0, 0))

    def test_decimal_point_truth_and_unequal_group_weighting(self):
        rows = [("x", 0, 2, 0, 1), ("y", 0, 2, 0, 1),
                ("z", Decimal("0.1"), Decimal("0.1"), 4, 0)]
        roster = [("x", "big"), ("y", "big"), ("z", "small")]
        out = bound_paired_aggregate_bias(rows, expected_roster=roster)
        self.assertEqual(out["row_weighted"]["mean_truth_bounds"], (Fraction(1, 30), Fraction(41, 30)))
        self.assertEqual(out["equal_group"]["mean_truth_bounds"], (Fraction(1, 20), Fraction(21, 20)))
        self.assertEqual(out["row_weighted"]["signed_bias_b_minus_a"], Fraction(-2, 3))
        self.assertEqual(out["equal_group"]["signed_bias_b_minus_a"], Fraction(-3, 2))

    def test_exhaustive_closed_integer_endpoints_and_bad_roster(self):
        roster = [("a", "wide"), ("b", "wide"), ("c", "small")]
        weights = [(Fraction(1, 3),) * 3, (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2))]
        for forecasts in product(range(3), repeat=3):
            rows = [(key, 0, 2, 1, b) for (key, _), b in zip(roster, forecasts)]
            report = bound_paired_aggregate_bias(rows, expected_roster=roster)
            for name, w in zip(("row_weighted", "equal_group"), weights):
                results = []
                for ys in product(range(3), repeat=3):
                    a = sum(weight * (1 - y) for weight, y in zip(w, ys))
                    b = sum(weight * (p - y) for weight, p, y in zip(w, forecasts, ys))
                    results.append(abs(b) - abs(a))
                self.assertEqual(report[name]["absolute_bias_b_minus_a_bounds"],
                                 (min(results), max(results)))
        for bad in (rows[:-1], [rows[0]] * 3, [("a", 2, 1, 0, 0), *rows[1:]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bound_paired_aggregate_bias(bad, expected_roster=roster)
