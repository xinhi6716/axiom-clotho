"""Outcome-free support comparisons for reference rows versus declared groups."""

from fractions import Fraction
from itertools import product
import sys
import unittest

from groupbench.input_support import audit_reference_group_neighbor_alerts


class ReferenceGroupNeighborAlertTests(unittest.TestCase):
    def test_recomputed_calibration_and_reference_multiplicity(self):
        reference = [0, *range(10, 42)]
        calibration = list(range(10, 49))
        for copies in range(41):
            for distinct in (False, True):
                groups = list(range(33)) + (list(range(33, 33 + copies)) if distinct else [0] * copies)
                got = audit_reference_group_neighbor_alerts(
                    reference + [0] * copies, calibration, [0],
                    reference_groups=groups, k=32, level=Fraction(1, 20))
                self.assertEqual(got["n_alerts_both"], int(copies <= 2))
                self.assertEqual(got["n_alerts_group_only"], int(copies >= 3 and not distinct))
                self.assertEqual(got["n_alerts_neither"], int(copies >= 3 and distinct))
                self.assertEqual(got["n_alerts_row_only"], 0)

    def test_closed_radius_oracle_and_order_permutations(self):
        queries, calibration, groups = [-2, 0, 2], [-1, 1, 1, 2], [0, 1, 0, 2]

        def score(values, x, k, grouped):
            for r in range(6):
                eligible = [g for p, g in zip(values, groups) if abs(p - x) <= r]
                if (len(set(eligible)) if grouped else len(eligible)) >= k:
                    return r
            raise AssertionError("fixture must be within tested radii")

        for values in product((-2, 0, 2), repeat=4):
            for k in (1, 2, 3):
                for level in (Fraction(1, 10), Fraction(2, 5), Fraction(4, 5)):
                    expected = [0] * 4
                    for x in queries:
                        flags = []
                        for grouped in (False, True):
                            q = score(values, x, k, grouped)
                            numerator = 1 + sum(score(values, c, k, grouped) >= q for c in calibration)
                            flags.append(Fraction(numerator, 1 + len(calibration)) <= level)
                        expected[2 * int(flags[0]) + int(flags[1])] += 1
                    for ref, ids in ((values, groups), (values[::-1], groups[::-1])):
                        got = audit_reference_group_neighbor_alerts(
                            ref, calibration, queries, reference_groups=ids, k=k, level=level)
                        self.assertEqual([got["n_alerts_" + q] for q in
                                          ("neither", "group_only", "row_only", "both")], expected)

    def test_binary64_extremes_and_invalid_inputs(self):
        largest = sys.float_info.max
        f = audit_reference_group_neighbor_alerts
        got = f([-largest], [largest / 2], [largest],
                reference_groups=["one"], k=1, level=Fraction(1, 2))
        self.assertEqual(got["n_alerts_both"], 1)
        got = f([1.0], [0.0], [-5e-324],
                reference_groups=["one"], k=1, level=Fraction(1, 2))
        self.assertEqual(got["n_alerts_both"], 1)
        for ref, cal, query, groups, k, level, error in (
            ([0, 0], [1], [1], [1, 1], 2, Fraction(1, 2), ValueError),
            ([0], [1], [1], [None], 1, Fraction(1, 2), ValueError),
            ([0], [1], [1], [1], True, Fraction(1, 2), ValueError),
            ([0], [1], [float("nan")], [1], 1, Fraction(1, 2), ValueError),
            ([2**53 + 1], [1], [1], [1], 1, Fraction(1, 2), ValueError),
            ([0], [1], [1], [1], 1, 0.5, TypeError),
            ([0], [1], [1], [1], 1, Fraction(1), ValueError),
            ([1]*1001, [1]*1000, [1]*1000, ["g"]*1001, 1, Fraction(1, 2), ValueError),
        ):
            with self.assertRaises(error):
                f(ref, cal, query, reference_groups=groups, k=k, level=level)
