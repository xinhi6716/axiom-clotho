from fractions import Fraction
from itertools import product
from math import nextafter
import unittest

from groupbench.metrics import lower_tail_rank_floor_audit


class LowerTailRankFloorAuditTests(unittest.TestCase):
    def test_five_percent_can_be_impossible_even_below_minimum(self):
        result = lower_tail_rank_floor_audit([0] * 18, [-1, 0], alpha=Fraction(1, 20))
        self.assertEqual(result["algebraic_p_floor"], Fraction(1, 19))
        self.assertEqual(result["query_p_floor"], Fraction(1, 19))
        self.assertFalse(result["attainable_for_some_real_score"])
        self.assertIsNone(result["strict_score_cutoff"])
        self.assertEqual(result["queries_flagged"], 0)

    def test_ties_can_prevent_observed_flags_when_algebraically_possible(self):
        calibration = [0] * 4 + [1] * 15
        alpha = Fraction(1, 20)
        result = lower_tail_rank_floor_audit(calibration, [0, 0, 1], alpha=alpha)
        self.assertTrue(result["attainable_for_some_real_score"])
        self.assertEqual(result["algebraic_p_floor"], alpha)
        self.assertEqual(result["query_p_floor"], Fraction(5, 20))
        self.assertEqual(result["calibration_minimum_fraction"], Fraction(4, 19))
        self.assertEqual(result["queries_tied_at_calibration_minimum"], 2)
        self.assertEqual(result["queries_flagged"], 0)
        self.assertEqual(result["query_minimum_minus_cutoff"], 0)
        control = lower_tail_rank_floor_audit(calibration, [0, 0, -0.25], alpha=alpha)
        self.assertEqual(control["queries_flagged"], 1)
        self.assertEqual(control["query_minimum_minus_cutoff"], Fraction(-1, 4))
        self.assertEqual(control["strict_score_cutoff"], 0)

    def test_higher_cutoff_and_all_counts_match_direct_computation(self):
        queries = [-1, 0, 1, 2, 3]
        for calibration in product(range(3), repeat=4):
            for alpha in (Fraction(1, 20), Fraction(1, 5), Fraction(2, 5), Fraction(4, 5)):
                out = lower_tail_rank_floor_audit(calibration, queries, alpha=alpha)
                pvalues = [Fraction(1 + sum(c <= x for c in calibration), 5) for x in queries]
                self.assertEqual(out["query_p_floor"], min(pvalues))
                self.assertEqual(out["queries_flagged"], sum(p <= alpha for p in pvalues))
                cutoff = out["strict_score_cutoff"]
                if cutoff is not None:
                    self.assertEqual(out["queries_flagged"], sum(x < cutoff for x in queries))

    def test_missing_invalid_or_rounded_inputs_rejected(self):
        for cal, queries, alpha in (([], [1], Fraction(1, 20)), ([1], [], Fraction(1, 20)),
                                    ([1], [0], 0.05), ([1], [True], Fraction(1, 20)),
                                    ([float("inf")], [0], Fraction(1, 20))):
            with self.subTest(cal=cal, queries=queries, alpha=alpha), self.assertRaises(ValueError):
                lower_tail_rank_floor_audit(cal, queries, alpha=alpha)

    def test_certified_lower_bound_distinguishes_structural_and_observed_floor(self):
        calibration = [0] * 4 + [1] * 15
        alpha = Fraction(1, 20)
        closed = lower_tail_rank_floor_audit(calibration, [0, 1], alpha=alpha, score_lower_bound=0)
        self.assertTrue(closed["attainable_for_some_real_score"])
        self.assertFalse(closed["attainable_in_declared_domain"])
        self.assertEqual(closed["declared_domain_p_floor"], Fraction(1, 4))
        lo = nextafter(0.0, -1.0)
        feasible = lower_tail_rank_floor_audit(calibration, [0, 1], alpha=alpha, score_lower_bound=lo)
        self.assertTrue(feasible["attainable_in_declared_domain"])
        self.assertEqual(feasible["declared_domain_p_floor"], alpha)
        self.assertEqual(feasible["queries_flagged"], 0)
        realized = lower_tail_rank_floor_audit(calibration, [lo], alpha=alpha, score_lower_bound=lo)
        self.assertEqual(realized["queries_flagged"], 1)
        for cal, query in (([-1], [0]), ([0], [-1])):
            with self.assertRaisesRegex(ValueError, "below the declared"):
                lower_tail_rank_floor_audit(cal, query, alpha=alpha, score_lower_bound=0)
