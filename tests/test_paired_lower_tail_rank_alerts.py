from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_lower_tail_rank_alerts


class PairedLowerTailRankAlertTests(unittest.TestCase):
    def test_same_marginal_rate_hides_discordance_and_group_shift(self):
        cal = [("c", 1, 2)]
        cr = [("c", "training")]
        query = [("n", 1, 2), ("b", 1, 1), ("a", 0, 2), ("both", 0, 1)]
        qr = [("both", "large"), ("a", "small"), ("b", "large"), ("n", "large")]
        result = paired_lower_tail_rank_alerts(cal, query, expected_calibration=cr,
                                               expected_queries=qr, alpha=Fraction(1, 2))
        self.assertEqual(result["pooled"], {
            "n_queries": 4, "cells": dict.fromkeys(("00", "01", "10", "11"), 1),
            "alert_rate_b_minus_a": 0, "discordance_rate": Fraction(1, 2)})
        self.assertEqual(result["equal_group_alert_rate_b_minus_a"], Fraction(-1, 3))
        self.assertEqual(result, paired_lower_tail_rank_alerts(
            cal, query[::-1], expected_calibration=cr, expected_queries=qr[::-1], alpha=Fraction(1, 2)))

    def test_exhaustive_count_matches_direct_conservative_rank(self):
        for ca, cb, qa, qb in product(range(3), repeat=4):
            cal = [("c1", ca, cb), ("c2", 1, 1)]
            for alpha in (Fraction(1, 20), Fraction(1, 3), Fraction(2, 3)):
                actual = paired_lower_tail_rank_alerts(
                    cal, [("q", qa, qb)], expected_calibration=[("c1", "c"), ("c2", "c")],
                    expected_queries=[("q", "q")], alpha=alpha)
                cell = "".join(str(int(Fraction(1 + (c <= q) + (1 <= q), 3) <= alpha))
                               for c, q in ((ca, qa), (cb, qb)))
                self.assertEqual(actual["pooled"]["cells"][cell], 1)

    def test_rejects_missing_repeated_malformed_and_cross_cohort_membership(self):
        cal = [("c", 0, 0)]
        query = [("q", 1, 1), ("r", 1, 1)]
        roster = [("q", "held"), ("r", "held")]
        for rows, expected in (
            (query[:1], roster), (query[:1] * 2, roster),
            (query, [("q", "cal"), ("r", "held")]),
            ([("c", 0, 0), query[-1]], [("c", "held"), ("r", "held")]),
            ([("q", 1, None), query[-1]], roster),
        ):
            with self.subTest(rows=rows, expected=expected), self.assertRaises(ValueError):
                paired_lower_tail_rank_alerts(cal, rows, expected_calibration=[("c", "cal")],
                                               expected_queries=expected, alpha=Fraction(1, 2))
