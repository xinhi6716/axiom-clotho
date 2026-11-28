"""Fixed horizon, paired roster, absorbing onset, and interval endpoints."""
from collections import Counter
from decimal import Decimal as D
from fractions import Fraction as F
from itertools import product
import random
import unittest

from groupbench.interval_evaluation import bound_paired_failure_risk_at_horizon as audit


class PairedFailureRiskTests(unittest.TestCase):
    def test_endpoint_accounting_and_original_denominator(self):
        rows = [
            ("first_at_h", 0, 1, 1, None),
            ("ambiguous_a", None, 2, 0, 1),
            ("ambiguous_b", 1, 2, None, None),
        ]
        roster = [("first_at_h", "source"), ("ambiguous_a", "source"), ("ambiguous_b", "new")]
        result = audit(rows, expected_roster=roster[::-1], horizon=D("1.00"))
        self.assertEqual((result["n_pairs"], result["n_declared_groups"]), (3, 2))
        self.assertEqual((result["a_indeterminate_pairs"], result["b_indeterminate_pairs"]), (1, 1))
        self.assertEqual(result["row_weighted"]["a_risk_bounds"], (F(1, 3), F(2, 3)))
        self.assertEqual(result["row_weighted"]["b_risk_bounds"], (F(1, 3), F(2, 3)))
        self.assertEqual(result["row_weighted"]["b_minus_a_risk_bounds"], (F(-1, 3), F(1, 3)))
        self.assertEqual(result["equal_group"]["b_minus_a_risk_bounds"], (F(-1, 4), F(1, 2)))
        at_origin = audit([("t", None, 0, 0, None)], expected_roster=[("t", "x")], horizon=0)
        self.assertEqual(at_origin["row_weighted"]["b_minus_a_risk_bounds"], (-1, -1))

    def test_exact_possible_times_and_swap_symmetry(self):
        rng = random.Random(562)
        windows = [(None, None), (None, 1), (0, 1), (1, 2), (None, 0), (0, None),
                   (1, None), (2, None), (0, 2), (None, 2)]
        grid = [F(i, 2) for i in range(7)]
        for _ in range(175):
            n = rng.randrange(1, 4)
            horizon = rng.randrange(3)
            roster = [(str(i), rng.choice(("x", "y"))) for i in range(n)]
            grouping = Counter(g for _, g in roster)
            rows, possibilities = [], []
            for key, _ in roster:
                a, b = rng.choice(windows), rng.choice(windows)

                def status(window):
                    lo, hi = window
                    possible = [t for t in grid if (lo is None or t > lo) and (hi is None or t <= hi)]
                    return {int(t <= horizon) for t in possible}

                rows.append((key, *a, *b))
                possibilities.append(set(y - x for x, y in product(status(a), status(b))))
            expected = audit(rows, expected_roster=roster, horizon=horizon)
            reverse = audit([(row[0], *row[3:], *row[1:3]) for row in rows[::-1]],
                            expected_roster=roster[::-1], horizon=horizon)
            for key in ("row_weighted", "equal_group"):
                weights = [F(1, n) if key == "row_weighted" else F(1, len(grouping) * grouping[g])
                           for _, g in roster]
                options = [sum(w * x for w, x in zip(weights, v)) for v in product(*possibilities)]
                low, high = min(options), max(options)
                self.assertEqual(expected[key]["b_minus_a_risk_bounds"], (low, high))
                self.assertEqual(reverse[key]["b_minus_a_risk_bounds"], (-high, -low))

    def test_failures_are_not_dropped_or_coerced(self):
        row = ("a", 0, 1, None, None)
        for rows, roster, horizon in (
            ([row], [("x", "g")], 1),
            ([row], [("a", "g"), ("x", "g")], 1),
            ([row, row], [("a", "g")], 1),
            ([row], [("a", "g"), ("a", "x")], 1),
            ([("a", 1, 1, 0, 1)], [("a", "g")], 1),
            ([("a", None, -1, 0, 1)], [("a", "g")], 1),
            ([("a", True, 1, None, None)], [("a", "g")], 1),
            ([("a", None, float("inf"), None, None)], [("a", "g")], 1),
            ([row], [("a", "g")], -1),
            ([row], [("a", "g")], False),
        ):
            with self.subTest(rows=rows, roster=roster, horizon=horizon), self.assertRaises(ValueError):
                audit(rows, expected_roster=roster, horizon=horizon)
