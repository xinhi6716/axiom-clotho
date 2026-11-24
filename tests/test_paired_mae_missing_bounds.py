"""Frozen-roster missing-target bounds, evaluated only on synthetic inputs."""
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

from groupbench.metrics import paired_mae_missing_bounds


ROSTER = [f"row-{i}" for i in range(100)]


class PairedMAEMissingBoundsTests(unittest.TestCase):
    def test_exact_100_row_ranking_reversal_and_attainable_endpoints(self):
        rows = [(f"row-{i}", 1.25 if i < 80 else None, 0, 3)
                for i in range(100)]
        expected = {
            "n_rows": 100, "n_observed": 80, "n_missing": 20,
            "available_mae_b_minus_a": Fraction(1, 2),
            "full_mae_b_minus_a_lower": Fraction(-1, 5),
            "full_mae_b_minus_a_upper": Fraction(1),
        }
        for roster in (ROSTER, ROSTER[::-1], tuple(ROSTER), frozenset(ROSTER)):
            self.assertEqual(paired_mae_missing_bounds(
                rows[::-1], expected_roster=roster), expected)
        for fill_a, key in ((False, "full_mae_b_minus_a_lower"),
                            (True, "full_mae_b_minus_a_upper")):
            completed = [(i, (a if fill_a else b) if y is None else y, a, b)
                         for i, y, a, b in rows]
            got = paired_mae_missing_bounds(completed, expected_roster=ROSTER)
            self.assertEqual(got["full_mae_b_minus_a_lower"], expected[key])
            self.assertEqual(got["full_mae_b_minus_a_upper"], expected[key])

    def test_all_missing_including_identical_and_oppositely_ordered_predictions(self):
        rows = [("a", None, 7, 7), ("b", None, 1, 4), ("c", None, 6, 0)]
        got = paired_mae_missing_bounds(rows, expected_roster=["a", "c", "b"])
        self.assertIsNone(got["available_mae_b_minus_a"])
        self.assertEqual((got["n_observed"], got["n_missing"]), (0, 3))
        self.assertEqual((got["full_mae_b_minus_a_lower"], got["full_mae_b_minus_a_upper"]),
                         (Fraction(-3), Fraction(3)))
        zero = paired_mae_missing_bounds([("a", None, 7, 7)], expected_roster=["a"])
        self.assertEqual(zero["full_mae_b_minus_a_lower"], Fraction())
        self.assertEqual(zero["full_mae_b_minus_a_upper"], Fraction())

    def test_exact_at_extreme_scales_without_false_zero_or_infinite_bounds(self):
        huge, tiny = float.fromhex("0x1.fffffffffffffp+1023"), math.ulp(0.0)
        a = paired_mae_missing_bounds([("a", huge, 0, tiny)], expected_roster=["a"])
        self.assertEqual(a["available_mae_b_minus_a"], -Fraction.from_float(tiny))
        self.assertIsInstance(a["available_mae_b_minus_a"], Fraction)
        b = paired_mae_missing_bounds([("a", None, huge, -huge)], expected_roster=["a"])
        self.assertEqual(b["full_mae_b_minus_a_upper"], Fraction.from_float(huge) * 2)

    def test_accumulates_without_row_fractions_and_matches_independent_oracle(self):
        scales = [math.ldexp(1., e) for e in (-1074, -600, -100, 0, 80, 500, 1023)]
        rows = [(str(i), None if i % 3 == 0 else a, -a, b)
                for i, (a, b) in enumerate((a, b) for a in scales for b in reversed(scales))]
        observed, radii = [], []
        for _, y, a, b in rows:
            exact_a, exact_b = map(Fraction.from_float, (a, b))
            if y is None:
                radii.append(abs(exact_b - exact_a))
            else:
                truth = Fraction.from_float(y)
                observed.append(abs(exact_b - truth) - abs(exact_a - truth))
        observed_sum, radius = sum(observed), sum(radii)
        with patch.object(Fraction, "from_float", side_effect=AssertionError("row Fraction conversion")):
            result = paired_mae_missing_bounds(rows, expected_roster=[row[0] for row in rows])
        self.assertEqual(result["available_mae_b_minus_a"], observed_sum / len(observed))
        self.assertEqual(result["full_mae_b_minus_a_lower"], (observed_sum - radius) / len(rows))
        self.assertEqual(result["full_mae_b_minus_a_upper"], (observed_sum + radius) / len(rows))

    def test_malformed_rosters_are_not_silently_rebuilt_from_available_rows(self):
        row = ("ok", 0, 0, 1)
        for rows, roster in (([], ["ok"]), ([row], []), ([row], ["ok", "ok"]),
                             ([row], ["ok", "absent"]), ([row], ["wrong"]),
                             ([row, row], ["ok"]), ([row], "ok"),
                             ([row], [" ok"]), ([row], [1]), ([row], ["\ud800"]),
                             ([row[:3]], ["ok"]), ([row], ["ok"] * 20_001)):
            with self.subTest(rows=rows, roster_size=len(roster)), self.assertRaises(ValueError):
                paired_mae_missing_bounds(rows, expected_roster=roster)

    def test_invalid_labels_or_forecasts_never_become_missing_truth(self):
        invalids = [float("nan"), float("inf"), -float("inf"), True, 2**53 + 1, 1j, "1"]
        for pos in (1, 2, 3):
            for invalid in invalids + ([None] if pos != 1 else []):
                row = ["ok", 0, 0, 1] if pos == 1 else ["ok", None, 0, 1]
                row[pos] = invalid
                with self.subTest(position=pos, invalid=invalid), self.assertRaises(ValueError):
                    paired_mae_missing_bounds([tuple(row)], expected_roster=["ok"])


if __name__ == "__main__":
    unittest.main()
