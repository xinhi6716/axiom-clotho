from fractions import Fraction
from itertools import product
from math import nextafter
import unittest

from groupbench.metrics import paired_fixed_lower_cutoff_tie_audit


class FixedCutoffTieAuditTests(unittest.TestCase):
    def test_exact_boundary_and_equal_aggregate_hide_per_group_switches(self):
        a_cut = 1
        b_cut = 0
        rows = [("x", 1, nextafter(0, 1)), ("y", nextafter(1, 2), -0.0),
                ("z", nextafter(1, 0), nextafter(0, -1))]
        roster = [("x", "A"), ("y", "B"), ("z", "B")]
        out = paired_fixed_lower_cutoff_tie_audit(rows, expected_roster=roster,
                                                  cutoff_a=a_cut, cutoff_b=b_cut)
        pooled = out["pooled"]
        self.assertEqual(pooled["below_equal_above_table"], ((1, 0, 0), (0, 0, 1), (0, 1, 0)))
        self.assertEqual(pooled["strict"]["alert_rate_b_minus_a"], 0)
        self.assertEqual(pooled["inclusive"]["alert_rate_b_minus_a"], 0)
        self.assertEqual(pooled["separate_tie_policy_alert_rate_b_minus_a_bounds"],
                         (Fraction(-1, 3), Fraction(1, 3)))
        self.assertEqual(out["per_group"]["A"]["inclusive"]["alert_rate_b_minus_a"], -1)
        self.assertEqual(out["per_group"]["B"]["inclusive"]["alert_rate_b_minus_a"], Fraction(1, 2))

    def test_all_joint_cells_match_direct_comparisons(self):
        rows = [(str(i), a, b) for i, (a, b) in enumerate(product((-1, 0, 1), repeat=2))]
        roster = [(key, "held") for key, *_ in rows]
        audit = paired_fixed_lower_cutoff_tie_audit(rows, expected_roster=roster, cutoff_a=0, cutoff_b=0)
        self.assertEqual(audit["pooled"]["below_equal_above_table"], ((1, 1, 1),) * 3)
        for name, fn in (("strict", lambda x: x < 0), ("inclusive", lambda x: x <= 0)):
            for i, j in product((0, 1), repeat=2):
                self.assertEqual(audit["pooled"][name]["cells"][f"{i}{j}"],
                                 sum((fn(a), fn(b)) == (i, j) for _, a, b in rows))
        self.assertEqual(audit, paired_fixed_lower_cutoff_tie_audit(
            rows[::-1], expected_roster=roster[::-1], cutoff_a=0, cutoff_b=0))

    def test_invalid_roster_and_cutoffs_fail(self):
        row = ("x", 0, 0)
        roster = [("x", "g")]
        for records, expected, cutoff in (
            ([row], roster + [("missing", "g")], 0), ([row, row], roster, 0),
            ([row], roster, True), ([row], roster, float("inf")),
            ([("x", 0, None)], roster, 0),
        ):
            with self.subTest(records=records, expected=expected, cutoff=cutoff), self.assertRaises(ValueError):
                paired_fixed_lower_cutoff_tie_audit(records, expected_roster=expected,
                                                    cutoff_a=cutoff, cutoff_b=0)
