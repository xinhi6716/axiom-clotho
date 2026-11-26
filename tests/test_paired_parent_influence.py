"""Synthetic paired parent influence, not population uncertainty."""
import json
from fractions import Fraction
from itertools import combinations, product
import unittest

from groupbench.metrics import paired_parent_mae_deletion_bounds, paired_parent_mae_influence


def fixture(count):
    roster = [(f"{wafer}-{i}", f"{wafer}.d{i}", wafer)
              for wafer in ("w0", "w1") for i in range(count)]
    rows = [(*entry, 0, int(entry[2] == "w0"), int(entry[2] == "w1"))
            for entry in roster]
    return roster, rows


class ParentInfluenceTests(unittest.TestCase):
    def test_more_correlated_devices_do_not_increase_declared_parent_support(self):
        for count in (8, 16):
            roster, rows = fixture(count)
            got = paired_parent_mae_influence(rows, expected_roster=roster[::-1])
            self.assertEqual(got["n_devices"], 2 * count)
            self.assertEqual(got["n_declared_parents"], 2)
            self.assertEqual(got["mae_b_minus_a"], 0)
            self.assertEqual(got["leave_one_parent_out_mae_b_minus_a"], {"min": -1, "max": 1})
            self.assertEqual(got["per_parent"], [
                {"parent": "w0", "n_devices": count, "mae_b_minus_a": -1},
                {"parent": "w1", "n_devices": count, "mae_b_minus_a": 1}])
            self.assertEqual(got, paired_parent_mae_influence(rows[::-1], expected_roster=roster))

    def test_parent_weights_do_not_track_device_counts(self):
        roster, rows = fixture(8)
        more_roster, more_rows = fixture(16)
        roster += [v for v in more_roster if v[2] == "w0" and v not in roster]
        rows += [v for v in more_rows if v[2] == "w0" and v not in rows]
        got = paired_parent_mae_influence(rows, expected_roster=roster)
        self.assertEqual(got["n_devices"], 24)
        self.assertEqual(got["n_declared_parents"], 2)
        self.assertEqual(got["mae_b_minus_a"], 0)
        self.assertEqual(got["leave_one_parent_out_mae_b_minus_a"], {"min": -1, "max": 1})
        roster.append(("r.w2", "w2.d0", "w2"))
        rows.append((*roster[-1], 0, 0, 3))
        got = paired_parent_mae_influence(rows, expected_roster=roster)
        self.assertEqual(got["n_declared_parents"], 3)
        self.assertEqual(got["mae_b_minus_a"], 1)
        self.assertEqual(got["leave_one_parent_out_mae_b_minus_a"], {"min": 0, "max": 2})

    def test_extra_reads_keep_same_device_support_and_equal_device_weighting(self):
        roster = [("r0", "d0", "w0"), ("r1", "d1", "w0"), ("r2", "d0", "w0")]
        rows = [(*roster[0], 0, 0, 0), (*roster[1], 0, 8, 0), (*roster[2], 0, 2, 0)]
        got = paired_parent_mae_influence(rows, expected_roster=roster)
        self.assertEqual(got["n_devices"], 2)
        self.assertEqual(got["n_declared_parents"], 1)
        self.assertEqual(got["mae_a"], 4.5)
        self.assertIsNone(got["leave_one_parent_out_mae_b_minus_a"])

    def test_parent_report_unicode_roundtrip_and_invalid_identifier_rejection(self):
        roster = [("r0", "d0", "\U0001f600"), ("r1", "d1", "\U0001f601")]
        rows = [(*entry, 0, 0, 1) for entry in roster]
        result = paired_parent_mae_influence(rows, expected_roster=roster)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual([entry["parent"] for entry in result["per_parent"]],
                         ["\U0001f600", "\U0001f601"])
        for invalid in ("\ud83d\ude00", "\ud83d", "\ude00"):
            for field in range(3):
                entry = list(roster[0])
                entry[field] = invalid
                with self.subTest(invalid=ascii(invalid), field=field):
                    with self.assertRaisesRegex(ValueError, "valid Unicode"):
                        paired_parent_mae_influence(rows, expected_roster=[tuple(entry), roster[1]])
                    with self.assertRaisesRegex(ValueError, "valid Unicode"):
                        paired_parent_mae_influence([(*entry, 0, 0, 1), rows[1]],
                                                   expected_roster=roster)

    def test_fail_closed_to_roster_errors(self):
        roster, rows = fixture(3)
        extra = ("extra", "w0.d0", "w1")
        for given, expected, error in (
            (rows[1:], roster, "missing row"),
            (rows + [rows[0]], roster, "Duplicate paired"),
            (rows, roster + [roster[0]], "Duplicate expected"),
            (rows + [(*extra, 0, 0, 0)], roster, "Unexpected"),
            (rows, roster + [extra], "exactly one parent"),
            ([(rows[0][0], rows[0][1], "w1", 0, 0, 0)] + rows[1:], roster, "rebound"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                paired_parent_mae_influence(given, expected_roster=expected)


class ParentDeletionBoundsTests(unittest.TestCase):
    @staticmethod
    def make(effects):
        roster = [(f'r{i}', f'd{i}', f'p{i}') for i in range(len(effects))]
        rows = [(*entry, 0, max(-effect, 0), max(effect, 0))
                for entry, effect in zip(roster, effects)]
        return rows, roster

    def test_one_parent_omission_can_miss_the_two_parent_reversal(self):
        rows, roster = self.make((-4, -4, 1, 1, 1))
        report = paired_parent_mae_deletion_bounds(rows, expected_roster=roster, max_deletions=2)
        bounds = report['mae_b_minus_a_bounds_by_budget']
        self.assertEqual(bounds, ((-1, -1), (Fraction(-3, 2), Fraction(-1, 4)),
                                  (Fraction(-7, 3), 1)))
        before = paired_parent_mae_influence(rows, expected_roster=roster)
        self.assertEqual(before['leave_one_parent_out_mae_b_minus_a'],
                         dict(zip(('min', 'max'), bounds[1])))
        swapped = [row[:-2] + row[-2:][::-1] for row in rows[::-1]]
        reverse = paired_parent_mae_deletion_bounds(swapped, expected_roster=roster[::-1],
                                                    max_deletions=2)
        self.assertEqual(reverse['mae_b_minus_a_bounds_by_budget'],
                         tuple((-high, -low) for low, high in bounds))

    def test_every_subset_for_each_small_support_pattern(self):
        for k in range(1, 6):
            for effects in product((-2, 0, 1), repeat=k):
                rows, roster = self.make(effects)
                bounds = paired_parent_mae_deletion_bounds(
                    rows, expected_roster=roster, max_deletions=k-1)['mae_b_minus_a_bounds_by_budget']
                possible = []
                for d in range(k):
                    possible.extend(Fraction(sum(kept), k-d)
                                    for kept in combinations(effects, k-d))
                    self.assertEqual(bounds[d], (min(possible), max(possible)))

    def test_nesting_weights_devices_and_parents_not_rows(self):
        rows, roster = self.make((-4, 1, 1))
        rows += [('x', 'd0', 'p0', 0, 0, 2), ('y', 'extra', 'p0', 0, 5, 0)]
        roster += [row[:3] for row in rows[-2:]]
        report = paired_parent_mae_deletion_bounds(rows, expected_roster=roster, max_deletions=2)
        self.assertEqual(report['mae_b_minus_a_bounds_by_budget'],
                         ((Fraction(-1, 3),) * 2, (-1, 1), (-3, 1)))

    def test_budget_and_frozen_hierarchy_checks(self):
        rows, roster = self.make((1, 0, -1))
        for budget in (-1, 3, True, 2., None):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'budget'):
                paired_parent_mae_deletion_bounds(rows, expected_roster=roster, max_deletions=budget)
        for bad in (rows[:-1], rows + [rows[0]], [(*rows[0][:2], 'p2', *rows[0][3:]), *rows[1:]]):
            with self.assertRaises(ValueError):
                paired_parent_mae_deletion_bounds(bad, expected_roster=roster, max_deletions=1)
