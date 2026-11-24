"""Finite-roster synthetic diagnostics; no fitted models or private data."""
import unittest

from groupbench.metrics import scheduled_phase_paired_mae


CELLS = [(device, phase) for device in ("d0", "d1") for phase in ("early", "late")]
WEIGHTS = {"early": 1, "late": 1}


def rows():
    return [(f"{device}/{phase}/{i}", device, phase, 0, a, 1)
            for device, phase, a, n in (
                ("d0", "early", 0, 4), ("d0", "late", 0, 4),
                ("d1", "early", 0, 1), ("d1", "late", 6, 1))
            for i in range(n)]


# Separate fixture declarations, not inferred from the predictions under test.
ROSTER = [(f"{device}/{phase}/{i}", device, phase)
          for device, phase, count in (("d0", "early", 4), ("d0", "late", 4),
                                       ("d1", "early", 1), ("d1", "late", 1))
          for i in range(count)]
DENSE_ROSTER = ROSTER + [(f"extra-{i}", "d1", "early") for i in range(8)]


class ScheduledPhaseMAETests(unittest.TestCase):
    def test_density_reversal_on_same_roster_and_invariance_within_cell(self):
        data = rows()
        original = scheduled_phase_paired_mae(data, expected_roster=ROSTER, expected_cells=CELLS, phase_weights=WEIGHTS)
        self.assertEqual(original["micro"], {"mae_a": 0.6, "mae_b": 1, "mae_b_minus_a": 0.4, "better": "A"})
        self.assertEqual(original["equal_device"]["mae_a"], 1.5)
        self.assertEqual(original["fixed_phase_equal_device"]["better"], "B")
        denser = data + [(f"extra-{i}", "d1", "early", 0, 0, 1) for i in range(8)]
        changed = scheduled_phase_paired_mae(denser, expected_roster=DENSE_ROSTER, expected_cells=CELLS, phase_weights=WEIGHTS)
        self.assertEqual(changed["micro"]["mae_a"], 1 / 3)
        self.assertEqual(changed["equal_device"]["mae_a"], 0.3)
        self.assertEqual(changed["equal_device"]["better"], "A")
        self.assertEqual(changed["fixed_phase_equal_device"], original["fixed_phase_equal_device"])
        self.assertEqual(changed, scheduled_phase_paired_mae(
            denser[::-1], expected_roster=DENSE_ROSTER[::-1], expected_cells=CELLS[::-1], phase_weights=WEIGHTS))

    def test_unequal_positive_weights_are_relative_within_each_device(self):
        got = scheduled_phase_paired_mae(rows(), expected_roster=ROSTER, expected_cells=CELLS,
                                         phase_weights={"early": 3, "late": 1})
        self.assertEqual(got["fixed_phase_equal_device"]["mae_a"], 0.75)
        self.assertEqual(got["fixed_phase_equal_device"]["mae_b_minus_a"], 0.25)

    def test_fail_closed_for_missing_extra_duplicate_or_malformed_rows(self):
        data = rows()
        for inputs, error in ((data[:-1], "missing cell"),
                              (data + [data[0]], "Duplicate"),
                              (data + [("new", "d2", "early", 0, 0, 0)], "outside"),
                              (data + [("new", "d1", "early", 0, 0)], "both forecasts")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                scheduled_phase_paired_mae(inputs, expected_roster=ROSTER, expected_cells=CELLS, phase_weights=WEIGHTS)

    def test_fail_closed_for_incomplete_expected_panel_or_bad_weights(self):
        with self.assertRaisesRegex(ValueError, "every phase"):
            scheduled_phase_paired_mae(rows(), expected_roster=ROSTER, expected_cells=CELLS[:-1], phase_weights=WEIGHTS)
        with self.assertRaisesRegex(ValueError, "unique"):
            scheduled_phase_paired_mae(rows(), expected_roster=ROSTER, expected_cells=CELLS + CELLS[:1], phase_weights=WEIGHTS)
        for invalid in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                scheduled_phase_paired_mae(rows(), expected_roster=ROSTER, expected_cells=CELLS,
                                           phase_weights={"early": invalid, "late": 1})


    def test_exact_expected_roster_catches_omissions_extras_and_rebinding(self):
        data = rows()
        row_id, device, phase, y, a, b = data[0]
        for inputs, error in ((data[1:], "missing row"),
                              (data + [("extra", device, phase, y, a, b)], "Unexpected"),
                              ([(row_id, device, "late", y, a, b)] + data[1:], "rebound")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                scheduled_phase_paired_mae(inputs, expected_roster=ROSTER,
                                           expected_cells=CELLS, phase_weights=WEIGHTS)
        with self.assertRaisesRegex(ValueError, "Duplicate expected"):
            scheduled_phase_paired_mae(data, expected_roster=ROSTER + ROSTER[:1],
                                       expected_cells=CELLS, phase_weights=WEIGHTS)
