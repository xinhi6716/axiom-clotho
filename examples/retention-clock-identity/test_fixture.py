import unittest
from run import CASES, Timestamp, classify


class ClockFixture(unittest.TestCase):
    def test_clock_identity_and_acquisition_uncertainty(self):
        cases = {name: classify(origin, terminal) for name, origin, terminal in CASES}
        self.assertEqual(cases, {
            "linked": ("eligible", (99_998, 100_002)),
            "reset": ("unscored_unlinked_clock", None),
            "clock-change": ("unscored_unlinked_clock", None),
            "missing-origin": ("unscored_missing_origin", None),
            "uncertain-edge": ("unscored_boundary_uncertain", (79_996, 80_004)),
            "outside": ("outside_window", (79_986, 79_994)),
        })

    def test_closed_limits_when_acquisition_is_exact(self):
        origin = Timestamp(2, "clock", "epoch", 0)
        for terminal in (80_002, 120_002):
            self.assertEqual(classify(origin, Timestamp(terminal, "clock", "epoch", 0))[0], "eligible")


if __name__ == "__main__":
    unittest.main()
