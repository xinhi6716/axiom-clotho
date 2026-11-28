"""Exact oracles for the normalized passive history example."""

from fractions import Fraction as F
import importlib.util
from itertools import product
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "examples" / "rram-passive-history" / "run.py"
spec = importlib.util.spec_from_file_location("rram_passive_history_example", SRC)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class PassiveHistoryTest(unittest.TestCase):
    def test_constant_drive_vs_independent_repeated_one_tick(self):
        values = [F(i, 4) for i in range(5)]
        for x, u, q, left, right in product(values, values, (F(1, 4), F(1, 2), F(4, 5)), range(6), range(6)):
            expected = x
            for _ in range(left + right):
                expected -= (expected - u) * (1 - q)
            actual = example.step(x, u, left + right, q)
            self.assertEqual(actual, expected)
            self.assertEqual(example.step(example.step(x, u, left, q), u, right, q), actual)
            self.assertLessEqual(min(x, u), actual)
            self.assertLessEqual(actual, max(x, u))
            for v in (-3, 0, 2):
                self.assertGreaterEqual(v * example.current(actual, v), 0)

    def test_exact_same_duty_counterexample(self):
        self.assertEqual(example.witness(), {
            "terminal_currents": ["13/8", "7/4"],
            "order_insensitive_baseline_mae_floor": "1/16",
            "same_law_constructed_world_mae": "0",
        })
        left, right = F(13, 8), F(7, 4)
        for candidate in [F(i, 16) for i in range(32)] + [(left + right) / 2]:
            mae = (abs(left-candidate) + abs(right-candidate)) / 2
            self.assertGreaterEqual(mae, F(1, 16))
        self.assertEqual(example.step(0, 1, 0), 0)
        self.assertEqual(example.step(0, 1, 256), 1-F(1, 2**256))
        self.assertGreater(0 + 2 * (1 - 0), 1)  # unstable Euler comparator

    def test_rejects_unverified_numeric_inputs(self):
        for args in ((2, 0, 1), (0, -1, 1), (0, 1, True), (0, 1, 257),
                     (0, 1, -1), (0, 1, 1, 0), (0, 1, 1, 1),
                     (F(1, 5), 0.5, 1)):
            with self.assertRaises(ValueError):
                example.step(*args)
        with self.assertRaises(ValueError):
            example.current(0, float("nan"))

    def test_relaxation_determinants_against_factorized_oracle(self):
        for q, r, a, b in product((F(1, 4), F(1, 2), F(3, 4)),
                                   (F(1, 4), F(1, 2), F(3, 4)),
                                   (F(-1), F(0), F(1, 3), F(2)),
                                   (F(-1), F(0), F(1, 3), F(2))):
            d = [a*q**i + b*r**i for i in range(5)]
            expected = tuple(a*b*(q*r)**i*(q-r)**2 for i in range(3))
            for shift, sign in ((F(7, 11), 1), (F(-9, 7), -1)):
                y = [shift] + [shift + sign * sum(d[:i]) for i in range(1, 6)]
                self.assertEqual(example.relaxation_increment_determinants(y), expected)
        # Fast then slow summed with same sign gives a held increment above
        # a one-rate extrapolation from its first two increments.
        x = [F(0), F(3, 2), F(17, 8), F(79, 32)]
        self.assertEqual(example.relaxation_increment_determinants(x), (F(1, 8),))
        self.assertEqual(F(11, 32) - F(5, 8)**2/F(3, 2), F(1, 12))
        for invalid in ([1,2,3], [0,1,2,3.0], [0]*257, [0,True,2,3]):
            with self.assertRaises(ValueError):
                example.relaxation_increment_determinants(invalid)
