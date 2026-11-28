"""Positive-control tests for scripted passive compliance."""

from fractions import Fraction as F
import importlib.util
from itertools import product
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "examples" / "rram-compliance-witness" / "run.py"
spec = importlib.util.spec_from_file_location("rram_compliance_example", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ComplianceWitnessTest(unittest.TestCase):
    def test_exact_witness(self):
        self.assertEqual(example.witness(), {
            "shared_command_current_record": ["1", "1"],
            "actual_device_voltages": ["1/2", "1/4"],
            "held_sublimit_currents": ["1/4", "1/2"],
            "command_as_terminal_mae": "1/4",
            "any_shared_prediction_mae_floor": "1/8",
            "synchronized_terminal_voltage_mae": "0",
        })

    def test_signed_passive_clamp_matches_resistance_and_bounds(self):
        for g, v, cap in product((0, F(1, 3), 1, 2, 4),
                                 (-2, -1, F(-1, 4), 0, F(1, 4), 1, 2),
                                 (F(1, 8), F(1, 2), 1, 2)):
            terminal, measured = example.ideal_sample(g, v, cap)
            self.assertEqual(measured, terminal*g)
            self.assertEqual(abs(measured), min(abs(g*v), cap))
            self.assertGreaterEqual(terminal*measured, 0)
            self.assertLessEqual(abs(terminal), abs(v))
            if terminal:
                self.assertEqual(example.infer_conductance(terminal, measured), g)

    def test_invalid_zero_and_negative_conductance_inputs(self):
        for case in ((-1, 1, 1), (1, 0, -1), (1, 1, 0), (True, 1, 1), (2.0, 1, 1)):
            with self.assertRaises(ValueError):
                example.ideal_sample(*case)
        for case in ((0, 0), (0, 1), (1, -1), (-1, 1), (1, .5)):
            with self.assertRaises(ValueError):
                example.infer_conductance(*case)
