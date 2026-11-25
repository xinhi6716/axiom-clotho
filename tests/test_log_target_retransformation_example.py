"""Check the original-scale estimand on an exact synthetic distribution."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/log-target-retransformation/run.py"


class LogTargetRetransformationTests(unittest.TestCase):
    def test_exact_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "direct inverse: quiet=1, variable=1, MSE=153/64",
            "pooled smearing: quiet=25/16, variable=25/16, MSE=531/256",
            "stratum smearing: quiet=1, variable=17/8, MSE=225/128",
        ])

    def test_bias_variance_oracle(self):
        mod = runpy.run_path(str(EXAMPLE))
        irreducible = (F(15, 8)**2) / 2
        true_means = {"quiet": F(1), "variable": F(17, 8)}
        for pred in mod["freeze_predictions"](mod["DEVELOPMENT"]).values():
            squared_bias = sum((pred[g] - v)**2 for g, v in true_means.items()) / 2
            self.assertEqual(mod["squared_error"](pred, mod["EVALUATION"]),
                             irreducible + squared_bias)

    def test_evaluation_change_cannot_change_frozen_factors(self):
        mod = runpy.run_path(str(EXAMPLE))
        frozen = mod["freeze_predictions"](mod["DEVELOPMENT"])
        before = frozen["stratum smearing"].copy()
        mutated = {g: (F(100),) for g in mod["EVALUATION"]}
        self.assertNotEqual(mod["squared_error"](before, mutated), F(225, 128))
        self.assertEqual(frozen["stratum smearing"], before)
        self.assertEqual(mod["freeze_predictions"](mod["DEVELOPMENT"]), frozen)
