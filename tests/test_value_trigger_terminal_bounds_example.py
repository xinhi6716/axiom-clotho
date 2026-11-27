import json
from pathlib import Path
import subprocess
import sys
import unittest


class ValueTriggerTerminalBoundsTests(unittest.TestCase):
    def test_value_trigger_terminal_bounds_cli(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "examples/value-trigger-terminal-bounds/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(result.stdout)
        assert out["attempted"] == [2, 2]
        assert out["terminal_observed"] == [1, 2]
        assert out["complete_case_difference"] == "-3"
        assert out["invalid_carryforward_difference"] == "-1/2"
        assert out["worlds_same_observed_log"] == {"0": "-7/2", "8": "1/2"}
        assert out["range_only"] == {
            "feasible_terminal": list(range(9)), "a_minus_b": ["-7/2", "1/2"]
        }
        assert out["range_and_slew"] == {
            "feasible_terminal": [5, 6, 7], "a_minus_b": ["-1", "0"]
        }
