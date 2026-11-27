import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/mux-order-elapsed-control/run.py"


class MuxOrderElapsedTests(unittest.TestCase):
    def test_cli(self):
        out = json.loads(subprocess.check_output([sys.executable, str(RUNNER)], cwd=ROOT))
        self.assertEqual(out["by_cell"], {
            "inside->outside": 1, "outside->inside": 1, "unresolved->unresolved": 1,
        })
        self.assertEqual(out["by_position"], {
            "inside->inside": 1, "outside->outside": 1, "unresolved->unresolved": 1,
        })

    def test_offset_bounds_and_boundary(self):
        code = runpy.run_path(str(RUNNER))
        self.assertEqual(code["elapsed"](19000, 20018, (-3, 3), (-2, 2)), (1013, 1023))
        self.assertEqual(code["state"]((1013, 1020)), "inside")
        self.assertEqual(code["state"]((0, 1020)), "unresolved")
        self.assertEqual(code["state"]((1020, 1021)), "unresolved")
        data = code["fixture"]()
        self.assertEqual(code["transitions"](data["forward"][::-1], data["reverse"], 0),
                         data["by_cell"])
