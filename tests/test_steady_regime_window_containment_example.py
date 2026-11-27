import json
from pathlib import Path
import subprocess
import sys
import unittest


class SteadyRegimeWindowContainmentTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/steady-regime-window-containment/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(proc.stdout)
        self.assertEqual(out["certified_common_regime"], [10, 20])
        rows = out["section_checks"]
        self.assertTrue(all(r["nominal_center_passes_nominal_regime"] for r in rows.values()))
        self.assertEqual(
            [(r["registered_envelope"], r["certified_whole_section"],
              r["compatible_failure_worlds"]) for r in rows.values()],
            [([9, 13], False, 2), ([10, 13], True, 0), ([12, 18], True, 0),
             ([18, 21], False, 2)],
        )
