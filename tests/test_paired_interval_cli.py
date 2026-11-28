"""Installed CLI aliases for matched saved interval scores."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_paired_interval_score import document


class PairedIntervalCliTests(unittest.TestCase):
    def test_actual_aliases_pair_exact_scores_and_refuse_a_mismatched_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
            def invoke(module, prefix, *args, no_site=False):
                return subprocess.run(
                    [sys.executable, *(["-S"] if no_site else []), "-m", module, *prefix, *args],
                    cwd=root, env=env, text=True, capture_output=True, timeout=20)
            a, b = document(-1), document(-10)
            a["alpha"] = b["alpha"] = 0.1
            b["rows"].reverse()
            b["expected_row_ids"].reverse()
            left, right = root / "reference with spaces.json", root / "candidate with spaces.json"
            left.write_text(json.dumps(a))
            right.write_text(json.dumps(b))
            before = {f.name: f.read_bytes() for f in root.iterdir()}
            for module, prefix in (("groupbench", ()), ("mlworkflow", ("tabular",))):
                with self.subTest(module=module):
                    help_result = invoke(module, prefix, "compare-intervals", "--help", no_site=True)
                    self.assertEqual((help_result.returncode, help_result.stderr), (0, ""))
                    self.assertIn("candidate minus reference", help_result.stdout)
                    out = invoke(module, prefix, "compare-intervals", left.name, right.name)
                    self.assertEqual((out.returncode, out.stderr), (0, ""))
                    result = json.loads(out.stdout)
                    self.assertEqual(result["row_weighted"]["b_minus_a"]["interval_score_exact"], "18")
                    self.assertEqual(result["equal_group"]["b_minus_a"]["interval_score_exact"], "90")
                    self.assertNotIn(left.name, out.stdout)
                    self.assertNotIn(right.name, out.stdout)
            self.assertEqual({f.name: f.read_bytes() for f in root.iterdir()}, before)

            for invalid in ("observed", "alpha", "too_large"):
                with self.subTest(invalid=invalid):
                    altered = deepcopy(b)
                    if invalid == "observed":
                        altered["rows"][0]["observed"] += 1
                    elif invalid == "alpha":
                        altered["alpha"] = 0.2
                    if invalid == "too_large":
                        # Sparse local fixture exercises the existing file-size check.
                        with right.open("wb") as handle:
                            handle.truncate(128 * 1024 * 1024)
                    else:
                        right.write_text(json.dumps(altered))
                    out = invoke("groupbench", (), "compare-intervals", left.name, right.name)
                    self.assertEqual((out.returncode, out.stdout), (2, ""))
                    self.assertIn("paired interval comparison rejected:", out.stderr)
                    self.assertNotIn(right.name, out.stderr)
