"""Exercise independently bound evaluation replay through the public CLI."""

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from groupbench import benchmark
from groupbench.cli import main


class EvaluationCliDigest(unittest.TestCase):
    def test_same_shape_replacement_is_not_the_frozen_evaluation(self):
        data = pd.DataFrame({"group": list("abcd")*2, "x": range(8), "y": range(8)})
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            args = ["validate-evaluation-bundle", work]

            def write(y, overwrite=False):
                benchmark(data.assign(y=y), target="y", group="group", baselines=("dummy",),
                          outer_splits=2, inner_splits=2).write(root, overwrite=overwrite)

            def run(*more):
                out, err = StringIO(), StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    rc = main([*args, *more])
                return rc, out.getvalue(), err.getvalue()

            write(range(8))
            trusted = sha256((root/"checksums.json").read_bytes()).hexdigest().upper()
            flag = "--expected-checksums-sha256"
            rc, out, err = run(flag, trusted, "--require-run-key")
            self.assertEqual((rc, err), (0, ""))
            self.assertIn("independently supplied SHA-256", out)
            self.assertIn("not scientific validity", out)
            write(range(8, 16), overwrite=True)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            rc, out, err = run(flag, trusted)
            self.assertEqual((rc, out), (2, ""))
            self.assertIn("independently recorded digest", err)
            self.assertNotIn(trusted, err)
            self.assertEqual(run()[0], 0)
            self.assertNotIn("independently supplied SHA-256", run()[1])
            for malformed in ("", "7", "z"*64):
                rc, out, err = run(flag, malformed)
                self.assertEqual((rc, out), (2, ""))
                self.assertIn("expected_checksums_sha256", err)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
