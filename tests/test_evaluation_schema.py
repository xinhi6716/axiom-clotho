"""Invented evidence only; saved evaluation checks never fit models."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd
from groupbench import benchmark, validate_evaluation_bundle
from groupbench.cli import main
from groupbench.verification import verify_bundle


class EvaluationSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = pd.DataFrame({'g': ['001', 'NA', 'nan', 'Inf', 'x', 'y'] * 2,
                              'x': range(12), 'y': [i*i/10 for i in range(12)]})
        cls.results = [benchmark(frame, target='y', group='g', baselines=('dummy',),
                                 outer_splits=3, inner_splits=2, categorical_encoding=encoding,
                                 objective='mae') for encoding in ('legacy-v1', 'distinct-v2')]
        frame['y'] = 3.0
        cls.constant = benchmark(frame, target='y', group='g', baselines=('dummy',), outer_splits=3, inner_splits=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results[0].write(self.root)

    def rehash(self):
        path = self.root / 'checksums.json'
        checks = json.loads(path.read_text())
        checks['files'] = {name: hashlib.sha256((self.root/name).read_bytes()).hexdigest() for name in checks['files']}
        path.write_text(json.dumps(checks))

    def mutate_json(self, name, action):
        path = self.root / name
        data = json.loads(path.read_text()); action(data)
        path.write_text(json.dumps(data)); self.rehash()

    def test_writer_versions_literal_groups_and_no_fit_or_write(self):
        for result in [*self.results, self.constant]:
            with tempfile.TemporaryDirectory() as temp:
                result.write(temp)
                before = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
                with patch('groupbench.model.GroupBenchRegressor._fit_prepared', side_effect=AssertionError('fit forbidden')):
                    checked = validate_evaluation_bundle(temp)
                self.assertEqual((checked.n_rows, checked.n_groups, checked.n_models), (12, 6, 2))
                self.assertEqual(before, {p.name: p.read_bytes() for p in Path(temp).iterdir()})

    def test_rehashed_false_metrics_fail_but_byte_contract_unchanged(self):
        self.mutate_json('summary.json', lambda d: d['models']['selected']['overall'].update(mae=123456.))
        self.assertEqual(verify_bundle(self.root).bundle_type, 'evaluation')
        with self.assertRaises(ValueError): validate_evaluation_bundle(self.root)

    def test_cli_requires_selected_predictions_to_match_declared_family_refit(self):
        from groupbench.model import GroupBenchRegressor

        command = [sys.executable, "-m", "groupbench.cli", "validate-evaluation-bundle", str(self.root)]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

        def cli():
            return subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)

        self.assertEqual(cli().returncode, 0)
        frame = pd.DataFrame({"g": ["a", "b", "c", "d"] * 2, "x": range(8), "y": range(8)})
        original = GroupBenchRegressor.predict_baselines

        def mismatched_selected(estimator, features):
            predictions = original(estimator, features)
            return {**predictions, "selected": predictions["dummy"] + 1.}

        with patch.object(GroupBenchRegressor, "predict_baselines", mismatched_selected):
            result = benchmark(frame, target="y", group="g", baselines=("dummy",),
                               outer_splits=2, inner_splits=2)
        result.write(self.root, overwrite=True)
        self.assertEqual(verify_bundle(self.root).bundle_type, "evaluation")
        self.assertTrue(all(fold["selected_family"] == "dummy" for fold in result.manifest["outer_folds"]))
        self.assertTrue((result.predictions["prediction_selected"] != result.predictions["prediction_dummy"]).all())
        # The writer scored the changed selected column normally, so all
        # reported metrics agree with the changed retained rows.
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        reply = cli()
        self.assertEqual(reply.returncode, 2, reply.stderr)
        self.assertIn("selected predictions differ from their declared family refit", reply.stderr)
        self.assertFalse(reply.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_cli_requires_group_median_predictions_to_match_refit_constant(self):
        frame = pd.DataFrame({"g": ["a", "b", "c", "d"] * 2, "x": range(8), "y": range(8)})
        result = benchmark(frame, target="y", group="g", baselines=("dummy", "group_median"),
                           outer_splits=2, inner_splits=2)
        result.write(self.root, overwrite=True)
        self.assertEqual(result.manifest["schema_version"], "1.2")
        command = [sys.executable, "-m", "groupbench.cli", "validate-evaluation-bundle", str(self.root)]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

        def replay():
            return subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)

        self.assertEqual(replay().returncode, 0)

        def change_constant(data):
            median = data["outer_folds"][0]["family_fit_details"]["group_median"]
            median["constant"] += 1.0

        self.mutate_json("manifest.json", change_constant)
        self.assertEqual(verify_bundle(self.root).bundle_type, "evaluation")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        reply = replay()
        self.assertEqual(reply.returncode, 2, reply.stderr)
        self.assertIn("group-median refit constant", reply.stderr)
        self.assertFalse(reply.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_count_bool_null_model_and_fold_corruption(self):
        mutations = [('summary.json', lambda d: d['models']['selected']['overall'].update(n_rows=True)),
                     ('summary.json', lambda d: d['models']['selected']['overall'].update(r2=None)),
                     ('summary.json', lambda d: d['models'].pop('dummy')),
                     ('manifest.json', lambda d: d['outer_folds'][0].update(test_rows=0)),
                     ('manifest.json', lambda d: d.update(schema_id='binary.evaluation')),
                     ('manifest.json', lambda d: d.update(schema_version='9.0'))]
        for name, action in mutations:
            with self.subTest(name=name, action=action):
                self.results[0].write(self.root, overwrite=True)
                self.mutate_json(name, action)
                with self.assertRaises(ValueError): validate_evaluation_bundle(self.root)

    def test_missing_duplicate_row_header_nonfinite_and_crossfold_group(self):
        original = (self.root/'predictions.csv').read_text()
        rows = original.splitlines()
        altered = [ '\n'.join(rows[:-1]), '\n'.join(rows+[rows[-1]]),
                    original.replace('prediction_dummy', 'prediction_other'),
                    original.replace(',0.0,', ',NaN,', 1)]
        import csv
        records = list(csv.reader(io.StringIO(original)))
        records[1][2] = str((int(records[1][2])+1)%3)
        stream = io.StringIO(); csv.writer(stream).writerows(records); altered.append(stream.getvalue())
        for text in altered:
            with self.subTest(text=text[:80]):
                (self.root/'predictions.csv').write_text(text); self.rehash()
                with self.assertRaises(ValueError): validate_evaluation_bundle(self.root)

    def test_nonzero_csv_token_cannot_disappear_in_replay(self):
        frame = pd.DataFrame({'g': ['a', 'b', 'c', 'd'] * 2,
                              'x': range(8), 'y': [0.] * 8})
        zero = benchmark(frame, target='y', group='g', baselines=('dummy',),
                         outer_splits=2, inner_splits=2)
        zero.write(self.root, overwrite=True)
        path = self.root / "predictions.csv"
        original = path.read_text()
        header, row, *rest = original.splitlines()
        fields = row.split(",")
        for column in (3, 4):
            for token in ("1e-400", "-5E-9999999999"):
                with self.subTest(column=column, numeric_token=token):
                    mutated = list(fields)
                    self.assertEqual(float(mutated[column]), 0.)
                    mutated[column] = token
                    path.write_text("\n".join([header, ",".join(mutated), *rest]) + "\n")
                    self.rehash()
                    self.assertEqual(verify_bundle(self.root).bundle_type, "evaluation")
                    with self.assertRaisesRegex(ValueError, "underflows to zero"):
                        validate_evaluation_bundle(self.root)
        fields[3] = "-0E-400"
        path.write_text("\n".join([header, ",".join(fields), *rest]) + "\n")
        self.rehash()
        self.assertEqual(validate_evaluation_bundle(self.root).n_rows, 8)

        # A representable subnormal is valid; identical values have zero error.
        subnormal = [",".join(item.split(",")[:3] + ["5e-324"] * 3)
                     for item in [row, *rest]]
        path.write_text("\n".join([header, *subnormal]) + "\n")
        self.rehash()
        self.assertEqual(validate_evaluation_bundle(self.root).n_rows, 8)

    def test_perfect_r2_cannot_exceed_one_within_recompute_tolerance(self):
        # A transparent noiseless fixture gives nonconstant targets in
        # every group and emits a replayable perfect regression bundle.
        frame = pd.DataFrame({"g": ["a", "b", "c", "d"] * 2,
                              "x": range(8), "y": range(8)})
        def perfect(features):
            prediction = features["x"].to_numpy(dtype=float)
            return {"dummy": prediction, "selected": prediction}
        with patch("groupbench.model.GroupBenchRegressor.predict_baselines", side_effect=perfect):
            result = benchmark(frame, target="y", group="g", baselines=("dummy",),
                               outer_splits=2, inner_splits=2)
        result.write(self.root, overwrite=True)
        self.assertEqual(validate_evaluation_bundle(self.root).n_rows, 8)
        for location in ("overall", "per_fold", "per_group"):
            with self.subTest(location=location):
                result.write(self.root, overwrite=True)
                def corrupt(summary):
                    row = summary["models"]["selected"][location]
                    if isinstance(row, list):
                        row = row[0]
                    self.assertEqual(row["r2"], 1.0)
                    row["r2"] = math.nextafter(1.0, math.inf)
                self.mutate_json("summary.json", corrupt)
                self.assertEqual(verify_bundle(self.root).bundle_type, "evaluation")
                with self.assertRaisesRegex(ValueError, "R2 must not exceed one"):
                    validate_evaluation_bundle(self.root)

    def test_checksum_symlink_and_bound(self):
        with patch('groupbench.evaluation_schema.MAX_CSV_BYTES', 8):
            with self.assertRaises(ValueError): validate_evaluation_bundle(self.root)
        p = self.root/'predictions.csv'; raw = p.read_bytes(); p.write_bytes(raw+b'\n')
        with self.assertRaises(ValueError): validate_evaluation_bundle(self.root)
        p.unlink(); other=self.root/'invented.csv'; other.write_bytes(raw); p.symlink_to(other)
        with self.assertRaises((ValueError, OSError)): validate_evaluation_bundle(self.root)

    def test_post_read_race_fails_even_when_buffer_hash_matches(self):
        from groupbench import evaluation_schema as module
        original = module._read_file
        def read_then_change(fd, name, limit, **kwargs):
            result = original(fd, name, limit, **kwargs)
            if name == 'manifest.json':
                path = self.root / name
                path.write_bytes(path.read_bytes() + b' ')
            return result
        with patch.object(module, '_read_file', side_effect=read_then_change):
            with self.assertRaisesRegex(ValueError, 'changed during validation'):
                validate_evaluation_bundle(self.root)

    def test_only_expected_payloads_read_no_model_or_source_access(self):
        from groupbench import evaluation_schema as module
        self.mutate_json('manifest.json', lambda d: d.update(source_path='/invented/not-to-be-read.csv'))
        original = module._read_file
        names = []
        def tracked(fd, name, limit, **kwargs):
            names.append(name)
            return original(fd, name, limit, **kwargs)
        with patch.object(module, '_read_file', side_effect=tracked), patch('groupbench.model.GroupBenchRegressor._fit_prepared', side_effect=AssertionError('fit forbidden')):
            validate_evaluation_bundle(self.root)
        self.assertEqual(names, ['checksums.json', 'manifest.json', 'summary.json', 'predictions.csv', 'report.html'])

    def test_tiny_positive_error_cannot_be_reported_as_zero_or_negative(self):
        frame = pd.DataFrame({'g': ['a', 'b', 'c', 'd'] * 2,
                              'x': range(8), 'y': [i * 1e-14 for i in range(8)]})
        result = benchmark(frame, target='y', group='g', baselines=('dummy',),
                           outer_splits=2, inner_splits=2, objective='mae')
        for location, field in (("overall", "group_mae"), ("per_fold", "group_rmse"),
                                ("per_group", "rmse")):
            for replacement in (0, -1e-14):
                with self.subTest(location=location, field=field, value=replacement):
                    result.write(self.root, overwrite=True)
                    def corrupt(summary):
                        row = summary["models"]["selected"][location]
                        if isinstance(row, list):
                            row = row[0]
                        self.assertGreater(row[field], 0)
                        self.assertLess(row[field], 1e-12)
                        row[field] = replacement
                    self.mutate_json("summary.json", corrupt)
                    self.assertEqual(verify_bundle(self.root).bundle_type, "evaluation")
                    with self.assertRaisesRegex(ValueError, "error magnitude sign or zero"):
                        validate_evaluation_bundle(self.root)

    def test_exact_zero_sign_is_irrelevant_but_small_nonzero_is_not_a_zero(self):
        self.constant.write(self.root, overwrite=True)
        metrics = lambda data: data["models"]["selected"]["overall"]
        self.mutate_json("summary.json", lambda data: metrics(data).update(mae=-0.0))
        validate_evaluation_bundle(self.root)
        self.mutate_json("summary.json", lambda data: metrics(data).update(mae=1e-14))
        with self.assertRaisesRegex(ValueError, "error magnitude sign or zero"):
            validate_evaluation_bundle(self.root)

    def test_tolerance_and_cli(self):
        self.mutate_json('summary.json', lambda d: d['models']['selected']['overall'].update(mae=d['models']['selected']['overall']['mae']+1e-13))
        validate_evaluation_bundle(self.root)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['validate-evaluation-bundle',str(self.root)]), 0)
        self.assertIn('not scientific validity', output.getvalue())


if __name__ == '__main__': unittest.main()
