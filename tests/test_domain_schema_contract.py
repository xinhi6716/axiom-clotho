"""Saved declaration checks only; hand-authored fixtures, no study draws."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from groupbench import benchmark, fit_predict
from groupbench.evaluation_schema import _check
from groupbench.prediction_schema import _metadata, _DOMAIN_FIELDS, _DOMAIN_AUDIT_FIELDS, validate_prediction_bundle_schema
from groupbench.verification import verify_bundle
from test_prediction_schema import write_bundle
from test_domain_inner import fixture, ARGS


class DomainSchemaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = fixture()
        cls.pred = fit_predict(f, f[['x']], **ARGS).manifest
        cls.evaluation = benchmark(f, outer_splits=3, **ARGS)
        legacy = ARGS | dict(domain=None, inner_strategy='group_kfold', inner_splits=2,
                             categorical_encoding='distinct-v2')
        cls.legacy_pred = fit_predict(f.drop(columns='d'), f[['x']], **legacy).manifest
        cls.legacy_eval = benchmark(f.drop(columns='d'), outer_splits=3, **legacy)

    def check_eval(self, manifest):
        r = self.evaluation
        return _check(manifest, r.summary, r.predictions.to_csv(index=False).encode())

    def test_all_explicit_legacy_audit_markers_rejected(self):
        for marker in _DOMAIN_AUDIT_FIELDS:
            with self.subTest(marker=marker):
                m = deepcopy(self.legacy_pred)
                m['inner_folds'][0][marker] = None
                with self.assertRaisesRegex(ValueError, 'domain audit'):
                    _metadata(m)
                r = self.legacy_eval
                m = deepcopy(r.manifest)
                m['outer_folds'][0]['inner_folds'][0][marker] = None
                with self.assertRaisesRegex(ValueError, 'domain audit'):
                    _check(m, r.summary, r.predictions.to_csv(index=False).encode())

    def test_config_stripped_downgrade_rejected(self):
        m = deepcopy(self.pred)
        for key in _DOMAIN_FIELDS:
            m['config'].pop(key, None)
        m.update(schema_version='1.2', protocol='group-fit-predict-distinct-v2')
        m['config']['categorical_encoding'] = 'distinct-v2'
        with self.assertRaisesRegex(ValueError, 'domain audit'):
            _metadata(m)
        r = self.evaluation
        m, s = deepcopy(r.manifest), deepcopy(r.summary)
        for key in _DOMAIN_FIELDS:
            m['config'].pop(key, None)
        m['config'].update(inner_splits=3, categorical_encoding='distinct-v2')
        m.update(schema_version='1.1', protocol='nested-group-kfold-distinct-v2')
        s.update(schema_version='1.1', protocol=m['protocol'])
        with self.assertRaisesRegex(ValueError, 'domain audit'):
            _check(m, s, r.predictions.to_csv(index=False).encode())

    def test_unrelated_legacy_metadata_retained(self):
        m = deepcopy(self.legacy_pred)
        m['inner_folds'][0]['user_note'] = {'domain': 'unrelated nested annotation'}
        self.assertEqual(_metadata(m)[0], '1.2')
        r = self.legacy_eval
        m = deepcopy(r.manifest)
        m['outer_folds'][0]['inner_folds'][0]['user_note'] = 'annotation'
        self.assertEqual(_check(m, r.summary, r.predictions.to_csv(index=False).encode())[0], '1.1')

    def test_domain_record_tampering_both_schemas(self):
        changes = [
            lambda a: a.update(inner_selection=a['inner_selection'][:1]),
            lambda a: a['inner_selection'].append(deepcopy(a['inner_selection'][0])),
            lambda a: a['inner_selection'].reverse(),
            lambda a: a['inner_selection'][0].update(candidate='fabricated'),
            lambda a: a['inner_selection'][0].update(family='fabricated'),
            lambda a: a['inner_selection'][1].update(params={'alpha': True}),
            lambda a: a['inner_selection'][0].update(selection_score=-10),
            lambda a: a['inner_selection'][0].update(selection_score=float('inf')),
            lambda a: a['inner_selection'][0].update(selection_score=True),
            lambda a: a['inner_selection'][0]['inner_fold_scores'].__setitem__(0, -1),
            lambda a: a['inner_selection'][0].update(objective='rmse'),
            lambda a: a.update(selected_candidate='fabricated'),
            lambda a: a.update(selected_family='fabricated'),
        ]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                m = deepcopy(self.pred); change(m)
                with self.assertRaises(ValueError):
                    _metadata(m)
                m = deepcopy(self.evaluation.manifest); change(m['outer_folds'][0])
                with self.assertRaises(ValueError):
                    self.check_eval(m)

    def test_eval_grid_and_family_winner_tampering(self):
        for change in [lambda m: m['config'].update(candidates=m['config']['candidates'][:1]),
                       lambda m: m['config']['candidates'][0].update(name='fabricated'),
                       lambda m: m['outer_folds'][0]['family_candidates'].update(dummy='fabricated')]:
            m = deepcopy(self.evaluation.manifest); change(m)
            with self.assertRaises(ValueError):
                self.check_eval(m)

    def test_frozen_contract_independent_of_installed_grid(self):
        with patch('groupbench.model.candidates_for', side_effect=AssertionError('runtime grid read')):
            self.assertEqual(_metadata(self.pred)[0], '1.4')
            self.assertEqual(self.check_eval(self.evaluation.manifest)[0], '1.3')

    def test_cli_rejects_changed_domain_selection_score_with_folds_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-m", "groupbench.cli", "validate-prediction-bundle", str(path)]
            for objective in ("mae", "group_mae"):
                with self.subTest(objective=objective):
                    import pandas as pd
                    frame = pd.concat([fixture(), fixture().iloc[[0]],
                                       fixture().iloc[[2]].assign(g="new-group")], ignore_index=True)
                    result = fit_predict(frame, frame[["x"]], **(ARGS | {"objective": objective}))
                    raw = result.predictions.to_csv(index=False).encode()
                    write_bundle(path, result.manifest, csv=raw)
                    self.assertEqual(subprocess.run(command, env=env, capture_output=True,
                                                    timeout=30).returncode, 0)
                    value = deepcopy(result.manifest)
                    chosen = next(r for r in value["inner_selection"]
                                  if r["candidate"] == value["selected_candidate"])
                    self.assertGreater(chosen["selection_score"], 0)
                    chosen["selection_score"] = 0.0
                    write_bundle(path, value, csv=raw)
                    self.assertEqual(verify_bundle(path).bundle_type, "prediction")
                    before = {p.name: p.read_bytes() for p in path.iterdir()}
                    reply = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(reply.returncode, 2, reply.stderr)
                    self.assertIn("pooled fold scores", reply.stderr)
                    self.assertFalse(reply.stdout)
                    self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

    def test_cli_domain_inner_counts_match_retained_group_sizes(self):
        import pandas as pd

        base = fixture()
        frame = pd.concat([base, base, base], ignore_index=True).assign(y=0.)
        result = benchmark(frame, outer_splits=3, **(ARGS | {"baselines": ("dummy",)}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-m", "groupbench.cli", "validate-evaluation-bundle", str(root)]

            def replay():
                return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

            result.write(root)
            self.assertEqual(replay().returncode, 0)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            first, second, *_ = manifest["outer_folds"][0]["inner_folds"]
            self.assertGreater(second["test_rows"], second["test_group_count"])
            self.assertGreater(first["train_rows"], first["train_group_count"])
            first["test_rows"] += 1
            first["train_rows"] -= 1
            second["test_rows"] -= 1
            second["train_rows"] += 1
            path.write_text(json.dumps(manifest))
            receipt_path = root / "checksums.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"]["manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(verify_bundle(root).bundle_type, "evaluation")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            reply = replay()
            self.assertEqual(reply.returncode, 2, reply.stderr)
            self.assertIn("retained outer group sizes", reply.stderr)
            self.assertFalse(reply.stdout)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_cli_rebuilds_domain_vector_fingerprint_from_outer_rows(self):
        frame = fixture().iloc[[0, 3, 1, 5, 2, 4]].reset_index(drop=True)
        frame["d"] = frame["d"].map({"A": "甲", "B": "乙", "C": "丙"})
        result = benchmark(frame, outer_splits=3, **(ARGS | {"baselines": ("dummy",)}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-m", "groupbench.cli", "validate-evaluation-bundle", str(root)]

            def replay():
                return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

            result.write(root)
            self.assertEqual(replay().returncode, 0)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            config = manifest["config"]
            # Same domain names/counts and exact hashes, but a sorted
            # domain vector does not describe the original retained order.
            encoded = json.dumps(sorted(frame["d"]), ensure_ascii=False, separators=(",", ":")).encode()
            alternate = hashlib.sha256(encoded).hexdigest()
            self.assertNotEqual(config["domain_values_sha256"], alternate)
            config["domain_values_sha256"] = alternate
            path.write_text(json.dumps(manifest))
            receipt_path = root / "checksums.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"]["manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(verify_bundle(root).bundle_type, "evaluation")
            frozen = {p.name: p.read_bytes() for p in root.iterdir()}
            reply = replay()
            self.assertEqual(reply.returncode, 2, reply.stderr)
            self.assertIn("domain-vector fingerprint", reply.stderr)
            self.assertFalse(reply.stdout)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, frozen)

    def test_cli_rejects_domain_configuration_with_stale_run_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.evaluation.write(root)
            command = [sys.executable, "-m", "groupbench.cli", "validate-evaluation-bundle", str(root)]
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

            def replay():
                return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

            self.assertEqual(replay().returncode, 0)
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            self.assertIn("x", manifest["config"]["features"])
            # The retained observations and audited domains are the same, but
            # this now declares a different training feature under the old key.
            manifest["config"]["features"] = ["different_measurement"]
            path.write_text(json.dumps(manifest))
            receipt_path = root / "checksums.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"]["manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(verify_bundle(root).bundle_type, "evaluation")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            reply = replay()
            self.assertEqual(reply.returncode, 2, reply.stderr)
            self.assertIn("run-key declaration", reply.stderr)
            self.assertFalse(reply.stdout)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_saved_selection_tie_does_not_hide_nonzero_decimal_underflow(self):
        m = deepcopy(self.pred)
        for record in m["inner_selection"]:
            record["selection_score"] = 0.0
            record["inner_fold_scores"] = [0.0] * len(record["inner_fold_scores"])
        m.update(selected_candidate="dummy_mean", selected_family="dummy")
        m["inner_selection"][0]["selection_score"] = 12345.125
        m["inner_selection"][0]["inner_fold_scores"][0] = 12346.125
        payload = json.dumps(m).encode()
        csv = b"row_position,prediction\n0,0.0\n1,0.0\n2,0.0\n3,0.0\n4,0.0\n5,0.0\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for token in (b"0e-400", b"-0.000e-400"):
                raw = payload.replace(b"12345.125", token).replace(b"12346.125", b"5e-324")
                write_bundle(root, csv=csv, raw=raw)
                self.assertEqual(validate_prediction_bundle_schema(root).schema_version, "1.4")
            for token in (b"1e-400", b"-1e-400", b"2e-324"):
                raw = payload.replace(b"12345.125", token).replace(b"12346.125", b"5e-324")
                with self.subTest(token=token):
                    write_bundle(root, csv=csv, raw=raw)
                    self.assertEqual(verify_bundle(root).bundle_type, "prediction")
                    before = {path.name: path.read_bytes() for path in root.iterdir()}
                    with self.assertRaisesRegex(ValueError, "underflow"):
                        validate_prediction_bundle_schema(root)
                    self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)

    def test_tie_order_declared_scores(self):
        m = deepcopy(self.pred)
        for record in m['inner_selection']:
            record['selection_score'] = 0
            record['inner_fold_scores'] = [0] * len(record['inner_fold_scores'])
        m.update(selected_candidate='dummy_mean', selected_family='dummy')
        self.assertEqual(_metadata(m)[0], '1.4')
        m.update(selected_candidate='ridge_alpha_0.1', selected_family='ridge')
        with self.assertRaises(ValueError):
            _metadata(m)

    def test_extra_trees_frozen_grid(self):
        f = fixture()
        r = fit_predict(f, f[['x']], **(ARGS | dict(baselines=('extra_trees', 'dummy'))))
        self.assertEqual(_metadata(r.manifest)[0], '1.4')


if __name__ == '__main__':
    unittest.main()
