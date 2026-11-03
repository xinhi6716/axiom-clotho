"""Fresh-process import contracts; handwritten metadata and no model execution."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = str(Path(__file__).resolve().parents[1] / "src")
EXPORTS = {
    "BenchmarkResult": "evaluation", "GroupBenchRegressor": "model",
    "GroupKFold": "splitting", "benchmark": "evaluation",
    "make_materials_fixture": "data", "PredictionResult": "prediction",
    "fit_predict": "prediction", "plan_benchmark": "planning", "plan_prediction": "planning",
    "GroupBenchClassifier": "binary.model", "BinaryBenchmarkResult": "binary.workflow",
    "BinaryPredictionResult": "binary.workflow", "benchmark_binary": "binary.workflow",
    "fit_predict_binary": "binary.workflow", "plan_binary_benchmark": "binary.workflow",
    "plan_binary_prediction": "binary.workflow", "make_binary_fixture": "binary.data",
    "PredictionSchemaResult": "prediction_schema", "validate_prediction_bundle_schema": "prediction_schema",
    "EvaluationSchemaResult": "evaluation_schema", "validate_evaluation_bundle": "evaluation_schema",
    "GroupForwardEvaluationError": "group_time_evaluation",
    "GroupForwardEvaluationPreview": "group_time_evaluation",
    "GroupForwardEvaluationResult": "group_time_evaluation",
    "preview_group_forward_evaluation": "group_time_evaluation",
    "evaluate_group_forward_baselines": "group_time_evaluation",
    "GroupForwardBundleError": "group_time_saved", "GroupForwardBundleValidation": "group_time_saved",
    "write_group_forward_evaluation_bundle": "group_time_saved",
    "validate_group_forward_evaluation_bundle": "group_time_saved",
}
SUBMODULES = ("baselines", "binary", "categorical", "data", "domain_selection", "evaluation",
              "evaluation_schema", "group_time_evaluation", "group_time_json", "group_time_plan",
              "group_time_saved", "input_support", "kernel", "metrics", "model", "package_source",
              "planning", "prediction", "prediction_schema", "report", "splitting", "verification")
METADATA = {"schema_id": "mlcalc.group_forward_window_metadata", "schema_version": 1,
            "time_unit": "tick", "clock_id": "handwritten-profile-clock", "gap": 0,
            "row_groups": ["old", "new"],
            "group_spans": [{"group": "old", "start": 0, "end": 1, "label_available_at": 1},
                            {"group": "new", "start": 5, "end": 6, "label_available_at": 6}],
            "windows": [{"lower": 5, "upper": 7}]}
# Frozen pre-change stdout for this handwritten metadata, including final newline.
PLAN_SHA256 = "091427595919463f714b5c16b8e74ef0b09d280e4a32d843f41da24c15103c19"
GUARD = '''import importlib.abc, sys
families = ('numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'threadpoolctl')
models = ('groupbench.model', 'groupbench.binary', 'groupbench.evaluation',
          'groupbench.prediction', 'groupbench.planning', 'groupbench.group_time_evaluation')
def forbidden(name):
    return any(name == item or name.startswith(item + '.') for item in families + models)
assert not any(forbidden(name) for name in sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('Unexpected numerical/model import: ' + fullname)
sys.meta_path.insert(0, Block())
'''


class FreshImportTests(unittest.TestCase):
    def child(self, code, *arguments, no_site=True):
        environment = dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return subprocess.run([sys.executable, "-B", *(["-S"] if no_site else []), "-c", code, *arguments],
                              capture_output=True, env=environment, timeout=20)

    def test_package_dir_is_light_and_unknown_attributes_fail(self):
        code = GUARD + '''
import groupbench
assert groupbench.__version__ == '0.3.0b7'
assert set(groupbench.__all__) <= set(dir(groupbench))
assert not any(name in vars(groupbench) for name in groupbench.__all__)
try:
    groupbench.not_a_public_api
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute resolved')
assert 'groupbench.not_a_public_api' not in sys.modules
assert not any(forbidden(name) for name in sys.modules)
'''
        result = self.child(code)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))

    def test_json_cli_and_raw_reader_imports_are_light(self):
        code = GUARD + '''
import hashlib
from pathlib import Path
from groupbench import cli, saved_scoring, group_time_json
from groupbench.cli import _open_csv_source
from groupbench.saved_scoring import _read_outcomes
assert _open_csv_source is cli._open_csv_source
assert {'grouped_metrics', 'paired_absolute_error_difference', 'contrast_float'} <= set(dir(saved_scoring))
assert not any(name in vars(saved_scoring) for name in saved_scoring._METRIC_NAMES)
raw, digest = _read_outcomes(Path(sys.argv[1]))
assert raw == b'\\xef\\xbb\\xbfhandwritten\\r\\n'
assert digest == hashlib.sha256(raw).hexdigest()
assert not any(forbidden(name) for name in sys.modules)
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"\xef\xbb\xbfhandwritten\r\n")
            result = self.child(code, str(path))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))

    def test_help_success_and_invalid_routes_without_dependencies(self):
        code = GUARD + '''
import json, runpy
module, args = json.loads(sys.argv[1])
sys.argv = [module, *args]
try:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
except SystemExit as error:
    status = error.code
else:
    status = 0
assert not any(forbidden(name) for name in sys.modules)
raise SystemExit(status)
'''
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / "tiny.json", Path(directory) / "PRIVATE.json"
            good.write_text(json.dumps(METADATA))
            bad.write_text('{"PRIVATE_SOURCE":true}')
            for module, prefix in (("groupbench", ["plan-group-forward-windows"]),
                                   ("groupbench.group_time_json", []),
                                   ("mlworkflow", ["tabular", "plan-group-forward-windows"])):
                for kind, arguments in (("help", ["--help"]), ("plan", [str(good)]), ("invalid", [str(bad)])):
                    with self.subTest(module=module, kind=kind):
                        result = self.child(code, json.dumps([module, prefix + arguments]))
                        if kind == "help":
                            self.assertEqual((result.returncode, result.stderr), (0, b""))
                            self.assertIn(b"usage:", result.stdout)
                        elif kind == "plan":
                            self.assertEqual((result.returncode, result.stderr), (0, b""))
                            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), PLAN_SHA256)
                        else:
                            self.assertEqual((result.returncode, result.stdout), (2, b""))
                            self.assertEqual(result.stderr, b"group forward-window plan rejected: invalid, unavailable or excessive metadata; no plan was written.\n")
                            self.assertNotIn(b"PRIVATE", result.stderr)
                        self.assertNotIn(b"RuntimeWarning", result.stderr)
                        self.assertNotIn(b"Traceback", result.stderr)

    def test_actual_direct_module_has_no_runpy_warning(self):
        environment = dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-S", "-m", "groupbench.group_time_json", "--help"],
                                capture_output=True, env=environment, timeout=15)
        self.assertEqual((result.returncode, result.stderr), (0, b""))
        self.assertIn(b"usage:", result.stdout)

    def test_required_model_dependency_failure_is_not_hidden_or_cached(self):
        code = '''import groupbench
for attempt in range(2):
    try:
        from groupbench import GroupBenchRegressor
    except ImportError:
        assert 'GroupBenchRegressor' not in vars(groupbench)
    else:
        raise AssertionError('Model dependencies unexpectedly optional')
'''
        result = self.child(code)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))

    def test_both_import_orders_keep_actual_export_and_submodule_identity(self):
        code = '''import importlib, inspect, json, pickle, sys
exports, submodules, first = json.loads(sys.argv[1])
import groupbench
assert groupbench.__all__ == list(exports)
assert set(exports) | set(submodules) <= set(dir(groupbench))
importlib.import_module('groupbench.' + first)
for name, module in exports.items():
    expected = getattr(importlib.import_module('groupbench.' + module), name)
    actual = getattr(groupbench, name)
    assert actual is expected is getattr(groupbench, name)
    assert vars(groupbench)[name] is expected
    assert actual.__module__ == expected.__module__
    try:
        assert inspect.signature(actual) == inspect.signature(expected)
    except ValueError:
        assert issubclass(actual, Exception)
    assert pickle.loads(pickle.dumps(actual)) is expected
namespace = {}
exec('from groupbench import *', namespace)
assert set(namespace) - {'__builtins__'} == set(exports)
assert all(namespace[name] is getattr(groupbench, name) for name in exports)
for name in submodules:
    assert getattr(groupbench, name) is importlib.import_module('groupbench.' + name)
from groupbench import cli, binary
assert cli is importlib.import_module('groupbench.cli')
assert binary is importlib.import_module('groupbench.binary')
from groupbench.model import GroupBenchRegressor
assert GroupBenchRegressor is groupbench.GroupBenchRegressor
'''
        for first in ("group_time_json", "group_time_evaluation"):
            with self.subTest(first=first):
                result = self.child(code, json.dumps([EXPORTS, SUBMODULES, first]), no_site=False)
                self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))


class LazyCLICompatibilityTests(unittest.TestCase):
    def test_direct_csv_helper_preserves_patched_lazy_attributes(self):
        from groupbench import cli
        frame = types.SimpleNamespace(columns=["group", "x"])
        fake_pandas = types.SimpleNamespace(read_csv=lambda *args, **kwargs: frame)
        with patch.object(cli, "pd", fake_pandas), patch.object(cli, "MAX_ROWS", 1):
            with patch.object(cli, "_open_csv_source", return_value=io.BytesIO(b"group,x\na,1\n")):
                result, digest = cli._read_csv_with_hash(Path("unused"))
            self.assertIs(result, frame)
            self.assertIs(cli.pd, fake_pandas)
            self.assertEqual(cli.MAX_ROWS, 1)
            self.assertEqual(digest, hashlib.sha256(b"group,x\na,1\n").hexdigest())
            with patch.object(cli, "_open_csv_source", return_value=io.BytesIO(b"group,x\na,1\nb,2\n")):
                with self.assertRaisesRegex(ValueError, "limit of 1 rows"):
                    cli._read_csv_with_hash(Path("unused"))

    def test_legacy_plan_defaults_and_monkeypatches_survive_loading(self):
        from groupbench import cli
        frame = object()
        for mode in ("run", "predict"):
            arguments = ["plan", mode, "train.csv"] + (["query.csv"] if mode == "predict" else [])
            arguments += ["--target", "y", "--group", "g"]
            original = arguments.copy()
            target = "plan_prediction" if mode == "predict" else "plan_benchmark"
            with patch.object(cli, target, return_value={"sentinel": mode}) as planner, \
                 patch.object(cli, "_read_training", return_value=(frame, ["x"], "a" * 64)), \
                 patch.object(cli, "_read_csv_with_hash", return_value=(frame, "b" * 64)), \
                 redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cli.main(arguments), 0)
                self.assertIs(getattr(cli, target), planner)
            self.assertEqual(arguments, original)
            self.assertEqual(err.getvalue(), "")
            self.assertEqual(out.getvalue(), json.dumps({"sentinel": mode}, indent=2) + "\n")
            options = planner.call_args.kwargs
            self.assertEqual(options["inner_splits"], 3)
            self.assertEqual(options["random_state"], 0)
            self.assertEqual(options["baselines"], ("dummy", "ridge", "extra_trees"))
            self.assertEqual(options["objective"], "group_mae")
            self.assertEqual(options["categorical_encoding"], "legacy-v1")
            self.assertEqual(options["inner_strategy"], "group_kfold")
            self.assertIsNone(options["domain"])
            if mode == "run": self.assertEqual(options["outer_splits"], 5)

    def test_metadata_dispatch_preserves_explicit_and_process_argv(self):
        from groupbench import cli, group_time_json
        arguments = ["plan-group-forward-windows", "name with spaces.json"]
        process = ["mlcalc", *arguments]
        with patch.object(group_time_json, "main", return_value=7) as delegated:
            with patch.object(cli, "_load_dependencies", side_effect=AssertionError("legacy import")):
                self.assertEqual(cli.main(arguments), 7)
                with patch.object(sys, "argv", process):
                    self.assertEqual(cli.main(), 7)
                    self.assertEqual(sys.argv, process)
        self.assertEqual(arguments, ["plan-group-forward-windows", "name with spaces.json"])
        self.assertEqual([call.args for call in delegated.call_args_list], [(["name with spaces.json"],)] * 2)


class SavedMetricCompatibilityTests(unittest.TestCase):
    def test_metric_attributes_are_original_cached_objects(self):
        from groupbench import metrics, saved_scoring
        from groupbench.saved_scoring import grouped_metrics, paired_absolute_error_difference, contrast_float
        names = ("grouped_metrics", "paired_absolute_error_difference", "contrast_float")
        self.assertTrue(set(names) <= set(dir(saved_scoring)))
        for name, value in zip(names, (grouped_metrics, paired_absolute_error_difference, contrast_float)):
            self.assertIs(value, getattr(metrics, name))
            self.assertIs(getattr(saved_scoring, name), value)
            self.assertIs(vars(saved_scoring)[name], value)
        with self.assertRaises(AttributeError):
            saved_scoring.not_a_metric

    def test_scoring_and_comparison_honor_all_three_metric_patches(self):
        from groupbench import cli, metrics, saved_scoring
        class Query:
            columns = ("x",)
            def __len__(self): return 1
        source_hash = "a" * 64
        schema = types.SimpleNamespace(n_predictions=1, schema_version="1.0")
        manifest = {"prediction_source_file_sha256": source_hash,
                    "config": {"target": "y", "features": ["x"]}}
        loaded = (schema, manifest, b"row_position,prediction\n0,1\n", {})
        contract = {key: "invented" for key in saved_scoring._COMPARISON_CONTRACT}
        contract.update(method_a="A", method_b="B", target_column="y")
        with patch.object(saved_scoring, "grouped_metrics", wraps=metrics.grouped_metrics) as grouped, \
             patch.object(saved_scoring, "paired_absolute_error_difference", wraps=metrics.paired_absolute_error_difference) as paired, \
             patch.object(saved_scoring, "contrast_float", wraps=metrics.contrast_float) as display, \
             patch.object(saved_scoring, "_load_prediction_bundle", return_value=loaded), \
             patch.object(saved_scoring, "_read_outcomes", return_value=(b"row_position,group,observed\n0,g,2\n", "b" * 64)), \
             patch.object(cli, "_read_csv_with_hash", return_value=(Query(), source_hash)):
            result = saved_scoring.compare_saved_predictions("a", "b", "query", "outcomes", contract)
            self.assertIs(saved_scoring.grouped_metrics, grouped)
            self.assertIs(saved_scoring.paired_absolute_error_difference, paired)
            self.assertIs(saved_scoring.contrast_float, display)
        self.assertEqual(grouped.call_count, 2)
        paired.assert_called_once_with(1, 1, 2)
        self.assertEqual(display.call_count, 4)
        self.assertEqual(result["row_mean_absolute_error_b_minus_a"], 0)
        self.assertEqual(result["fits_performed"], 0)


if __name__ == "__main__":
    unittest.main()
