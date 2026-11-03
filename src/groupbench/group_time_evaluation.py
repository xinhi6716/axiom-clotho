"""Bounded, in-memory retrospective evaluation of two fixed temporal baselines.

This is not selection, full OOF evaluation, a saved-artifact schema, or a
prediction API. Declarations and content bindings do not certify source truth,
causality, feature availability, independence, or execution. See
``docs/GROUP_FORWARD_BASELINE_EVALUATION.md`` for the complete v1 boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import platform
import re
import struct
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from . import __version__
# Only the transport's pure document/bound helpers are reused, not its parser,
# CLI, reader, or integer-only feature/target transport semantics.
from .group_time_json import _document, _output_bound
from .group_time_plan import (
    ForwardWindowFold, GroupForwardWindowPlan, GroupSpan, TimeWindow,
    MAX_TICK, MIN_TICK, plan_group_forward_windows,
)
from .metrics import _metric_vector, grouped_metrics
from .model import Candidate, make_pipeline, _validate_fitted_numeric_state
from .package_source import observe_package_source

PROTOCOL = "group-forward-fixed-regression-v1"
COVERAGE_MODE = "partial_predeclared"
MAX_ROWS = 20_000
MAX_FEATURES = 100
MAX_WINDOWS = 32
MAX_WORK = 250_000
MAX_PLAN_BYTES = 16 * 1024 * 1024
MAX_MATRIX_CELLS = 20_000_000
_RECIPE_NAMES = ("dummy_mean", "ridge_alpha_1")
_LIMITATIONS = (
    "Retrospective fixed regression recipes only: no winner, tuning, refit, classification, or live prediction of unlabeled rows.",
    "Partial predeclared test coverage, never full OOF: training participation is not test coverage; no predictions for untested rows.",
    "Caller declarations do not establish honest predeclaration, source timing, causality, group independence, or prediction-time feature availability.",
    "Feature and metadata roles enforce the declared boundary, not detection of renamed metadata, target-derived predictors, or undeclared aliases; the caller is responsible for these.",
    "Bindings check retained content consistency, not source truth, atomic input snapshots, executed-code identity, or proof a model ran. Concurrent source mutation during snapshotting is unsupported.",
    "Training uses unweighted rows; group weighting applies only to reported metrics. Group RMSE is the mean of per-group RMSE.",
    "A Ridge iteration-cap hit rejects execution; a result below the cap is not a numerical-accuracy guarantee. Rescale extreme numeric units before evaluation.",
    "Finite resource limits are scheduling limits, not exact solver-operation, memory, or wall-clock guarantees. Dependency and hardware changes may change numeric results.",
    "Available safe local package-source observation is required. NumPy-backed real numeric columns only; nullable or masked extension payloads are unsupported.",
)


class GroupForwardEvaluationError(ValueError):
    """Categorical rejection; counts describe actual attempted pipeline fits."""

    def __init__(self, message, *, attempted_fits=0, completed_fits=0):
        super().__init__(message)
        self.attempted_fits = attempted_fits
        self.completed_fits = completed_fits


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError("Unsupported immutable protocol value.")


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


class _Document:
    __slots__ = ()

    def to_dict(self):
        """Return a detached mutable copy, never live evaluation buffers."""
        return {field.name: _thaw(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class GroupForwardEvaluationPreview(_Document):
    schema_id: str
    schema_version: int
    protocol: str
    coverage_mode: str
    bindings: Mapping
    plan: Mapping
    feature_schema: tuple
    metadata_columns: tuple[str, ...]
    target_name: str
    group_name: str
    recipes: tuple
    fit_budget: Mapping
    coverage: Mapping
    windows: tuple
    resource_limits: Mapping
    environment: Mapping
    fits_performed: int
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupForwardEvaluationResult(_Document):
    schema_id: str
    schema_version: int
    protocol: str
    coverage_mode: str
    bindings: Mapping
    plan: Mapping
    feature_schema: tuple
    metadata_columns: tuple[str, ...]
    target_name: str
    group_name: str
    recipes: tuple
    fit_budget: Mapping
    coverage: Mapping
    predictions: tuple
    metrics: tuple
    environment: Mapping
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Job:
    window_index: int
    recipe_index: int
    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Prepared:
    X: pd.DataFrame
    y: np.ndarray
    groups: tuple
    plan: GroupForwardWindowPlan
    jobs: tuple[_Job, ...]
    common: dict
    cells: int


def _reject(message):
    raise GroupForwardEvaluationError(message)


def _name(value, category):
    if type(value) is not str or not value or value.strip() != value:
        _reject(f"{category} requires a nonblank unpadded built-in string name.")


def _names(value, category):
    if type(value) not in (list, tuple) or not value:
        _reject(f"{category} requires a nonempty explicit list or tuple.")
    for name in value:
        _name(name, category)
    if len(set(value)) != len(value):
        _reject(f"{category} must contain distinct names.")
    return tuple(value)


def _identity(value, position):
    if type(value) is str and value.strip():
        return
    if type(value) is int and MIN_TICK <= value <= MAX_TICK:
        return
    _reject(f"Row group at position {position} requires an exact planner identity.")


def _same_typed(left, right):
    """Compare every canonical field without bool/int or container coercion."""
    if type(left) is not type(right):
        return False
    if is_dataclass(right):
        return all(_same_typed(getattr(left, f.name), getattr(right, f.name))
                   for f in fields(right))
    if type(right) in (list, tuple):
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right))
    if type(right) is dict:
        return (left.keys() == right.keys()
                and all(_same_typed(left[key], value) for key, value in right.items()))
    if type(right) is float:
        return struct.pack("<d", left) == struct.pack("<d", right)
    return left == right


def _encode(value):
    """Versioned callers domain-separate this explicit type/length encoding."""
    if value is None:
        return b"n"
    if type(value) is bool:
        return b"t" if value else b"f"
    if type(value) is int:
        payload, tag = str(value).encode("ascii"), b"i"
    elif type(value) is float:
        payload, tag = struct.pack("<d", value), b"d"
    elif type(value) is str:
        payload, tag = value.encode("utf-8", errors="surrogatepass"), b"s"
    elif type(value) is bytes:
        payload, tag = value, b"b"
    elif type(value) in (list, tuple):
        payload = b"".join(_encode(item) for item in value)
        tag = b"l" if type(value) is list else b"q"
        return tag + struct.pack("<Q", len(value)) + struct.pack("<Q", len(payload)) + payload
    elif type(value) is dict:
        if any(type(key) is not str for key in value):
            _reject("Binding maps require string field names.")
        payload = b"".join(_encode(key) + _encode(value[key]) for key in sorted(value))
        return b"m" + struct.pack("<Q", len(value)) + struct.pack("<Q", len(payload)) + payload
    else:
        _reject("Unsupported binding value type.")
    return tag + struct.pack("<Q", len(payload)) + payload


def _digest(domain, value):
    return hashlib.sha256(_encode("mlcalc.temporal-binding-v1:" + domain) + _encode(value)).hexdigest()


def _recipes():
    preprocessing = [
        {"step": "SimpleImputer", "strategy": "median", "keep_empty_features": True},
        {"step": "StandardScaler", "with_mean": True, "with_std": True},
    ]
    return [
        {"name": "dummy_mean", "preprocessing": preprocessing,
         "estimator": "DummyRegressor", "parameters": {"strategy": "mean"},
         "training_weight": "unweighted_rows"},
        {"name": "ridge_alpha_1", "preprocessing": preprocessing,
         "estimator": "Ridge", "parameters": {
             "alpha": 1.0, "solver": "lsqr", "fit_intercept": True,
             "tol": 1e-8, "max_iter": 1000, "copy_X": True, "positive": False},
         "training_weight": "unweighted_rows", "iteration_cap_policy": "reject_at_cap"},
    ]


def _candidate(index):
    # Fresh private dictionaries; no mutable externally supplied candidate.
    if index == 0:
        return Candidate("dummy_mean", "dummy", {"strategy": "mean"})
    return Candidate("ridge_alpha_1", "ridge", {
        "alpha": 1.0, "fit_intercept": True, "tol": 1e-8,
        "max_iter": 1000, "copy_X": True, "positive": False})


def _numeric(series, *, missing):
    # Nullable extension arrays are masked payloads, not NumPy numeric buffers.
    dtype = series.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind not in "iuf" or dtype.itemsize > 8:
        _reject("Numeric columns require NumPy real integer/float dtypes of at most 64 bits, without masked payloads.")
    original = series.to_numpy(copy=True)
    if np.ma.isMaskedArray(original):
        _reject("Masked numeric payloads are unsupported.")
    mask = np.isnan(original) if dtype.kind == "f" else np.zeros(len(original), dtype=bool)
    if mask.any() and not missing:
        _reject("Targets must be observed finite numbers at every input row.")
    values = np.zeros(len(original), dtype=np.float64)
    try:
        if (~mask).any():
            # Reuse checked finite/precision conversion, including exact large ints.
            values[~mask] = _metric_vector(original[~mask])
    except ValueError:
        _reject("Numeric input contains unsupported, nonfinite, or inexact float64 values.")
    values[mask] = np.nan
    return values, mask


def _rebuild_plan(plan, groups):
    if type(plan) is not GroupForwardWindowPlan:
        _reject("Plan must have the exact supported typed planner schema.")
    if (type(plan.row_groups) is not tuple or type(plan.group_spans) is not tuple
            or type(plan.folds) is not tuple
            or not 1 <= len(plan.folds) <= MAX_WINDOWS
            or not 1 <= len(plan.group_spans) <= len(groups)):
        _reject("Plan containers or declared counts are unsupported.")
    if not _same_typed(plan.row_groups, groups):
        _reject("Plan row groups must match actual typed groups at every position.")
    if len(plan.folds) * (len(groups) + len(plan.group_spans)) > MAX_WORK:
        _reject("Plan exceeds the window-by-row/group work limit.")
    if any(type(span) is not GroupSpan for span in plan.group_spans):
        _reject("Plan span records require exact supported types.")
    if any(type(fold) is not ForwardWindowFold or type(fold.window) is not TimeWindow
           for fold in plan.folds):
        _reject("Plan fold/window records require exact supported types.")
    # Bound raw text before JSON escaping/complete document allocation. The
    # conservative transport bound below subsequently accounts for every copy.
    labels = (plan.time_unit, plan.clock_id)
    if any(type(label) is not str for label in labels):
        _reject("Plan clock/unit declarations require exact string types.")
    for position, span in enumerate(plan.group_spans):
        _identity(span.group, position)
    size = (sum(len(value) for value in groups if type(value) is str)
            + sum(len(span.group) for span in plan.group_spans if type(span.group) is str)
            + sum(map(len, labels)))
    if size > MAX_PLAN_BYTES:
        _reject("Plan exceeds the conservative document size bound.")
    metadata = {"row_groups": groups,
                "group_spans": [{"group": span.group} for span in plan.group_spans],
                "windows": tuple(fold.window for fold in plan.folds),
                "time_unit": plan.time_unit, "clock_id": plan.clock_id}
    if _output_bound(metadata) > MAX_PLAN_BYTES:
        _reject("Plan exceeds the conservative document size bound.")
    # All allocation/work bounds are checked before reconstructing membership.
    try:
        rebuilt = plan_group_forward_windows(
            groups, group_spans=tuple(GroupSpan(span.group, span.start, span.end, span.label_available_at)
                                     for span in plan.group_spans),
            windows=tuple(TimeWindow(fold.window.lower, fold.window.upper) for fold in plan.folds),
            time_unit=plan.time_unit, clock_id=plan.clock_id, gap=plan.gap)
    except (ValueError, TypeError, OverflowError):
        _reject("Plan does not define a supported complete forward schedule.")
    if not _same_typed(plan, rebuilt):
        _reject("Plan differs from its complete canonical reconstruction.")
    all_rows, seen_rows, seen_groups = set(range(len(groups))), set(), set()
    for fold in rebuilt.folds:
        partitions = [set(member.rows) for member in (fold.train, fold.test, fold.unused)]
        group_parts = [{(type(g), g) for g in member.groups}
                       for member in (fold.train, fold.test, fold.unused)]
        if (set.union(*partitions) != all_rows
                or sum(map(len, partitions)) != len(groups)
                or sum(map(len, group_parts)) != len(rebuilt.total.groups)
                or set.union(*group_parts) != {(type(g), g) for g in rebuilt.total.groups}
                or seen_rows & partitions[1] or seen_groups & group_parts[1]):
            _reject("Canonical plan has invalid partitioning or repeated test coverage.")
        seen_rows.update(partitions[1])
        seen_groups.update(group_parts[1])
    if seen_rows != set(rebuilt.tested.rows):
        _reject("Canonical tested membership is inconsistent.")
    return rebuilt


def _verified_plan(plan, groups):
    try:
        return _rebuild_plan(plan, groups)
    except GroupForwardEvaluationError:
        raise
    except (AttributeError, TypeError, OverflowError, RecursionError):
        _reject("Plan contains malformed or unsupported typed fields.")


def _coverage(plan):
    def membership(value):
        return {"rows": list(value.rows), "groups": list(value.groups),
                "n_rows": value.n_rows, "n_groups": value.n_groups}
    return {"total": membership(plan.total), "tested": membership(plan.tested),
            "untested": membership(plan.untested), "untested_predictions_present": False}


def _limits(cells):
    return {"max_rows": MAX_ROWS, "max_features": MAX_FEATURES,
            "max_windows": MAX_WINDOWS, "max_groups": MAX_ROWS,
            "max_window_row_group_work": MAX_WORK, "max_plan_document_bytes": MAX_PLAN_BYTES,
            "max_matrix_cell_visits": MAX_MATRIX_CELLS, "matrix_cell_visits": cells,
            "max_pipeline_fits": 2 * MAX_WINDOWS, "threads": 1,
            "ridge_max_iter": 1000}


def _budget(jobs, *, completed=False, cells):
    windows = tuple(dict.fromkeys(job.window_index for job in jobs))
    count = len(jobs)
    return {"unit": "pipeline.fit calls", "expected_total": count,
            "attempted_total": count if completed else 0,
            "completed_total": count if completed else 0,
            "per_window": [{"window_index": i,
                            "expected": sum(job.window_index == i for job in jobs),
                            "attempted": sum(job.window_index == i for job in jobs) if completed else 0,
                            "completed": sum(job.window_index == i for job in jobs) if completed else 0}
                           for i in windows],
            "inner_selection_fits": 0, "full_data_refits": 0, "selected_model_refits": 0,
            "bootstraps": 0, "retries": 0, "resource_limits": _limits(cells)}


def _prepare(X, y, row_groups, plan, feature_allowlist, metadata_columns):
    if type(X) is not pd.DataFrame or type(y) is not pd.Series or type(row_groups) is not pd.Series:
        _reject("Inputs require a DataFrame and named target/group Series.")
    n, p = X.shape
    if not 1 <= n <= MAX_ROWS or not 1 <= p <= MAX_FEATURES:
        _reject("Input row/feature counts exceed v1 limits.")
    for value in (X, y, row_groups):
        index = value.index
        if (len(value) != n or type(index) is not pd.RangeIndex or index.name is not None
                or index.start != 0 or index.stop != n or index.step != 1):
            _reject("All inputs require the same unnamed canonical RangeIndex(0, N, 1).")
    features = _names(feature_allowlist, "Feature allowlist")
    metadata = _names(metadata_columns, "Metadata columns")
    _name(y.name, "Target")
    _name(row_groups.name, "Group")
    if (tuple(X.columns) != features or any(type(column) is not str for column in X.columns)
            or y.name == row_groups.name or y.name in features or row_groups.name in features
            or row_groups.name not in metadata or y.name in metadata or set(features) & set(metadata)):
        _reject("Feature order, target/group names, or declared metadata roles conflict.")
    if row_groups.dtype != np.dtype("O"):
        _reject("Row groups require object dtype to preserve exact typed identities.")
    groups = tuple(row_groups.to_numpy(copy=True))
    for position, value in enumerate(groups):
        _identity(value, position)
    verified = _verified_plan(plan, groups)
    jobs = tuple(_Job(fold.index, recipe_index, fold.train.rows, fold.test.rows)
                 for fold in verified.folds for recipe_index in range(len(_RECIPE_NAMES)))
    cells = sum((len(job.train) + len(job.test)) * p for job in jobs)
    if cells > MAX_MATRIX_CELLS:
        _reject("Evaluation exceeds the matrix-cell visit scheduling limit.")
    values, masks, schema = [], [], []
    for name in features:
        array, mask = _numeric(X[name], missing=True)
        values.append(array)
        masks.append(mask)
        schema.append({"name": name, "source_dtype": str(X[name].dtype), "normalized_dtype": "float64"})
    matrix = np.column_stack(values)
    missing = np.column_stack(masks)
    target, _ = _numeric(y, missing=False)
    # A canonical zero payload plus explicit mask ignores NaN payload variation
    # while keeping every observed signed zero and little-endian float64 bit.
    binding_values = matrix.copy()
    binding_values[missing] = 0.0
    document = _document(verified)
    source = observe_package_source()
    if source.get("status") != "available":
        _reject("Available safe local package-source observation is required for execution binding.")
    environment = {"package_version": __version__, "package_source": source,
                   "python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "sklearn": sklearn.__version__,
                   "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__}
    recipes = _recipes()
    plan_digest = _digest("plan", document)
    input_digest = _digest("input", {
        "feature_schema": schema, "shape": (n, p),
        "feature_values_le_float64": binding_values.astype("<f8", copy=False).tobytes(order="C"),
        "feature_missing_mask_uint8": missing.astype(np.uint8).tobytes(order="C"),
        "target_values_le_float64": target.astype("<f8", copy=False).tobytes(),
        "target_source_dtype": str(y.dtype), "row_groups": groups,
        "index_rule": "unnamed_RangeIndex_0_N_1", "metadata_columns": metadata,
        "target_name": y.name, "group_name": row_groups.name})
    execution_digest = _digest("execution", {
        "plan_sha256": plan_digest, "input_sha256": input_digest,
        "protocol": PROTOCOL, "coverage_mode": COVERAGE_MODE,
        "recipes": recipes, "environment": environment,
        "resource_limits": _limits(cells)})
    common = {"schema_version": 1, "protocol": PROTOCOL, "coverage_mode": COVERAGE_MODE,
              "bindings": {"encoding": "mlcalc.temporal-binding-v1", "plan_sha256": plan_digest,
                           "input_sha256": input_digest, "execution_sha256": execution_digest},
              "plan": document, "feature_schema": schema, "metadata_columns": metadata,
              "target_name": y.name, "group_name": row_groups.name, "recipes": recipes,
              "coverage": _coverage(verified), "environment": environment,
              "limitations": _LIMITATIONS}
    matrix.setflags(write=False)
    target.setflags(write=False)
    return _Prepared(pd.DataFrame(matrix, columns=features, copy=True), target, groups,
                     verified, jobs, common, cells)


def preview_group_forward_evaluation(X, y, *, row_groups, plan, feature_allowlist,
                                     metadata_columns) -> GroupForwardEvaluationPreview:
    """Validate and bind inputs and the entire schedule, without creating a model."""
    prepared = _prepare(X, y, row_groups, plan, feature_allowlist, metadata_columns)
    windows = [{"window_index": fold.index,
                **{name: {"n_rows": getattr(fold, name).n_rows,
                          "n_groups": getattr(fold, name).n_groups}
                   for name in ("train", "test", "unused")}}
               for fold in prepared.plan.folds]
    return GroupForwardEvaluationPreview(**{
        key: _freeze(value) for key, value in {
            "schema_id": "mlcalc.group_forward_baseline_preview", **prepared.common,
            "fit_budget": _budget(prepared.jobs, cells=prepared.cells), "windows": windows,
            "resource_limits": _limits(prepared.cells), "fits_performed": 0}.items()})


def _finite(value, category):
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        _reject(f"Execution produced nonfinite or invalid {category}.")


def _learned_state(pipeline, recipe_index, train, test):
    _validate_fitted_numeric_state(pipeline)
    preprocessing = pipeline.named_steps["preprocess"]
    for data in (train, test):
        transformed = preprocessing.transform(data)
        _finite(transformed, "transformed numeric values")
        if transformed.shape != data.shape:
            _reject("Numeric preprocessing changed the declared matrix shape.")
    estimator = pipeline.named_steps["model"]
    if recipe_index == 0:
        _finite(estimator.constant_, "dummy constant")
    else:
        _finite(estimator.coef_, "Ridge coefficients")
        _finite(estimator.intercept_, "Ridge intercept")
        iterations = np.asarray(estimator.n_iter_)
        if (iterations.shape != (1,) or iterations.dtype.kind not in "iu"
                or not ((iterations >= 0) & (iterations < 1000)).all()):
            _reject("Ridge reached its iteration cap or returned invalid iteration state.")


def _checked_grouped(y, prediction, groups):
    identities = tuple(dict.fromkeys((type(group), group) for group in groups))
    lookup = {identity: i for i, identity in enumerate(identities)}
    codes = np.asarray([lookup[(type(group), group)] for group in groups], dtype=np.int64)
    pooled, records = grouped_metrics(y, prediction, codes)
    if len(records) != len(identities):
        _reject("Metric surrogate group coverage is inconsistent.")
    for code, record in enumerate(records):
        if type(record["group"]) is not str or record["group"] != str(code):
            _reject("Metric surrogate group mapping is inconsistent.")
        record["group"] = identities[code][1]
    for record in [pooled, *records]:
        for key, value in record.items():
            if key != "group" and value is not None:
                _finite(value, "metric arithmetic")
    return pooled, records


def _metrics(records, plan):
    result = []
    for name in _RECIPE_NAMES:
        y = np.asarray([record["observed"] for record in records], dtype=np.float64)
        predictions = np.asarray([record["prediction_" + name] for record in records], dtype=np.float64)
        groups = tuple(record["group"] for record in records)
        pooled, per_group = _checked_grouped(y, predictions, groups)
        per_window = []
        for fold in plan.folds:
            indices = [i for i, record in enumerate(records) if record["window_index"] == fold.index]
            scores, _ = _checked_grouped(y[indices], predictions[indices], tuple(groups[i] for i in indices))
            per_window.append({"window_index": fold.index, **scores})
        result.append({"recipe": name, "pooled": pooled, "per_window": per_window,
                       "per_group": per_group})
    return result


def _validate_result(result, prepared):
    """Internal consistency, not validation of arbitrary external saved input."""
    document = result.to_dict()
    for key, value in prepared.common.items():
        # to_dict uses lists for all frozen ordered containers.
        if not _same_typed(document[key], _thaw(_freeze(value))):
            _reject("Result metadata differs from its prepared declaration.")
    if (document["schema_id"] != "mlcalc.group_forward_baseline_evaluation"
            or not _same_typed(document["fit_budget"], _budget(prepared.jobs, completed=True, cells=prepared.cells))):
        _reject("Result schema or fit counts are inconsistent.")
    expected_positions = prepared.plan.tested.rows
    records = document["predictions"]
    if len(records) != len(expected_positions):
        _reject("Result prediction coverage is inconsistent.")
    windows = {position: fold.index for fold in prepared.plan.folds for position in fold.test.rows}
    keys = {"row_position", "group", "window_index", "observed",
            "prediction_dummy_mean", "prediction_ridge_alpha_1"}
    for record, position in zip(records, expected_positions):
        if (type(record) is not dict or record.keys() != keys
                or not _same_typed(record["row_position"], position)
                or not _same_typed(record["group"], prepared.groups[position])
                or not _same_typed(record["window_index"], windows[position])
                or not _same_typed(record["observed"], float(prepared.y[position]))):
            _reject("Result tested-row identity or window mapping is inconsistent.")
        for name in _RECIPE_NAMES:
            prediction = record["prediction_" + name]
            if type(prediction) is not float:
                _reject("Result predictions require finite float64 values.")
            _finite(prediction, "predictions")
    if not _same_typed(document["metrics"], _metrics(records, prepared.plan)):
        _reject("Result metrics disagree with retained tested predictions.")


def evaluate_group_forward_baselines(X, y, *, row_groups, plan, feature_allowlist,
                                     metadata_columns, expected_execution_sha256
                                     ) -> GroupForwardEvaluationResult:
    """Execute exactly two fresh train-only pipelines per predeclared window.

    A preview's matching execution digest is mandatory before pipeline creation.
    A failed run returns no partial success and performs no retry or replacement.
    Error counts distinguish attempted fits from returned pipeline.fit calls.
    """
    prepared = _prepare(X, y, row_groups, plan, feature_allowlist, metadata_columns)
    if (type(expected_execution_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_execution_sha256) is None
            or expected_execution_sha256 != prepared.common["bindings"]["execution_sha256"]):
        _reject("Execution binding is missing, malformed, or stale; preview these exact inputs again.")
    attempted = completed = 0
    predictions = {}
    try:
        with threadpool_limits(limits=1):
            for job in prepared.jobs:
                train, test = prepared.X.iloc[list(job.train)], prepared.X.iloc[list(job.test)]
                pipeline = make_pipeline(train, _candidate(job.recipe_index), 0)
                attempted += 1
                pipeline.fit(train, prepared.y[list(job.train)])
                completed += 1
                _learned_state(pipeline, job.recipe_index, train, test)
                prediction = np.asarray(pipeline.predict(test))
                if prediction.shape != (len(job.test),) or prediction.dtype.kind != "f" or prediction.dtype.itemsize > 8:
                    _reject("Prediction vector must be one-dimensional float64-compatible test-row values.")
                _finite(prediction, "predictions")
                for position, value in zip(job.test, prediction):
                    record = predictions.setdefault(position, {
                        "row_position": position, "group": prepared.groups[position],
                        "window_index": job.window_index, "observed": float(prepared.y[position])})
                    key = "prediction_" + _RECIPE_NAMES[job.recipe_index]
                    if key in record:
                        _reject("Duplicate tested prediction in execution schedule.")
                    record[key] = float(value)
            records = [predictions[position] for position in prepared.plan.tested.rows]
            result = GroupForwardEvaluationResult(**{
                key: _freeze(value) for key, value in {
                    "schema_id": "mlcalc.group_forward_baseline_evaluation", **prepared.common,
                    "fit_budget": _budget(prepared.jobs, completed=True, cells=prepared.cells),
                    "predictions": records, "metrics": _metrics(records, prepared.plan)}.items()})
            if attempted != len(prepared.jobs) or completed != len(prepared.jobs):
                _reject("Execution fit counts disagree with the prepared schedule.")
            _validate_result(result, prepared)
            return result
    except Exception:
        # Do not echo private values, estimator messages, or paths. This run is
        # failed even when earlier pipelines completed; never publish them.
        raise GroupForwardEvaluationError(
            "Temporal evaluation failed; no successful partial result was returned. "
            "Check numeric units and the fixed recipe's finite-state/iteration requirements.",
            attempted_fits=attempted, completed_fits=completed) from None
