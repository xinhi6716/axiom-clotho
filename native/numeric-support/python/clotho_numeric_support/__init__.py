"""Explicit optional numeric-only diagnostics; never an automatic backend.

Training must be normalized float64 features from Clotho's prepare_data or
validate_features. Keep inputs stable while snapshots are created. This package
does not fit models, read targets, encode categories, or change default routes.
"""

import importlib
from dataclasses import dataclass
import sys
import sysconfig

import numpy as np
import pandas as pd

from groupbench.data import MAX_FEATURES, MAX_ROWS, validate_prediction_features
from groupbench import input_support as _python_support


__all__ = ["prediction_input_support_numeric", "NumericSupportReference"]
_PROTOCOL = "numeric-source-support-v1"
_BAD_NATIVE = "Malformed native numeric-support result; no report was produced."


def _load_native():
    if sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"):
        raise RuntimeError("The optional numeric backend currently requires GIL-enabled CPython.")
    if getattr(_python_support, "NUMERIC_SUPPORT_PROTOCOL", None) != _PROTOCOL:
        raise RuntimeError("This optional backend requires the matching current Clotho numeric-support protocol.")
    try:
        native = importlib.import_module("clotho_numeric_support._native")
    except ImportError:
        raise RuntimeError("The optional native numeric extension is not installed or could not load; no compiler or fallback was invoked.") from None
    if (getattr(native, "PROTOCOL", None) != _PROTOCOL
            or getattr(native, "MAX_ROWS", None) != MAX_ROWS
            or getattr(native, "MAX_FEATURES", None) != MAX_FEATURES):
        raise RuntimeError("Native numeric-support protocol or bounds do not match this Clotho installation.")
    return native


def _training_schema(training):
    if not isinstance(training, pd.DataFrame):
        raise TypeError("Normalized training features must be a pandas DataFrame.")
    if not 1 <= len(training) <= MAX_ROWS or not 1 <= training.shape[1] <= MAX_FEATURES:
        raise ValueError("Normalized training must contain 1 to 20,000 rows and 1 to 100 features.")
    if not training.columns.is_unique or any(not isinstance(name, str) for name in training.columns):
        raise ValueError("Normalized training feature names must be unique strings.")
    if any(training[name].dtype != np.dtype("float64") for name in training):
        raise ValueError("This optional backend accepts only normalized float64 training features. Use Clotho prepare_data or validate_features first; use the Python diagnostic for categorical or mixed features.")


def _count(value, limit):
    return type(value) is int and 0 <= value <= limit


def _report(result, training, incoming):
    return _report_dimensions(result, training.columns, len(training), len(incoming))


def _report_dimensions(result, names, n, m):
    p = len(names)
    if type(result) is not tuple or len(result) != 5:
        raise RuntimeError(_BAD_NATIVE)
    nt, nq, any_flag, flags, rows = result
    if (type(nt) is not int or nt != n or type(nq) is not int or nq != m
            or not _count(any_flag, m) or type(flags) is not tuple or len(flags) != 4
            or not all(_count(value, m) for value in flags) or flags[2] != 0
            or not max(flags) <= any_flag <= min(m, sum(flags))
            or type(rows) is not list or len(rows) != p):
        raise RuntimeError(_BAD_NATIVE)
    features = []
    for name, counts in zip(names, rows):
        if type(counts) is not tuple or len(counts) != 5:
            raise RuntimeError(_BAD_NATIVE)
        source_missing, query_missing, unsupported, below, above = counts
        if (not _count(source_missing, n) or not _count(query_missing, m)
                or not _count(unsupported, m - query_missing)):
            raise RuntimeError(_BAD_NATIVE)
        all_missing = source_missing == n
        if all_missing:
            valid = below is None and above is None and unsupported == m - query_missing
        else:
            valid = (_count(below, m - query_missing) and _count(above, m - query_missing)
                     and below + above <= m - query_missing and unsupported == 0)
        if not valid:
            raise RuntimeError(_BAD_NATIVE)
        features.append({
            "name": name, "kind": "numeric", "training_missing_count": source_missing,
            "training_support": "unavailable_all_missing" if all_missing else "observed_non_missing_values",
            "prediction_missing_count": query_missing,
            "prediction_non_missing_without_training_values_count": unsupported,
            "below_training_min_count": below, "above_training_max_count": above,
        })
    if any_flag < max(feature["prediction_missing_count"]
                      + feature["prediction_non_missing_without_training_values_count"]
                      + (feature["below_training_min_count"] or 0)
                      + (feature["above_training_max_count"] or 0) for feature in features):
        raise RuntimeError(_BAD_NATIVE)
    for flag, counts in (
        (flags[0], [feature["prediction_missing_count"] for feature in features]),
        (flags[1], [(feature["below_training_min_count"] or 0)
                    + (feature["above_training_max_count"] or 0) for feature in features]),
        (flags[3], [feature["prediction_non_missing_without_training_values_count"] for feature in features]),
    ):
        if not max(counts) <= flag <= min(m, sum(counts)):
            raise RuntimeError(_BAD_NATIVE)
    return {
        "schema_version": "1.0", "reference": "full_training_features_before_imputation",
        "action": "diagnostic_only_no_input_changes", "n_training_rows": n, "n_prediction_rows": m,
        "rows_with_any_flag": any_flag,
        "row_flag_counts": dict(zip(("missing_value", "numeric_outside_training_range", "unseen_category",
                                     "non_missing_value_in_all_missing_training_feature"), flags)),
        "features": features,
    }


def prediction_input_support_numeric(training, new_rows):
    """Match the Python diagnostic for normalized numeric training features.

    Raw prediction admission/alignment stays in the existing Python validator.
    Every numeric column becomes a private immutable little-endian binary64
    byte snapshot. Rust reads bytes safely while attached to Python; it never
    borrows a mutable NumPy buffer. Names stay in Python without UTF-8 coercion.
    No native error silently falls back to a different backend.
    """
    _training_schema(training)
    incoming = validate_prediction_features(
        new_rows, training.columns, [str(training[name].dtype) for name in training])
    native = _load_native()
    source_bytes = tuple(training[name].to_numpy(dtype="<f8", copy=False).tobytes() for name in training)
    query_bytes = tuple(incoming[name].to_numpy(dtype="<f8", copy=False).tobytes() for name in incoming)
    return _report(native.counts(source_bytes, query_bytes), training, incoming)


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class NumericSupportReference:
    """Explicit immutable numeric support snapshot for repeated query batches.

    Use normalized float64 training features, exactly as for the one-shot API.
    Rust retains only support bounds and counts; no DataFrame, training bytes,
    targets or fitted model are retained. Rebuild explicitly when the training
    reference changes. This is a diagnostic, not a predictor or an OOD gate.
    Keep inputs stable while creating the snapshot or each query's byte copies.
    """

    _reference: object
    _names: tuple
    _n_training_rows: int

    def __init__(self, training):
        if hasattr(self, "_reference"):
            raise TypeError("NumericSupportReference is already initialized; construct a new reference explicitly.")
        _training_schema(training)
        native = _load_native()
        if getattr(native, "REFERENCE_PROTOCOL", None) != "numeric-source-reference-v1":
            raise RuntimeError("The installed native extension does not support this training-reference protocol; rebuild the matching optional package.")
        names = tuple(training.columns)
        source_bytes = tuple(training[name].to_numpy(dtype="<f8", copy=False).tobytes() for name in names)
        reference = native.NumericReference(source_bytes)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_n_training_rows", len(training))
        object.__setattr__(self, "_reference", reference)

    def compare(self, new_rows):
        """Return the existing complete report for this batch only, without fitting.

        Admission, alignment and all-missing query accommodation use Clotho's
        existing validator. Previous queries never expand the reference.
        """
        incoming = validate_prediction_features(new_rows, self._names, ["float64"] * len(self._names))
        query_bytes = tuple(incoming[name].to_numpy(dtype="<f8", copy=False).tobytes() for name in incoming)
        return _report_dimensions(self._reference.counts(query_bytes), self._names,
                                  self._n_training_rows, len(incoming))

    def __reduce_ex__(self, protocol):
        raise TypeError("NumericSupportReference has no persistence format; rebuild from trusted normalized training features.")
