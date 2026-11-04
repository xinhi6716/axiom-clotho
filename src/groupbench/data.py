"""Schema validation, stable fingerprints, and wholly synthetic example data."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from numbers import Integral
from pathlib import Path
import reprlib
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MAX_ROWS = 20_000
MAX_FEATURES = 100


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32 - 100:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 101.")
    return int(seed)


def _validate_numeric_feature_precision(column: pd.Series) -> None:
    """Reject typed values that float64 conversion would silently narrow.

    Check before conversion, including pandas nullable integer columns. Missing
    values remain available for the existing fold-local imputation policy.
    Upstream conversion or parsing may already have discarded precision.
    """
    message = ("Numeric feature values must be representable in float64: integer "
               "values must be exact and floating dtypes wider than float64 "
               "are not supported. Explicitly transform the source values "
               "after reviewing their precision.")
    dtype = column.dtype
    if pd.api.types.is_integer_dtype(dtype):
        for value in column.dropna():
            try:
                converted = float(value)
                if not np.isfinite(converted) or int(converted) != int(value):
                    raise ValueError(message)
            except (ValueError, OverflowError):
                raise ValueError(message) from None
    elif pd.api.types.is_float_dtype(dtype):
        # Nullable and sparse dtypes expose their underlying NumPy dtype in
        # different attributes. Unknown extension dtypes retain the existing
        # conversion policy rather than failing during this precision check.
        numeric_dtype = getattr(dtype, "numpy_dtype", getattr(dtype, "subtype", dtype))
        try:
            width = np.dtype(numeric_dtype).itemsize
        except (TypeError, ValueError):
            width = None
        if width is not None and width > np.dtype(np.float64).itemsize:
            raise ValueError(message)


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy; no imputation or fitted statistics are computed."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named columns.")
    if not 1 <= len(X) <= MAX_ROWS:
        raise ValueError(f"X must contain 1 to {MAX_ROWS:,} rows in this MVP.")
    if not 1 <= X.shape[1] <= MAX_FEATURES:
        raise ValueError(f"X must contain 1 to {MAX_FEATURES} features in this MVP.")
    if not X.columns.is_unique or any(not isinstance(c, str) for c in X.columns):
        raise ValueError("Feature names must be unique strings.")
    X = X.copy().reset_index(drop=True)
    for col in X:
        if is_numeric_dtype(X[col]) and not is_bool_dtype(X[col]):
            if np.iscomplexobj(X[col]):
                raise ValueError(f"Complex values are not supported: {col!r}.")
            _validate_numeric_feature_precision(X[col])
            X[col] = X[col].astype(float)
            if np.isinf(X[col].to_numpy()).any():
                raise ValueError(f"Infinite feature values are not supported: {col!r}.")
        elif isinstance(X[col].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(X[col]):
            raise ValueError(f"Convert datetime feature {col!r} to explicit numeric/categorical features first.")
        else:
            # This is stateless normalization, not a learned vocabulary.
            X[col] = X[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            # Check observed normalized values, not unused category metadata.
            # pandas uniqueness/hashing can merge distinct NUL-containing
            # strings even when the model encoder keeps them separate.
            if any(isinstance(value, str) and "\x00" in value for value in X[col]):
                raise ValueError("Categorical feature values cannot contain NUL (U+0000) characters; explicitly repair the source values before use.")
    return X


def _prediction_column_details(actual, expected):
    """Bounded schema-only guidance; never include row values or infer renames."""
    actual, expected = list(actual), list(expected)
    actual_set, expected_set = set(actual), set(expected)
    missing = [name for name in expected if name not in actual_set]
    unexpected = [name for name in actual if name not in expected_set]
    display = reprlib.Repr()
    display.maxstring = 80

    def describe(label, names):
        # repr escapes terminal controls; cap both the count and each name.
        shown = ", ".join(display.repr(name) for name in names[:5]) or "none"
        if len(names) > 5:
            shown += f" (+{len(names) - 5} more)"
        return f"{label} ({len(names)}): {shown}."

    return (describe("Missing columns", missing) + " "
            + describe("Unexpected columns", unexpected)
            + " Match names exactly (including case and spaces) and supply only the selected features."
            + " No columns were renamed or dropped.")


def validate_prediction_features(X, feature_names, feature_types):
    """Use the same stateless schema/type checks before planning and inference."""
    return _align_normalized_prediction_features(validate_features(X), feature_names, feature_types)


def _align_normalized_prediction_features(X, feature_names, feature_types):
    """Align an internal normalized copy without repeating raw-value validation.

    Only call with the unchanged result of ``validate_features``. This is not
    a public admission boundary: raw dtype/precision, infinity and category
    checks must already have run. Column selection creates a separate frame,
    so all-missing type accommodation never changes the normalized input used
    for historical prediction fingerprints.
    """
    if set(X.columns) != set(feature_names):
        raise ValueError("Prediction columns must exactly match training feature names. "
                         + _prediction_column_details(X.columns, feature_names))
    X = X[list(feature_names)]
    for col, expected_type in zip(X.columns, feature_types):
        if X[col].isna().all():
            X[col] = X[col].astype(expected_type)
    if [str(X[c].dtype) for c in X] != list(feature_types):
        raise ValueError("Prediction feature types differ from training; check numeric/categorical columns.")
    return X


def _validate_target_precision(original):
    """Guard typed numeric values; retain the legacy text/object parse policy."""
    message = ("Integer targets must be exactly representable in float64, and floating target "
               "dtypes wider than float64 are not supported. Rescale or explicitly convert "
               "targets after reviewing their precision.")
    try:
        if original.dtype.kind == "f" and original.dtype.itemsize > 8:
            raise ValueError(message)
        if original.dtype.kind not in "iuO":
            return
        for value in original.flat:
            if isinstance(value, Integral):
                converted = float(value)
                # A comparison against a float can itself round the integer.
                if not np.isfinite(converted) or int(converted) != int(value):
                    raise ValueError(message)
            elif isinstance(value, np.floating) and value.dtype.itemsize > 8:
                raise ValueError(message)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None


def validate_target(y, n_rows: int) -> np.ndarray:
    # np.asarray discards MaskedArray masks and exposes the hidden payload.
    # A masked label is missing even when its stored value happens to be finite.
    if np.ma.isMaskedArray(y) and np.ma.getmaskarray(y).any():
        raise ValueError("The target must have one finite numeric value per row; missing targets are not dropped silently.")
    if np.iscomplexobj(y):
        raise ValueError("Complex target values are not supported.")
    try:
        # Do not promote a mixed Python integer/float sequence before checking
        # each scalar's precision. Existing arrays already have explicit dtypes.
        original = (np.asarray(y) if isinstance(y, (np.ndarray, pd.Series))
                    else np.asarray(y, dtype=object))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The target must be a one-dimensional, finite numeric vector.") from exc
    # NumPy converts datetime/timedelta values to unit-dependent numbers,
    # including NaT to a finite minimum integer. Never infer target units
    # or mistake that missing-value sentinel for an observed outcome.
    if original.dtype.kind in "mM" or (original.dtype.kind == "O" and any(
            isinstance(value, (date, timedelta, np.datetime64, np.timedelta64))
            for value in original.flat)):
        raise ValueError("Datetime and timedelta targets are not supported; convert observed outcomes to explicit numeric units and preserve missing targets.")
    # Object containers can hide NumPy complex scalars from np.iscomplexobj;
    # float(complex_numpy_scalar) can discard their imaginary part with a warning.
    if original.dtype.kind == "O" and any(isinstance(value, (complex, np.complexfloating))
                                          for value in original.flat):
        raise ValueError("Complex target values are not supported.")
    _validate_target_precision(original)
    try:
        a = np.asarray(y, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The target must be a one-dimensional, finite numeric vector.") from exc
    if a.ndim != 1 or len(a) != n_rows or not np.isfinite(a).all():
        raise ValueError("The target must have one finite numeric value per row; missing targets are not dropped silently.")
    return a


def _numeric_group_identifier(value) -> str:
    """Canonicalize exact stored values, never rounded display strings."""
    numerator, denominator = ((int(value), 1) if isinstance(value, Integral)
                              else value.as_integer_ratio())
    if denominator == 1:
        try:
            return str(numerator)
        except ValueError:
            # Python can limit decimal conversion of very large integers.
            return hex(numerator)
    # Preserve ordinary float64 labels while widening float16/32 exactly.
    narrowed = float(value)
    if narrowed.as_integer_ratio() == (numerator, denominator):
        return str(narrowed)
    # Wider floating types can differ below float64 resolution. An exact
    # hexadecimal ratio avoids collisions and decimal integer-string limits
    # for longdouble subnormals, without changing the underlying identity.
    return f"{numerator:#x}/{denominator:#x}"


def validate_groups(groups, n_rows: int) -> np.ndarray:
    # Preserve missing-ID metadata before np.asarray exposes hidden payloads.
    # A masked identifier cannot establish a physical/source grouping boundary.
    if np.ma.isMaskedArray(groups) and np.ma.getmaskarray(groups).any():
        raise ValueError("groups must have one non-missing group identifier per row.")
    a = np.asarray(groups, dtype=object)
    if a.ndim != 1 or len(a) != n_rows or pd.isna(a).any():
        raise ValueError("groups must have one non-missing group identifier per row.")
    if any(not isinstance(v, (str, int, float, np.integer, np.floating)) or isinstance(v, bool) for v in a):
        raise ValueError("Group identifiers must be strings or finite numbers, not containers or booleans.")
    if any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in a):
        raise ValueError("Group identifiers must be finite.")
    # Refuse ambiguous mixed identifiers instead of merging e.g. 1 and '1'.
    kinds = {"text" if isinstance(v, str) else "number" for v in a}
    if len(kinds) > 1:
        raise ValueError("Group identifiers must not mix strings and numbers.")
    if kinds == {"number"}:
        return np.asarray([_numeric_group_identifier(v) for v in a], dtype=object)
    # NULs are not safe identifiers: NumPy string conversion can trim a
    # trailing NUL, and pandas grouping can merge embedded-NUL strings too.
    # Reject before either path instead of silently repairing group identity.
    if any("\x00" in v for v in a):
        raise ValueError("Group identifiers cannot contain NUL (U+0000) characters; explicitly repair the source identifiers before use.")
    if any(not v.strip() for v in a):
        raise ValueError("Group identifiers cannot be blank strings.")
    return a.astype(str).astype(object)


def validate_training_columns(frame: pd.DataFrame, target: str, group: str):
    """Require string column selectors before reading rows or fitting."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique column names.")
    # Non-string pandas labels can pass lookup but fail JSON serialization
    # only after fitting. Do not coerce labels: that can change their identity.
    if not isinstance(target, str) or not isinstance(group, str):
        raise ValueError("target and group must be string column names; explicitly rename non-string columns before use.")
    if target == group or target not in frame or group not in frame:
        raise ValueError("target and group must identify two distinct existing columns.")


def prepare_data(frame: pd.DataFrame, target: str, group: str, features=None):
    validate_training_columns(frame, target, group)
    if isinstance(features, (str, bytes)):
        raise ValueError("features must be a sequence of column names, not a bare string or bytes. "
                         "For one feature, pass a one-item list such as ['feature_name'].")
    names = list(features) if features is not None else [c for c in frame if c not in (target, group)]
    if target in names or group in names:
        raise ValueError("The target and grouping column cannot be input features.")
    if len(names) != len(set(names)) or any(c not in frame for c in names):
        raise ValueError("features must be unique existing column names.")
    # Frame-based workflows emit feature names in saved bundle metadata. Keep
    # admission compatible with that reader before any fitting or writing;
    # standalone diagnostics and in-memory estimators have no bundle contract.
    if any(isinstance(c, str) and (not c or "\x00" in c) for c in names):
        raise ValueError("Selected feature names must be nonempty strings without NUL (U+0000) characters; explicitly rename the source columns before use.")
    X = validate_features(frame[names])
    y = validate_target(frame[target], len(X))
    groups = validate_groups(frame[group], len(X))
    return X, y, groups


def data_fingerprint(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> str:
    """Hash normalized values, their row order, column names, and feature types."""
    schema = json.dumps([(str(c), str(X[c].dtype)) for c in X], ensure_ascii=False)
    payload = X.copy()
    # Hash each component separately to avoid reserved-column collisions.
    h = hashlib.sha256(schema.encode("utf-8"))
    for part in (payload, pd.Series(y), pd.Series(groups)):
        h.update(pd.util.hash_pandas_object(part, index=False).values.tobytes())
    return h.hexdigest()


def validate_source_sha256(value, name="source_sha256"):
    """Validate optional caller-declared digest syntax, not source authenticity."""
    if value is not None and (not isinstance(value, str)
                              or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None):
        raise ValueError(f"{name} must be None or exactly 64 hexadecimal SHA-256 characters.")
    return value


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_materials_fixture(n_groups: int = 24, rows_per_group: int = 12, random_state: int = 7) -> pd.DataFrame:
    """Generate fictitious process/property data, not a physical materials model.

    Groups represent manufacturing batches. Shared unobserved batch effects make
    observations within a batch dependent. No real measurements are embedded.
    """
    seed = validate_seed(random_state)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 2 for v in (n_groups, rows_per_group)):
        raise ValueError("n_groups and rows_per_group must be integers of at least two.")
    if n_groups * rows_per_group > MAX_ROWS:
        raise ValueError(f"The fixture must not exceed {MAX_ROWS:,} rows.")
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    batch = np.repeat(np.arange(n_groups), rows_per_group)
    temperature = rng.uniform(400, 900, n)
    pressure = rng.uniform(1, 5, n)
    duration = rng.uniform(10, 90, n)
    material = rng.choice(["alloy-A", "alloy-B", "alloy-C"], n)
    batch_effect = rng.normal(0, 8, n_groups)[batch]
    effect = np.select([material == "alloy-A", material == "alloy-B"], [4, -5], default=9)
    target = (70 + 0.11 * temperature + 6 * pressure + 0.15 * duration
              + effect + 3 * np.sin(temperature / 70) + batch_effect + rng.normal(0, 2, n))
    frame = pd.DataFrame({
        "batch": [f"batch-{g:03d}" for g in batch],
        "temperature_c": temperature,
        "pressure_mpa": pressure,
        "duration_min": duration,
        "material_family": material,
        "response": target,
    })
    for col in ("temperature_c", "duration_min", "material_family"):
        frame.loc[rng.random(n) < 0.04, col] = np.nan
    return frame


def duplicate_diagnostics(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict:
    """Count identical normalized feature/target rows without changing the data.

    Identical records across groups can defeat a nominally disjoint split.
    This diagnostic is not a proof of leakage and never drops records.
    """
    values = X.copy()
    values.columns = list(range(X.shape[1]))
    values[X.shape[1]] = y
    duplicated = values.duplicated(keep=False).to_numpy()
    # pandas' groupby compares actual values; missing feature values form a
    # shared category here, rather than making an otherwise duplicate vanish.
    codes = values.groupby(list(values.columns), dropna=False, sort=False).ngroup().to_numpy()
    pairs = pd.DataFrame({"signature": codes, "group": groups})
    group_counts = pairs.groupby("signature")["group"].nunique()
    cross_signatures = group_counts[group_counts > 1].index
    across = pairs["signature"].isin(cross_signatures).to_numpy()
    return {
        "identical_feature_target_rows": int(duplicated.sum()),
        "identical_feature_target_rows_across_groups": int(across.sum()),
        "cross_group_duplicate_patterns": len(cross_signatures),
        "action": "diagnostic_only_no_rows_removed",
    }
