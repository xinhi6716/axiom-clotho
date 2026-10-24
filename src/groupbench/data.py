"""Schema validation, stable fingerprints, and wholly synthetic example data."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MAX_ROWS = 20_000
MAX_FEATURES = 100


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32 - 100:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 101.")
    return int(seed)


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
            X[col] = X[col].astype(float)
            if np.isinf(X[col].to_numpy()).any():
                raise ValueError(f"Infinite feature values are not supported: {col!r}.")
        elif isinstance(X[col].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(X[col]):
            raise ValueError(f"Convert datetime feature {col!r} to explicit numeric/categorical features first.")
        else:
            # This is stateless normalization, not a learned vocabulary.
            X[col] = X[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X


def validate_prediction_features(X, feature_names, feature_types):
    """Use the same stateless schema/type checks before planning and inference."""
    X = validate_features(X)
    if set(X.columns) != set(feature_names):
        raise ValueError("Prediction columns must exactly match training feature names.")
    X = X[list(feature_names)]
    for col, expected_type in zip(X.columns, feature_types):
        if X[col].isna().all():
            X[col] = X[col].astype(expected_type)
    if [str(X[c].dtype) for c in X] != list(feature_types):
        raise ValueError("Prediction feature types differ from training; check numeric/categorical columns.")
    return X


def validate_target(y, n_rows: int) -> np.ndarray:
    if np.iscomplexobj(y):
        raise ValueError("Complex target values are not supported.")
    try:
        a = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
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
    if any(not v.strip() for v in a):
        raise ValueError("Group identifiers cannot be blank strings.")
    return a.astype(str).astype(object)


def prepare_data(frame: pd.DataFrame, target: str, group: str, features=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique column names.")
    if target == group or target not in frame or group not in frame:
        raise ValueError("target and group must identify two distinct existing columns.")
    names = list(features) if features is not None else [c for c in frame if c not in (target, group)]
    if target in names or group in names:
        raise ValueError("The target and grouping column cannot be input features.")
    if len(names) != len(set(names)) or any(c not in frame for c in names):
        raise ValueError("features must be unique existing column names.")
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
