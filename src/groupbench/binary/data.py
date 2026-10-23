"""Explicit binary labels, using existing stateless feature/group contracts."""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
import pandas as pd

from ..data import MAX_ROWS, prepare_data, validate_seed


def label_contract():
    return {"encoding": "binary01", "classes": [0, 1], "positive_class": 1,
            "threshold": 0.5, "comparison": ">=", "semantics": "user_defined"}


def validate_binary_target(y, n_rows, *, require_both=True):
    # Preserve original sequence element types: np.asarray([0, True]) would
    # otherwise erase the Boolean before any value check can reject it.
    if isinstance(y, (pd.Series, np.ndarray)):
        if (not pd.api.types.is_numeric_dtype(y.dtype)
                or pd.api.types.is_bool_dtype(y.dtype)
                or pd.api.types.is_complex_dtype(y.dtype)):
            raise ValueError("Binary targets must be explicit numeric 0/1, not Boolean, string or object labels.")
    try:
        original = np.asarray(y, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("Binary targets must be a one-dimensional numeric 0/1 vector.") from exc
    if original.ndim != 1 or len(original) != n_rows or not n_rows:
        raise ValueError("Binary targets must contain one explicit numeric 0/1 value per row.")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in original):
        raise ValueError("Binary targets must be explicit numeric 0/1, not Boolean, string, complex or missing labels.")
    # Compare before narrowing: longdouble fractional labels must not round
    # into 0/1, and huge invalid Python integers must not overflow conversion.
    if any(v != 0 and v != 1 for v in original):
        raise ValueError("Binary targets must contain only finite numeric 0/1; rows are never dropped or remapped.")
    result = np.asarray(original, dtype=np.int64)
    if require_both:
        require_classes(result, "Full training data")
    return result


def require_classes(y, context):
    for label in (0, 1):
        if not np.any(y == label):
            raise ValueError(f"{context} lacks class {label}; every training partition must contain both classes. No automatic resplitting is performed.")


def class_counts(y):
    return {"n_true_0": int(np.sum(y == 0)), "n_true_1": int(np.sum(y == 1))}


def prepare_binary_data(frame, target, group, features=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique column names.")
    if target == group or target not in frame or group not in frame:
        raise ValueError("target and group must identify two distinct existing columns.")
    y = validate_binary_target(frame[target], len(frame))
    X, _, groups = prepare_data(frame, target, group, features)
    return X, y, groups


def validate_splits(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 2 <= value <= 10:
        raise ValueError(f"{name} must be an integer between 2 and 10 for binary classification.")
    return int(value)


def make_binary_fixture(n_groups=24, rows_per_group=12, random_state=7):
    """Original invented data, not semiconductor measurements or physics.

    Label 1 means fictitious fail and label 0 fictitious pass. A fixed draw is
    returned, never resampled to make a split feasible or a model look good.
    """
    seed = validate_seed(random_state)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 2
           for v in (n_groups, rows_per_group)):
        raise ValueError("n_groups and rows_per_group must be integers of at least two.")
    if n_groups * rows_per_group > MAX_ROWS:
        raise ValueError(f"The fixture must not exceed {MAX_ROWS:,} rows.")
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    ids = np.repeat(np.arange(n_groups), rows_per_group)
    temperature, pressure = rng.normal(size=(2, n))
    tool = rng.choice(["invented-A", "invented-B", "invented-C"], size=n)
    latent = rng.normal(0, 0.6, n_groups)[ids]
    logits = -0.6 + 0.8 * temperature - 0.5 * pressure + 0.4 * (tool == "invented-B") + latent
    probability = 1 / (1 + np.exp(-logits))
    frame = pd.DataFrame({"lot": [f"lot-{i:03d}" for i in ids],
                          "synthetic_temperature": temperature,
                          "synthetic_pressure": pressure,
                          "synthetic_tool_family": tool,
                          "fail": rng.binomial(1, probability)})
    for col in ("synthetic_temperature", "synthetic_tool_family"):
        frame.loc[rng.random(n) < 0.04, col] = np.nan
    return frame
