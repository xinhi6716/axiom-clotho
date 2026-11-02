"""Explicit standalone constants; no registry or default-model changes."""

from __future__ import annotations

from collections import Counter
from math import lcm

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .data import MAX_FEATURES, MAX_ROWS, validate_groups, validate_target


_MAX_MASK_VISITS = MAX_ROWS * (MAX_FEATURES + 2) + 1


def _reject_masked(value, name):
    """Inspect retained masks before numeric conversion can erase them.

    Unlike binary-only inputs, numeric-convertible object arrays are accepted
    here, so inspect their cells as well as built-in list/tuple containers.
    Incremental DFS avoids width-sized work stacks. Cycles, excessive depth
    and excessive container work reject; arbitrary iterables are not consumed.
    """
    pending = [(iter((value,)), 0, None)]
    active, finished = set(), set()
    visits = 0
    while pending:
        iterator, depth, identity = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            if identity is not None:
                active.remove(identity)
                finished.add(identity)
            continue
        visits += 1
        if visits > _MAX_MASK_VISITS:
            raise ValueError(f"{name} nested-container inspection exceeds the supported bound.")
        if np.ma.isMaskedArray(item):
            # getmask preserves nomask without allocating a full all-False mask.
            mask = np.asarray(np.ma.getmask(item))
            if mask.dtype.kind != "b":
                raise ValueError(f"{name} requires unstructured numeric inputs.")
            if mask.any():
                raise ValueError(f"{name} cannot contain masked values.")
        if isinstance(item, list):
            children = list.__iter__(item)
        elif isinstance(item, tuple):
            children = tuple.__iter__(item)
        elif isinstance(item, np.ndarray) and item.dtype.kind == "O":
            children = iter(np.ndarray.flat.__get__(item))
        elif sparse.issparse(item) and item.format in ("csr", "csc"):
            # Sparse shape/type validation can discard masks on numeric and
            # index buffers. Inspect their retained metadata before conversion.
            children = iter((item.data, item.indices, item.indptr))
        elif isinstance(item, (pd.DataFrame, pd.Series, pd.Index)):
            cells = item.to_numpy(copy=False)
            if cells.dtype.kind != "O":
                continue
            children = iter(cells.flat)
        else:
            continue
        identity = id(item)
        if identity in active:
            raise ValueError(f"{name} cannot contain cyclic containers.")
        if identity in finished:
            continue
        if depth >= 64:
            raise ValueError(f"{name} container nesting exceeds the supported depth.")
        active.add(identity)
        pending.append((children, depth + 1, identity))


def _matrix_and_names(X):
    if sparse.issparse(X) and X.format not in ("csr", "csc"):
        # check_array otherwise converts other sparse formats implicitly. Keep
        # the documented CSR/CSC boundary explicit, before any mask can be lost.
        raise ValueError("Only CSR or CSC sparse inputs are supported; other formats are not converted implicitly.")
    _reject_masked(X, "X")
    columns = getattr(X, "columns", None)
    names = None
    if columns is not None:
        names = list(columns)
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("Named feature columns must be distinct strings.")
    # Default finite-value checks work across the declared sklearn>=1.4 range;
    # do not use the version-dependent force_all_finite/ensure_all_finite name.
    matrix = check_array(X, accept_sparse=("csr", "csc"), dtype="numeric")
    if len(matrix.shape) != 2 or not 1 <= matrix.shape[0] <= MAX_ROWS:
        raise ValueError(f"X must contain 1 to {MAX_ROWS:,} rows.")
    return matrix, names


def _lower_group_median(target, groups):
    """Exact cumulative weights; no floating-point median/tie arithmetic."""
    counts = Counter(groups)
    denominator = 1
    for size in sorted(set(counts.values())):
        denominator = lcm(denominator, size)
    total_weight = len(counts) * denominator
    cumulative = 0
    for index in np.argsort(target, kind="stable"):
        cumulative += denominator // counts[groups[index]]
        if 2 * cumulative >= total_weight:
            value = float(target[index])
            # Signed zeros represent the same response; remove order-dependent
            # sign bits without averaging targets or introducing a new value.
            return (0.0 if value == 0 else value), len(counts)
    raise RuntimeError("Internal error: positive training weights did not reach their total.")


class GroupBalancedMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict an equal-training-group lower weighted median constant.

    ``fit`` requires the groups of the rows actually passed to that fit. Each
    row receives internal weight ``1 / training_group_size``. This is a median
    of all weighted targets, not a median of group means or group medians.
    Exactly half cumulative weight selects the lower observed target.

    X must be a finite real numeric 2D dense/CSR/CSC matrix (usually the output
    of fold-local preprocessing). Its values do not affect the constant. Named
    training columns must be distinct strings; prediction preserves their exact
    names/order, or just feature count for an unnamed fit. At most MAX_ROWS rows
    are accepted per call. One training group is valid for this estimator;
    grouped cross-validation has its own stricter sufficiency requirements.

    Any non-None ``sample_weight`` is rejected: caller weights would change or
    double-weight the declared objective. Prediction takes no groups. For a
    legacy sklearn Pipeline named ``model``, pass ``model__groups=training_groups``
    to fit. Automatic metadata-routing/CV dispatch is not promised.

    Also available as the explicit optional GroupBench family group_median.
    It does not change the default family list.
    ``score`` is the inherited row R-squared score, NOT group MAE.
    A failed refit preserves the previous complete fitted state.
    """

    def fit(self, X, y, *, groups=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported; weights come only from current training-group sizes.")
        if groups is None:
            raise ValueError("groups is required for the current training rows.")
        matrix, names = _matrix_and_names(X)
        _reject_masked(y, "y")
        _reject_masked(groups, "groups")
        target = validate_target(y, matrix.shape[0])
        normalized_groups = validate_groups(groups, matrix.shape[0])
        constant, n_groups = _lower_group_median(target, normalized_groups)
        # Do not retain targets, IDs or row weights. Publish only after every
        # validation and the exact calculation succeeded.
        state = {
            "constant_": constant, "n_features_in_": int(matrix.shape[1]),
            "n_samples_seen_": int(matrix.shape[0]), "n_groups_seen_": n_groups,
            "weight_rule_": "equal_training_group_then_equal_row",
            "tie_break_": "lower_observed_target",
        }
        if names is not None:
            state["feature_names_in_"] = np.asarray(names, dtype=object)
        self.__dict__.update(state)
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict(self, X):
        check_is_fitted(self, "constant_")
        matrix, names = _matrix_and_names(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count must match training.")
        if hasattr(self, "feature_names_in_"):
            if names != self.feature_names_in_.tolist():
                raise ValueError("Prediction feature names and order must exactly match training.")
        return np.full(matrix.shape[0], self.constant_, dtype=float)
