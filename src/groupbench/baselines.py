"""Explicit optional regression primitives; no default-model changes."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import date, timedelta
from itertools import repeat
from math import fsum, isclose, lcm, sqrt
from numbers import Real
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_is_fitted
from threadpoolctl import threadpool_limits

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


# A transformed original feature may create up to 64 one-hot columns. This is
# the same width bound as the optional kernel primitive, not a raw-data limit.
MAX_RIDGE_FEATURES = MAX_FEATURES * 64
_WEIGHT_RTOL = 64 * np.finfo(np.float64).eps


@contextmanager
def _ridge_numerical_stage(stage):
    """Surface invalid intermediates, even when a solver could hide them."""
    try:
        with warnings.catch_warnings(), np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"):
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", LinAlgWarning)
            warnings.simplefilter("error", ConvergenceWarning)
            yield
    except (RuntimeWarning, LinAlgWarning, ConvergenceWarning, FloatingPointError,
            OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Group-balanced Ridge numerical failure during {stage}; rescale feature and target units. "
            "No candidate is skipped, retried, or replaced."
        ) from exc


def _ridge_real_input(value):
    """Reject temporal/complex features before float conversion loses type."""
    pending = [(iter((value,)), 0)]
    finished = {}
    visits = 0
    while pending:
        iterator, depth = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        visits += 1
        if visits > _MAX_MASK_VISITS:
            raise ValueError("X nested-container inspection exceeds the supported bound.")
        if isinstance(item, (date, timedelta, np.datetime64, np.timedelta64,
                             complex, np.complexfloating)):
            raise ValueError("X must contain real numeric values, not temporal or complex values.")
        if getattr(getattr(item, "dtype", None), "kind", None) in ("M", "m", "c"):
            raise ValueError("X must contain real numeric values, not temporal or complex values.")
        if isinstance(item, (list, tuple)):
            children = list.__iter__(item) if isinstance(item, list) else tuple.__iter__(item)
        elif isinstance(item, np.ndarray) and item.dtype.kind == "O":
            children = iter(np.ndarray.flat.__get__(item))
        elif isinstance(item, (pd.DataFrame, pd.Series, pd.Index)):
            cells = item.to_numpy(copy=False)
            children = iter((cells,))
        elif sparse.issparse(item):
            children = iter((item.data,))
        else:
            continue
        # The mask pass has already rejected cycles. This separate pass also
        # remains bounded and does not consume arbitrary user iterables.
        if id(item) in finished:
            continue
        if depth >= 64:
            raise ValueError("X container nesting exceeds the supported depth.")
        finished[id(item)] = item
        pending.append((children, depth + 1))


def _ridge_sparse_copy(X):
    """Validate raw compressed structure before any compiled sparse operation."""
    rows, width = X.shape
    data, indices, indptr = (np.asarray(getattr(X, key)) for key in ("data", "indices", "indptr"))
    major, minor = (rows, width) if X.format == "csr" else (width, rows)
    if (any(buffer.ndim != 1 for buffer in (data, indices, indptr))
            or indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu"
            or len(data) != len(indices) or len(indptr) != major + 1
            or indptr[0] != 0 or indptr[-1] != len(data)
            or np.any(indptr[1:] < indptr[:-1])
            or np.any(indptr < 0) or np.any(indptr > len(data))
            or np.any(indices < 0) or np.any(indices >= minor)):
        raise ValueError("X has invalid CSR/CSC sparse structure.")
    # Never call check_format on the caller: SciPy may rewrite its buffers.
    factory = sparse.csr_matrix if X.format == "csr" else sparse.csc_matrix
    result = factory((np.array(data, dtype=np.float64, copy=True),
                      np.array(indices, dtype=np.int64, copy=True),
                      np.array(indptr, dtype=np.int64, copy=True)), shape=X.shape)
    result.check_format(full_check=True)
    result.sum_duplicates()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError("Group-balanced Ridge has non-finite sparse features after canonicalization; rescale feature units.")
    return result


def _ridge_matrix_and_names(X):
    if sparse.issparse(X) and X.format not in ("csr", "csc"):
        raise ValueError("Only CSR or CSC sparse inputs are supported; other formats are not converted implicitly.")
    _reject_masked(X, "X")
    _ridge_real_input(X)
    shape = getattr(X, "shape", None)
    if shape is not None and (len(shape) != 2 or not 1 <= shape[0] <= MAX_ROWS
                              or not 1 <= shape[1] <= MAX_RIDGE_FEATURES):
        raise ValueError(f"X must contain 1 to {MAX_ROWS:,} rows and 1 to {MAX_RIDGE_FEATURES:,} transformed features.")
    names = None
    if getattr(X, "columns", None) is not None:
        names = list(X.columns)
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("Named feature columns must be distinct strings.")
    with _ridge_numerical_stage("feature validation and sparse canonicalization"):
        source = _ridge_sparse_copy(X) if sparse.issparse(X) else X
        matrix = check_array(source, accept_sparse=("csr", "csc"), dtype=np.float64, copy=True)
    if not 1 <= matrix.shape[0] <= MAX_ROWS or not 1 <= matrix.shape[1] <= MAX_RIDGE_FEATURES:
        raise ValueError(f"X must contain 1 to {MAX_ROWS:,} rows and 1 to {MAX_RIDGE_FEATURES:,} transformed features.")
    return matrix, names


def _group_balanced_weights(groups):
    """Current-slice weights; float64 checks use 64 epsilon relative tolerance.

    The weights have exact integer count denominators, each at most n**2.
    fsum checks total n and every group total n/G; no clipping or repair is
    performed. Equal group sizes produce literal float64 ones.
    """
    rows = len(groups)
    counts = Counter(groups)
    n_groups = len(counts)
    sizes = list(counts.values())
    if (not 1 <= n_groups <= rows <= MAX_ROWS
            or any(type(size) is not int or size <= 0 for size in sizes)
            or sum(sizes) != rows):
        raise ValueError("Invalid current training-group counts.")
    by_group = {group: rows / (n_groups * size) for group, size in counts.items()}
    weights = (np.ones(rows, dtype=np.float64) if min(sizes) == max(sizes)
               else np.fromiter((by_group[group] for group in groups), dtype=np.float64, count=rows))
    if (not np.isfinite(weights).all() or np.any(weights <= 0)
            or np.any(weights < 1.0 / rows) or np.any(weights > rows)):
        raise ValueError("Invalid internally derived training weights.")
    total = fsum(weights)
    if (not isclose(total, rows, rel_tol=_WEIGHT_RTOL, abs_tol=0.0)
            or any(not isclose(fsum(repeat(by_group[group], count)), rows / n_groups,
                               rel_tol=_WEIGHT_RTOL, abs_tol=0.0)
                   for group, count in counts.items())):
        raise ValueError("Internal training-weight totals failed validation.")
    details = {"training_groups": n_groups,
               "min_training_group_size": min(sizes), "max_training_group_size": max(sizes),
               "min_row_weight": float(np.min(weights)), "max_row_weight": float(np.max(weights)),
               "sum_row_weights": total}
    return weights, details


def _ridge_magnitude(value):
    values = value.data if sparse.issparse(value) else np.asarray(value)
    return max(float(np.max(values, initial=0.0)), -float(np.min(values, initial=0.0)))


def _ridge_range_preflight(matrix, target=None, alpha=None):
    """Conservative finite-range envelope, not a convergence certificate.

    An eighth-root bound with dimension headroom keeps weighted sums, row
    scaling, squared norms and repeated LSQR norm arithmetic away from float64
    overflow. For fitting, also reserve room for the ridge solution bound
    ||beta|| <= sqrt(n)*max(abs(y))/sqrt(alpha). Very small alpha may therefore
    require smaller target units. Alpha itself needs headroom for damped norm
    additions. Tail underflow is allowed; overflow/invalid arithmetic is not.
    """
    rows, width = matrix.shape
    limit = (np.finfo(np.float64).max / (1024.0 * max(rows, width) ** 2)) ** 0.125
    if _ridge_magnitude(matrix) > limit:
        raise ValueError("Group-balanced Ridge features exceed the safe numerical range; rescale feature units.")
    if target is not None:
        if alpha > np.finfo(np.float64).max / 1024.0:
            raise ValueError("Group-balanced Ridge alpha exceeds the safe numerical range; rescale feature and target units and alpha.")
        target_limit = limit * min(1.0, sqrt(alpha) / sqrt(rows))
        if _ridge_magnitude(target) > target_limit:
            raise ValueError("Group-balanced Ridge targets exceed the safe numerical range for alpha; rescale feature and target units.")


def _checked_balanced_ridge(model, width):
    coef, intercept = np.asarray(model.coef_), np.asarray(model.intercept_)
    if coef.shape != (width,) or intercept.shape != () or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("Group-balanced Ridge has invalid or non-finite fitted coefficients/intercept; rescale feature and target units.")


class GroupBalancedRidgeRegressor(RegressorMixin, BaseEstimator):
    """Equal-training-group weighted squared-loss Ridge, not group MAE.

    Each row in this fit has weight n/(G*n_g), with total n and group totals
    n/G. Only the final regression is weighted: preceding imputation, scaling
    and categorical preprocessing remain row-weighted. Fit requires groups;
    prediction uses features only. Caller sample_weight is always rejected.

    Finite real dense/CSR/CSC input is limited to 20,000 rows and 6,400 columns.
    Named input preserves exact feature names/order. A conservative numeric
    envelope can require rescaling otherwise finite values. Failed fits retain
    the previous complete state. Raw targets, groups and weights are not kept.
    Inherited score is row R-squared; finite solver output is not a convergence
    certificate or a promise of improved group MAE.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, *, groups=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported; weights come only from current training-group sizes.")
        if groups is None:
            raise ValueError("groups is required for the current training rows.")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise ValueError("alpha must be a positive finite real number.")
        try:
            alpha = float(self.alpha)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("alpha must be a positive finite real number.") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite real number.")
        matrix, names = _ridge_matrix_and_names(X)
        rows, width = matrix.shape
        _reject_masked(y, "y")
        _reject_masked(groups, "groups")
        target = validate_target(y, rows).copy()
        normalized_groups = validate_groups(groups, rows)
        weights, weight_details = _group_balanced_weights(normalized_groups)
        _ridge_range_preflight(matrix, target, alpha)
        model = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-4,
                      max_iter=None, copy_X=True, positive=False)
        with threadpool_limits(limits=1), _ridge_numerical_stage("fitting"):
            model.fit(matrix, target, sample_weight=weights)
        _checked_balanced_ridge(model, width)
        details = {
            "implementation": "group-balanced-ridge-v1", "training_rows": rows,
            "transformed_features": width, "alpha": alpha, "solver": "lsqr", "tol": 1e-4,
            "max_iter": None, "fit_intercept": True, "copy_X": True, "positive": False,
            "threads": 1, "fitting_loss": "weighted_squared_error",
            "weight_rule": "equal_training_group_then_equal_row", "normalization": "sum_training_rows",
            "preprocessing_weighting": "unweighted_training_rows", **weight_details,
        }
        state = {"ridge_": model, "coef_": model.coef_, "intercept_": float(model.intercept_),
                 "n_features_in_": width, "n_samples_seen_": rows,
                 "n_groups_seen_": weight_details["training_groups"], "fit_details_": details}
        if names is not None:
            state["feature_names_in_"] = np.asarray(names, dtype=object)
        self.__dict__.update(state)
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        return self

    def predict(self, X):
        check_is_fitted(self, ("ridge_", "fit_details_"))
        matrix, names = _ridge_matrix_and_names(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count must match training.")
        if hasattr(self, "feature_names_in_") and names != self.feature_names_in_.tolist():
            raise ValueError("Prediction feature names and order must exactly match training.")
        _checked_balanced_ridge(self.ridge_, self.n_features_in_)
        _ridge_range_preflight(matrix)
        # A separate checked envelope protects the prediction dot product even
        # for externally modified, finite learned coefficients.
        maximum = np.finfo(np.float64).max / 4.0
        coefficient = _ridge_magnitude(self.ridge_.coef_)
        if (abs(float(self.ridge_.intercept_)) > maximum or coefficient > 1.0 and
                _ridge_magnitude(matrix) > maximum / self.n_features_in_ / coefficient):
            raise ValueError("Group-balanced Ridge prediction exceeds the safe numerical range; rescale feature and target units.")
        with threadpool_limits(limits=1), _ridge_numerical_stage("prediction"):
            prediction = self.ridge_.predict(matrix)
        if np.shape(prediction) != (matrix.shape[0],) or not np.isfinite(prediction).all():
            raise ValueError("Group-balanced Ridge produced invalid or non-finite predictions; rescale feature and target units.")
        return prediction
