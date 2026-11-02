"""Bounded local kernel regression primitives, not an enabled model family.

Inputs are finite, already training-partition-preprocessed features. This module
does not add a family to GroupBench's registry or change its default models.
"""

from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
from numbers import Integral, Real
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_is_fitted
from threadpoolctl import threadpool_limits

from .data import MAX_FEATURES, MAX_ROWS, validate_seed, validate_target


MAX_RANK = 64
MAX_TRANSFORMED_FEATURES = MAX_FEATURES * 64
MAX_FEATURE_MAP_BYTES = 16 * 1024 * 1024
_FLOAT_BYTES = np.dtype(np.float64).itemsize


def _reject_unsafe_containers(values, name):
    """Check masks/temporal values before array coercion can erase their meaning.

    Walk only supported concrete containers, not arbitrary iterables. Iterator
    frames avoid materializing a pending entry for every array cell. Retain
    completed containers so transient pandas views cannot reuse visited IDs.
    """
    stack = [(iter((values,)), None)]
    active, finished = {}, {}
    while stack:
        items, parent = stack[-1]
        try:
            value = next(items)
        except StopIteration:
            stack.pop()
            if parent is not None:
                finished[id(parent)] = active.pop(id(parent))
            continue
        if isinstance(value, (date, timedelta, np.datetime64, np.timedelta64)):
            raise ValueError(f"{name} cannot contain datetime or timedelta values; use explicit numeric units and preserve missing values.")
        if np.ma.isMaskedArray(value):
            mask = np.asarray(np.ma.getmask(value))
            if mask.dtype.kind != "b":
                raise ValueError(f"{name} require unstructured numeric arrays.")
            if mask.any():
                raise ValueError(f"{name} cannot contain masked entries; no values are filled or rows dropped.")
        dtype = getattr(value, "dtype", None)
        if getattr(dtype, "kind", None) in ("m", "M"):
            raise ValueError(f"{name} cannot contain datetime or timedelta values; use explicit numeric units and preserve missing values.")
        if isinstance(value, (list, tuple)):
            children = list.__iter__(value) if isinstance(value, list) else tuple.__iter__(value)
        elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
            children = value.data.flat if np.ma.isMaskedArray(value) else value.flat
        elif isinstance(value, (pd.Series, pd.Index)) and value.dtype.kind == "O":
            children = value.to_numpy(copy=False).flat
        elif isinstance(value, pd.DataFrame):
            children = (column for _, column in pd.DataFrame.items(value))
        elif sparse.issparse(value):
            if value.format != "csr":
                raise ValueError(f"{name} sparse input must use CSR format; no format conversion is inferred.")
            children = iter((value.data, value.indices, value.indptr))
        else:
            continue
        identity = id(value)
        if identity in active:
            raise ValueError(f"{name} cannot contain cyclic containers.")
        if identity in finished:
            continue
        if len(stack) >= 65:
            raise ValueError(f"{name} container nesting exceeds supported depth.")
        active[identity] = value
        stack.append((iter(children), value))


def _map_bytes(rows, components):
    """Check one dense float64 map before allocation, not total process RSS."""
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
           for value in (rows, components)):
        raise ValueError("Kernel map dimensions must be positive integers.")
    size = int(rows) * int(components) * _FLOAT_BYTES
    if size > MAX_FEATURE_MAP_BYTES:
        raise ValueError("Kernel feature map exceeds the 16 MiB per-array limit; no rank reduction or row sampling is attempted.")
    return size


@contextmanager
def _numerical_stage(stage, *, allow_tail_underflow=False):
    """Reject numerical corruption without suppressing unrelated warnings."""
    try:
        with warnings.catch_warnings(), np.errstate(
                over="raise", invalid="raise", divide="raise",
                under="ignore" if allow_tail_underflow else "raise"):
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", LinAlgWarning)
            yield
    except (RuntimeWarning, LinAlgWarning, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Kernel regression numerical failure during {stage}; check feature and target units. "
            "No candidate is skipped, retried, or replaced."
        ) from exc


def _features(X):
    _reject_unsafe_containers(X, "Kernel features")
    with _numerical_stage("feature validation"):
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
    if len(X.shape) != 2 or X.shape[0] > MAX_ROWS or X.shape[1] > MAX_TRANSFORMED_FEATURES:
        raise ValueError(f"Kernel features are limited to {MAX_ROWS} rows and {MAX_TRANSFORMED_FEATURES} transformed columns.")
    if sparse.issparse(X):
        # Equivalent CSR storage must not bypass the coordinate bound through
        # repeated indices. Never mutate the caller's sparse representation.
        X = X.copy()
        with _numerical_stage("sparse canonicalization"):
            X.check_format(full_check=True)
            if not X.has_canonical_format:
                X.sum_duplicates()
                X.sort_indices()
        _finite(X, "canonical sparse features")
    # Compiled squared-norm routines can overflow without a NumPy warning,
    # after which exp(-infinity) conceals the failure as a finite zero kernel.
    # Leave ample headroom for all dot-product/distance intermediates. This
    # check reduces values directly and allocates no full squared matrix.
    bound = np.sqrt(np.finfo(np.float64).max / (16.0 * X.shape[1]))
    values = X.data if sparse.issparse(X) else X
    if np.max(values, initial=0.0) > bound or np.min(values, initial=0.0) < -bound:
        raise ValueError("Kernel regression numerical failure: features exceed the safe RBF distance range; rescale feature units.")
    return X


def _finite(value, name, shape=None):
    data = value.data if sparse.issparse(value) else np.asarray(value)
    actual_shape = value.shape if sparse.issparse(value) else data.shape
    if (shape is not None and actual_shape != shape) or not np.isfinite(data).all():
        raise ValueError(f"Kernel regression has invalid or non-finite {name}; check feature and target units.")


def _checked_map(mapper, rows, width, components):
    if (mapper.kernel != "rbf" or mapper.gamma != 1.0 / width
            or mapper.n_components != components or mapper.n_jobs != 1):
        raise ValueError("Kernel regression has invalid fitted map configuration.")
    _finite(mapper.components_, "fitted basis", (components, width))
    _finite(mapper.normalization_, "fitted normalization", (components, components))
    indices = np.asarray(mapper.component_indices_)
    if (indices.shape != (components,) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= rows)
            or len(np.unique(indices)) != components):
        raise ValueError("Kernel regression has invalid fitted basis indices.")


def _checked_embedding(values, rows, components):
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64):
        raise ValueError("Kernel regression requires a dense float64 feature map.")
    _finite(values, "feature map", (rows, components))
    return values


def _checked_ridge(model, components):
    _finite(model.coef_, "fitted ridge coefficients", (components,))
    intercept = np.asarray(model.intercept_)
    _finite(intercept, "fitted ridge intercept", ())


class NystroemRidgeRegressor(RegressorMixin, BaseEstimator):
    """Explicit RBF approximation plus Ridge on finite preprocessed features.

    The basis uses only supplied training rows. Gamma is 1 / transformed width,
    and the component count is min(rank_cap, training rows), not algebraic rank.
    Sampling and fitting weight rows equally; this is not group-balanced fitting.
    No groups or targets are needed for prediction. No executable state is saved.
    Failed fitting preserves any previous complete fitted state.
    """

    def __init__(self, alpha=1.0, rank_cap=64, random_state=0):
        self.alpha = alpha
        self.rank_cap = rank_cap
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise ValueError("alpha must be a positive finite real number.")
        try:
            alpha = float(self.alpha)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("alpha must be a positive finite real number.") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite real number.")
        if (isinstance(self.rank_cap, bool) or not isinstance(self.rank_cap, Integral)
                or not 1 <= self.rank_cap <= MAX_RANK):
            raise ValueError(f"rank_cap must be an integer from 1 to {MAX_RANK}.")
        seed = validate_seed(self.random_state)
        X = _features(X)
        rows, width = X.shape
        _reject_unsafe_containers(y, "Kernel targets")
        # Reuse the shared numeric/precision policy before any float coercion.
        # The local walker only prevents masks/temporal containers being erased.
        y = validate_target(y, rows)
        components = min(int(self.rank_cap), rows)
        map_size = _map_bytes(rows, components)
        gamma = 1.0 / width
        mapper = Nystroem(kernel="rbf", gamma=gamma, n_components=components,
                          random_state=seed, n_jobs=1)
        # Explicit SVD avoids sklearn's Cholesky-to-SVD fallback on failure.
        model = Ridge(alpha=alpha, solver="svd", fit_intercept=True)
        with threadpool_limits(limits=1):
            with _numerical_stage("basis fitting", allow_tail_underflow=True):
                mapper.fit(X)
            _checked_map(mapper, rows, width, components)
            expected = X[mapper.component_indices_]
            if sparse.issparse(expected) and sparse.issparse(mapper.components_):
                matching = (expected != mapper.components_).nnz == 0
            else:
                expected = expected.toarray() if sparse.issparse(expected) else expected
                actual = mapper.components_.toarray() if sparse.issparse(mapper.components_) else mapper.components_
                matching = np.array_equal(expected, actual)
            if not matching:
                raise ValueError("Kernel fitted basis must consist only of the indexed training rows.")
            with _numerical_stage("training transformation", allow_tail_underflow=True):
                embedded = mapper.transform(X)
            _checked_embedding(embedded, rows, components)
            with _numerical_stage("ridge fitting"):
                model.fit(embedded, y)
            _checked_ridge(model, components)
        details = {
            "implementation": "bounded-nystroem-rbf-ridge-v1", "training_rows": rows,
            "transformed_features": width, "rank_cap": int(self.rank_cap),
            "effective_n_components": components, "gamma_policy": "inverse_transformed_width",
            "gamma": gamma, "random_state": seed, "basis_sampling": "uniform_training_rows_without_replacement",
            "basis_indices_sha256": hashlib.sha256(
                np.asarray(mapper.component_indices_, dtype="<i8").tobytes()).hexdigest(),
            "alpha": alpha, "solver": "svd", "threads": 1,
            "training_map_bytes": map_size, "map_byte_limit": MAX_FEATURE_MAP_BYTES,
        }
        self.__dict__.update(mapper_=mapper, ridge_=model, n_features_in_=width,
                             n_training_rows_=rows, effective_n_components_=components,
                             gamma_=gamma, fit_details_=details)
        return self

    def predict(self, X):
        check_is_fitted(self, ("mapper_", "ridge_", "fit_details_"))
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Kernel prediction feature width must match the fitted transformed features.")
        components = self.effective_n_components_
        _map_bytes(X.shape[0], components)
        _checked_map(self.mapper_, self.n_training_rows_, self.n_features_in_, components)
        _checked_ridge(self.ridge_, components)
        with threadpool_limits(limits=1):
            with _numerical_stage("prediction transformation", allow_tail_underflow=True):
                embedded = self.mapper_.transform(X)
            _checked_embedding(embedded, X.shape[0], components)
            with _numerical_stage("ridge prediction"):
                prediction = self.ridge_.predict(embedded)
        _finite(prediction, "predictions", (X.shape[0],))
        return prediction
