"""A small sklearn-like selector whose every learned transform stays in-fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .categorical import LEGACY, categorical_imputer, encoding_options, validate_categorical_encoding
from .data import (validate_features, validate_groups, validate_prediction_features,
                   validate_seed, validate_target)
from .metrics import grouped_metrics
from .splitting import GroupKFold, audit_split


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict


def candidates_for(baselines) -> list[Candidate]:
    if isinstance(baselines, str) or not baselines or len(set(baselines)) != len(baselines):
        raise ValueError("baselines must be a non-empty sequence of distinct model family names.")
    known = {"dummy", "ridge", "extra_trees"}
    if set(baselines) - known:
        raise ValueError(f"Unknown baseline. Choose from {sorted(known)}.")
    result = []
    for family in baselines:
        if family == "dummy":
            result.append(Candidate("dummy_mean", family, {"strategy": "mean"}))
        elif family == "ridge":
            result.extend(Candidate(f"ridge_alpha_{alpha:g}", family, {"alpha": alpha}) for alpha in (0.1, 1.0, 10.0))
        else:
            result.extend(Candidate(f"extra_trees_depth_{depth or 'none'}", family, {"max_depth": depth, "n_estimators": 64, "min_samples_leaf": 2}) for depth in (6, None))
    return result


@dataclass(frozen=True)
class SelectionPlan:
    """The candidate, split and refit schedule consumed by fitting and preview."""

    candidates: tuple[Candidate, ...]
    splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    refit_families: tuple[str, ...]
    seed: int
    objective: str
    categorical_encoding: str = LEGACY

    @property
    def fit_counts(self) -> dict:
        # Count the same iterations consumed in _fit_prepared, not a separate
        # formula tied to today's default models or requested split counts.
        inner = sum(len(self.splits) for _ in self.candidates)
        refits = len(self.refit_families)
        return {"inner_selection_fits": inner, "family_refits": refits,
                "total_fits": inner + refits}


def selection_plan(X, groups, *, baselines, inner_splits, random_state, objective, categorical_encoding=LEGACY):
    """Validate and materialize a selection schedule without fitting anything."""
    validate_categorical_encoding(categorical_encoding)
    if objective not in ("group_mae", "mae"):
        raise ValueError("objective must be 'group_mae' or 'mae'.")
    seed = validate_seed(random_state)
    candidates = tuple(candidates_for(baselines))
    splits = tuple(GroupKFold(inner_splits, seed).split(X, groups=groups))
    return SelectionPlan(candidates, splits, tuple(baselines), seed, objective, categorical_encoding)


def make_pipeline(X: pd.DataFrame, candidate: Candidate, random_state: int, categorical_encoding=LEGACY) -> Pipeline:
    validate_categorical_encoding(categorical_encoding)
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", categorical_imputer(categorical_encoding)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=64, sparse_output=True)),
        ]), categorical))
    preprocessing = ColumnTransformer(transforms, remainder="drop", sparse_threshold=1.0)
    if candidate.family == "dummy":
        model = DummyRegressor(**candidate.params)
    elif candidate.family == "ridge":
        model = Ridge(**candidate.params, solver="lsqr")
    else:
        model = ExtraTreesRegressor(**candidate.params, random_state=random_state, n_jobs=1)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


class GroupBenchRegressor(RegressorMixin, BaseEstimator):
    """Select a local baseline with group-disjoint inner cross-validation.

    X must be a named pandas DataFrame without target or group identifiers.
    ``fit`` selects by inner out-of-fold error, then refits on all supplied rows.
    Evaluate generalization with ``benchmark``, not the returned training fit.
    No remote service, model weights, or accelerator is used.
    A failed fit preserves the previous complete fitted state, if one exists.
    """

    def __init__(self, baselines=("dummy", "ridge", "extra_trees"), inner_splits=3,
                 random_state=0, objective="group_mae", categorical_encoding=LEGACY):
        self.baselines = baselines
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.objective = objective
        self.categorical_encoding = categorical_encoding

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("groups is required; there is no row-random fallback.")
        X = validate_features(X)
        y = validate_target(y, len(X))
        groups = validate_groups(groups, len(X))
        plan = selection_plan(X, groups, baselines=self.baselines, inner_splits=self.inner_splits,
                              random_state=self.random_state, objective=self.objective,
                              categorical_encoding=self.categorical_encoding)
        return self._fit_prepared(X, y, groups, plan)

    def _fit_prepared(self, X, y, groups, plan: SelectionPlan):
        """Execute a validated schedule; used by both nested and full-data fits."""
        candidates, splits, seed = plan.candidates, plan.splits, plan.seed
        audit = [{"fold": i, **audit_split(train, test, groups)} for i, (train, test) in enumerate(splits)]
        records = []
        # Limit BLAS/OpenMP as well as the tree estimator to one thread.
        with threadpool_limits(limits=1):
            for candidate in candidates:
                oof = np.full(len(X), np.nan)
                fold_scores = []
                for train, test in splits:
                    pipeline = make_pipeline(X.iloc[train], candidate, seed, **encoding_options(plan.categorical_encoding))
                    pipeline.fit(X.iloc[train], y[train])
                    prediction = pipeline.predict(X.iloc[test])
                    oof[test] = prediction
                    scores, _ = grouped_metrics(y[test], prediction, groups[test])
                    fold_scores.append(scores[plan.objective])
                scores, _ = grouped_metrics(y, oof, groups)
                records.append({
                    "candidate": candidate.name, "family": candidate.family,
                    "params": candidate.params, "selection_score": scores[plan.objective],
                    "objective": plan.objective, "inner_fold_scores": fold_scores,
                })
            # A fixed candidate order gives deterministic tie-breaking.
            winner = min(range(len(records)), key=lambda i: records[i]["selection_score"])
            family_models, family_candidates = {}, {}
            for family in plan.refit_families:
                best = min((i for i, c in enumerate(candidates) if c.family == family), key=lambda i: records[i]["selection_score"])
                chosen = candidates[best]
                pipeline = make_pipeline(X, chosen, seed, **encoding_options(plan.categorical_encoding))
                pipeline.fit(X, y)
                family_models[family] = pipeline
                family_candidates[family] = chosen.name
        # Publish state only after selection and every family refit succeeds.
        # Otherwise a failed refit could leave old selected predictions paired
        # with a new audit or incomplete baseline models.
        best = candidates[winner]
        self.__dict__.update(
            inner_audit_=audit, family_models_=family_models,
            family_candidates_=family_candidates, best_candidate_=best.name,
            best_family_=best.family, best_estimator_=family_models[best.family],
            selection_results_=records, feature_names_in_=np.array(X.columns, dtype=object),
            n_features_in_=X.shape[1], feature_types_=[str(X[c].dtype) for c in X],
            n_groups_seen_=len(pd.unique(groups)), categorical_encoding_=plan.categorical_encoding)
        return self

    def _validated_predict_frame(self, X):
        check_is_fitted(self, "best_estimator_")
        return validate_prediction_features(X, self.feature_names_in_, self.feature_types_)

    @staticmethod
    def _finite_predictions(prediction):
        prediction = np.asarray(prediction, dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Model produced non-finite predictions. Check or rescale numeric inputs and avoid extreme extrapolation.")
        return prediction

    def predict(self, X):
        X = self._validated_predict_frame(X)
        with threadpool_limits(limits=1):
            return self._finite_predictions(self.best_estimator_.predict(X))

    def predict_baselines(self, X) -> dict[str, np.ndarray]:
        X = self._validated_predict_frame(X)
        with threadpool_limits(limits=1):
            result = {family: self._finite_predictions(model.predict(X)) for family, model in self.family_models_.items()}
        result["selected"] = result[self.best_family_].copy()
        return result
