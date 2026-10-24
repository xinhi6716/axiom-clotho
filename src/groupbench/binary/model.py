"""Bounded binary selection with train-only preprocessing and group folds."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from ..categorical import LEGACY, categorical_imputer, encoding_options, validate_categorical_encoding
from ..data import validate_features, validate_groups, validate_prediction_features, validate_seed
from ..splitting import GroupKFold, audit_split
from .data import class_counts, require_classes, validate_binary_target, validate_splits
from .metrics import checked_probabilities, grouped_binary_metrics


@dataclass(frozen=True)
class BinaryCandidate:
    name: str
    family: str
    params: dict


def binary_candidates(baselines):
    known = {"dummy", "logistic", "extra_trees"}
    if (not isinstance(baselines, Sequence) or isinstance(baselines, str) or not baselines
            or any(not isinstance(v, str) for v in baselines)
            or len(set(baselines)) != len(baselines) or set(baselines) - known):
        raise ValueError("Binary baselines must be a nonempty ordered distinct sequence from dummy, logistic, extra_trees.")
    result = []
    for family in baselines:
        if family == "dummy":
            result.append(BinaryCandidate("dummy_prior", family, {"strategy": "prior"}))
        elif family == "logistic":
            for c in (0.1, 1.0, 10.0):
                result.append(BinaryCandidate(f"logistic_C_{c:g}", family,
                    {"C": c, "solver": "lbfgs", "tol": 1e-4, "max_iter": 1000,
                     "fit_intercept": True, "class_weight": None}))
        else:
            for depth in (6, 12):
                result.append(BinaryCandidate(f"extra_trees_depth_{depth}", family,
                    {"n_estimators": 64, "max_depth": depth, "max_leaf_nodes": 256,
                     "min_samples_leaf": 2, "max_features": "sqrt", "criterion": "gini",
                     "bootstrap": False, "class_weight": None, "n_jobs": 1}))
    return tuple(result)


@dataclass(frozen=True)
class BinarySelectionPlan:
    candidates: tuple[BinaryCandidate, ...]
    splits: tuple
    refit_families: tuple[str, ...]
    seed: int
    categorical_encoding: str = LEGACY

    @property
    def fit_counts(self):
        inner = sum(len(self.splits) for _ in self.candidates)
        return {"inner_selection_fits": inner, "family_refits": len(self.refit_families),
                "total_fits": inner + len(self.refit_families)}


def binary_selection_plan(X, y, groups, *, baselines, inner_splits, random_state, context="Full training", categorical_encoding=LEGACY):
    validate_categorical_encoding(categorical_encoding)
    seed = validate_seed(random_state)
    count = validate_splits(inner_splits, "inner_splits")
    candidates = binary_candidates(baselines)
    require_classes(y, context)
    splits = tuple(GroupKFold(count, seed).split(X, groups=groups))
    for fold, (train, _) in enumerate(splits):
        require_classes(y[train], f"{context}, inner fold {fold} training")
    return BinarySelectionPlan(candidates, splits, tuple(baselines), seed, categorical_encoding)


def binary_pipeline(X, candidate, seed, categorical_encoding=LEGACY):
    validate_categorical_encoding(categorical_encoding)
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", categorical_imputer(categorical_encoding)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=64, sparse_output=True))]), categorical))
    preprocessing = ColumnTransformer(transforms, remainder="drop", sparse_threshold=1.0)
    if candidate.family == "dummy":
        model = DummyClassifier(**candidate.params)
    elif candidate.family == "logistic":
        # Finite C + default regularization + lbfgs is L2 in supported sklearn.
        # Avoid 1.8's deprecated penalty/n_jobs and newer-only l1_ratio choices.
        model = LogisticRegression(**candidate.params, random_state=seed)
    else:
        model = ExtraTreesClassifier(**candidate.params, random_state=seed)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def _fit_pipeline(X, y, candidate, seed, context, categorical_encoding=LEGACY):
    pipeline = binary_pipeline(X, candidate, seed, **encoding_options(categorical_encoding))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            pipeline.fit(X, y)
    except (ConvergenceWarning, RuntimeWarning, ValueError, FloatingPointError) as exc:
        raise ValueError(f"Binary fit failed for {candidate.name} at {context} ({type(exc).__name__}); no candidate is skipped or retried.") from exc
    return pipeline


def _probabilities(pipeline, X):
    return checked_probabilities(pipeline.predict_proba(X), len(X), pipeline.named_steps["model"].classes_)


class GroupBenchClassifier(ClassifierMixin, BaseEstimator):
    """Explicit numeric 0/1 classifier; fixed group-Brier selection, local only.

    Training weights rows equally; selection weights groups equally. predict
    uses p1>=0.5 (ties are class 1). Inherited score is pooled label accuracy,
    not the selection objective. No model persistence or calibration is added.
    A failed fit preserves the previous complete fitted state, if one exists.
    """
    def __init__(self, baselines=("dummy", "logistic", "extra_trees"), inner_splits=3, random_state=0, categorical_encoding=LEGACY):
        self.baselines = baselines
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.categorical_encoding = categorical_encoding

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("groups is required; no row-random fallback is used.")
        X = validate_features(X)
        y = validate_binary_target(y, len(X))
        groups = validate_groups(groups, len(X))
        plan = binary_selection_plan(X, y, groups, baselines=self.baselines,
                                     inner_splits=self.inner_splits, random_state=self.random_state,
                                     categorical_encoding=self.categorical_encoding)
        return self._fit_prepared(X, y, groups, plan)

    def _fit_prepared(self, X, y, groups, plan):
        records, family_models, family_candidates = [], {}, {}
        audit = [{"fold": fold, **audit_split(train, test, groups),
                  "training_class_counts": class_counts(y[train]),
                  "validation_class_counts": class_counts(y[test])}
                 for fold, (train, test) in enumerate(plan.splits)]
        with threadpool_limits(limits=1):
            for candidate in plan.candidates:
                oof = np.full((len(X), 2), np.nan)
                fold_scores, fold_metrics = [], []
                for fold, (train, test) in enumerate(plan.splits):
                    pipeline = _fit_pipeline(X.iloc[train], y[train], candidate, plan.seed, f"inner fold {fold}", **encoding_options(plan.categorical_encoding))
                    probabilities = _probabilities(pipeline, X.iloc[test])
                    oof[test] = probabilities
                    scores, _ = grouped_binary_metrics(y[test], probabilities[:, 1], groups[test])
                    fold_scores.append(scores["group_brier"])
                    fold_metrics.append({"fold": fold, **scores})
                scores, _ = grouped_binary_metrics(y, oof[:, 1], groups)
                records.append({"candidate": candidate.name, "family": candidate.family,
                                "params": dict(candidate.params), "selection_score": scores["group_brier"],
                                "objective": "group_brier", "inner_fold_scores": fold_scores,
                                "inner_oof_metrics": scores, "inner_fold_metrics": fold_metrics})
            winner = min(range(len(records)), key=lambda i: records[i]["selection_score"])
            for family in plan.refit_families:
                best = min((i for i, c in enumerate(plan.candidates) if c.family == family),
                           key=lambda i: records[i]["selection_score"])
                chosen = plan.candidates[best]
                family_models[family] = _fit_pipeline(X, y, chosen, plan.seed, "family refit", **encoding_options(plan.categorical_encoding))
                family_candidates[family] = chosen.name
        # Publish all state only after every fit succeeds.
        best_candidate = plan.candidates[winner]
        self.__dict__.update(
            family_models_=family_models, family_candidates_=family_candidates,
            best_candidate_=best_candidate.name, best_family_=best_candidate.family,
            best_estimator_=family_models[best_candidate.family], selection_results_=records,
            inner_audit_=audit, classes_=np.array([0, 1], dtype=np.int64),
            feature_names_in_=np.array(X.columns, dtype=object), n_features_in_=X.shape[1],
            feature_types_=[str(X[c].dtype) for c in X], n_groups_seen_=len(pd.unique(groups)),
            fit_counts_=plan.fit_counts, categorical_encoding_=plan.categorical_encoding)
        return self

    def _validated_predict_frame(self, X):
        check_is_fitted(self, "best_estimator_")
        return validate_prediction_features(X, self.feature_names_in_, self.feature_types_)

    def predict_proba(self, X):
        X = self._validated_predict_frame(X)
        with threadpool_limits(limits=1):
            return _probabilities(self.best_estimator_, X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def predict_proba_baselines(self, X):
        X = self._validated_predict_frame(X)
        with threadpool_limits(limits=1):
            result = {family: _probabilities(model, X) for family, model in self.family_models_.items()}
        result["selected"] = result[self.best_family_].copy()
        return result
