"""Separate binary evaluation, prediction, and no-fit planning contracts."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn

from .. import __version__
from ..package_source import observe_package_source
from ..categorical import LEGACY, DISTINCT, encoding_options, preprocessing_description
from ..data import data_fingerprint, duplicate_diagnostics, validate_prediction_features, validate_seed
from ..input_support import prediction_input_support
from ..splitting import GroupKFold, audit_split
from .data import class_counts, label_contract, prepare_binary_data, validate_splits
from .metrics import grouped_binary_metrics, metric_contract
from .model import GroupBenchClassifier, binary_selection_plan

DEFAULT_BASELINES = ("dummy", "logistic", "extra_trees")
LIMITATIONS = [
    "Original synthetic fixtures test software only, not semiconductor or other real predictive performance.",
    "The user defines label 1 and independently justifies the group boundary and feature timing before evaluation.",
    "Group separation covers only the supplied axis, not temporal validation, crossed dependence, proxy leakage or duplicate freedom.",
    "The fixed threshold p1 >= 0.5 is a reproducibility convention, not a cost-optimized operational decision rule.",
    "Probabilities are not established calibrated failure risks, causal effects, uncertainty intervals or OOD safety guarantees.",
    "Brier propriety is a mathematical score property, not empirical calibration or perfect float64 resolution.",
    "Log loss clips only its metric copy at float64 epsilon; distinct extreme raw probabilities can have the same diagnostic loss.",
    "Fold variation and group wins/losses are descriptive, not confidence intervals, significance or independent trials.",
    "No autonomous physical/process/wafer disposition or other consequential decision is supported.",
    "Bundles contain group IDs, labels, probabilities and feature names; review before sharing.",
]
INPUT_LIMITATION = "Input-support flags describe individual source features only, not uncertainty, calibration, multivariate support or prediction reliability; no flags is not evidence of safety."
PLAN_LIMITATIONS = [
    "No estimators or preprocessing were constructed/fitted; no predictions, scores or result files were written.",
    "Counts describe intended pipeline.fit calls if execution succeeds, not a hard time or memory budget.",
    "Each fit includes preprocessing and one estimator; individual trees/transform steps are not counted separately.",
    "Every requested family is refitted, including dummy; selected reuses the winning family with no extra fit.",
    "Class feasibility and counts use labels, but group assignments never do; impossible schedules are not rerouted.",
    "Validation does not establish training success, leakage absence, calibration or predictive performance.",
]


def _identity(kind, categorical_encoding=LEGACY):
    return {"schema_id": f"mlcalc.binary.{kind}",
            "schema_version": "1.1" if categorical_encoding == DISTINCT else "1.0",
            "task": "binary_classification", "label_contract": label_contract()}


def _environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "platform": platform.platform()}


def _resources():
    return {"max_rows_per_frame": 20000, "max_source_features": 100, "max_categories_per_feature": 64,
            "max_encoded_features": 6400, "max_outer_splits": 10, "max_inner_splits": 10,
            "max_candidates": 6, "max_logistic_iterations": 1000, "tree_estimators": 64,
            "max_tree_depth": 12, "max_leaves_per_tree": 256, "threads": 1,
            "max_payload_bytes": 64 * 1024 * 1024, "hard_time_or_memory_limit": False}


def _config(X, plan, target, group, *, outer_splits=None, seed=None):
    result = {"target": target, "group": group, "features": list(X.columns),
              "inner_splits": len(plan.splits), "random_state": plan.seed if seed is None else seed,
              "baselines": list(plan.refit_families), "selection_objective": "group_brier",
              "candidates": [{"name": c.name, "family": c.family, "params": dict(c.params)} for c in plan.candidates],
              "training_weighting": "equal_rows_no_class_or_sample_weights",
              "logistic_regularization": "l2_default_finite_C_lbfgs",
              "model_random_state": "selection_seed", "threads": 1, "resource_limits": _resources()}
    result.update(encoding_options(plan.categorical_encoding))
    if outer_splits is not None:
        result["outer_splits"] = outer_splits
    return result


def _prepare_evaluation(frame, *, target, group, features, outer_splits, inner_splits, random_state, baselines, categorical_encoding=LEGACY):
    seed = validate_seed(random_state)
    outer = validate_splits(outer_splits, "outer_splits")
    validate_splits(inner_splits, "inner_splits")
    X, y, groups = prepare_binary_data(frame, target, group, features)
    schedule = []
    for fold, (train, test) in enumerate(GroupKFold(outer, seed).split(X, groups=groups)):
        plan = binary_selection_plan(X.iloc[train], y[train], groups[train], baselines=baselines,
                                     inner_splits=inner_splits, random_state=seed + fold + 1,
                                     context=f"Outer fold {fold} training", categorical_encoding=categorical_encoding)
        schedule.append((train, test, plan))
    return X, y, groups, seed, tuple(schedule)


def _prepare_prediction(frame, new_rows, *, target, group, features, inner_splits, random_state, baselines, categorical_encoding=LEGACY):
    X, y, groups = prepare_binary_data(frame, target, group, features)
    incoming = validate_prediction_features(new_rows, X.columns, [str(X[c].dtype) for c in X])
    plan = binary_selection_plan(X, y, groups, baselines=baselines,
                                 inner_splits=inner_splits, random_state=random_state, categorical_encoding=categorical_encoding)
    return X, y, groups, incoming, plan


def _split_summary(train, test, y, groups):
    result = {k: v for k, v in audit_split(train, test, groups).items()
              if k not in ("train_groups", "test_groups")}
    result.update(training_class_counts=class_counts(y[train]), validation_class_counts=class_counts(y[test]))
    return result


def _selection_summary(plan, y, groups):
    return {"inner_seed": plan.seed,
            "inner_folds": [{"fold": i, **_split_summary(train, test, y, groups)}
                            for i, (train, test) in enumerate(plan.splits)],
            "refit_families": list(plan.refit_families), "fit_counts": plan.fit_counts}


def _fit_budget(plans, *, evaluation):
    inner = sum(p.fit_counts["inner_selection_fits"] for p in plans)
    refits = sum(p.fit_counts["family_refits"] for p in plans)
    return {"unit": "pipeline.fit calls", "inner_selection_fits": inner,
            "outer_family_refits": refits if evaluation else 0,
            "full_data_family_refits": 0 if evaluation else refits,
            "selected_extra_refits": 0, "total_fits": inner + refits}


def _plan_base(X, y, groups, plan, target, group, source_sha256, mode):
    return {**_identity("execution_plan", plan.categorical_encoding), "mode": mode, "groupbench_version": __version__,
            "fits_performed": 0, "source_file_sha256": source_sha256,
            "n_training_rows": len(X), "n_training_groups": len(pd.unique(groups)),
            "training_class_counts": class_counts(y),
            "feature_schema": [{"name": c, "dtype": str(X[c].dtype), "missing_count": int(X[c].isna().sum())} for c in X],
            "config": _config(X, plan, target, group), "metric_contract": metric_contract(),
            "limitations": list(PLAN_LIMITATIONS)}


def plan_binary_benchmark(frame, *, target, group, features=None, outer_splits=5, inner_splits=3,
                          random_state=0, baselines=DEFAULT_BASELINES, source_sha256=None, categorical_encoding=LEGACY):
    X, y, groups, seed, schedule = _prepare_evaluation(
        frame, target=target, group=group, features=features, outer_splits=outer_splits,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, categorical_encoding=categorical_encoding)
    result = _plan_base(X, y, groups, schedule[0][2], target, group, source_sha256, "run")
    result["config"] = _config(X, schedule[0][2], target, group, outer_splits=len(schedule), seed=seed)
    result["outer_folds"] = [{"fold": i, **_split_summary(train, test, y, groups),
                               **_selection_summary(plan, y[train], groups[train])}
                              for i, (train, test, plan) in enumerate(schedule)]
    result["fit_budget"] = _fit_budget([plan for _, _, plan in schedule], evaluation=True)
    if categorical_encoding == DISTINCT:
        result["protocol"] = "nested-group-binary-brier-distinct-v2"
    return result


def plan_binary_prediction(frame, new_rows, *, target, group, features=None, inner_splits=3,
                           random_state=0, baselines=DEFAULT_BASELINES, source_sha256=None,
                           prediction_source_sha256=None, categorical_encoding=LEGACY):
    X, y, groups, incoming, plan = _prepare_prediction(
        frame, new_rows, target=target, group=group, features=features,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, categorical_encoding=categorical_encoding)
    result = _plan_base(X, y, groups, plan, target, group, source_sha256, "predict")
    result.update(n_prediction_rows=len(incoming), prediction_source_file_sha256=prediction_source_sha256,
                  input_support=prediction_input_support(X, incoming), selection=_selection_summary(plan, y, groups),
                  fit_budget=_fit_budget([plan], evaluation=False))
    result["limitations"].append(INPUT_LIMITATION)
    if categorical_encoding == DISTINCT:
        result["protocol"] = "group-binary-brier-fit-predict-distinct-v2"
    return result


@dataclass
class BinaryBenchmarkResult:
    summary: dict
    manifest: dict
    predictions: pd.DataFrame

    @property
    def leaderboard(self):
        return pd.DataFrame([{"model": name, **v["overall"]} for name, v in self.summary["models"].items()]).sort_values(
            ["group_brier", "model"]).reset_index(drop=True)

    def write(self, output_dir):
        from .report import write_binary_evaluation
        return write_binary_evaluation(self, output_dir)


@dataclass
class BinaryPredictionResult:
    predictions: pd.DataFrame
    manifest: dict
    estimator: GroupBenchClassifier

    def write(self, output_dir):
        from .report import write_binary_prediction
        return write_binary_prediction(self, output_dir)


def _adverse(models):
    if "dummy" not in models:
        return {"available": False, "reason": "Dummy was not requested; no hidden fits were performed."}
    selected, dummy = models["selected"], models["dummy"]
    baseline = {row["group"]: row["brier"] for row in dummy["per_group"]}
    rows = [{"group": row["group"], "selected_minus_dummy_brier": row["brier"] - baseline[row["group"]]}
            for row in selected["per_group"]]
    delta = [row["selected_minus_dummy_brier"] for row in rows]
    return {"available": True, "group_brier_delta": selected["overall"]["group_brier"] - dummy["overall"]["group_brier"],
            "groups_better": sum(v < 0 for v in delta), "groups_tied": sum(v == 0 for v in delta),
            "groups_worse": sum(v > 0 for v in delta), "per_group": rows,
            "interpretation": "Selected minus dummy; positive is worse. Descriptive only, not independent trials or significance."}


def benchmark_binary(frame, *, target, group, features=None, outer_splits=5, inner_splits=3,
                     random_state=0, baselines=DEFAULT_BASELINES, source_sha256=None, categorical_encoding=LEGACY):
    started = time.perf_counter()
    X, y, groups, seed, schedule = _prepare_evaluation(
        frame, target=target, group=group, features=features, outer_splits=outer_splits,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, categorical_encoding=categorical_encoding)
    names = list(baselines) + ["selected"]
    oof = {name: np.full((len(X), 2), np.nan) for name in names}
    fold_ids = np.full(len(X), -1, dtype=int)
    audits = []
    for fold, (train, test, plan) in enumerate(schedule):
        estimator = GroupBenchClassifier(baselines=baselines, inner_splits=inner_splits, random_state=plan.seed, categorical_encoding=categorical_encoding)
        estimator._fit_prepared(X.iloc[train].reset_index(drop=True), y[train], groups[train], plan)
        probabilities = estimator.predict_proba_baselines(X.iloc[test])
        for name in names:
            oof[name][test] = probabilities[name]
        fold_ids[test] = fold
        audits.append({"fold": fold, **audit_split(train, test, groups),
                       "training_class_counts": class_counts(y[train]), "validation_class_counts": class_counts(y[test]),
                       "inner_seed": plan.seed, "selected_candidate": estimator.best_candidate_,
                       "selected_family": estimator.best_family_, "family_candidates": estimator.family_candidates_,
                       "inner_selection": estimator.selection_results_, "inner_folds": estimator.inner_audit_})
    if np.any(fold_ids < 0):
        raise RuntimeError("Internal binary schedule error: a row has no held-out fold.")
    predictions = pd.DataFrame({"row_position": np.arange(len(X)), "group": groups,
                                "outer_fold": fold_ids, "observed_label": y})
    models = {}
    for name in names:
        overall, per_group = grouped_binary_metrics(y, oof[name][:, 1], groups)
        per_fold = [{"fold": fold, **grouped_binary_metrics(y[test], oof[name][test, 1], groups[test])[0]}
                    for fold, (_, test, _) in enumerate(schedule)]
        models[name] = {"overall": overall, "per_group": per_group, "per_fold": per_fold}
        predictions[f"probability_0_{name}"] = oof[name][:, 0]
        predictions[f"probability_1_{name}"] = oof[name][:, 1]
        predictions[f"predicted_label_{name}"] = (oof[name][:, 1] >= 0.5).astype(np.int64)
    config = _config(X, schedule[0][2], target, group, outer_splits=len(schedule), seed=seed)
    fingerprint = data_fingerprint(X, y, groups)
    protocol = ("nested-group-binary-brier-distinct-v2" if categorical_encoding == DISTINCT
                else "nested-group-binary-brier-v1")
    schema = [{"name": c, "dtype": str(X[c].dtype), "missing_count": int(X[c].isna().sum())} for c in X]
    run_input = {**_identity("evaluation.manifest", categorical_encoding), "protocol": protocol, "data_sha256": fingerprint,
                 "config": config, "metric_contract": metric_contract(), "feature_schema": schema}
    run_key = hashlib.sha256(json.dumps(run_input, sort_keys=True, allow_nan=False).encode()).hexdigest()
    diagnostics = duplicate_diagnostics(X, y, groups)
    limitations = list(LIMITATIONS) + ["The outer leaderboard must not be reused for tuning without new independent evaluation."]
    if diagnostics["identical_feature_target_rows_across_groups"]:
        limitations.append("Identical feature/target rows span groups; inspect duplicates. No records were removed.")
    manifest = {**run_input, "groupbench_version": __version__, "run_key": run_key,
                "package_source": observe_package_source(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_file_sha256": source_sha256,
                "n_rows": len(X), "n_groups": len(pd.unique(groups)), "training_class_counts": class_counts(y),
                "data_diagnostics": diagnostics, "environment": _environment(), "network_required_at_runtime": False,
                "preprocessing": "Each fit: numeric median imputation/scaling; categorical constant imputation and train-only sparse one-hot (max 64 categories).",
                "outer_folds": audits, "fit_budget": _fit_budget([plan for _, _, plan in schedule], evaluation=True),
                "elapsed_seconds": time.perf_counter() - started, "limitations": limitations}
    summary = {**_identity("evaluation.summary", categorical_encoding), "models": models, "selection_objective": "group_brier",
               "primary_entry": "selected", "metric_contract": metric_contract(),
               "adverse_comparison_to_dummy": _adverse(models), "limitations": limitations}
    if categorical_encoding == DISTINCT:
        manifest["preprocessing"] = preprocessing_description()
        summary["protocol"] = protocol
    return BinaryBenchmarkResult(summary, manifest, predictions)


def fit_predict_binary(frame, new_rows, *, target, group, features=None, inner_splits=3,
                       random_state=0, baselines=DEFAULT_BASELINES, source_sha256=None,
                       prediction_source_sha256=None, categorical_encoding=LEGACY):
    X, y, groups, incoming, plan = _prepare_prediction(
        frame, new_rows, target=target, group=group, features=features,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, categorical_encoding=categorical_encoding)
    estimator = GroupBenchClassifier(baselines=baselines, inner_splits=inner_splits, random_state=plan.seed, categorical_encoding=categorical_encoding)
    estimator._fit_prepared(X, y, groups, plan)
    p = estimator.predict_proba(incoming)
    manifest = {**_identity("prediction.manifest", categorical_encoding), "groupbench_version": __version__, "mode": "fit_predict",
                "package_source": observe_package_source(),
                "protocol": "group-binary-brier-fit-predict-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_data_sha256": data_fingerprint(X, y, groups),
                "new_features_sha256": hashlib.sha256(pd.util.hash_pandas_object(incoming, index=False).values.tobytes()).hexdigest(),
                "source_file_sha256": source_sha256, "prediction_source_file_sha256": prediction_source_sha256,
                "n_training_rows": len(X), "n_training_groups": len(pd.unique(groups)), "n_predictions": len(p),
                "training_class_counts": class_counts(y),
                "feature_schema": [{"name": c, "dtype": str(X[c].dtype)} for c in X],
                "config": _config(X, plan, target, group), "metric_contract": metric_contract(),
                "selected_candidate": estimator.best_candidate_, "selected_family": estimator.best_family_,
                "inner_selection": estimator.selection_results_, "inner_folds": estimator.inner_audit_,
                "data_diagnostics": duplicate_diagnostics(X, y, groups), "input_support": prediction_input_support(X, incoming),
                "environment": _environment(), "network_required_at_runtime": False,
                "fit_budget": _fit_budget([plan], evaluation=False),
                "limitations": list(LIMITATIONS) + [
                    "These full-training new-row predictions are not held-out evaluation; inner scores are selection diagnostics, not unbiased performance.",
                    INPUT_LIMITATION, "No executable model is saved; retain the in-memory estimator or repeat fit_predict_binary."]}
    predictions = pd.DataFrame({"row_position": np.arange(len(p)), "probability_0": p[:, 0],
                                "probability_1": p[:, 1], "predicted_label": (p[:, 1] >= 0.5).astype(np.int64)})
    if categorical_encoding == DISTINCT:
        manifest.update(protocol="group-binary-brier-fit-predict-distinct-v2",
                        preprocessing=preprocessing_description())
    return BinaryPredictionResult(predictions, manifest, estimator)
