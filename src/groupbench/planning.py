"""Read-only planning from the same validated schedules used by execution."""

from __future__ import annotations

import pandas as pd

from . import __version__
from .evaluation import _prepare_benchmark
from .prediction import _prepare_prediction
from .splitting import audit_split


_LIMITATIONS = [
    "No models or preprocessing were fitted; no predictions, scores or result files were written.",
    "Counts describe intended pipeline.fit calls if execution succeeds, not a hard resource budget or time/memory estimate.",
    "Each pipeline fit includes preprocessing and one estimator; individual trees and preprocessing steps are not counted separately.",
    "Every requested family is refitted once per selection, including dummy. The selected entry reuses its winning family refit, with no extra fit.",
    "Schema and split validation do not establish training success, leakage absence, predictive performance or extrapolation safety.",
    "This plan describes the current input and options; changing inputs, options or software can change later execution.",
]


def _split_counts(train, test, groups):
    # Do not expose group identifiers or row values in a planning report.
    return {key: value for key, value in audit_split(train, test, groups).items()
            if key not in ("train_groups", "test_groups")}


def _selection_summary(plan, groups):
    return {
        "inner_seed": plan.seed,
        "inner_folds": [{"fold": fold, **_split_counts(train, test, groups)}
                        for fold, (train, test) in enumerate(plan.splits)],
        "refit_families": list(plan.refit_families),
        "fit_counts": plan.fit_counts,
    }


def _base_report(X, groups, plan, *, mode, target, group, source_sha256):
    return {
        "schema_version": "1.0", "kind": "execution_plan", "mode": mode,
        "groupbench_version": __version__, "fits_performed": 0,
        "source_file_sha256": source_sha256,
        "n_training_rows": len(X), "n_training_groups": len(pd.unique(groups)),
        "feature_schema": [{"name": col, "dtype": str(X[col].dtype),
                            "missing_count": int(X[col].isna().sum())} for col in X],
        "config": {"target": target, "group": group, "features": list(X.columns),
                   "inner_splits": len(plan.splits), "baselines": list(plan.refit_families),
                   "selection_objective": plan.objective, "threads": 1},
        "candidates": [{"name": c.name, "family": c.family, "params": dict(c.params)}
                       for c in plan.candidates],
        "limitations": list(_LIMITATIONS),
    }


def plan_benchmark(frame: pd.DataFrame, *, target: str, group: str, features=None,
                   outer_splits: int = 5, inner_splits: int = 3, random_state: int = 0,
                   baselines=("dummy", "ridge", "extra_trees"), objective="group_mae",
                   source_sha256: str | None = None) -> dict:
    """Validate a nested evaluation and return its exact no-fit work schedule."""
    X, _, groups, seed, schedule = _prepare_benchmark(
        frame, target=target, group=group, features=features, outer_splits=outer_splits,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, objective=objective)
    result = _base_report(X, groups, schedule[0][2], mode="run", target=target,
                          group=group, source_sha256=source_sha256)
    result["config"].update(outer_splits=len(schedule), random_state=seed)
    result["outer_folds"] = [
        {"fold": fold, **_split_counts(train, test, groups),
         **_selection_summary(plan, groups[train])}
        for fold, (train, test, plan) in enumerate(schedule)
    ]
    result["fit_budget"] = {
        "unit": "pipeline.fit calls",
        "inner_selection_fits": sum(plan.fit_counts["inner_selection_fits"] for _, _, plan in schedule),
        "outer_family_refits": sum(plan.fit_counts["family_refits"] for _, _, plan in schedule),
        "full_data_family_refits": 0, "selected_extra_refits": 0,
        "total_fits": sum(plan.fit_counts["total_fits"] for _, _, plan in schedule),
    }
    return result


def plan_prediction(frame: pd.DataFrame, new_rows: pd.DataFrame, *, target: str,
                    group: str, features=None, inner_splits: int = 3, random_state: int = 0,
                    baselines=("dummy", "ridge", "extra_trees"), objective="group_mae",
                    source_sha256: str | None = None,
                    prediction_source_sha256: str | None = None) -> dict:
    """Validate a full-data selection/refit and new-row schema without fitting."""
    X, _, groups, incoming, plan = _prepare_prediction(
        frame, new_rows, target=target, group=group, features=features,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, objective=objective)
    result = _base_report(X, groups, plan, mode="predict", target=target,
                          group=group, source_sha256=source_sha256)
    result["config"]["random_state"] = plan.seed
    result["n_prediction_rows"] = len(incoming)
    result["prediction_source_file_sha256"] = prediction_source_sha256
    result["selection"] = _selection_summary(plan, groups)
    result["fit_budget"] = {
        "unit": "pipeline.fit calls",
        "inner_selection_fits": plan.fit_counts["inner_selection_fits"],
        "outer_family_refits": 0,
        "full_data_family_refits": plan.fit_counts["family_refits"],
        "selected_extra_refits": 0, "total_fits": plan.fit_counts["total_fits"],
    }
    return result
