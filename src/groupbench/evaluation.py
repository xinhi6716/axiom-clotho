"""Nested grouped evaluation and a portable provenance record."""

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

from . import __version__
from .data import validate_source_sha256
from .package_source import observe_package_source
from .categorical import LEGACY, DISTINCT, encoding_options, preprocessing_description
from .data import data_fingerprint, duplicate_diagnostics, prepare_data, validate_seed
from .metrics import grouped_metrics
from .model import GroupBenchRegressor, candidates_for, selection_plan
from .splitting import GroupKFold, audit_split


@dataclass
class BenchmarkResult:
    """In-memory output; ``write`` creates only local JSON/CSV/HTML artifacts."""

    summary: dict
    manifest: dict
    predictions: pd.DataFrame

    @property
    def leaderboard(self) -> pd.DataFrame:
        rows = [{"model": name, **value["overall"]} for name, value in self.summary["models"].items()]
        return pd.DataFrame(rows).sort_values(["group_mae", "model"]).reset_index(drop=True)

    def write(self, output_dir, *, overwrite: bool = False):
        from .report import write_report
        return write_report(self, output_dir, overwrite=overwrite)


def _prepare_benchmark(frame, *, target, group, features, outer_splits,
                       inner_splits, random_state, baselines, objective, categorical_encoding=LEGACY):
    """Shared no-fit validation and exact seeded nested schedule."""
    seed = validate_seed(random_state)
    X, y, groups = prepare_data(frame, target, group, features)
    candidates_for(baselines)
    splits = tuple(GroupKFold(outer_splits, seed).split(X, groups=groups))
    schedule = []
    # Validate every actual inner seed before any training, rather than checking
    # only an approximation of the requested protocol.
    for fold, (train, test) in enumerate(splits):
        selection = selection_plan(X.iloc[train], groups[train], baselines=baselines,
                                   inner_splits=inner_splits, random_state=seed + fold + 1,
                                   objective=objective, categorical_encoding=categorical_encoding)
        schedule.append((train, test, selection))
    return X, y, groups, seed, schedule


def benchmark(frame: pd.DataFrame, *, target: str, group: str, features=None,
              outer_splits: int = 5, inner_splits: int = 3,
              random_state: int = 0, baselines=("dummy", "ridge", "extra_trees"),
              objective: str = "group_mae", source_sha256: str | None = None,
              categorical_encoding: str = LEGACY) -> BenchmarkResult:
    """Estimate performance on unseen groups using genuinely nested selection.

    The ``selected`` entry is selected only on inner validation in each outer
    fold. Other entries tune within their own family on those same inner folds.
    The outer leaderboard is descriptive, not an extra model-selection layer.
    Every observation receives exactly one held-out prediction per entry.
    """
    validate_source_sha256(source_sha256)
    started = time.perf_counter()
    X, y, groups, seed, schedule = _prepare_benchmark(
        frame, target=target, group=group, features=features, outer_splits=outer_splits,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, objective=objective, categorical_encoding=categorical_encoding)
    splits = [(train, test) for train, test, _ in schedule]
    candidates = schedule[0][2].candidates
    outer_splits = len(schedule)
    inner_splits = len(schedule[0][2].splits)
    names = list(baselines) + ["selected"]
    oof = {name: np.full(len(X), np.nan) for name in names}
    fold_ids = np.full(len(X), -1, dtype=int)
    outer_audit = []
    for fold, (train, test, plan) in enumerate(schedule):
        estimator = GroupBenchRegressor(baselines=baselines, inner_splits=inner_splits,
                                        random_state=seed + fold + 1, objective=objective, categorical_encoding=categorical_encoding)
        estimator._fit_prepared(X.iloc[train].reset_index(drop=True), y[train], groups[train], plan)
        predictions = estimator.predict_baselines(X.iloc[test])
        for name in names:
            oof[name][test] = predictions[name]
        fold_ids[test] = fold
        outer_audit.append({
            "fold": fold, **audit_split(train, test, groups),
            "inner_seed": seed + fold + 1,
            "selected_candidate": estimator.best_candidate_,
            "selected_family": estimator.best_family_,
            "family_candidates": estimator.family_candidates_,
            "inner_selection": estimator.selection_results_,
            "inner_folds": estimator.inner_audit_,
        })
    if np.any(fold_ids < 0):
        raise RuntimeError("Internal error: some observations have no held-out fold.")
    model_metrics = {}
    prediction_frame = pd.DataFrame({"row_position": np.arange(len(X)), "group": groups,
                                     "outer_fold": fold_ids, "observed": y})
    for name in names:
        scores, per_group = grouped_metrics(y, oof[name], groups)
        per_fold = [{"fold": fold, **grouped_metrics(y[test], oof[name][test], groups[test])[0]}
                    for fold, (_, test) in enumerate(splits)]
        model_metrics[name] = {"overall": scores, "per_group": per_group, "per_fold": per_fold}
        prediction_frame[f"prediction_{name}"] = oof[name]
    config = {
        "target": target, "group": group, "features": list(X.columns),
        "outer_splits": outer_splits, "inner_splits": inner_splits, "random_state": seed,
        "baselines": list(baselines), "selection_objective": objective,
        "candidates": [{"name": c.name, "family": c.family, "params": c.params} for c in candidates],
        "threads": 1,
    }
    config.update(encoding_options(categorical_encoding))
    fingerprint = data_fingerprint(X, y, groups)
    run_key = hashlib.sha256((fingerprint + json.dumps(config, sort_keys=True)).encode()).hexdigest()
    diagnostics = duplicate_diagnostics(X, y, groups)
    warnings = [
        "Synthetic examples validate plumbing, not scientific predictive performance.",
        "Group separation protects only the supplied grouping axis; it does not prove absence of other leakage.",
        "The outer leaderboard must not be reused for tuning without a new independent evaluation.",
        "Fold variability is not a confidence interval; no uncertainty calibration is implemented.",
        "Outputs include group IDs, observed targets, and predictions; review before sharing.",
    ]
    if diagnostics["identical_feature_target_rows_across_groups"]:
        warnings.append(f"Found {diagnostics['identical_feature_target_rows_across_groups']} identical feature/target rows spanning different groups. Inspect duplicates before trusting evaluation; no rows were removed.")
    manifest = {
        "schema_version": "1.0", "groupbench_version": __version__,
        "package_source": observe_package_source(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_key": run_key, "data_sha256": fingerprint, "source_file_sha256": source_sha256,
        "n_rows": len(X), "n_groups": len(pd.unique(groups)),
        "data_diagnostics": diagnostics,
        "feature_schema": [{"name": c, "dtype": str(X[c].dtype), "missing_count": int(X[c].isna().sum())} for c in X],
        "config": config,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "platform": platform.platform()},
        "protocol": "nested-group-kfold-v1", "network_required_at_runtime": False,
        "preprocessing": "Within each fit: numeric median imputation + scaling; categorical constant imputation + train-only one-hot encoding (max 64 categories per feature).",
        "outer_folds": outer_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": warnings,
    }
    summary = {"schema_version": "1.0", "models": model_metrics,
               "selection_objective": objective, "primary_entry": "selected", "limitations": warnings}
    if categorical_encoding == DISTINCT:
        manifest.update(schema_version="1.1", protocol="nested-group-kfold-distinct-v2",
                        preprocessing=preprocessing_description())
        summary.update(schema_version="1.1", protocol=manifest["protocol"])
    return BenchmarkResult(summary=summary, manifest=manifest, predictions=prediction_frame)
