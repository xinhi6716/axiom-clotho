# Fixed whole-group temporal regression evaluation v1

This Python-only, in-memory API answers: how did two fixed recipes perform on
these declared future-group windows? It is separate from `benchmark`, selection,
full-OOF results, saved bundles, CLI commands, and the metadata-only planner.
There is no model choice, tuning, full-data refit, fitted-model return, or file
writer. Existing evaluation and prediction bundle schemas do not accept this
schema. No scientific-performance claim follows from the software fixtures.

## Preview, then evaluate the same snapshot

```python
import pandas as pd
from groupbench import (
    preview_group_forward_evaluation, evaluate_group_forward_baselines,
)
from groupbench.group_time_plan import GroupSpan, TimeWindow, plan_group_forward_windows

row_groups = pd.Series([1, "1", "a", "warm2", "b", "cross", 1, "a", "late"],
                       dtype=object, name="group")
y = pd.Series([0., 3., 4., 4., 5., 10., 2., 6., 9.], name="target")
X = pd.DataFrame({"measurement": y.to_numpy(copy=True)})
plan = plan_group_forward_windows(
    tuple(row_groups),
    group_spans=[GroupSpan(1, 0, 1, 1), GroupSpan("warm2", 2, 3, 3),
                 GroupSpan("1", 10, 10, 12), GroupSpan("a", 11, 12, 12),
                 GroupSpan("b", 20, 21, 21), GroupSpan("cross", 8, 12, 12),
                 GroupSpan("late", 40, 41, 41)],
    windows=[TimeWindow(10, 15), TimeWindow(20, 25)],
    time_unit="day", clock_id="handwritten-fixture-clock", gap=2,
)
arguments = dict(row_groups=row_groups, plan=plan,
                 feature_allowlist=["measurement"],
                 metadata_columns=["group", "start", "end", "label_available_at"])
preview = preview_group_forward_evaluation(X, y, **arguments)
result = evaluate_group_forward_baselines(
    X, y, **arguments,
    expected_execution_sha256=preview.bindings["execution_sha256"],
)
assert preview.fits_performed == 0
assert result.fit_budget["completed_total"] == 4
assert result.coverage["tested"]["rows"] == (1, 2, 4, 7)
assert result.coverage["untested"]["rows"] == (0, 3, 5, 6, 8)
```

This example deliberately copies the target into a feature to make the numeric
software fixture easy to check. It is not an example of a scientifically valid
predictor, independent measurement, or deployment evaluation.

## Exact input and role boundary

All three inputs must have the same unnamed canonical `RangeIndex(0, N, 1)`.
There is no reset, join, drop, sort, or implicit alignment. Row positions refer
only to this caller-retained mapping. The DataFrame and the two named Series
must have their exact pandas classes; subclasses are outside v1.

The explicit, nonempty feature allowlist must equal all DataFrame columns in
order. Names are distinct nonblank, unpadded built-in strings. The target and
group names are distinct, and neither can be a feature. `metadata_columns` is an
explicit distinct list/tuple containing the group name and every source timing,
span, availability, gap, clock/unit, row-key, fold/window, and other non-predictor
column name. It must be disjoint from features and target. Actual metadata
arrives separately from X. These declared roles do **not** identify undeclared
aliases, copied metadata, target-derived features, or unavailable predictors.
The caller is responsible for the original mapping and feature availability.

Feature and target columns require NumPy-backed real integer/floating dtypes
with width at most 64 bits. Booleans, complex values, object/string/category,
datetimes, timedeltas, wider floats, pandas nullable/masked extension arrays,
infinity, and integers that cannot convert exactly to float64 are rejected.
Ordinary floating-point NaNs are allowed only in features. Every target,
including all untested rows, must be observed and finite. This API explicitly
means regression; low target cardinality does not select classification.

The group Series must have object dtype and exact built-in nonblank strings or
signed-64-bit integers. Integer `1`, string `"1"`, and string `"01"` are distinct.
Valid whitespace and NUL-containing identities are preserved, not normalized or
sent directly to pandas grouping. Group values are checked against the plan at
every row. Numeric and identity snapshots are copied; concurrent source mutation
during snapshotting is unsupported. This does not make the initial mapping true.

## Reconstruct the complete plan

Frozen dataclasses are not certificates. Preview and execution check exact
supported container/scalar types, preflight resource limits, and independently
rebuild the complete plan with `plan_group_forward_windows`. Every field must
match with type-sensitive equality, including fixed markers, rows, group order,
exclusions, folds, and tested/untested membership. Only fresh reconstructed
records are used. The whole-group partitions are disjoint and exhaustive in
each window. Every tested row and tested group appears once across windows.
Repeated training is allowed; earlier test groups train later only when their
declared label availability and gap satisfy that later cutoff.

Training requires `label_available_at + gap < window.lower`; testing requires
the entire inclusive observation span inside the half-open test window.
Warmup, crossing, and late groups can remain untested even if some train later.
No predictions are manufactured for these rows.

## Fixed numeric recipes and failure behavior

For each window, in declared order, run `dummy_mean`, then `ridge_alpha_1`.
Each pipeline is fresh and receives only that window's training features and
targets. Preprocessing is independently trained median `SimpleImputer` with
`keep_empty_features=True`, then `StandardScaler(with_mean=True, with_std=True)`.
An entirely missing training feature is retained with zero imputation. No global
imputation, fitted-state sharing, or full-data preprocessing occurs.

- `dummy_mean`: `DummyRegressor(strategy="mean")`, a row-weighted training mean.
- `ridge_alpha_1`: `Ridge(alpha=1.0, solver="lsqr", fit_intercept=True, tol=1e-8,
  max_iter=1000, copy_X=True, positive=False)`, ordinary unweighted row loss.

Training does not use group weights; group weighting applies only to reporting.
Execution uses `threadpool_limits(limits=1)` and no RNG. There are exactly two
pipeline fits per window, including preprocessing. All selection, bootstrap,
retry, selected-model, and full-data refit counts are zero.

Nonfinite learned imputer/scaler state, estimator coefficients/intercept or
constant, transformed train/test values, predictions, and metric arithmetic
reject the entire run. Metric overflow and underflow reject. A Ridge iteration
count reaching 1000 rejects conservatively; fewer iterations do not certify
numerical accuracy. Use practical feature and target units. There is no window
skipping, fallback recipe, partial successful result, or retry.
`GroupForwardEvaluationError.attempted_fits` counts invoked pipeline fits;
`completed_fits` counts returned `pipeline.fit` calls, even when a subsequent
state/prediction/metric check fails. Preparation failures report both as zero.
Error messages identify categories rather than source values.

## Resource policy

V1 fixes these software bounds: 1–20,000 rows, 1–100 features, 1–32 windows,
1–N groups, `W * (N + G) <= 250,000`, the metadata adapter's conservative 16 MiB
canonical plan-document bound, and
`2 * sum_w(n_train[w] + n_test[w]) * P <= 20,000,000` matrix-cell visits.
The visit count describes scheduling, not exact allocations or solver work.
There are at most 64 pipeline fits and 1000 Ridge solver iterations per fit.
No wall-clock or successful-fit guarantee follows from these limits.

Preview is zero-fit and zero-pipeline-construction. Its budget is derived from
the same materialized window/recipe jobs used by execution, not a separate
formula. It includes per-window train/test/unused counts and full partial
coverage, recipe, dependency, binding, and resource declarations.

## Result protocol and integrity bindings

Preview identity is `mlcalc.group_forward_baseline_preview`, version 1. Result
identity is `mlcalc.group_forward_baseline_evaluation`, version 1, protocol
`group-forward-fixed-regression-v1`, coverage mode `partial_predeclared`.
Returned dataclasses and their nested mappings/tuples are immutable; `to_dict()`
returns a detached mutable copy. No estimators, raw feature buffers, leaderboard,
`selected` key, or legacy writer are returned.

The result contains exactly `schema_id`, `schema_version`, `protocol`,
`coverage_mode`, `bindings`, `plan`, `feature_schema`, `metadata_columns`,
`target_name`, `group_name`, `recipes`, `fit_budget`, `coverage`, `predictions`,
`metrics`, `environment`, and `limitations`. Its complete canonical metadata
plan retains the planner's zero-fit count; execution counts occur only in the
evaluator budget. Coverage includes row/group counts and memberships for total,
tested, and untested data, with `untested_predictions_present=false`.

Prediction records occur in original global tested-row order and contain exactly
`row_position`, typed `group`, `window_index`, `observed`,
`prediction_dummy_mean`, and `prediction_ridge_alpha_1`. Metric entries retain
recipe order and contain `recipe`, `pooled`, `per_window`, `per_group`.
Per-window records add `window_index`; per-group records retain exact typed IDs.
Scores include row MAE, RMSE, R², bias, equal-group MAE, arithmetic mean of group
RMSE, and worst-group MAE. Constant or singleton target partitions have null R².
Pooled scores are computed directly from all tested records, not averaged over
windows. Encounter-order integer surrogate codes are used only inside existing
checked grouped metrics and restored exactly afterward.

Bindings are versioned, domain-separated SHA-256 encodings with explicit type,
length, map order, and little-endian float64 bytes. Feature NaN payloads are
canonicalized to zero plus a separate missing mask; observed signed zeros remain
distinct. `plan_sha256` binds the entire canonical metadata document.
`input_sha256` binds normalized features, original numeric dtypes, names/order,
targets, typed groups, index rule, and role declarations. `execution_sha256`
binds both digests, protocol/recipes, resource policy, and observed package and
dependency environment. Execution re-prepares everything and rejects a stale or
malformed expected digest **before pipeline construction**. Available safe local
source observation is mandatory; source-free or unsupported safe-open loaders
reject rather than treating an unavailable marker as a code identity.

An internal check recomputes metrics from retained tested predictions, verifies
all metadata, coverage, row/group/window mapping and fit counts, and rejects any
inconsistency before returning. This is not a saved-result import API. Hashes and
checks do not authenticate source truth, attest executed code, establish causal
validity, or prove any model ran. Identical inputs need not yield bit-identical
numeric results across dependency or hardware versions.
