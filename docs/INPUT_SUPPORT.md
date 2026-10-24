# New-row input-support diagnostics v1

`plan_prediction` / `mlcalc plan predict` and `fit_predict` / `mlcalc predict`
include the same aggregate `input_support` object. The former does not fit any
estimator or preprocessing; the latter reports it alongside successful full-data
refit predictions. `GroupBenchRegressor.predict` still returns only the prediction
array. This feature does not add an estimator method, model state or model export.

## Reference and exact meaning

The reference is **all supplied, validated training feature rows**, after the
existing stateless normalization and explicit feature allowlist, but **before
imputation, scaling or encoding**. Targets, group labels, excluded columns,
predictions and prediction-batch values do not define reference values. This is
not a held-out evaluation diagnostic and is not used to select any candidate.

- Numeric features use their finite, non-missing minimum and maximum. Counts use
  strict `<` and `>` comparisons, so equality is not flagged, including constant
  features. Comparisons do not subtract endpoints, avoiding range-overflow
  arithmetic. Existing input validation still rejects infinite numeric features.
- Categorical features compare exact, case-sensitive normalized strings with
  non-missing source values. Existing normalization turns booleans and other
  supported non-numeric source values into strings. There is no whitespace
  trimming, fuzzy matching or new type conversion policy.
- Missing prediction values are counted separately and are never called unseen
  categories or out-of-range values. They are counted even if missing values
  occurred in training. Existing imputation still applies.
- An all-missing training feature has `training_support` set to
  `unavailable_all_missing`. Its new non-missing values are counted under
  `prediction_non_missing_without_training_values_count`. Numeric below/above
  counts are `null`, because no numerical range exists. Every non-missing value
  of an all-missing categorical source is also unseen. These overlapping counts
  are intentional, not extra rows.
- Categorical support is **not encoder vocabulary support**. An observed source
  category remains observed even if the encoder groups it into an infrequent
  bucket under its existing 64-category cap. A literal string
  `__GROUPBENCH_MISSING__` is an ordinary source category here. If it was absent
  from non-missing training values it is flagged as unseen, even when missing
  training values cause the existing imputer/encoder to represent that string.
  This does not change or solve the preprocessing marker collision.

The shared existing prediction validation still rejects missing/extra/duplicate
columns and incompatible types before fitting. Reordered columns are accepted;
entirely missing new columns retain the existing accommodation to the training
type. Indices are not alignment keys. Prediction CSVs retain exactly
`row_position,prediction`, with zero-based positional row numbers and no new
flags or identifiers. There is no special row-ID column inference.

## Output fields and bounds

The nested diagnostic object has its own `schema_version: "1.0"`, with:

- `reference: "full_training_features_before_imputation"` and
  `action: "diagnostic_only_no_input_changes"`.
- `n_training_rows`, `n_prediction_rows` and `rows_with_any_flag`.
- `row_flag_counts`: counts of new rows with at least one `missing_value`,
  `numeric_outside_training_range`, `unseen_category`, or
  `non_missing_value_in_all_missing_training_feature`. A row can count in more
  than one category; `rows_with_any_flag` counts its union only once.
- `features`: one record per selected feature, in training feature order, with
  `name`, `kind` (`numeric` or `categorical`), `training_missing_count`,
  `training_support` (`observed_non_missing_values` or
  `unavailable_all_missing`), `prediction_missing_count` and
  `prediction_non_missing_without_training_values_count`.
  Numeric records also have `below_training_min_count` and
  `above_training_max_count`; categorical records have `unseen_category_count`.

Existing limits of 20,000 rows per input and 100 selected features bound the
work. The diagnostic returns only one record per feature and fixed aggregate
counts, never a list of row-level flags, training extrema, categories, raw values
or row/group identifiers. Category membership is computed locally from at most
20,000 training source values per feature and not persisted. This is not a new
hard memory/time cap. Feature names and counts can still reveal context; review
before sharing. Other manifest fields already include group identifiers.

## Artifact compatibility and migration

- Successful **prediction manifests** (`mode: "fit_predict"`) advance from outer
  schema `"1.0"` to `"1.1"`, adding `input_support` and a limitation statement.
- **Prediction plans** (`kind: "execution_plan", mode: "predict"`) likewise
  advance to `"1.1"` with that additive object and limitation statement.
- Evaluation manifests, summaries and evaluation plans remain `"1.0"`; their
  output fields, HTML report, evaluation protocol and run keys are unchanged.
- Prediction bundles still contain exactly `predictions.csv`, `manifest.json`
  and `checksums.json`. Checksums still list exactly the two payload files, so
  no verifier allowlist expansion is needed. Old and new bundles use the same
  byte-integrity verification path, which does not parse manifest semantics.
- Strict consumers should branch on the outer mode/schema and explicitly
  accept `"1.1"` when consuming predictions/plans. Existing prediction fields
  keep their meaning. A missing `input_support` in old `"1.0"` data means
  **unavailable**, not zero flags. Do not modify archived bundles merely to
  label them `"1.1"`; rerun with the original inputs/options if diagnostics are
  needed. Changing a manifest's bytes requires recomputing its checksum.

## Limitations

These descriptive single-feature checks do not estimate prediction error,
uncertainty, calibration, confidence or out-of-distribution probability. They
cannot detect novel feature combinations, correlations, scientifically invalid
features, missingness mechanisms, unit mistakes, causal changes or all forms of
distribution shift. No flags is not evidence of reliability or safety. Flags
do not prove a prediction is wrong. No row is removed, prediction rejected or
model adjusted merely for these conditions; existing validation and non-finite
prediction failures remain in force. Independent evaluation and domain judgment
are still required, particularly before consequential use.

## Source diagnostics versus categorical encoding

These diagnostics remain source-based under both `legacy-v1` and explicit
`distinct-v2`, with no change to their schema or counts. In legacy-v1, a real
`__GROUPBENCH_MISSING__` source category can collide with missing after imputation,
even when the source diagnostic calls that literal unseen. Diagnostics do not
repair a fitted encoding. Use the [explicit v2 option](CATEGORICAL_ENCODING.md)
to separate their internal keys; existing unknown/infrequent buckets can still
merge encoded representations intentionally.
