# Evaluation protocol v1

## Estimand

The default experiment estimates the performance of a fixed, inner-selected regression procedure on rows from groups absent from its training set. A group should represent the domain's independence boundary. It does not estimate future-time performance unless the chosen grouping and population justify that interpretation.

## Algorithm

1. Validate columns, finite one-dimensional targets, and non-missing group identifiers. Preserve row order; exclude target and group from features.
2. Factorize group IDs in first-occurrence order. Seed-shuffle group IDs, stably sort by descending group size, then assign each group to a currently lightest fold, breaking equal-load ties with the seeded generator. Use no target information.
3. For every outer fold, run the same splitter on its training groups, with seed `outer_seed + fold + 1`. Fail before training if any requested inner split is impossible.
4. For each candidate and inner fold, instantiate a fresh preprocessing pipeline and model. Numeric columns receive train-fitted median imputation and standardization. Categorical columns receive a fixed missing marker and train-fitted sparse one-hot encoding, with at most 64 categories per feature and unknown-category handling.
5. Aggregate each candidate's inner out-of-fold predictions. Default selection minimizes the arithmetic mean of group-level MAEs. Optional `objective="mae"` minimizes pooled row MAE. Fixed candidate ordering resolves exact ties.
6. Select the best configuration inside each family; refit these family representatives on the whole outer training fold. Select the overall winner by the same inner scores, not outer errors.
7. Predict the untouched outer test fold. The `selected` prediction is the winning family's prediction. Every row is evaluated once per entry.
8. Aggregate all outer held-out predictions into pooled, group-balanced, per-group, and per-fold metrics. Report all choices and group memberships.

Numeric/categorical column typing is a schema decision and not target-derived. All learned values, including medians, scaling statistics, vocabularies, category-frequency grouping, and model parameters, stay inside each fit. The duplicate diagnostic is post-evaluation and does not modify selection or data.

`GroupBenchRegressor.fit` publishes learned state only after all inner selection
and family refits succeed. If a fit fails or is interrupted before completion,
an already-fitted estimator retains its previous models, feature schema,
selection results and split audit; a fresh estimator remains unfitted. A later
successful fit replaces that state. Exceptions still propagate without silent
candidate skipping or retry. This does not roll back constructor parameters
changed by the caller, provide concurrent-fit/predict safety, or make artifact
writes transactional.

## Group identifiers

The Python API identifies numeric groups by their exact stored values, not rounded display text or NumPy scalar comparison promotion. Equal values across integer/float types share one group; distinct values remain distinct. Ordinary float64 IDs retain decimal labels. Fractional values beyond float64 precision use exact hexadecimal-ratio labels; integers beyond Python's decimal-conversion limit use hexadecimal labels.

String IDs remain literal: `"1"`, `"01"` and `"1.0"` are different groups. NUL (U+0000) is prohibited anywhere in a group ID, including embedded, trailing, or repeated NULs. NumPy string conversion can trim trailing NULs, and pandas grouping can merge embedded-NUL strings, so the DataFrame API now rejects these identifiers before planning, splitting, fitting, or training-data fingerprinting. This validation tightening does not preserve previously accepted NUL-containing group input: explicitly repair the source identifiers and verify their intended grouping, rather than stripping characters automatically. Other Unicode strings and the numeric normalization contract are unchanged; non-group DataFrame features are not newly sanitized. Mixed string/numeric IDs are rejected. CSV CLI group columns are read as literal strings, so these numeric Python API normalization rules do not reinterpret CSV group labels.

## Metrics

- **MAE:** mean absolute prediction error over rows.
- **RMSE:** square root of mean squared prediction error over rows.
- **Group MAE:** arithmetic mean of each group's MAE, giving groups equal weight.
- **Group RMSE:** arithmetic mean of each group's RMSE. It is intentionally not the square root of group-balanced MSE.
- **Worst-group MAE:** the largest group MAE, descriptive and sensitive to group size.
- **R²:** one minus squared-error sum divided by the target total sum of squares; undefined for fewer than two rows or constant targets, reported as JSON `null`.
- **Bias:** mean prediction minus observation.

Fold metrics are descriptive. No bootstrap interval, standard error, confidence interval, or independence claim is derived from their variance.

## Prediction mode

`fit_predict` uses only the training data for grouped inner selection and preprocessing. It refits selected family representatives on all labeled rows, then predicts the separate feature-only frame. No target is required or accepted as an extra prediction column. Its inner scores are not an unbiased test estimate. Use the nested `benchmark` result separately.

## Reproducibility contract

Given the same normalized ordered data, parameters, and dependency versions, the tested CPU implementation produces the same assignments, selections, and predictions. Pandas indices are not alignment keys: inputs are positional and predictions carry zero-based original row positions. Changing row order can change deterministic tie handling. Cross-version or cross-hardware bitwise equality is not promised.

The manifest contains raw group IDs and schema, but not raw feature values. Prediction CSVs contain observed values in evaluation mode. A checksum is an integrity check against accidental modification, not an attestation, authentication signature, or proof that a claimed run occurred.

## Known boundaries

A correct implementation cannot rescue an incorrect scientific protocol. Precomputed global normalization, duplicate/near-duplicate measurements, crossed dependence, a group-identifying proxy, or post-outcome predictors can invalidate evaluation. Exact feature+target copies across group labels trigger a diagnostic; near-duplicates and semantic leakage remain the user's responsibility. A real benchmark publication requires documented datasets and rights, independently justified grouping, multiple seeds where appropriate, realistic deployment conditions, resource reporting, and independent scientific review.

## Separate explicit binary protocol

The original regression contracts above are unchanged. Numeric 0/1 classification is a separately selected additive workflow with its own schemas and fixed group-Brier objective; see [BINARY_PROTOCOL.md](BINARY_PROTOCOL.md). Regression targets containing 0/1 do not trigger automatic classification.

## Opt-in categorical encoding v2

The compatibility default remains the v1 procedure above, including its known
collision between a real `__GROUPBENCH_MISSING__` category and a missing value.
Explicit `categorical_encoding="distinct-v2"` / `--categorical-encoding distinct-v2`
uses injective marker-prefix escaping, with separate protocol identities and
compatible schema minor versions. It does not change grouping, numeric transforms,
model grids, objectives, fit counts, or intentional unknown/infrequent compression.
See [CATEGORICAL_ENCODING.md](CATEGORICAL_ENCODING.md) for exact semantics,
legacy compatibility, migration, and limitations.
