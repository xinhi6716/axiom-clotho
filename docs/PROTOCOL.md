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
