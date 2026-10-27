# Evaluation protocol v1

## Estimand

The default experiment estimates the performance of a fixed, inner-selected regression procedure on rows from groups absent from its training set. A group should represent the domain's independence boundary. It does not estimate future-time performance unless the chosen grouping and population justify that interpretation.

## Read the split scope

Both ordinary regression and binary evaluation hold out whole values of the
single `--group` column. Choose that boundary before looking at scores:

| Question | What the ordinary report supports |
| --- | --- |
| New groups along the supplied axis? | Each outer test group's ID is absent from its training fold. Zero ID overlap does not establish physical independence or exclude other leakage. |
| New domain or site? | Only if the supplied grouping boundary and population justify that interpretation. A batch holdout can still share sites; other domain/site axes are not separately audited. A column named `domain` is not evidence by itself. |
| Random-row performance? | No separate random-row comparison is run. The seed allocates whole groups, not individual rows within a group. Do not compare these scores to row-random results as though they answer the same question. |

Current-source reports put this scope beside the model comparison. If recorded
row and unique-group counts are equal, a nonblocking note explains that every
group has one row: there are no repeated-observation groups for the split to
protect. Independent single-row groups are allowed; the note does not diagnose
bad IDs, reject data, or claim this algorithm equals a row-random splitter.
Neither report wording nor a successful run proves domain independence,
future-time transfer or causal effects. Frozen a3 reports are unchanged.

Before fitting, run `mlcalc plan run` (or `mlcalc binary plan run`) with the same
input, target, group and feature arguments. Missing/blank group IDs, a missing
group column, or selecting the group as a feature already fail validation.
Every actual outer training partition also needs enough distinct groups for
the inner folds; too few groups cause an error, never a row-random fallback.
For example, two domains cannot support two outer folds plus two inner folds.
Do not relabel related rows or silently substitute a weaker boundary to pass.

For a separately declared fixed new environment, see the existing
[environment-heldout comparison companion](../examples/environment-heldout-comparison/README.md).
It requires one row per device, an explicit held-out environment and no shared
declared dependency block across training/future. Its target-free prediction
and separate scoring are not the ordinary nested-CV command or causal evidence.

## Algorithm

1. Validate columns, finite one-dimensional targets, and non-missing group identifiers. Preserve row order; exclude target and group from features.
2. Factorize group IDs in first-occurrence order. Seed-shuffle group IDs, stably sort by descending group size, then assign each group to a currently lightest fold, breaking equal-load ties with the seeded generator. Use no target information.
3. For every outer fold, run the same splitter on its training groups, with seed `outer_seed + fold + 1`. Fail before training if any requested inner split is impossible.
4. For each candidate and inner fold, instantiate a fresh preprocessing pipeline and model. Numeric columns receive train-fitted median imputation and standardization. Categorical columns receive a fixed missing marker and train-fitted sparse one-hot encoding, with at most 64 categories per feature and unknown-category handling.
5. Aggregate each candidate's inner out-of-fold predictions. Default selection minimizes the arithmetic mean of group-level MAEs. Optional `objective="mae"` minimizes pooled row MAE. Fixed candidate ordering resolves exact ties.
6. Select the best configuration inside each family; refit these family representatives on the whole outer training fold. Select the overall winner by the same inner scores, not outer errors.
7. Predict the untouched outer test fold. The `selected` prediction is the winning family's prediction. Every row is evaluated once per entry.
8. Aggregate all outer held-out predictions into pooled, group-balanced, per-group, and per-fold metrics. Report all choices and group memberships.

Categorical feature values containing NUL (U+0000) are unsupported under both encoding protocols. Shared DataFrame validation rejects these observed values before planning, fitting, input-support diagnostics, or training/prediction fingerprinting, because pandas uniqueness/hashing can merge them even when one-hot encoding separates them. This explicitly tightens the previously accepted DataFrame input contract; callers must repair source values rather than relying on silent stripping or escaping. Unused pandas category metadata is not a feature value and is not rejected when observed values are clean. Ordinary inputs, model protocols, and artifact schema identities are unchanged. The CSV reader already rejects NUL anywhere in a file.

Numeric/categorical column typing is a schema decision and not target-derived. All learned values, including medians, scaling statistics, vocabularies, category-frequency grouping, and model parameters, stay inside each fit. The duplicate diagnostic is post-evaluation and does not modify selection or data.

`GroupBenchRegressor.fit` publishes learned state only after all inner selection
and family refits succeed. If a fit fails or is interrupted before completion,
an already-fitted estimator retains its previous models, feature schema,
selection results and split audit; a fresh estimator remains unfitted. A later
successful fit replaces that state. Exceptions still propagate without silent
candidate skipping or retry. This does not roll back constructor parameters
changed by the caller, provide concurrent-fit/predict safety, or make artifact
writes transactional.

## Column selectors

Regression and binary DataFrame workflows require `target` and `group` to be
string names of distinct existing columns. Non-string selectors are rejected
before planning, splitting or fitting, rather than accepting a pandas lookup
that later fails artifact JSON serialization. This explicitly tightens earlier
accidental acceptance; rename non-string source columns before use. There is no
automatic label coercion or change to accepted string-name identity, feature
selection, fingerprints, model protocols, or artifact schema versions. The rule
concerns column names, not the numeric/string group values described below.

## Group identifiers

The Python API identifies numeric groups by their exact stored values, not rounded display text or NumPy scalar comparison promotion. Equal values across integer/float types share one group; distinct values remain distinct. Ordinary float64 IDs retain decimal labels. Fractional values beyond float64 precision use exact hexadecimal-ratio labels; integers beyond Python's decimal-conversion limit use hexadecimal labels.

String IDs remain literal: `"1"`, `"01"` and `"1.0"` are different groups. NUL (U+0000) is prohibited anywhere in a group ID, including embedded, trailing, or repeated NULs. NumPy string conversion can trim trailing NULs, and pandas grouping can merge embedded-NUL strings, so the DataFrame API now rejects these identifiers before planning, splitting, fitting, or training-data fingerprinting. This validation tightening does not preserve previously accepted NUL-containing group input: explicitly repair the source identifiers and verify their intended grouping, rather than stripping characters automatically. Other Unicode strings and the numeric normalization contract are unchanged; group validation does not strip or escape identifiers. Mixed string/numeric IDs are rejected. CSV CLI group columns are read as literal strings, so these numeric Python API normalization rules do not reinterpret CSV group labels.

## Metrics

- **MAE:** mean absolute prediction error over rows.
- **RMSE:** square root of mean squared prediction error over rows.
- **Group MAE:** arithmetic mean of each group's MAE, giving groups equal weight.
- **Group RMSE:** arithmetic mean of each group's RMSE. It is intentionally not the square root of group-balanced MSE.
- **Worst-group MAE:** the largest group MAE, descriptive and sensitive to group size.
- **R²:** one minus squared-error sum divided by the target total sum of squares; undefined for fewer than two rows or constant targets, reported as JSON `null`.
- **Bias:** mean prediction minus observation.

Fold metrics are descriptive. No bootstrap interval, standard error, confidence interval, or independence claim is derived from their variance.

### Direct scoring of saved arrays

`groupbench.metrics.grouped_metrics` and
`groupbench.binary.metrics.grouped_binary_metrics` enforce the same group-ID
validation as the high-level workflows, before NumPy dtype promotion or pandas
grouping. They preserve distinct exact numeric IDs and reject mixed text/numeric,
missing, blank, boolean, nonfinite, NUL-containing, or non-vector group inputs.
This closes earlier accidental acceptance at the direct scoring boundary; it
does not change metric formulas, equal-group weighting, or already validated
benchmark inputs.

For example, raw IDs `[2**60, 2**60 + 1, 0.5]` define three groups, not two.
With all-zero targets and predictions `[0, 10, 0]`, group MAE is `10/3` and
worst-group MAE is `10`, rather than `2.5` and `5`. Preserve IDs in a raw Python
list, object array/Series, or literal strings as appropriate. If a caller has
already converted those integers to a float array, the first two stored IDs
are equal: scoring cannot discover or recover the lost original identities.

Inputs are positional, including pandas Series; index labels are not joined or
used to align targets, predictions, and groups. Callers must supply matching row
order and the intended complete cohort. Direct scoring neither validates
held-out fold provenance nor establishes independence of the declared groups.

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
