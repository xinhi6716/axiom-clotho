# Standalone Rust numeric kNN regression

This dependency-free library baseline can fit numeric training rows and predict
continuous targets entirely in Rust. It is additive and is **not** connected to
Clotho's Python CLI, saved-study format, grouped model selection or native adapter.
It does not establish real semiconductor accuracy or complete the Rust migration.

## Run

```sh
cargo test --manifest-path rust/Cargo.toml --offline --locked
cargo run --manifest-path rust/Cargo.toml -p clotho-core --example knn_process --offline --locked
```

Use `clotho_core::knn::KnnRegressor::fit(&features, &targets, k)` and then
`model.predict(&query_features)`. Rows are `Vec<f64>`; the fitted model owns its
training data. The example is explicitly synthetic, not an experimental dataset.

## Algorithm and contract

For feature j, fit s_j = max_i |x_ij| on training data only, replacing zero with
one. Euclidean distances are computed between x_j/s_j and q_j/s_j using `hypot`
for stable accumulation. Average the k nearest training targets with equal
weights. Exact distance ties select lower training-row indices first. This is
maximum-absolute scaling, **not** standard deviation or range scaling. It can be
sensitive to training outliers and feature offsets. Reordering tied training rows
can change predictions. Duplicates retain their individual weight.

Inputs must be nonempty, rectangular, numeric and finite. Each table is limited
to 20,000 rows and 100 features; k is between 1 and the training row count.
Prediction requires exactly the fitted feature width. Callers must keep feature
order and units identical, exclude targets/IDs from features and handle missing
values/categorical features separately. All-zero training features use scale 1;
changing those query features affects distances, but does not certify support.
Extremely large queries relative to tiny fitted scales may return NumericOverflow,
never a silent infinity. Errors contain no input values.

Target averaging normalizes by the largest neighbor-target magnitude before
summing with Neumaier compensation; final normalized roundoff is bounded to [-1, 1]. This avoids overflow
for finite targets but does not promise exact arithmetic or preserved subnormal
precision. Prediction validates all rows and returns no partial output on error.

## Validation and scientific use

Unit tests cover hand-computed predictions, training-only scaling, batch
independence, deterministic ties, constant features, invalid shapes/k/nonfinite
values, extreme finite targets and explicit overflow rejection. These are
software checks, not model-quality benchmarks. The synthetic example checks 1.5.

For material/process data, hold out entire appropriate groups (device, wafer,
batch, composition family or source paper) before fitting. Tune k only inside
training folds. Fit scaling separately in each fold. Compare against a training
mean and stronger baselines on identical data/splits; retain predictions, seeds,
units, source provenance and uncertainty. Do not interpret nearest neighbors as
causal identification or use extrapolated predictions for safety-critical control.

Exact search costs O(n_query * (n_train * n_features + n_train log n_train)) time,
O(n_train * n_features) fitted storage and O(n_train + n_features) working memory
per query. No accelerator, approximate index, intervals, serialization, CLI or
real-data benchmark is included. Large requests within the validation ceiling can
still be expensive. No network requests or external runtime dependencies occur.

Scaling that collapses any nonzero training/query feature or selected target to
zero is explicitly rejected with NumericOverflow (a numeric-range error covering
both overflow and underflow). This conservative rule avoids silent nearest-neighbor
changes for extreme dynamic ranges. Ordinary binary64 rounding still applies.

## Recorded software verification

On 2026-11-07, the complete Rust workspace passed 47 integration tests (including
9 kNN tests) under Rust 1.70.0 and 1.99.1 on Linux. The synthetic example passed
under both toolchains; stable Clippy with `--all-targets -- -D warnings` and
format checking of the new Rust files passed. Independent review identified
cancellation and scaling-underflow cases; both received fixes and regressions.
This record does not claim Python-suite, hosted CI, Windows/macOS, real-data
accuracy, statistical superiority or independent scientific reproduction.
