# Standalone Rust numeric kNN regression

This dependency-free Rust library fits numeric training rows and predicts
continuous targets. The [native Rust CLI](RUST_NATIVE_WORKFLOW.md) uses it for
`fit --model knn`, `predict` from a native artifact, and grouped `evaluate`.
Native artifacts contain the selected training rows and targets; `predict`
reconstructs the estimator from them. Keep these files private. This path does
not reuse Python saved studies or establish scientific model accuracy.

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
one. For each query/training pair, subtract in the original units and then
scale the difference: `(q_j - x_ij) / s_j`. If direct subtraction overflows,
try `q_j / s_j - x_ij / s_j` instead. Accumulate the coordinates with `hypot`.
Average the k nearest targets with equal weights. Equal *computed* distances
select lower training-row indices first. This is
maximum-absolute scaling, **not** standard deviation or range scaling. It can be
sensitive to training outliers and feature offsets. Reordering tied training rows
can change predictions. Duplicates retain their individual weight.

For example, train on `x,y` pairs `(7,0)`, `(7.000000000000001,1)`, and `(12,2)`
with `k=1`. Query in the order `7.000000000000001`, then `7`: the native
fit/reload regression expects predictions `1`, then `0`. Scaling the coordinates
separately first would collapse the first two training inputs to the same
binary64 value and incorrectly predict `0` for both. Subtracting first avoids
this false tie, but accumulated distances still round to binary64.

Inputs must be nonempty, rectangular, numeric and finite. Each table is limited
to 20,000 rows and 100 features; k is between 1 and the training row count.
Prediction requires exactly the fitted feature width. Callers must keep feature
order and units identical, exclude targets/IDs from features and handle missing
values/categorical features separately. All-zero training features use scale 1;
changing those query features affects distances, but does not certify support.
Extremely large queries relative to tiny fitted scales may return NumericOverflow,
never a silent infinity. Errors contain no input values.

The selected finite binary64 targets are summed as exact integer multiples of
the smallest positive subnormal. Divide by k and round the mean to binary64
once, to nearest with ties to even. Reordering the same selected neighbors does
not change this mean. Exact cancellation returns positive zero; a negative
nonzero mean that rounds to zero returns negative zero.

The existing target-normalization range check still rejects a selected nonzero
target if dividing it by the largest selected target magnitude would round it
to zero. This check is an acceptance rule, not how the mean is computed. The
exact mean neither changes which neighbors are selected nor recovers decimal
precision lost at input. Prediction returns no partial output on error.

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
per query. The core library has no serializer; the CLI handles saved training
specifications as described above. No accelerator, approximate index, intervals
or real-data benchmark is included. Large requests within the validation ceiling can
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
