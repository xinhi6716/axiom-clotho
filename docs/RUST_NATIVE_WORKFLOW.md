# Native Rust modelling workflow

This is a runnable Rust-only path through CSV loading, fitting, saved model
specifications, prediction and held-out grouped evaluation. It launches **no
Python process** and makes **no runtime network calls**. It is additive: it does
not yet replace every existing Clotho feature. See the
[migration inventory](RUST_MIGRATION_STATUS.md) before claiming completion.

The regression examples below are complemented by [native binary classification
and explicit mixed-feature preprocessing](RUST_BINARY_AND_MIXED.md). That guide
covers logistic/majority models, missing/categorical inputs and the v3 model schema.
For bounded candidate selection with independent inner and outer group folds,
see [native nested numeric regression comparison](RUST_NESTED_SELECTION.md).

## Build and run

Rust 1.70 or newer is required. The core is dependency-free. The CLI uses the
MIT/Apache-2.0 licensed csv, serde and serde_json ecosystem, pinned in Cargo.lock.
Initial Cargo dependency acquisition uses crates.io; once fetched, the commands
below work offline. No Python environment is needed.

```sh
cargo fetch --manifest-path rust/Cargo.toml --locked
cargo build --manifest-path rust/Cargo.toml --release --locked --offline
cargo test --manifest-path rust/Cargo.toml --workspace --locked --offline
rust/target/release/clotho --help

rust/target/release/clotho evaluate \
  --csv rust/clotho-cli/examples/synthetic_process.csv \
  --target synthetic_response \
  --features anneal_temperature_c,film_thickness_nm \
  --group wafer --folds 4 --model ridge --alpha 1 \
  --out rust-evaluation-example

rust/target/release/clotho fit \
  --csv rust/clotho-cli/examples/synthetic_process.csv \
  --target synthetic_response \
  --features anneal_temperature_c,film_thickness_nm \
  --model ridge --alpha 1 --out rust-model-example.json

rust/target/release/clotho predict \
  --artifact rust-model-example.json \
  --csv rust/clotho-cli/examples/synthetic_process.csv \
  --out rust-prediction-example.csv
```

These eight rows are **synthetic arithmetic examples, not experimental material
or semiconductor measurements**. Predictions on the training CSV demonstrate
I/O only; they are not held-out quality evidence. Use the separate grouped
`evaluate` command for held-out estimates. Scientific use still requires a
justified group definition, appropriate independent data and strong baselines.

## Regression models

- `--model mean`: training-target mean; the mandatory low-complexity baseline.
- `--model knn --k 5`: uniform nearest-neighbor regression, training-only
  maximum-absolute feature scaling. See [kNN details](RUST_KNN_BASELINE.md).
- `--model ridge --alpha 1`: training-only population standardization, intercept
  not penalized, objective SSE + alpha × sum of squared standardized coefficients.
  Uses column-pivoted QR and conditioning checks; alpha=0 requests unregularized
  least squares and can correctly fail for rank-deficient designs. Finite numeric
  inputs only. Constant columns require positive regularization.

Each evaluation fold fits its own scaler and estimator solely on its training
rows. No full-data normalization or target-mean fitting occurs before splitting.
Hyperparameters are fixed by the caller, not selected by looking at test results.
Repeatedly choosing settings from this report is model selection; use separate
inner/outer data or an untouched final evaluation instead of reporting a tuned
result as unbiased. Native nested tuning is not yet implemented.

## Data contract

- Explicit comma-separated `--features` prevents automatic inclusion of IDs or
  targets. Target and group columns cannot be features. Other potential leakage
  columns must still be excluded by the researcher.
- Headers must be unique, nonempty, have no surrounding whitespace and no NUL.
  The feature option supports simple comma-separated column names, not names
  containing commas. CSV quoting, CRLF and UTF-8 are parsed by the csv crate.
- The default input path requires finite, present numeric values. Explicit
  `--preprocess mixed` enables training-only median/one-hot preprocessing as
  described in the extension guide. No units inference or target conversion occurs.
- Prediction aligns the saved feature names with CSV headers; column reordering
  and unrelated extra columns are accepted. Required missing columns fail.
- At most 64 MiB per input file, 20,000 rows and 100 selected features. CLI
  evaluation permits 2–20 folds; the group count must be at least the fold count.
  The core fold function has separate 20,000-row / 100-fold limits. These limits
  bound allocation, not a speed guarantee. k must fit every training fold.
- Group values are exact strings, including significant whitespace; missing,
  whitespace-only or NUL-containing group IDs fail. Do not mix naming conventions
  for the same wafer or source paper, or the group protection can be invalidated.
- Predictors/metrics reject unsupported numeric ranges rather than writing
  nonfinite JSON. Ordinary binary64 rounding and documented extreme-value
  precision limits still apply.

## Evaluation protocol and outputs

Groups are ordered largest-first, with lexical group-ID ties, then assigned to
the least-loaded fold with fold-index ties. Every row is held out exactly once;
no group appears in both train and test of a fold. This is a **new deterministic
Rust protocol**, not the original NumPy-randomized Python split protocol.

The new directory contains:

- `predictions.csv`: original zero-based row index, fold, true target, held-out
  prediction. It excludes original feature values but still contains private
  target data.
- `metrics.json`: selected features/model/settings, exact split rule, row/group
  counts, pooled row-weighted MAE/RMSE/R², per-fold and per-group metrics, mean
  group MAE, mean group RMSE, and worst group MAE. The two means give each group
  equal weight; mean group RMSE is not the same as pooled RMSE. R² is null for
  exactly constant observed targets, including singleton groups.
- `COMPLETE`: written last. If a filesystem failure leaves a directory without
  this marker, treat that output as incomplete. Training and serialization finish
  before output creation, but publication is not a multi-file atomic transaction.

All three commands refuse existing output paths, including symlinks. They never
replace an earlier study. Unix files are created with permissions 0600 and study
directories 0700. On Windows permissions follow the parent directory's ACL;
private ACL configuration is the user's responsibility. Unix mode bits do not
replace encryption, disk backups, or control of the containing directory.

## Saved model specifications

`fit` writes schema `clotho.numeric.v3`, feature order, target, estimator settings
and **the full selected training features and targets**. Mixed mode also stores
raw cells and categorical roles. Keep this file private. Numeric v1/v2 files
remain readable through the explicit legacy compatibility paths.
`predict` validates and reconstructs the estimator from these rows. This is a
portable, inspectable refit specification, **not a serialized coefficient-only
model**; prediction startup includes fitting and needs the same protocol/version.
No pickle, executable object graph, Python code or external model download is
accepted. Unknown schema/fields, invalid shape and invalid numeric values fail.
Float-roundtrip JSON parsing preserves saved binary64 values on reload.

Legacy `.pkl` bundles, Python saved studies and their schema versions are not
accepted. Do not rename them to JSON or assume compatibility. These native
artifacts have no cryptographic integrity manifest yet; keep trusted copies and
record the source input hashes/commit separately for scientific reproduction.

## Current limits

No random forests, mixed/classification nested search, uncertainty intervals,
forward-window evaluation, measurement ingestion,
HTML report, legacy saved-artifact importer or source-manifest verification is
implemented in this native path. The existing Python path remains available so
those features are not silently deleted during migration. Passing software tests
is not proof of material prediction accuracy, full migration, or industrial
readiness. No new material benchmark or external scientific validation is claimed.

## Verification record (2026-11-07)

- Complete Rust workspace: **95 integration tests passed** on Linux with Rust
  **1.70.0 and 1.99.1** (11 CLI, 84 core). No unit/doc tests are defined.
- Stable `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`
  passed; formatting checks of all newly changed Rust source/test files passed.
- Release build and all nine example paths (evaluate, fit, predict for each of
  mean/ridge/kNN) passed. The eight-row synthetic example produced grouped RMSE
  0.669662 (mean), 0.096315 (ridge alpha=1), 0.396761 (kNN k=3). These values are
  synthetic smoke-test observations, not a semiconductor benchmark.
- Independent review found and corrected unbounded fold allocation, private Unix
  output permissions, equal-group underflow, and non-exact default JSON float
  parsing. Regression tests cover these cases. The review also checked held-out
  target/scaler fitting and target/group exclusion.
- Python behavior parity, Python test suite, Windows/macOS, hosted CI, real-data
  prediction accuracy, packaging release and full migration remain unverified.
