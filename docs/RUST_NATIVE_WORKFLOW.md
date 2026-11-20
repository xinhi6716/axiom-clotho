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
For fit-free comparison of saved declared series and optional exact query-byte
binding, see [baseline-relative scoring](RUST_BASELINE_SKILL.md).

## Build and run

Rust 1.70 or newer is required. The core is dependency-free. The CLI uses the
MIT/Apache-2.0 licensed csv, serde, serde_json and RustCrypto sha2 ecosystem,
pinned in Cargo.lock. SHA-256 uses portable software compression; the lockfile
also retains an MSRV-compatible libc 0.2.172 for platform-specific transitive
dependencies. Regenerating the lockfile requires rechecking the Rust 1.70 minimum.
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
- At most 64 MiB per input file and 100 selected features. Query CSVs, mixed
  training, kNN/classification training, evaluation and nested selection retain
  the 20,000-row limit. Opt-in strict-numeric `fit --model mean|ridge --large-training true` additionally
  accepts up to 32,768 training rows subject to the budgets below. CLI
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


## Bounded narrow-table training extension

With explicit `--large-training true`, strict-numeric mean/ridge training can exceed 20,000 rows up to 32,768 rows.
Only this extension is subject to `n * p <= 1,000,000`; ridge additionally
requires `n * p * p <= 10,000,000`, where `p` is the selected feature count.
These are allocation/work proxies, not measured bytes, flops or a runtime
promise. Previously supported tables of at most 20,000 rows and 100 features
retain their existing behavior, including wider ridge tables. The extension
therefore deliberately has a narrower feature budget rather than globally
changing the legacy limit. The augmented ridge solve also includes its intercept
and regularization rows, so actual work is larger than the proxy count.

Default CLI parsing retains the original 20,000-row limit and error message.
The opt-in flag must be exactly `true`, on `fit --model mean|ridge` with neither
`--preprocess` nor `--categorical`; other uses fail. Core mean/ridge training uses the same
training-only budgets; prediction still rejects more than 20,000 query rows.
Mixed preprocessing, kNN, logistic/majority, fold construction, evaluate, nested
selection, prepared labels, saved comparisons and scalar report pooling keep
their existing caps. This is not a larger-fold evaluation feature.

Saved artifacts still contain the raw training data rather than fitted
coefficients. Native `predict` refits those rows, so loading a narrow extended
artifact uses the same training budget even for a small query. Extended artifacts
are checked against the existing 64 MiB reload limit before writing; a failure
creates no artifact. Legacy input byte caps are unchanged. Repeated prediction incurs
repeated fit cost; chunking queries does not remove that cost.

No training rows are sampled, truncated or split to evade limits. Feature order,
scaling, solver, conditioning checks and target behavior are unchanged. This
capacity extension does not establish generalization, physical independence,
Lin-data feasibility, or access to held-out data. Resource measurements must
name the synthetic shape, feature count, algorithm and environment rather than
claiming a general performance guarantee.


Opt-in raw CSV limits are checked **before** String cell materialization: 16 MiB
bytes, at most 128 raw fields per record and 1,000,000 raw cells total including
headers and unused columns. An allocation-free lexical preflight enforces strict
UTF-8/BOM, LF/CRLF, escaped-quote syntax, no blank records, and the 32,768-data-row
limit. Quoted commas/newlines are parsed as field content, not extra cells/rows.
This tighter raw input contract applies even below 20,000 rows when the flag is
present. Ordinary calls without the flag keep their existing CSV behavior.
Raw cell/byte budgets complement, rather than replace, selected-feature training
budgets. They bound the new allocation path but do not guarantee a fixed RSS or
run time across machines. Stored artifacts retain the existing schema; older
versions may reject their larger training tables on reload.

### Opt-in Mean inference summary

Strict numeric Mean alone can export `clotho.mean_inference.v1` without raw
training rows. Both commands require explicit opt-in; existing defaults and
legacy training-containing artifacts are unchanged:

```sh
clotho fit --csv train.csv --features x,z --target y --model mean \
  --mean-inference true --out NEW-mean.json
clotho predict --artifact NEW-mean.json --mean-inference true \
  --csv query.csv --out NEW-predictions.csv
```

The strict named JSON object has exactly `schema_version`, `model` (`mean`),
`features`, `target`, and finite binary64 `mean`. Feature order is declared by
1–100 unique names excluding the target; names are nonblank, without surrounding
whitespace or control characters, and at most 256 UTF-8 bytes each. The artifact
is at most 64 KiB. Unknown/duplicate fields, positional arrays, unsupported
versions/models and nonzero numeric tokens rounding to zero are rejected.
Loading reconstructs a validated Mean evaluator, not training arrays, and does
not refit. Queries retain the normal numeric feature alignment, finite checks
and 20,000-row limit even though Mean ignores feature values. The existing
`--row-id` / `--saved-contract` v2 prediction output also works in this mode.
`--threshold`, mixed preprocessing, Ridge and other estimators are unsupported.
`--large-training true` may be combined at fit time under its existing limits.

Serialization roundtrips the fitted binary64 mean; it does not promise decimal
exactness, anonymization, confidentiality, producer authenticity or scientific
validity. A single-row training mean reveals that target; feature/target names
may also be sensitive. Outputs use the existing private, create-new-only writer:
validation and serialization complete before opening the output. This does not
promise transactional publication on filesystem failure (a partial new file can
remain on write failure). No raw-row artifact is automatically converted,
replaced, deleted, uploaded or made safe to publish. Ridge export is deliberately
not implemented: its numerically stable state includes first-row feature anchors.

### Numeric CSV underflow validation

Selected numeric feature and target cells now reject a decimal nonzero value
that parses to binary64 zero, for example `1e-999` or `2e-324`. Previously these
could silently become zero-valued features, targets, or binary labels. This is
an intentional validation tightening for fit, prediction, evaluation and
selection, including Mean inference queries and numeric mixed-mode columns.
Failures occur before creating the requested output. Mixed artifact numeric
strings are checked by the same parser when reloaded.

Actual zero (including signed zero and zero with an exponent) and representable
subnormal values remain accepted. Ordinary decimal rounding to a nonzero
binary64 value is unchanged. Explicit categorical literals such as `1e-999`,
mixed-mode empty cells, group IDs, and unused columns retain their existing
meaning. Errors do not echo the offending private cell. This check concerns
input conversion, not arithmetic underflow inside a fitted model. It does not
retroactively recover precision already lost in legacy numeric JSON artifacts,
or change CLI option parsing or the separate domain-extraction workflows.

Imported `compare-saved` v1 and v2 JSON series also reject nonzero numeric
literals that round to binary64 zero (for example `1e-999` or `2e-324`) before
scoring any of the observed, model or baseline inputs. Otherwise an imported
nonzero error could silently become a perfect baseline. This validation occurs
before output creation, on the original bounded JSON bytes, without changing
strict/duplicate field checks or v2 query digest requirements. True zero,
representable subnormals, and numeric-looking text inside quoted metadata/IDs
remain valid. Ordinary rounding to a nonzero binary64 value is unchanged.
This is input conversion validation, not arbitrary-precision scoring or a
change to grouped/pooling mathematics. Previously rounded source artifacts
cannot have their lost precision reconstructed by this check.

Legacy training-containing model JSON loaded by `predict` now uses the same
quote-aware numeric loss-to-zero check as saved-series import. A nonzero JSON
number such as `1e-999` is rejected before serde can turn it into zero, including
training feature/target arrays and `alpha`. This checks every numeric token in
the artifact, even a parameter a particular estimator would not consume; it
does not silently reinterpret authored values. Validation completes before any
prediction output is created. True zeros, representable subnormals, and quoted
numeric-looking feature names remain supported. Existing schema, duplicate-field,
shape and finite-value checks still run. Mixed-mode strings are not JSON numeric
tokens and retain their existing numeric/categorical parsing rules. This does
not recover precision already lost in old files, change ordinary nonzero
rounding, or alter estimator arithmetic and legacy refitting behavior.

kNN distances subtract original feature coordinates before dividing by the
training-only max-absolute scale. This prevents separately rounded normalized
values from inventing a tie between distinct adjacent values (for example 7
and the next representable value above 7, with scale 12). Opposite extreme
values use a scaled-subtraction fallback when direct subtraction overflows.
Nonzero differences lost to zero or nonfinite distances are rejected. Actual
distance ties still use training-row order. Training/query limits, feature
scaling, uniform averaging, and artifact format are unchanged; the in-memory
model retains raw coordinates rather than pre-rounded normalized coordinates.
This improves pairwise distance fidelity, not arbitrary-precision distance
ordering when accumulation itself rounds nearby distances to the same value.

kNN neighbor-target averaging returns the exact arithmetic mean of the
selected, accepted finite binary64 targets, correctly rounded once to binary64
(nearest, ties to even). Fixed-size integer accumulators sum the inputs in
units of the minimum positive subnormal before exact division by the neighbor
count. This preserves small residuals through cancellation without
intermediate floating-point overflow and does not depend on the order of an
unchanged selected neighbor set. An exact zero mean returns positive zero; a
negative nonzero mean that rounds to zero returns negative zero.

The target-normalization acceptance guard still rejects nonzero targets that
would normalize to zero. This change does not alter neighbor selection or
row-order tie breaking, prediction bounds, artifact formats, or Mean/Ridge
estimators. Correct rounding refers to the actual binary64 inputs, not to
decimal values before ingestion or to arbitrary-precision distance ordering.
