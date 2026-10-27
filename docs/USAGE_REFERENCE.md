# Usage reference

[English overview](../README.md) · [繁體中文總覽](../README.zh-TW.md)

Detailed CLI, input, evaluation and output contracts. This reference retains the technical material previously on the repository front page.

Compare regression models on held-out groups, then refit locally to predict new rows.

A local-first ML calculator for small materials, semiconductor, and process datasets. Its first engine, **GroupBench**, asks a practical question:

> What is the prediction error on held-out batches?

Read a CSV → hold out entire groups → compare local baselines → inspect an offline report → refit and predict new rows. No account, hosted compute API, model-weight download, or telemetry.

## Version scope

This reference describes the source checkout, not everything in the frozen download.
The following small matrix tracks three post-release fixes; it is not a complete
feature catalogue. `present` means included in the reviewed source snapshot below,
not certified on every platform. Package version alone does not identify a checkout.

<!-- mlcalc-version-scope:start -->
| Field | Value |
| --- | --- |
| documentation_scope | current-source |
| frozen_tag | v0.2.0a2 |
| frozen_source | `20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a` |
| fixes_snapshot | `73dc3f81885a1f8d9acca14198bce7ebe6aad755` |

| Capability | Frozen download | Reviewed source |
| --- | --- | --- |
| regression-output-path-preflight | absent | present |
| validated-csv-header-identity | absent | present |
| reject-bare-string-api-features | absent | present |
<!-- mlcalc-version-scope:end -->

These rows mean early regression output-directory rejection, preservation of
validated CSV column identity across parsers, and rejection of a bare string as a
Python API feature list. [Review the source changes](https://github.com/xinhi6716/ml-super-calculator/compare/v0.2.0a2...73dc3f81885a1f8d9acca14198bce7ebe6aad755)
and [choose the version you mean](../README.md#choose-the-version-you-mean).
Later source commits can coexist with this fixed, reviewed snapshot; they do not
retroactively modify the a2 wheel. This is an original sklearn-based toolkit,
not a pretrained foundation model, full AutoML system or TabPFN equivalent.
Synthetic examples demonstrate workflow, not real-world accuracy.

## Run the demo

Python 3.10+ is declared; the verified development environment is Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0. Other supported versions have not yet been tested in a CI matrix.

From this checkout:

```bash
python -m pip install -e .
mlcalc demo --out runs/demo
```

Open `runs/demo/report.html` in your browser. It is a self-contained local file, with no remote assets or scripts. The example contains 288 **wholly synthetic** process observations in 24 batches.

For an already provisioned environment, run without installation or network access:

```bash
PYTHONPATH=src python -m groupbench demo --out runs/demo
```

Prefer a visible input file? The repository includes [synthetic training rows](../examples/synthetic_materials.csv) and [new process settings](../examples/new_process_settings.csv), both generated from the documented fixture:

```bash
mlcalc run examples/synthetic_materials.csv --target response --group batch --out runs/example-evaluation
mlcalc predict examples/synthetic_materials.csv examples/new_process_settings.csv --target response --group batch --out runs/example-predictions
```

The `groupbench` command is an alias for `mlcalc`. This project is installable from the checkout; no public package-registry release is assumed.

## Distinguish missing values from literal marker strings

The compatibility default `legacy-v1` still merges a real categorical string
`__GROUPBENCH_MISSING__` with missing values. Select
`--categorical-encoding distinct-v2` (Python: `categorical_encoding="distinct-v2"`)
in regression/binary run, prediction, or no-fit planning to use collision-free
internal keys. Existing unknown/infrequent-category compression still applies.
The opt-in has explicit protocol/schema identities; default results and run keys
are unchanged. See [the exact mapping and migration contract](../docs/CATEGORICAL_ENCODING.md).

## Source companion: fixed environment feature comparison

The [environment-heldout comparison](../examples/environment-heldout-comparison/README.md)
fits five fixed local pipelines for two prespecified numeric feature sets and a
shared training-label-shuffle diagnostic. It requires one row per device and no
declared dependency block crossing the boundary. Target-free prediction and
exact-cohort scoring are separate commands. This is a predictive stress workflow,
not causal inference or a causal negative-control test; default Calculator
commands and nested evaluation are unchanged.

## Runnable semiconductor integration

The [invented measurement-to-heldout walkthrough](../examples/semiconductor-heldout-walkthrough/README.md)
connects a trusted local Semiconductor Toolbox checkout to Calculator: I–V
measurements → device features → actual lot holdout audit → five training-only
fits → target-free predictions → separate exact-device scores and a readable interpretation. All fabrication,
batch, wafer and device identities are explicitly invented. The repeated numeric
templates are a software example, not independent experimental or causal evidence.

## Simulation-only causal-design rehearsal

The [causal-design rehearsal](../examples/causal-design-rehearsal/README.md)
separates illustrative pre-setting context, device-level setting, intermediary
and readout under explicit invented equations. It checks total-versus-direct
adjustment roles, compares fixed observational regressions with separate paired
intervention oracles, and refuses rank-deficient adjustment. This is a local
design smoke test, not a calibrated device model or real-data causal estimator.

<a id="explicit-binary-passfail-workflow-unreleased"></a>

## Explicit binary pass/fail workflow (prerelease)

For a numeric 0/1 target, explicitly select the new binary namespace. Existing regression commands remain regression even for a 0/1 target.

```bash
mlcalc binary plan run examples/synthetic_binary_process.csv --target fail --group lot
mlcalc binary run examples/synthetic_binary_process.csv --target fail --group lot --out runs/binary
mlcalc binary predict examples/synthetic_binary_process.csv examples/new_binary_process.csv --target fail --group lot --out runs/binary-new
```

Fixed group-balanced Brier selects among prior Dummy, regularized Logistic, and bounded Extra Trees. Both labels must occur in every training partition; single-class held-out groups remain included. Raw probabilities, class counts, false negatives/positives and adverse group results are retained. Prediction labels use p1 >= 0.5, not an optimized manufacturing threshold. Binary output directories must be fresh; no overwrite, calibration, model loading or automatic physical decisions. The fixture is wholly invented, not real semiconductor performance.

See [binary protocol, precise metrics, schemas and limits](../docs/BINARY_PROTOCOL.md) and [Python example](../examples/binary_python_api.py). Default no-fit plans show 105 evaluation or 21 prediction pipeline fits, with no models or output files created. Training/new-row data stay local.

## Use your own regression data

Your CSV should have:

- A finite numeric target, such as a measured property.
- A group identifier, such as batch, specimen, wafer, or source experiment. Choose the unit that must not cross the evaluation boundary.
- Numeric and/or categorical input features available **before** the target is measured. Missing feature values are supported.

```bash
mlcalc run measurements.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/mobility
```

The target and group columns are excluded from features automatically. Use `--features` or `--features-file` as an explicit allowlist to exclude specimen IDs, post-outcome measurements, operator notes, or other leakage-prone columns. If neither is given, all other columns become features; the software cannot infer which of them are scientifically valid.

Group labels are preserved as text when reading CSV, including leading zeros. NUL (U+0000) characters in group identifiers are rejected in both the DataFrame API and CSV workflows, before splitting or fitting, because string conversions and grouping can otherwise merge distinct IDs. This is an input-validation tightening: previously accepted NUL-containing DataFrame group IDs must be explicitly repaired in the source, not silently stripped or escaped by the calculator. Group validation never strips or escapes identifier characters. Blank or missing groups, non-finite targets, duplicate headers, unsupported datetime/complex features, and impossible split requests fail with an actionable error. Rows with missing targets are **not** silently dropped.

CSV files must use UTF-8 (an optional UTF-8 BOM is supported), with unique, non-blank column names. Each non-blank record must have exactly as many fields as the header: malformed short/long records are rejected **before fitting**, rather than silently padded or shifted into the wrong columns. Represent missing values with explicit empty fields, such as `001,,2`, and quote values containing commas or newlines. Empty lines and lines containing only ASCII spaces/tabs outside quoted values are ignored. Structure errors report physical line numbers and field counts, not cell contents.

The CLI uses commas between fields and a dot for decimal numbers, such as `1.5`; it does not detect semicolon delimiters or convert decimal commas. Adjust the source export settings before importing such data. A quoted feature value such as `"1,5"` remains categorical text, not the number `1.5`. Before fitting, inspect `feature_schema` → `dtype` in the existing `plan run` or `plan predict` output to check that intended numeric features were read as numeric.

NUL characters (U+0000) are rejected anywhere in a CSV, including headers and quoted fields, before pandas can truncate them and change group identifiers, features or targets. Diagnostics identify only the physical source line, not its contents. Physical lines include newlines inside quoted fields; they are not CSV record numbers. This input check is shared by regression and binary `run`, `predict` and `plan` commands.

Categorical feature values containing NUL (U+0000) are also rejected by the DataFrame API, under both categorical encodings, before planning, fitting, input-support checks, or training/prediction fingerprints. This explicit validation tightening prevents pandas source-category and hashing operations from merging distinct values even when one-hot encoding keeps them separate. No characters are stripped or escaped automatically: repair the source values explicitly. Only observed values are checked; an unused NUL-containing pandas category does not invalidate otherwise clean rows. Numeric features and ordinary strings keep their existing behavior. CSV continues to reject NUL anywhere in the file.

CSV input now requires a regular local file and is limited to **64 MiB
(67,108,864 bytes) per file**, including the UTF-8 BOM, line endings, quoted
multiline text and ignored blank lines. Training and prediction files have
separate caps. This is an explicit compatibility tightening: an otherwise valid
file with at most 20,000 rows can still exceed the byte cap and is rejected,
not truncated or sampled. No optional override bypasses the cap. The DataFrame
API is unchanged and does not inherit this file-byte limit.

Both pre-open metadata and the opened descriptor are checked. Local symlinks
and hardlinks to regular files remain accepted; directories, FIFOs and device
files are rejected. Nonblocking open protects against a FIFO replacement on
platforms supporting O_NONBLOCK. The same open stream is hashed and parsed,
with bounded reads and at most one excess byte read to detect growth beyond
the cap. No model fitting or output writing starts when either input fails.
These are input byte/type guards, not a hard peak-memory, wall-clock or training
budget, an atomic filesystem snapshot, or protection from all concurrent file
changes or network-mounted storage delays. Only Linux has been verified; use
trusted, idle local files and OS limits where needed.

The CLI checks a bounded input snapshot before pandas type inference, preserving the original quoted text and group identifiers. Python's CSV parser field-size limit applies to both headers and data cells (131,072 characters per field under its default setting); a parser-limit or quoting failure returns an actionable error. This per-field guard is specific to CSV input, not the DataFrame API, and is not a hard memory limit for the whole file.

### Preview the work before fitting

Check inputs and see the exact intended split sizes, candidate hyperparameters,
and fit-count budget before spending time on model fitting:

```bash
mlcalc plan run examples/synthetic_materials.csv --target response --group batch
mlcalc plan predict examples/synthetic_materials.csv examples/new_process_settings.csv --target response --group batch
```

`plan run` and `plan predict` accept the matching command's target, group,
`--features` / `--features-file`, baselines, seed, selection objective and split
options. They print **JSON to stdout only**, with no `--out` or `--overwrite`
option. They do not fit estimators or preprocessing, load saved models, create
an output directory, or write predictions/results. Input validation is shared
with execution, including malformed CSV rejection, target/group exclusion,
feature allowlists, new-row schema/types and sufficient groups for every split.
Group identifiers and row values are omitted from the plan.

The schedule uses the same actual split seeds, candidate grid and per-family
refits consumed by `run` / `predict`, not a separate estimate of the defaults.
With the default six candidates, three families and three inner folds:

- `plan run` with five outer folds: **90 inner-selection fits + 15 outer-family
  refits = 105 pipeline fits**.
- `plan predict`: **18 inner-selection fits + 3 full-data family refits = 21
  pipeline fits**.

Dummy is fitted like each requested family. The `selected` entry reuses its
winning family's refit; it causes no extra fit. One pipeline fit includes
preprocessing and one estimator, not one count per tree or transform. Custom
baseline subsets and split counts change the reported budget automatically.

These counts describe intended work **if every fit succeeds**. They are not an
execution-time estimate, a hard compute/memory cap, performance evidence or a
guarantee that training will succeed. Planning does not validate a future output
path, identify scientifically invalid features or select a winner. Run again
with the same inputs/options to execute; changed files or software may change
the plan. If redirecting stdout yourself, choose a separate file: your shell can
truncate a destination before the CLI starts.

The Python equivalents return plain dictionaries without fitting or writing:

```python
from groupbench import plan_benchmark, plan_prediction

plan = plan_benchmark(training, target="response", group="batch")
print(plan["fit_budget"])
# With feature-only new_rows:
plan = plan_prediction(training, new_rows, target="response", group="batch")
```

### Predict new rows

First evaluate the procedure with `run`. Then supply a second CSV containing **exactly the feature columns**, without target or group columns:

```bash
mlcalc predict measurements.csv new_process_settings.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/new-predictions
```

This performs grouped inner selection on all labeled rows, refits locally, and writes `predictions.csv`, `manifest.json`, and `checksums.json`. It does **not** claim new test-set performance. New rows retain a zero-based `row_position`; predictions are in target units. Unknown categorical values are handled, but there are no uncertainty intervals or out-of-distribution guarantees.

No executable model file is loaded or saved. The Python API can retain the fitted estimator in memory; the CLI refits when called again.

### Check whether new inputs were observed in training

`plan predict` includes an `input_support` summary before any fitting. `predict`
and Python `fit_predict` put the same summary in `manifest.json` /
`forecast.manifest["input_support"]`. It counts, per feature and across new rows:

- Numeric values strictly below/above the finite, non-missing training minimum/maximum. Values equal to a boundary are not flagged.
- Non-missing categorical values absent from the normalized training source values.
- Missing prediction values, even when training also had missing values.
- Non-missing prediction values in features that were entirely missing in training. Their training support is explicitly unavailable; numeric range counts are `null`, not a measured zero.

Only selected training features define these checks, before imputation. New rows
never expand their own reference range or category set. Counts can overlap; the
summary's `rows_with_any_flag` counts each affected row once. Reports include
feature names and aggregate counts, not training extrema, category inventories,
raw values or row identifiers. Existing `row_position` and prediction CSV
columns are unchanged. No inputs are removed or modified because of these flags.

These are simple individual-feature checks, **not calibrated uncertainty, an
out-of-distribution test, or evidence that a prediction is reliable**. A row can
have no flags yet combine features in a way never observed in training. Counts
describe source values, not the encoder's learned/infrequent category buckets;
missing values and a literal `__GROUPBENCH_MISSING__` string remain distinct in
this diagnostic even though preprocessing can encode them alike. Existing
imputation, unknown-category handling, models, seeds, splits, selection and fit
counts are unchanged. Use independent evaluation and domain judgment.

Prediction manifests and prediction plans now have schema version **1.1**;
evaluation manifests, summaries and plans remain **1.0**. No files were added to
either bundle. Strict consumers should explicitly accept prediction schema 1.1
and its additive `input_support` field; older 1.0 files have no support summary
and must not be treated as having zero flags. See the
[diagnostic semantics and schema migration note](../docs/INPUT_SUPPORT.md).

### Reuse a declared feature schema

Both `run` and `predict` accept `--features-file PATH` instead of `--features`.
It reads the version-1 `feature_schema.json` emitted by Semiconductor Toolbox's
[synthetic RRAM measurement-to-features example](https://github.com/xinhi6716/semiconductor-machine-learning-toolbox/blob/b485b68021160cd0eb1124084464b355c24a7b23/examples/measurement-to-features/README.md). Only the names in
`model_feature_allowlist`, in their declared order, become predictors. IDs,
counts, conditions, row keys, units and other metadata are **not** added as
features. An explicit `--target` and `--group` are still required; the JSON does
not infer them or supply labels. For a separately reviewed, labeled table:

```bash
mlcalc run labeled_devices.csv \
  --target separately_measured_endpoint --group split_group \
  --features-file feature_schema.json --out runs/device-evaluation

mlcalc predict labeled_devices.csv new_predictors_only.csv \
  --target separately_measured_endpoint --group split_group \
  --features-file feature_schema.json --out runs/device-predictions
```

These commands are workflow templates, not commands that train on the upstream
example as-is. That example has **no target** and only six invented devices in
two invented batches, insufficient for this nested evaluation. Do not manufacture
a scientific endpoint from its HRS/LRS resistances or their derived ratio. A real
endpoint, feature timing, dependency grouping and adequate sample size require
separate review. A valid declaration is **not proof against scientific leakage**.

The input is strict UTF-8 JSON, at most **64 KiB**, with exactly these fields:
`schema_version` (integer `1`), `row_key`, `model_feature_allowlist`,
`metadata_columns`, `target_columns`, `feature_units`, and `warnings`.
The four name lists have at most 100 unique, non-blank, unpadded strings;
`row_key` and the feature allowlist must be nonempty. Predictor, metadata and
target roles must be disjoint; row keys must be metadata. `feature_units` maps
exactly the allowed feature names to non-blank strings. `warnings` is a list of
non-blank strings. Units and warnings are declarations, not physically validated
or acted upon. Unknown fields/versions, duplicate keys or names, non-finite JSON
constants and malformed/empty declarations fail without fitting or writing.
Names are matched exactly against the training CSV, and the usual target/group
exclusion remains enforced. Prediction rows must still contain **exactly** the
selected features. The existing output manifest records these selected names in
`config.features`; no separate feature-file hash or authentication is claimed.

This is a bounded local JSON reader, not a general manifest/model loader: no
referenced files, executable imports, URLs or network resources are loaded.
Like CSV source paths, an explicitly supplied symlink to a regular local file
is followed. Directories, FIFOs and other non-regular files are rejected; the
feature schema is also protected from input/output path aliases. Use trusted,
idle local files. This is not a hostile-filesystem sandbox or an atomic read
guarantee, and platform-specific race behavior is not universally verified.
The separate `verify` command below has stricter POSIX **no-symlink** semantics.

## Python API

Pass `features` as a collection of complete column names, not a comma-separated string. For a column literally named `xy`, use `features=["xy"]` (or `("xy",)`), **not** `features="xy"`; `features=["x", "y"]` selects two different columns. `features=None` uses all columns other than the target and group, so supply an explicit allowlist when other columns are IDs or post-outcome information. The CLI's `--features x,y` syntax is different from the Python API.

Current source from [73dc3f8](https://github.com/xinhi6716/ml-super-calculator/commit/73dc3f81885a1f8d9acca14198bce7ebe6aad755) rejects bare strings and bytes before planning or model work in regression and binary workflows. The frozen `v0.2.0a2` wheel predates this guard and can interpret a string as separate characters. Use a name collection on either version; see [download versus source](../README.md#choose-the-version-you-mean).

`target` and `group` must name two distinct existing columns using strings.
Non-string column selectors (including Python/NumPy integers and tuples) are
rejected before planning or fitting; explicitly rename those source columns
first. This tightens accidental earlier support that could fail only when
serializing a result after fitting. Names are not coerced, stripped, or escaped.
This does not restrict numeric **values** in the group column, and unrelated
metadata columns can still be excluded with an explicit feature allowlist.


```python
from groupbench import benchmark, fit_predict, make_materials_fixture

training = make_materials_fixture()
result = benchmark(
    training,
    target="response",
    group="batch",
    outer_splits=5,
    inner_splits=3,
    random_state=7,
)
print(result.leaderboard)
result.write("runs/python-evaluation")

# Independent synthetic rows only for demonstrating the input shape.
new_rows = make_materials_fixture(random_state=19).drop(
    columns=["response", "batch"]
).iloc[:4]
forecast = fit_predict(
    training, new_rows, target="response", group="batch", random_state=7
)
print(forecast.predictions)
forecast.write("runs/python-predictions")
```

A familiar estimator interface is also available:

```python
from groupbench import GroupBenchRegressor

X = training.drop(columns=["response", "batch"])
model = GroupBenchRegressor(inner_splits=3, random_state=7)
model.fit(X, training["response"], groups=training["batch"])
predictions = model.predict(new_rows)
print(model.best_candidate_, model.selection_results_)
```

`X` must be a named pandas DataFrame. All arrays are aligned **positionally**, not by pandas index labels. `predict` accepts reordered columns but rejects missing/extra columns and incompatible numeric/categorical types. This is a sklearn-like estimator with `get_params`, `set_params`, and `clone`; it is not claimed to pass every general-purpose sklearn estimator check or metadata-routing workflow.

## How the evaluation is split and scored

1. **Outer folds hold out whole groups.** Every row receives exactly one held-out prediction per model entry. Folds balance row counts approximately, using group sizes but never target values.
2. **Inner folds select models without seeing the outer test groups.** The `selected` entry evaluates the entire selection procedure. Each family also gets its own inner-selected configuration.
3. **Preprocessing is fitted inside each training partition.** Numeric median imputation/scaling and categorical imputation/one-hot vocabulary never learn from the corresponding validation rows.
4. **Scores explain their weighting.** Group MAE gives each group equal weight. Row MAE/RMSE pool all observations. Constant-target or one-row R² is `null`, not a misleading perfect score.
5. **The split is inspectable.** Exact outer/inner group memberships, candidate scores, chosen settings, per-row predictions, versions, seeds, and data hashes are included.
6. **Duplicates are surfaced, not silently removed.** Identical normalized feature/target rows spanning different groups create a diagnostic warning. It cannot detect near-duplicates or all forms of leakage.

Do not pick a new model or tune features by repeatedly looking at the outer leaderboard and then cite that same leaderboard as unbiased evidence. Use a new independent test set or a fresh preregistered evaluation for those changes.

Metric arithmetic uses float64 and fails explicitly on overflow, underflow, or
invalid intermediate operations, even when the input targets and predictions
are finite. This conservative check can reject a scale whose final mathematical
score would be representable; it does not silently emit infinite/NaN scores,
round a tiny squared error down to zero, or call an underflowed target variance
constant. Rescale targets and predictions consistently to practical units and
rerun. A metric failure aborts selection/evaluation with `ValueError` (CLI exit
status 2), rather than dropping a group or candidate. Exactly constant targets
and one-row groups still have `null` R² when the other metric arithmetic is safe.
The ordinary score formulas, row/group weighting, models, and split protocol
are unchanged; this is not arbitrary-precision arithmetic or a guarantee that
estimator training itself supports extreme scales.

## Baselines and resource budget

| Family | Candidates | Settings |
| --- | ---: | --- |
| Dummy mean | 1 | Training-target mean |
| Ridge | 3 | `alpha = 0.1, 1, 10`, LSQR solver |
| Extra Trees | 2 | 64 trees; maximum depth 6 or unlimited; minimum leaf size 2 |

Defaults: 5 outer folds × (6 candidates × 3 inner folds + 3 family refits) = **105 small fits**. A standalone `fit` or `predict` command uses 21 fits. Families share folds and preprocessing, but **do not receive equal compute budgets**. This is a transparent baseline portfolio, not a fair-compute competition or exhaustive hyperparameter search.

All model fits run with tree `n_jobs=1` and threadpool limits of one. The MVP admits at most 20,000 rows and 100 input features; one-hot encoding uses at most 64 categories per feature. CSV reads stop at 20,001 rows before rejecting oversize files. These are size guards, not a hard time/memory quota or a promise that the upper bound is fast on every machine. Begin with a small dataset. You can use `--baselines dummy,ridge`, fewer folds, or an OS-level time limit for tighter budgets.

### Why numeric units still matter

The [A versus nA example](../examples/numeric-unit-equivalence/README.md) compares
the same invented current feature in two units, with and without the existing
train-only preprocessing. Its twelve fixed fits, raw predictions, and unit-gap
checks are a teaching example—not a real-device benchmark or a new product fix.
It does not change the default fit budgets above.

## Outputs

An evaluation run writes five local files:

- `report.html`: readable comparison, worst-group errors, split audit, and limitations.
- `summary.json`: pooled, per-group, and per-fold metrics for every entry.
- `predictions.csv`: original row positions, group labels, observed targets, fold assignments, and predictions.
- `manifest.json`: protocol/configuration, complete nested split audit, candidate selections, schema, versions, hashes, and diagnostics.
- `checksums.json`: SHA-256 checksums of the four other files.

The run key hashes normalized ordered data and protocol configuration; it is not a cryptographic proof of execution. Versions are separately recorded. Repeated runs with identical input order, configuration, and dependency versions are deterministic in the tested environment. Timestamps and elapsed time naturally differ; bitwise agreement across platforms or dependency versions is not guaranteed.

For CLI CSV inputs, `source_file_sha256` and, in prediction mode, `prediction_source_file_sha256` hash the exact byte stream read for validation, including any UTF-8 BOM, original line endings, and ignored blank lines. Each input is opened once and hashed incrementally before decoding; the CLI does not reopen its pathname to compute provenance after parsing. A later file replacement cannot change the recorded hash. This is not a filesystem-atomic snapshot or lock: avoid modifying inputs while they are being read, and note that the training and new-row files are read separately. The DataFrame API's optional source hashes remain caller-supplied provenance, not independently verified file hashes.

Existing files are preserved unless `--overwrite` is explicit. Atomic per-file no-clobber writes protect against a racing writer; the multi-file bundle is not a filesystem transaction. An interrupted final write may leave a partial bundle, which must be checked against `checksums.json`. The CLI refuses to overwrite either input through a direct, symlink, or hardlink output alias. Prediction mode refuses an evaluation output directory even with overwrite enabled, so an old report cannot be mistaken for the new predictions.

### Verify an existing bundle offline

```bash
mlcalc verify runs/example-evaluation
mlcalc verify runs/example-predictions
# The groupbench alias and python -m groupbench support the same command.
```

`verify` only reads an existing directory. It does not fit a model, open the report in a browser, parse payload contents, load serialized models, read source datasets, access the network, or create/repair files. Exit status **0** means every expected file's observed bytes match `checksums.json`; **2** means verification or command usage failed. A missing file, changed byte, malformed/duplicate checksum record, unsupported file set, or unsafe file path fails verification.

The existing checksum format is exactly `{"algorithm": "sha256", "files": {"filename": "64 lowercase hexadecimal characters"}}`. Evaluation bundles must list exactly `report.html`, `summary.json`, `manifest.json`, and `predictions.csv`. Prediction bundles must list exactly `manifest.json` and `predictions.csv`, with no stale `report.html` or `summary.json` entries in the directory. Other unlisted files are not opened or verified. This command does not validate the manifest, CSV, report, or model semantics and does not recognize arbitrary checksum manifests.

Verification requires POSIX directory-relative file access and no-follow opens. It fails closed on platforms without these primitives. Use a direct relative or absolute directory path without `..`, a network-style `//` root, or symlink components; checksum/payload entries must be regular files, not symlinks, directories, or FIFOs. Metadata is capped at **16 KiB** and each of at most four payloads at **64 MiB**, with **64 KiB** streaming reads. Larger bundles are rejected rather than partially verified. These are byte/read bounds, not a wall-clock deadline or a sandbox; use trusted local storage and OS limits where needed.

**Integrity is not authenticity or scientific correctness.** Someone who changes both a payload and its checksum can pass this check. It does not prove a run occurred, establish an unbiased scientific protocol, validate predictions, or assess the checksum author's trustworthiness. Verify an idle directory: detected file changes fail, but sequential reads and metadata checks are not a lock, an atomic directory snapshot, or a guarantee against all concurrent modifications. Keep an independently trusted copy of checksums when provenance matters.

**Privacy:** reports include group IDs, target values, predictions, and feature names. Keep private datasets and run directories out of public commits. CSV values are faithfully preserved, not spreadsheet-formula sanitized: import group IDs as text rather than opening an untrusted CSV with formula evaluation enabled. Do not upload a report without reviewing its contents.

## Limits to understand before trusting a result

- Group disjointness protects only the supplied grouping axis. Related compositions, overlapping specimens, time leakage, or feature engineering performed on all data can still leak information.
- Holding out a combined `batch × specimen` label does not automatically isolate both crossed factors. Use a scientifically appropriate split strategy; multi-axis and temporal holdouts are not implemented yet.
- Default commands perform regression; binary classification requires the explicit `binary` namespace described above. Foundation-model training, calibrated uncertainty, causal inference, scientific-unit validation, extrapolation guarantees, and production monitoring are not included.
- Sparse small groups can produce unstable rankings. Fold variability is not a confidence interval, and a single seed is not a robustness study.
- The synthetic fixture is not a validated semiconductor/materials simulator. No private research data, unpublished manuscript, or review content is embedded.
- `fit_predict` produces predictions, not a performance estimate. Validate on appropriate independent data before consequential use.

See [the evaluation protocol](../docs/PROTOCOL.md), [roadmap](../ROADMAP.md), and [contribution guide](../CONTRIBUTING.md).

## Design inspiration, not copied modeling

TabPFN is a useful example of a clear `fit`/`predict` interface, compact quickstarts, and approachable local-model packaging. We use those **product ergonomics** as inspiration only. This repository contains no TabPFN code, weights, checkpoint downloads, or API client, and claims no comparable foundation-model capability or state-of-the-art result.

Current official licensing was checked on 2026-10-23: the [TabPFN source repository](https://github.com/PriorLabs/TabPFN) uses [Apache 2.0 for its code](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE); model weights have separate terms, including non-commercial terms for recent releases. Those terms do not become this project's license. See the [reference note](../docs/REFERENCES.md) and [official quickstart](https://docs.priorlabs.ai/quickstart).

## Develop

```bash
python -m unittest discover -s tests -v
python examples/python_api.py
```

The suite uses Python's standard `unittest`; no test-runner download is needed. For an uninstalled checkout, prefix either command with `PYTHONPATH=src`.

MIT for original project code and the synthetic fixture generator. Dependencies retain their own licenses. The roadmap is a long-term maintenance direction, not a promise that its future features already exist.

### Compare saved validation budgets locally

For a strictly descriptive two-method × two-budget comparison, with matched
repeats aggregated within declared units, equal-unit weighting, strict
wins/ties/harms and a complete adverse-case ledger, see
[Saved budget comparison](../docs/SAVED_BUDGET_COMPARISON.md).
This optional utility does not retrain models or verify provenance/independence.

For already saved observations and prediction-interval bounds, see
[Saved interval evaluation](../docs/SAVED_INTERVAL_EVALUATION.md). Its separate
stdout-only Python/module CLI reports empirical coverage, width and (with a
fixed declared alpha) standard interval score, both row-weighted and equal-group,
on one explicit complete cohort. It rejects failed/missing rows, preserves exact
numeric inputs and group identities, and marks rounded output explicitly.
It does not generate intervals, calibrate models or guarantee future coverage;
existing training and saved bundles are unchanged.

### Optional saved prediction consistency

For an existing regression prediction bundle, `mlcalc validate-prediction-bundle
RUN_DIR` checks supported manifest/CSV count, position and input-support count
relationships in the exact checksummed bytes. It does not fit or load a model.
This is separate from the unchanged byte-only `mlcalc verify`. Supported schema
combinations, smaller parser bounds and explicitly unchecked metadata are in
[the prediction bundle contract](../docs/PREDICTION_BUNDLE_SCHEMA.md). A pass is not
scientific validation, authenticity, or an atomic filesystem snapshot.
