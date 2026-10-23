# ML Super Calculator · ML 超級計算器

**From a local CSV to an honest model comparison and usable predictions.**

A local-first ML calculator for small materials, semiconductor, and process datasets. Its first engine, **GroupBench**, asks a practical question:

> Will this model still work on a batch it has never seen?

Read a CSV → hold out entire groups → compare local baselines → inspect an offline report → refit and predict new rows. No account, hosted compute API, model-weight download, or telemetry.

中文：這不是另一個只會呼叫雲端模型的外殼。第一版先把小型材料／製程表格資料的「分組驗證、模型比較、在本機重新訓練與預測」做扎實。依批次、試片或實驗來源切開驗證，讓同一批資料不會同時出現在訓練與測試裡。這是長期產品的可用起點，不是已完成的基礎模型。

**Status:** v0.1.0 alpha. Regression only. An original sklearn-based selection/evaluation toolkit, not a pretrained foundation model, full AutoML system, or TabPFN equivalent. Synthetic examples demonstrate the workflow, not real-world accuracy.

## Try it in one minute

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

Prefer a visible input file? The repository includes [synthetic training rows](examples/synthetic_materials.csv) and [new process settings](examples/new_process_settings.csv), both generated from the documented fixture:

```bash
mlcalc run examples/synthetic_materials.csv --target response --group batch --out runs/example-evaluation
mlcalc predict examples/synthetic_materials.csv examples/new_process_settings.csv --target response --group batch --out runs/example-predictions
```

The `groupbench` command is an alias for `mlcalc`. This project is installable from the checkout; no public package-registry release is assumed.

## Use your own data

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

The target and group columns are excluded from features automatically. Use `--features` as an explicit allowlist to exclude specimen IDs, post-outcome measurements, operator notes, or other leakage-prone columns. If omitted, all other columns become features; the software cannot infer which of them are scientifically valid.

Group labels are preserved as text when reading CSV, including leading zeros. Blank or missing groups, non-finite targets, duplicate headers, unsupported datetime/complex features, and impossible split requests fail with an actionable error. Rows with missing targets are **not** silently dropped.

CSV files must use UTF-8 (an optional UTF-8 BOM is supported), with unique, non-blank column names. Each non-blank record must have exactly as many fields as the header: malformed short/long records are rejected **before fitting**, rather than silently padded or shifted into the wrong columns. Represent missing values with explicit empty fields, such as `001,,2`, and quote values containing commas or newlines. Empty lines and lines containing only ASCII spaces/tabs outside quoted values are ignored. Structure errors report physical line numbers and field counts, not cell contents.

The CLI checks a bounded input snapshot before pandas type inference, preserving the original quoted text and group identifiers. Python's CSV parser field-size limit applies to both headers and data cells (131,072 characters per field under its default setting); a parser-limit or quoting failure returns an actionable error. This per-field guard is specific to CSV input, not the DataFrame API, and is not a hard memory limit for the whole file.

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

## Python API

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

## What makes the evaluation trustworthy?

1. **Outer folds hold out whole groups.** Every row receives exactly one held-out prediction per model entry. Folds balance row counts approximately, using group sizes but never target values.
2. **Inner folds select models without seeing the outer test groups.** The `selected` entry evaluates the entire selection procedure. Each family also gets its own inner-selected configuration.
3. **Preprocessing is fitted inside each training partition.** Numeric median imputation/scaling and categorical imputation/one-hot vocabulary never learn from the corresponding validation rows.
4. **Scores explain their weighting.** Group MAE gives each group equal weight. Row MAE/RMSE pool all observations. Constant-target or one-row R² is `null`, not a misleading perfect score.
5. **The split is inspectable.** Exact outer/inner group memberships, candidate scores, chosen settings, per-row predictions, versions, seeds, and data hashes are included.
6. **Duplicates are surfaced, not silently removed.** Identical normalized feature/target rows spanning different groups create a diagnostic warning. It cannot detect near-duplicates or all forms of leakage.

Do not pick a new model or tune features by repeatedly looking at the outer leaderboard and then cite that same leaderboard as unbiased evidence. Use a new independent test set or a fresh preregistered evaluation for those changes.

## Baselines and resource budget

| Family | Candidates | Settings |
| --- | ---: | --- |
| Dummy mean | 1 | Training-target mean |
| Ridge | 3 | `alpha = 0.1, 1, 10`, LSQR solver |
| Extra Trees | 2 | 64 trees; maximum depth 6 or unlimited; minimum leaf size 2 |

Defaults: 5 outer folds × (6 candidates × 3 inner folds + 3 family refits) = **105 small fits**. A standalone `fit` or `predict` command uses 21 fits. Families share folds and preprocessing, but **do not receive equal compute budgets**. This is a transparent baseline portfolio, not a fair-compute competition or exhaustive hyperparameter search.

All model fits run with tree `n_jobs=1` and threadpool limits of one. The MVP admits at most 20,000 rows and 100 input features; one-hot encoding uses at most 64 categories per feature. CSV reads stop at 20,001 rows before rejecting oversize files. These are size guards, not a hard time/memory quota or a promise that the upper bound is fast on every machine. Begin with a small dataset. You can use `--baselines dummy,ridge`, fewer folds, or an OS-level time limit for tighter budgets.

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
mlcalc verify runs/example
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
- No classification, foundation-model training, calibrated uncertainty, causal inference, scientific-unit validation, extrapolation guarantee, or production monitoring is included.
- Sparse small groups can produce unstable rankings. Fold variability is not a confidence interval, and a single seed is not a robustness study.
- The synthetic fixture is not a validated semiconductor/materials simulator. No private research data, unpublished manuscript, or review content is embedded.
- `fit_predict` produces predictions, not a performance estimate. Validate on appropriate independent data before consequential use.

See [the evaluation protocol](docs/PROTOCOL.md), [roadmap](ROADMAP.md), and [contribution guide](CONTRIBUTING.md).

## Design inspiration, not copied modeling

TabPFN is a useful example of a clear `fit`/`predict` interface, compact quickstarts, and approachable local-model packaging. We use those **product ergonomics** as inspiration only. This repository contains no TabPFN code, weights, checkpoint downloads, or API client, and claims no comparable foundation-model capability or state-of-the-art result.

Current official licensing was checked on 2026-10-23: the [TabPFN source repository](https://github.com/PriorLabs/TabPFN) uses [Apache 2.0 for its code](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE); model weights have separate terms, including non-commercial terms for recent releases. Those terms do not become this project's license. See the [reference note](docs/REFERENCES.md) and [official quickstart](https://docs.priorlabs.ai/quickstart).

## Develop

```bash
python -m unittest discover -s tests -v
python examples/python_api.py
```

The suite uses Python's standard `unittest`; no test-runner download is needed. For an uninstalled checkout, prefix either command with `PYTHONPATH=src`.

MIT for original project code and the synthetic fixture generator. Dependencies retain their own licenses. The roadmap is a long-term maintenance direction, not a promise that its future features already exist.
