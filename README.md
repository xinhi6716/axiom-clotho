<p align="right"><strong>English</strong> · <a href="README.zh-TW.md">繁體中文</a></p>

# ML Super Calculator

**Compare models on held-out batches. Predict new rows locally.**

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/readme-workflow.mobile.svg">
  <img src="docs/assets/readme-workflow.svg" alt="GroupBench workflow: prepare a table, compare on held-out groups, inspect the report">
</picture>

Small materials, semiconductor and process datasets rarely arrive as independent rows. GroupBench, the first engine in this calculator, keeps your declared batches together when comparing regression baselines.

Training, reports and predictions stay on your computer. No account, hosted compute API, model-weight download or telemetry.

[Download prerelease](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a2) · [Choose a version](#choose-the-version-you-mean) · [Try the demo](#try-the-demo) · [Bring your data](#bring-your-data) · [Read the results](#read-the-results) · [Examples](#examples) · [Detailed reference (English)](docs/USAGE_REFERENCE.md)

> **Prerelease:** v0.2.0a2, including an explicitly selected binary-classification workflow. This is an original sklearn-based selection and evaluation toolkit, not a pretrained foundation model, full AutoML system or TabPFN equivalent. Synthetic examples show how to use it, not real-world accuracy.

## Try the demo

### Choose the version you mean

- **Published download:** `v0.2.0a2` is the frozen [20d17c1 source snapshot](https://github.com/xinhi6716/ml-super-calculator/tree/20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a). Its wheel and source archive do not change when `main` changes.
- **Current-source fixes:** [73dc3f8](https://github.com/xinhi6716/ml-super-calculator/tree/73dc3f81885a1f8d9acca14198bce7ebe6aad755) adds early checks for invalid regression output-directory paths, preserves validated CSV column names across parsers, and rejects bare-string Python API feature lists. These three fixes are **not in the a2 download**. [See the exact changes](https://github.com/xinhi6716/ml-super-calculator/compare/v0.2.0a2...73dc3f81885a1f8d9acca14198bce7ebe6aad755).

To use those fixes, install from that source revision (or a later reviewed checkout) using the checkout instructions below. Record the Git revision as well as the package version; a version label alone does not identify post-release source changes. This is not a new wheel release.

**Install the published wheel:** download the [0.2.0a2 wheel](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a2/ml_super_calculator-0.2.0a2-py3-none-any.whl) and compare it with [SHA256SUMS.txt](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a2/SHA256SUMS.txt). In your Python virtual environment, from the download directory:

```bash
python -m pip install ./ml_super_calculator-0.2.0a2-py3-none-any.whl
mlcalc demo --out runs/demo
```

Installation may download dependencies from your configured package index; the wheel does not bundle them. Runtime modeling stays local. These are GitHub Release downloads, not a PyPI or GitHub Packages publication. For the file-based examples below, use the [source archive](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a2/ml_super_calculator-0.2.0a2.tar.gz) or a checkout.

**Or install from a checkout:**

For a fresh, pinned checkout, choose a new destination directory; the commands below do not update an existing checkout. Git and Python are required.

```bash
git clone https://github.com/xinhi6716/ml-super-calculator.git mlcalc-source
cd mlcalc-source
git checkout --detach 73dc3f81885a1f8d9acca14198bce7ebe6aad755
git rev-parse HEAD
python -m pip install -e .
mlcalc demo --out runs/demo
```

Open `runs/demo/report.html`. The report is a self-contained local file with no remote assets or scripts. The demo uses **288 wholly synthetic observations in 24 batches**.

Python 3.10+ is declared. Development has been verified with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0; there is no tested cross-version CI matrix yet. `groupbench` is an alias for `mlcalc`.

<details>
<summary>Already have the dependencies? Run without installing or using the network.</summary>

```bash
PYTHONPATH=src python -m groupbench demo --out runs/demo
```

</details>

## Bring your data

Start with a CSV containing three things:

| Column | What belongs here |
| --- | --- |
| Target | A finite numeric outcome, such as a measured property. |
| Group | The batch, specimen, wafer or source experiment that must stay on one side of the split. |
| Features | Numeric or categorical inputs available **before** the outcome is measured. Missing feature values are supported. |

```bash
mlcalc run measurements.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/mobility
```

The target and group are excluded from predictors. Use `--features` or `--features-file` to keep IDs, post-outcome measurements and notes out too. Without an allowlist, **every other column becomes a feature**; the calculator cannot decide which inputs are scientifically valid.

Prefer a ready-made input? Use the [synthetic training CSV](examples/synthetic_materials.csv) and [new process settings](examples/new_process_settings.csv). Their values are invented.

### Plan first, then fit

```bash
mlcalc plan run examples/synthetic_materials.csv --target response --group batch
mlcalc plan predict examples/synthetic_materials.csv examples/new_process_settings.csv --target response --group batch
```

Plans validate inputs and print split sizes, candidate settings and intended fit counts as JSON to stdout. They do not fit models or write results. Defaults require **105 evaluation fits** or **21 prediction fits** if all fits succeed. These counts are not a runtime estimate or a hard compute budget. [Planning details](docs/USAGE_REFERENCE.md#preview-the-work-before-fitting)

### Predict new rows after evaluation

Supply a second CSV with **exactly the selected feature columns**, without target or group columns:

```bash
mlcalc predict measurements.csv new_process_settings.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/new-predictions
```

Prediction performs grouped inner selection on all labeled rows and refits locally. It writes predictions in target units, not a new test-performance estimate. No executable model is loaded or saved; the CLI refits each time. Simple input-support flags can highlight unseen values, but they are **not uncertainty intervals or an out-of-distribution test**. [Prediction and input-support details](docs/USAGE_REFERENCE.md#predict-new-rows)

## Read the results

| File | Start here when you want to… |
| --- | --- |
| `report.html` | Compare models and inspect the worst-performing groups. |
| `summary.json` | Read pooled, per-group and per-fold metrics. |
| `predictions.csv` | Inspect observed targets, predictions and fold assignments row by row. |
| `manifest.json` | Check splits, candidate selections, versions, seeds, hashes and diagnostics. |
| `checksums.json` | Check the bytes of the four other evaluation files. |

Regression compares **Dummy mean, Ridge and Extra Trees** with the same folds and preprocessing. Hyperparameters are selected inside each outer training partition. The `selected` entry evaluates the selection procedure, rather than choosing a winner after inspecting the outer test scores. Families do **not** receive equal compute budgets.

Group MAE gives each group equal weight; row MAE and RMSE pool observations. Constant-target or one-row R² is `null`. Do not repeatedly tune against the outer leaderboard and then present those same scores as unbiased evidence. [Full evaluation protocol](docs/PROTOCOL.md)

```bash
mlcalc verify runs/demo
```

`verify` checks saved bytes against checksums without fitting or loading models. It requires POSIX no-follow file access and rejects unsupported platforms. A pass proves neither authenticity nor scientific correctness. The separate `validate-prediction-bundle` command checks supported prediction manifest/CSV relationships. [Verification limits](docs/USAGE_REFERENCE.md#verify-an-existing-bundle-offline) · [Prediction bundle contract](docs/PREDICTION_BUNDLE_SCHEMA.md)

**Before sharing:** reports contain group IDs, target values, predictions and feature names. Keep private data and run directories out of public commits. CSV values are not spreadsheet-formula sanitized; review files before opening them in spreadsheet software or uploading them.

## Examples

| Try this | What it demonstrates |
| --- | --- |
| [Python API](examples/python_api.py) | `benchmark`, `fit_predict` and local output files. A sklearn-like estimator is also documented in the [API reference](docs/USAGE_REFERENCE.md#python-api). |
| [Environment-heldout comparison](examples/environment-heldout-comparison/README.md) | Five fixed pipelines, two prespecified feature sets and a training-label-shuffle diagnostic. Predictive stress testing, not causal inference. |
| [Semiconductor walkthrough](examples/semiconductor-heldout-walkthrough/README.md) | Invented I–V measurements through a trusted local Toolbox checkout to device features, lot holdout, training-only fits, target-free prediction and separate scoring. |
| [Causal-design rehearsal](examples/causal-design-rehearsal/README.md) | Invented equations and paired intervention oracles to rehearse adjustment choices. Not a calibrated device model or real-data causal estimator. |
| [Saved budget comparison](docs/SAVED_BUDGET_COMPARISON.md) | A descriptive two-method × two-budget comparison, including adverse cases. No retraining or provenance verification. |
| [Saved interval evaluation](docs/SAVED_INTERVAL_EVALUATION.md) | Coverage, width and interval score from already saved bounds. Does not generate or calibrate intervals. |

### Binary pass/fail, explicitly selected

```bash
mlcalc binary plan run examples/synthetic_binary_process.csv --target fail --group lot
mlcalc binary run examples/synthetic_binary_process.csv --target fail --group lot --out runs/binary
mlcalc binary predict examples/synthetic_binary_process.csv examples/new_binary_process.csv --target fail --group lot --out runs/binary-new
```

This prerelease workflow expects numeric 0/1 labels and selects among prior Dummy, regularized Logistic and bounded Extra Trees using group-balanced Brier. Both labels must appear in every training partition; single-class held-out groups are retained. The threshold is fixed at p1 ≥ 0.5, not optimized for manufacturing decisions. Binary output directories must be fresh. Default regression commands stay regression even for 0/1 targets. [Binary protocol and limits](docs/BINARY_PROTOCOL.md)

## Know the boundaries

- **Choose groups with care.** Disjoint batch labels do not isolate unrecorded dependencies or both factors in a crossed `batch × specimen` label. Multi-axis and temporal holdouts are not implemented.
- **Small groups can change the ranking.** Fold variation is not a confidence interval; one seed is not a robustness study. Validate independently before consequential use.
- **Keep inputs bounded and local.** At most 20,000 rows and 100 features; CSVs must be regular local UTF-8 files, at most 64 MiB each. Missing targets, malformed records, NUL characters and invalid splits fail rather than being silently repaired. These guards are not hard memory or time limits.
- **Check categorical markers.** The compatibility default `legacy-v1` can merge a literal `__GROUPBENCH_MISSING__` string with missing values. Choose `distinct-v2` for collision-free internal keys. [Encoding contract](docs/CATEGORICAL_ENCODING.md)
- **Separate predictions from scientific claims.** No calibrated uncertainty, causal inference, scientific-unit validation, extrapolation guarantee, foundation-model training or production monitoring is included. Synthetic fixtures are not validated materials simulators.

The [complete usage reference](docs/USAGE_REFERENCE.md) retains the detailed input, schema, arithmetic, resource, output and compatibility contracts. Documentation examples do not establish real-device accuracy. A valid feature schema is not proof against leakage.

## Develop and contribute

```bash
python -m unittest discover -s tests -v
python examples/python_api.py
```

The suite uses the standard-library `unittest`. For an uninstalled checkout, prefix either command with `PYTHONPATH=src`. Read the [contribution guide](CONTRIBUTING.md), [roadmap](ROADMAP.md) and [security notes](SECURITY.md).

TabPFN inspired the approachable `fit`/`predict` interface and packaging, not the modeling implementation. This repository includes no TabPFN code, weights, downloads or API client and makes no equivalent-capability claim. [Reference and licensing note](docs/REFERENCES.md)

[MIT](LICENSE) covers original project code and the synthetic fixture generator. Dependencies retain their own licenses. The roadmap describes intended work, not features already delivered.

<details>
<summary>Looking for a section from the previous README?</summary>

The detailed sections now live in the usage reference. Existing section anchors are retained below.

<a id="run-the-demo"></a>

- [Run the demo](docs/USAGE_REFERENCE.md#run-the-demo)

<a id="distinguish-missing-values-from-literal-marker-strings"></a>

- [Distinguish missing values from literal marker strings](docs/USAGE_REFERENCE.md#distinguish-missing-values-from-literal-marker-strings)

<a id="source-companion-fixed-environment-feature-comparison"></a>

- [Source companion: fixed environment feature comparison](docs/USAGE_REFERENCE.md#source-companion-fixed-environment-feature-comparison)

<a id="runnable-semiconductor-integration"></a>

- [Runnable semiconductor integration](docs/USAGE_REFERENCE.md#runnable-semiconductor-integration)

<a id="simulation-only-causal-design-rehearsal"></a>

- [Simulation-only causal-design rehearsal](docs/USAGE_REFERENCE.md#simulation-only-causal-design-rehearsal)

<a id="explicit-binary-passfail-workflow-unreleased"></a>

- [Explicit binary pass/fail workflow (prerelease)](docs/USAGE_REFERENCE.md#explicit-binary-passfail-workflow-unreleased)

<a id="use-your-own-regression-data"></a>

- [Use your own regression data](docs/USAGE_REFERENCE.md#use-your-own-regression-data)

<a id="preview-the-work-before-fitting"></a>

- [Preview the work before fitting](docs/USAGE_REFERENCE.md#preview-the-work-before-fitting)

<a id="predict-new-rows"></a>

- [Predict new rows](docs/USAGE_REFERENCE.md#predict-new-rows)

<a id="check-whether-new-inputs-were-observed-in-training"></a>

- [Check whether new inputs were observed in training](docs/USAGE_REFERENCE.md#check-whether-new-inputs-were-observed-in-training)

<a id="reuse-a-declared-feature-schema"></a>

- [Reuse a declared feature schema](docs/USAGE_REFERENCE.md#reuse-a-declared-feature-schema)

<a id="python-api"></a>

- [Python API](docs/USAGE_REFERENCE.md#python-api)

<a id="how-the-evaluation-is-split-and-scored"></a>

- [How the evaluation is split and scored](docs/USAGE_REFERENCE.md#how-the-evaluation-is-split-and-scored)

<a id="baselines-and-resource-budget"></a>

- [Baselines and resource budget](docs/USAGE_REFERENCE.md#baselines-and-resource-budget)

<a id="outputs"></a>

- [Outputs](docs/USAGE_REFERENCE.md#outputs)

<a id="verify-an-existing-bundle-offline"></a>

- [Verify an existing bundle offline](docs/USAGE_REFERENCE.md#verify-an-existing-bundle-offline)

<a id="limits-to-understand-before-trusting-a-result"></a>

- [Limits to understand before trusting a result](docs/USAGE_REFERENCE.md#limits-to-understand-before-trusting-a-result)

<a id="design-inspiration-not-copied-modeling"></a>

- [Design inspiration, not copied modeling](docs/USAGE_REFERENCE.md#design-inspiration-not-copied-modeling)

<a id="develop"></a>

- [Develop](docs/USAGE_REFERENCE.md#develop)

<a id="compare-saved-validation-budgets-locally"></a>

- [Compare saved validation budgets locally](docs/USAGE_REFERENCE.md#compare-saved-validation-budgets-locally)

<a id="optional-saved-prediction-consistency"></a>

- [Optional saved prediction consistency](docs/USAGE_REFERENCE.md#optional-saved-prediction-consistency)

</details>
