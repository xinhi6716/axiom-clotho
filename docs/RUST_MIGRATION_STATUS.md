# Rust migration inventory and acceptance gates

## Status and evidence boundary

This inventory was prepared on **2026-11-07** against the public repository
`xinhi6716/axiom-clotho`, with `main` resolved to
[`6fa3ac180160d7f900f5030cfb4d0d6b4e525abe`](https://github.com/xinhi6716/axiom-clotho/tree/6fa3ac180160d7f900f5030cfb4d0d6b4e525abe).
The recursive Git tree was complete (`truncated: false`). The review inspected
all 48 owned Python modules under `src/`, all 11 bundled Python components,
public command parsers, selected protocol documentation and regression-test
names. This is a feature inventory and implementation plan, **not a claim that
all Python behavior has been ported or that the Python suite was run here**.

The verified upstream Rust baseline consists of `input_support.rs`,
`numeric_bytes.rs` and `knn.rs`. New work in this change must be described
separately below with its own verification evidence. A directory, an API name,
a CLI wrapper, or a passing unrelated test does not establish migration completion.

### Meaning of the status labels

- **Native, bounded:** algorithm implemented in Rust for a specifically documented
  boundary. This does not imply that the larger Python workflow is replaced.
- **Partial boundary:** useful native implementation exists, but input admission,
  orchestration, contracts or workflow integration remain Python-owned.
- **Missing at reviewed baseline:** no equivalent implementation in the reviewed
  Rust core. This is not a judgment that the feature should be removed.
- **New contract:** a deliberately different native algorithm or schema. It can be
  useful without being a parity port; it must have a distinct identity and tests.

There is deliberately no “percentage migrated.” Source lines, wrappers and
estimator counts are poor measures of preserved scientific and safety behavior.

## 1. What the existing Rust code actually covers

| Boundary | Reviewed implementation | Coverage and exclusions |
| --- | --- | --- |
| Marginal training/query support | `clotho_core::input_support::prediction_input_support` | Native, bounded. Numeric and categorical normalized named columns; missing counts, range/category support and unions of row flags. 1–20,000 rows and 1–100 features. Does not parse CSV, normalize raw Python values, impute, fit or predict. |
| Numeric byte transport | `numeric_bytes::prediction_input_support_bytes` | Native, bounded. Borrowed little-endian binary64 bytes; NaN means missing only at this boundary; infinity/shape/width errors reject. Does not provide categorical transport or a general saved-file format. |
| Repeated support queries | `NumericSupportReference::from_training`, `compare` | Native, bounded. Immutable owned training summary reused across query batches. Queries never expand the reference. Not a model, serialized model or OOD/calibration guarantee. |
| Optional Python bridge | `native/numeric-support` | Partial boundary. Explicit PyO3 package for numeric support/reference APIs. Python still performs DataFrame admission, alignment, byte snapshots and report validation. GIL-enabled CPython and matching protocol required; no automatic build or fallback. This is not a whole-program Rust rewrite. |
| Numeric kNN regression | `knn::KnnRegressor::fit`, `predict` | Native, bounded **new estimator**. Finite numeric dense rows, training-only max-absolute scaling, exact search, deterministic row-index distance ties, equal-neighbor averaging with numeric-range guards. No Python-registry counterpart, missing/categorical preprocessing, grouped selection, CLI, bundles, saved scoring or intervals at this reviewed commit. |

Existing evidence is scoped: `docs/RUST_KNN_BASELINE.md` records 47 Rust
integration tests (9 kNN) on Rust 1.70.0 and 1.99.1, the synthetic example,
Clippy and formatting. `docs/RUST_CORE.md` describes exact cross-language
support-report parity cases. Those records are not fresh verification of any
later work, a complete Python parity run, portability certification, real-data
accuracy evidence or an end-to-end Rust release.

### Additive native workflow in this change

This change adds `regression.rs` (mean and standardized ridge), `evaluation.rs`
(regression metrics and deterministic group-disjoint folds), and `clotho-cli`
(CSV fit, saved-specification predict, and grouped evaluate). See the
[native workflow and verification record](RUST_NATIVE_WORKFLOW.md). Model schema
`clotho.numeric.v1` and evaluation schema `clotho.grouped_evaluation.v1` are new;
they do not import Python artifacts or reproduce its randomized split assignment.
The baseline missing entries below describe the audited prior commit, while these
new numeric capabilities are explicitly additive. Retain the distinction between:

1. useful Rust functionality under its own documented contract;
2. Python-equivalent behavior backed by matched acceptance tests; and
3. Python functionality still awaiting migration.

The next native extension adds `classification.rs`, `classification_metrics.rs`
and `preprocessing.rs`, integrated into fit/predict/evaluate with model schema v3
and binary evaluation schema `clotho.grouped_binary_evaluation.v1`. It supports
explicit mixed data and logistic/prevalence models while preserving v1/v2 numeric
reads. See [scope and contracts](RUST_BINARY_AND_MIXED.md). This is not a parity
claim for sklearn estimators, encoders, random splits or legacy artifacts.

A further native slice adds `selection.rs` and `clotho compare`: bounded nested
grouped numeric regression comparison with explicit candidates, fit budgeting,
failure accounting and untouched outer-fold predictions. See the
[protocol and remaining boundaries](RUST_NESTED_SELECTION.md). Mixed/classification
nested selection and Python workflow/artifact compatibility remain unimplemented.

## 2. Public command surface to preserve or explicitly replace

The Python distribution is `ml-super-calculator` 0.3.0b7. Installed scripts
`clotho` and `mlcalc-workflow` resolve to `mlworkflow.cli:main`;
`mlcalc` and `groupbench` resolve to `groupbench.cli:main`.
A Rust executable must not silently impersonate all four compatibility contracts
when only a subset is implemented.

| Command / route | User-visible responsibility | Baseline Rust status |
| --- | --- | --- |
| `clotho init --workflow rram\|fet --out ...` | Fresh header-only input scaffold, intentionally incomplete choices and README; no invented measurements or fitting. | Missing |
| `clotho prepare` | Explicit RRAM v1 or FET v2 measurement extraction, exact partition/target joins, provenance and complete prepared bundle. | Missing |
| `clotho plan [--budget-summary]` | Validate actual nested selection plus prediction-refit schedules and `max_fits`; stdout JSON, no fits/writes. Over-budget summary exits 2 without raising the cap. | Missing |
| `clotho run` | Nested grouped comparison, full-training selection/refit, target-free future prediction, linked outputs and completion receipt. | Missing |
| `clotho score` | Later outcomes joined to the complete saved cohort; no model import, training, selection or refit. | Missing |
| `clotho features` | Standalone strict RRAM extractor, including explicit device-condition export; separate from supervised device-only v1. | Missing |
| `clotho demo --run-invented-example` | Explicitly invented, pinned, multi-stage measurement/holdout/prediction/scoring example with fresh-output preflight. | Missing |
| `clotho components` | Verify pinned component files, licenses, hashes and source identities. | Missing |
| `clotho tabular ...` | Access the general calculator surface below, currently by Python delegation. | Missing as native execution |
| `mlcalc demo`, `fixture` | Synthetic grouped materials data and evaluation example, explicit synthetic interpretation. | Only a different small Rust kNN example exists |
| `mlcalc run`, `predict` | General local regression CSV evaluation or full-training refit prediction; exact selectors/allowlists, provenance and output preflight. | Missing |
| `mlcalc plan run`, `plan predict` | Same validated execution schedules as fitting, but no fitting or artifact writes. | Missing |
| `mlcalc binary run\|predict\|plan run\|plan predict` | Explicit numeric 0/1 classification, not automatic task switching. | Missing |
| `mlcalc verify` | Bounded, read-only SHA-256 verification of fixed supported file inventories. | Missing |
| `validate-prediction-bundle`, `validate-evaluation-bundle` | Structural/metric consistency of supported saved schemas, not authentication or model execution proof. | Missing |
| `score-predictions` | Complete delayed-outcome regression cohort scoring. | Missing |
| `compare-predictions` | Paired saved regression comparison on identical query bytes and explicit semantic declarations. | Missing |
| `evaluate-binary` | Strict JSON transport and complete saved-probability cohort scoring. | Missing |
| `evaluate-intervals` | Saved interval coverage, width and interval score; no fitting/calibration. | Missing |
| `plan-group-forward-windows` | Strict bounded metadata JSON, whole-group forward memberships/exclusions, stdout only. | Missing |

Also retain standalone/module APIs not all exposed as top-level commands:
saved budget comparison (`python -m groupbench.budget_comparison`),
observed-history evaluation, fixed group-forward baseline evaluation and
persistence, quantity/process/evidence declaration checks, and optional support
diagnostics. Their absence from a top-level help listing does not make them
disposable.

## 3. Feature-by-feature preservation inventory

### A. Data admission, identity and numerical safety

**Sources:** `groupbench/data.py`, `categorical.py`,
`feature_manifest.py`, raw CSV handling in `cli.py`.

- Bounded tables; exact unique string column names; distinct target/group selectors.
  Target/group/domain metadata must never accidentally become predictors.
- Explicit ordered feature allowlists and bounded version-1 feature manifests.
- Strict CSV header identity, BOM/encoding/record validation, NUL rejection,
  source-byte hashing, a 64 MiB CSV bound and row/feature bounds.
- Literal text group IDs, including leading zeros. Python API numeric group IDs
  preserve exact identity across numeric types; mixed text/numeric groups, missing,
  blank, Boolean, nonfinite and NUL-containing IDs reject. Do not convert IDs
  through `f64`, trim them, or merge `"01"`, `"1"` and `"1.0"`.
- Stateless numeric/target normalization, precision-loss rejection for integral
  values, nonfinite rejection, missing-feature handling and missing-target rejection.
- Prediction column set/type checking and reorder-to-training-schema behavior,
  including the all-missing incoming-column accommodation.
- Categorical legacy-v1 compatibility and explicit distinct-v2 marker escaping.
  Literal missing-marker text and actual missing values differ under v2.
- Stable normalized data fingerprints, original row positions, source digests and
  duplicate diagnostics that **do not change** the training/evaluation cohort.

**Rust baseline:** normalized support checks only. Raw admission, normalization,
CSV, fingerprint and categorical-preprocessing contracts are missing.

**Acceptance focus:** `test_csv_*`, `test_training_column_names.py`,
`test_feature_numeric_precision.py`, `test_target_numeric_precision.py`,
`test_target_missingness.py`, `test_*group_identity.py`,
`test_masked_group_identifiers.py`, `test_categorical_*.py`,
`test_api_feature_allowlist.py`, `test_cli_feature_allowlist.py`,
`test_feature_manifest.py`, `test_prediction_normalization.py`.

### B. Group splitting, no-fit planning and leakage protection

**Sources:** `splitting.py`, `planning.py`, `domain_selection.py`,
selection schedules in `model.py` and `binary/model.py`.

- Whole-group, approximately row-balanced folds. First-encounter factorization;
  seeded shuffle; stable descending group-size sort; seeded least-load tie choice.
  No target-derived allocation or row-random fallback.
- Same ordered data, configuration and compatible runtime reproduce assignments.
  Ordinary Python splitting uses NumPy's `default_rng`; a different Rust RNG
  must not be called exact split parity.
- Nested outer evaluation and inner selection. Outer fold `i` uses inner seed
  `outer_seed + i + 1`; every actual inner schedule is validated before fitting.
- Group-overlap audits, positional alignment, exact-once outer predictions.
- Explicit leave-one-domain-out **inner** selection for supported regression
  families, preserving ordinary grouped outer folds and excluding domain metadata.
- Candidate order, family-refit schedule and actual fit counts shared by plan and
  execution, rather than an independent estimate. No partial fitting when an
  impossible nested schedule is already detectable.

**Rust baseline:** missing.

**Acceptance focus:** `test_groupbench.py` split/leakage cases,
`test_split_target_alignment.py`, `test_planning.py`,
`test_evaluation_partition_parity.py`, `test_domain_*.py`,
`test_workflow_fit_budget.py`. In particular, preserve
`test_outer_targets_do_not_affect_their_predictions`,
`test_impossible_inner_fails_before_any_fit`, and
`test_new_rows_cannot_affect_selection`.

### C. Regression estimation, preprocessing and state

**Sources:** `model.py`, `baselines.py`, `kernel.py`.

| Existing family | Actual contract to preserve |
| --- | --- |
| `dummy` | Training row-mean constant (`dummy_mean`). |
| `ridge` | Alphas 0.1, 1, 10; existing Ridge uses LSQR. |
| `extra_trees` | 64 trees, depths 6 or unbounded, minimum leaf size 2, seeded and single-job. |
| `group_median` | Equal-training-group lower weighted median constant, with explicit tie rule. |
| `group_balanced_ridge` | Equal-training-group weighted squared loss, alphas 0.1, 1, 10; does not group-weight preprocessing or optimize MAE directly; group-kfold inner strategy only. |
| `nystroem_ridge` | Explicit bounded RBF approximation plus ridge, rank cap 64 and recorded fitted details. |

All learned preprocessing stays in each training partition:
numeric median imputation (including all-empty features) then StandardScaler;
categorical missing handling then sparse one-hot encoding with unknown handling
and at most 64 categories per feature. Candidate selection uses pooled inner
OOF predictions, default group MAE or explicit pooled MAE. Exact score ties use
candidate order. Refit the best member of **each** requested family; the overall
selected result uses inner scores, never outer errors.

Validate learned numeric/tree state even when output predictions happen to be
finite. Failures do not silently skip candidates or retry with weaker settings.
Publish complete learned state only after every selection/refit step succeeds;
an unsuccessful refit preserves a previous complete fitted model.

**Rust baseline:** kNN is a native new estimator, not a replacement for any row
above. Its maximum-absolute scaling is not the Python StandardScaler contract.
All six Python regression families and their full preprocessing/selection
contracts remain missing at the reviewed baseline.

**Acceptance focus:** `test_regression_*.py`, `test_group_median_*.py`,
`test_group_balanced_ridge_*.py`, `test_kernel_*.py`,
`test_numeric_unit_equivalence.py`, `test_optional_delayed_walkthrough.py`.
Establish documented tolerances for numerical solvers and new backend identities
before claiming equivalence; deterministic seeded trees require more than similar
hyperparameter names.

### D. Regression evaluation and prediction

**Sources:** `evaluation.py`, `prediction.py`, `metrics.py`,
`input_support.py`, `package_source.py`.

- Nested group-disjoint family comparisons plus inner-selected procedure.
- Full-training selection/refit and separate feature-only query inference.
  Training fit scores are not held-out generalization estimates.
- Pooled MAE, RMSE, bias and R²; R² is null for constant/singleton targets.
  R² uses anchored centering to protect large common offsets.
- Equal-group MAE and **mean of group RMSEs**, not square root of group MSE;
  worst-group MAE; per-group/per-fold summaries.
- Strict direct-metric input types and arithmetic failure behavior, independent
  of a caller's NumPy warning configuration. Reject corruption instead of
  returning flattering zero/infinite scores.
- Training-only marginal support and optional outer-fold support; immutable
  reference/query separation. These are diagnostics, not OOD or physical gates.
- Candidate/selection/split audit, aligned row positions, run-key/config/data
  fingerprints, package-source observation and honest unavailable status.

**Rust baseline:** marginal source support is native; kNN predictions are native
but outside this workflow. Nested evaluation, metrics, model selection and run
provenance remain missing.

**Acceptance focus:** `test_metric_*.py`, `test_regression_metric_inputs.py`,
`test_input_support.py`, `test_outer_fold_input_support.py`,
`test_evaluation_run_key.py`, `test_package_source.py`,
`test_prediction_and_safety.py`, `test_prediction_schema_guidance.py`.

### E. Explicit binary classification

**Sources:** all eight `groupbench/binary/` modules.

- Numeric labels exactly 0/1; both classes in every training partition.
  Regression 0/1 targets must not auto-switch task type.
- Dummy prior, logistic C values 0.1/1/10 (L2, LBFGS, tolerance 1e-4,
  maximum 1,000 iterations), and 64-tree extra-trees configurations
  (depth 6/12, max 256 leaves, minimum leaf size 2).
- Fold-local preprocessing and fixed equal-group Brier selection. Training
  weights rows equally; prediction uses p1 >= 0.5, including ties.
- Checked n-by-2 probabilities in class order [0,1], no range/sum repair,
  rejection of unsupported Boolean/masked/precision inputs and convergence errors.
- Brier, clipped-natural-log loss, clipping/endpoint counts, confusion metrics
  and undefined threshold metrics as null; explicit positive-loss-underflow policy.
- Nested benchmark, prediction, no-fit plans, fresh-only binary reports/bundles,
  strict JSON saved-probability evaluation and exact cohort checks.

**Rust baseline:** missing in full.

**Acceptance focus:** `test_binary*.py`, `test_nullable_probabilities.py`,
`test_saved_binary*.py`. A binary metric utility alone does not constitute a
binary training/prediction port.

### F. Saved artifacts, local safety and offline reports

**Sources:** `report.py`, `verification.py`, `prediction_schema.py`,
`evaluation_schema.py`, binary reporting and workflow contract writers.

- Self-contained escaped offline HTML, strict finite JSON, CSV, manifests,
  summaries and checksums; no remote assets or executable-model deserialization.
- Exact supported inventory/path names and schema versions. Existing regression,
  binary, RRAM v1, FET v2 and temporal artifacts have different contracts.
- Preflight before expensive fitting; do not overwrite source through direct,
  symlink or hardlink aliases. Fresh-output semantics where required; explicit
  overwrite applies only to documented known outputs.
- Same-directory staged writes and per-file no-clobber protections. Do not imply
  multi-file atomicity when only individual replacements are atomic.
- Bounded read-only verification with safe regular-file/no-follow handling,
  no loading/rendering/execution of payloads. POSIX-specific checks must not
  silently weaken on another platform.
- Structural schema checks include ordered feature/row/fold consistency,
  selection/candidate details, numeric and metric consistency and version gates.
- Measurement bundles use fixed inventories, INCOMPLETE/COMPLETE receipts and
  exact hashes. Partial or interrupted output remains unusable.

**Rust baseline:** missing. A CSV prediction writer is not legacy bundle parity.

**Acceptance focus:** `test_artifact_bytes.py`, `test_bundle_*.py`,
`test_cli_output_preflight.py`, `test_binary_output_preflight.py`,
`test_prediction_schema*.py`, `test_evaluation_schema.py`,
`test_source_digest_validation.py`, `test_lightweight_verification_cli.py`,
`test_lazy_import_boundaries.py`, `test_report_*.py`,
`test_clotho_report_*.py`, race/symlink cases in
`test_prediction_and_safety.py`.

### G. Delayed outcomes and descriptive comparison

**Sources:** `saved_scoring.py`, `interval_evaluation.py`,
`budget_comparison.py`, `observed_history.py`,
`binary/saved_evaluation.py`.

- Score a complete saved regression cohort by row-position/group contract;
  reject missing, extra, duplicate and inconsistent rows; no refitting.
- Compare saved regression bundles only on the same exact query bytes and
  declared target/unit/semantic contract. Preserve exact B-minus-A
  absolute-error contrasts before final display rounding.
- Saved interval coverage, mean width, interval score and tail-miss diagnostics;
  strict bounded numbers and full cohort; separate optional reference diagnostic.
  This evaluates supplied intervals; it does not generate or calibrate them.
- Descriptive two-method/two-budget comparison without rerunning training.
- Frozen observed-history one-step evaluation of one saved model and two
  baselines on the same origins, with support diagnostics.
- Saved binary evaluation is separate and retains its fixed probability contract.

**Rust baseline:** missing.

**Acceptance focus:** `test_saved_prediction_scoring.py`,
`test_saved_prediction_comparison.py`, `test_saved_paired_contrast.py`,
`test_saved_comparison_cli.py`, `test_budget_comparison*.py`,
`test_saved_intervals.py`, `test_interval_*.py`,
`test_observed_history*.py`.

### H. Explicit time-window evaluation

**Sources:** `group_time_plan.py`, `group_time_json.py`,
`group_time_evaluation.py`, `group_time_saved.py`.

- Declared signed-64-bit ticks, clock/unit, whole-group spans and nonoverlapping
  half-open test windows. Training requires label_available_at + gap < lower;
  testing requires the entire span inside [lower, upper).
- Exact text/integer identities, overflow checks, explicit exclusions, stable
  membership order and nonempty train/test folds; metadata-only plan has no fitting.
- Strict bounded JSON transport, read-only stdout command.
- Separate bounded in-memory retrospective evaluation of two fixed baselines,
  with two fresh training-only pipeline fits per declared window; not adaptive
  tuning or ordinary full OOF evaluation.
- Persist/validate declared temporal-result bundles, exact content binding and
  fresh-output handling, without claiming proof of execution or causality.

**Rust baseline:** missing.

**Acceptance focus:** `test_group_time_plan.py`, `test_group_time_json.py`,
`test_group_time_evaluation.py`, `test_group_time_saved.py`.

### I. Measurement-to-prediction workflows

**Sources:** `mlworkflow/contracts.py`, `regression.py`,
`adapters/__init__.py`, `adapters/fet_transfer.py`, `scaffold.py`.

**RRAM v1:** explicit HRS/LRS measurement/readpoint conditions, one device row,
three supported feature choices (HRS resistance median, LRS resistance median,
resistance ratio median), exact device-keyed partitions/training-only endpoints,
explicit batch/wafer/device grouping and train/predict group disjointness.
The standalone extractor's device-condition option is **not** accepted by this
supervised device-only schema.

**FET v2:** one complete acquisition per composite (lot, wafer, device),
with (condition, replicate, branch) retained; explicit fixed signed biases,
temperature, monotonic transfer branch, criterion-current and log-slope settings;
criterion gate-source voltage and log-current-slope features. Exact composite
partitions/endpoints and lot/wafer/device tuple-prefix grouping; preserve all-row
cohort and identity map. No fallback branch, averaging or silent row dropping.

**Both:** explicit finite targets, features, units, scientific declarations,
grouped baseline/budget choices; no future outcomes accepted by prepare/plan/run.
Prepared/run/score fixed inventories, captured input bytes and identity maps,
completion receipts and source lineage. Later outcomes must exactly match all
saved prediction devices; scoring is zero-fit. Blank init scaffolds deliberately
fail preparation until real choices and measurements are supplied.

**Rust baseline:** missing. Port the actual extraction/join/contracts, not just
spawn the existing Python workflow from Rust.

**Acceptance focus:** `test_measurement_regression.py`,
`test_fet_measurement_adapter.py`, `test_fet_bundle_versions.py`,
`test_workflow_init.py`, `test_workflow_fit_budget.py`,
`test_workflow_cli_errors.py`, `test_workflow_demo_preflight.py`,
`test_workflow_tabular_dependencies.py`.

### J. Bundled algorithms, source identity and scientific declarations

**Sources:** `mlworkflow/components.py`, `_vendor/manifest.json`,
the 11 bundled Python files; `quantity_contract.py`, `process_contract.py`.

Bundled executable functionality includes signed I–V readpoint interpolation,
paired declared RRAM windows, constant-current transfer criterion, windowed
log-current slope, physical holdout metadata audit, RRAM/FET feature builders,
synthetic RRAM generator, exact-cohort scorecard, fixed environment feature
comparison and the invented semiconductor walkthrough. Native replacements need
their own tested implementation identities. Preserve MIT notices, upstream
attribution and immutable source pins; do not pretend a rewritten algorithm is
the unchanged pinned upstream component.

Independent helpers also preserve:
- electrical quantity/statistic/condition/unit declarations and exact bounded
  decimal unit normalization; explicit guarded decimal-to-binary64 conversion;
- ALD process role, timing, identity and thickness/cycle derivation declarations;
- precise evidence-quarantine scopes and inventory checks, without importing
  quarantined values into modeling or inferring unverified source facts.

**Rust baseline:** missing.

**Acceptance focus:** `test_quantity_contract.py`,
`test_process_contract.py`, `test_evidence_quarantine.py`,
`test_ml_workflow.py`, `test_semiconductor_walkthrough.py`,
`test_vendor_documentation.py`, `test_environment_feature_probe.py`,
`test_joint_dependency_example.py`, `test_causal_design_rehearsal.py`.
The latter examples are supporting workflows, not proof of causal or physical validity.

## 4. Implementation order

This order gives users a real native workflow early without declaring the rest
finished. All existing useful capabilities remain on the migration backlog.

| Priority | Deliverable | Completion gate |
| --- | --- | --- |
| P0 | Freeze this inventory, assign native protocol/schema identities, build Rust test/CI entry points and explicit unsupported-command behavior. | No accidental Python execution/network/model download; no full-parity advertising. |
| P1 | Native numeric table admission, exact text group IDs, grouped metrics, deterministic native split schedules, no-fit plans, mean plus kNN grouped selection/evaluation/prediction. | End-to-end numeric CLI/API tests; leakage invariance; train-only scaling; fit-budget agreement; no-write plan; safe fresh output. If RNG/estimator differs, label as a new contract. |
| P2 | Full typed/missing/categorical preprocessing and Python regression families, starting with mean/group-median/ridge/group-balanced ridge, then bounded kernel and extra trees. | Oracle fixtures per transform/model; solver tolerance and failure policy; failed-refit state tests; candidate/family/tie parity. |
| P3 | Versioned artifacts, offline reports, integrity/schema verification, delayed-outcome scoring and same-query comparison. | Cross-language golden bundles or explicit new schema; malformed/corrupted/racy filesystem tests; zero-fit readers; exact-cohort validation. |
| P4 | RRAM/FET native extraction and prepare/plan/run/score/init workflows; component provenance and declared physical joins. | Complete synthetic end-to-end fixtures; no early future outcomes; exact identity/units; native extraction numerical/error parity; completion receipts. |
| P5 | Explicit binary families/metrics/plans/artifacts and saved probability scoring. | Both-class training checks, fixed group-Brier and probability safety, isolated namespace, full workflow tests. |
| P6 | Temporal plans/evaluation/persistence, intervals, budget/history analyses, quantity/process/quarantine helpers and companion examples. | Match each documented contract; no silent downgrade to ordinary group CV or pooled scoring. |
| P7 | Install/distribution and supported-platform matrix; performance evidence and migration guide. | No Python runtime needed for advertised native routes; installation smoke outside source; supported old artifact behavior tested; all retained capabilities accounted for. |

P1 is a useful standalone product slice. It is **not** permission to delete the
Python implementation or call the whole rewrite complete. P2–P6 are functional
work, not documentation-only tasks.

## 5. Required cross-cutting acceptance gates

1. **Native execution:** run the advertised Rust route with Python absent from
   PATH. No subprocess Python bridge, FFI into Python estimators, hidden network
   fetch, hosted inference, model-weight download or runtime compiler invocation.
2. **Observable contracts:** preserve supported commands, fields, units, errors,
   exit codes, limits and saved versions, or provide a clearly versioned replacement
   and explicit unsupported-case errors. Never accept options that are ignored.
3. **Scientific isolation:** changing outer-test targets must not change their
   predictions/selection; changing query rows must not change fitted training state.
   Fit each transform independently in each training fold. No target inference.
4. **Schedule integrity:** no-fit plans and executions consume the same schedules;
   count real pipeline fits, including all family refits; validate every fold before
   fitting; never weaken grouping to rescue infeasibility.
5. **Numeric integrity:** finite predictions are insufficient. Test learned state,
   cancellation, overflow/underflow, large common offsets, tied scores, exact IDs,
   integer-to-float precision, unsupported masks/types and missing values.
6. **Artifact integrity:** fresh/output-alias/symlink/race/interruption behavior;
   bounded strict readers; complete exact cohorts; escaped offline HTML; distinct
   checksum-integrity, schema-consistency and scientific-validity statements.
7. **Reproducibility:** identify algorithm, RNG, seed policy, preprocessing,
   feature order, selection objective, versions and source. Different Rust RNG or
   solver output must not masquerade as unchanged Python byte parity.
8. **Test scope:** run focused tests plus the complete applicable Rust and Python
   compatibility suites on final code. Report passed, failed and unrun stages
   separately. Existing source test names are a checklist, not executed evidence.
9. **Distribution:** retain old scripts/imports while compatibility is promised;
   document old-bundle reading, schema changes and unsupported platforms. No
   publication/release claim until the exact built artifact and its checks pass.
10. **Scientific claims:** invented fixtures establish software behavior only.
    A Rust rewrite by itself does not establish real semiconductor accuracy,
    calibrated uncertainty, physical correctness, causality or superiority.

## 6. Source map for maintaining the inventory

Use the pinned revision above when comparing this inventory, not an unqualified
future `main`. The principal references are:

- [Entry points and dependency baseline](https://github.com/xinhi6716/axiom-clotho/blob/6fa3ac180160d7f900f5030cfb4d0d6b4e525abe/pyproject.toml)
- [General regression protocol](PROTOCOL.md), [usage reference](USAGE_REFERENCE.md),
  [binary protocol](BINARY_PROTOCOL.md), [optional regression families](OPTIONAL_REGRESSION_FAMILIES.md)
- [RRAM workflow](CLOTHO_CSV_BETA.md), [FET workflow](FET_DEVICE_REGRESSION.md),
  [blank-input scaffold](WORKFLOW_INIT.md)
- [Rust support boundary](RUST_CORE.md), [optional numeric adapter](RUST_NATIVE_NUMERIC_SUPPORT.md),
  [numeric reference](RUST_NUMERIC_REFERENCE.md), [kNN baseline](RUST_KNN_BASELINE.md)
- [Pinned source modules](https://github.com/xinhi6716/axiom-clotho/tree/6fa3ac180160d7f900f5030cfb4d0d6b4e525abe/src)
  and [test inventory](https://github.com/xinhi6716/axiom-clotho/tree/6fa3ac180160d7f900f5030cfb4d0d6b4e525abe/tests)

Some linked documents describe historical release boundaries. For migration
decisions, verify behavior against the pinned source and relevant tests rather
than treating every historical paragraph as the current executable contract.

