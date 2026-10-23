# Explicit binary classification protocol v1

This additive local workflow accepts numeric 0/1 labels. It never infers the task from a regression target and never uploads data. The original synthetic example uses 1 for fictitious fail and 0 for fictitious pass; real users must define their own label meaning, pre-outcome features, and scientifically justified grouping before analysis.

## Interfaces

```bash
mlcalc binary plan run examples/synthetic_binary_process.csv --target fail --group lot
mlcalc binary run examples/synthetic_binary_process.csv --target fail --group lot --out runs/binary-evaluation
mlcalc binary plan predict examples/synthetic_binary_process.csv examples/new_binary_process.csv --target fail --group lot
mlcalc binary predict examples/synthetic_binary_process.csv examples/new_binary_process.csv --target fail --group lot --out runs/binary-prediction
mlcalc verify runs/binary-evaluation
```

Use `PYTHONPATH=src python -m groupbench` instead of `mlcalc` in an already provisioned source environment. Both executable aliases remain supported. Binary commands accept the existing mutually exclusive `--features` / `--features-file` allowlists. Plans print JSON to stdout only; no model/transformer construction, fitting, output-directory creation, or report writes occur. No binary `--objective`, `--threshold`, `--overwrite`, or model-loading option exists.

Python exports: `GroupBenchClassifier`, `BinaryBenchmarkResult`, `BinaryPredictionResult`, `benchmark_binary`, `fit_predict_binary`, `plan_binary_benchmark`, `plan_binary_prediction`, and `make_binary_fixture`. See [the example](../examples/binary_python_api.py). The estimator's inherited `score` is ordinary pooled label accuracy, not its selection objective.

## Labels and missing classes

Targets must be one-dimensional finite real numeric values exactly 0 or 1. Validate original values before narrowing or casting: reject booleans (including `[0, True]`), complex values, in-memory string/object encodings, missing values, fractional probabilities, -1/1 and multiclass labels. `[0, 1.0]` is valid. No row is dropped, thresholded, or automatically remapped. Already-cast numeric arrays are interpreted as supplied; earlier caller conversions cannot be recovered.

CSV has no intrinsic types: ordinary pandas inference may parse 0, 1, 0.0, 1e0 or quoted numeric tokens as numeric. Such exact numeric 0/1 values are accepted; Boolean/text columns are not. This is not a new lexical CSV grammar. Existing UTF-8/header/field-count/row/feature/group validations still apply.

Both classes must occur in every actual fitting partition, including every nested inner training partition and family refit. The entire schedule is checked before the first fit, even for Dummy-only runs. An infeasible schedule fails with the fold and missing class. It never retries seeds, stratifies, splits groups, drops hard groups, or falls back to row-random validation. One-class validation/test groups remain valid and included in probability metrics.

## Frozen procedure

1. Reuse the existing deterministic, approximately row-balanced whole-group splitter without target information. Default outer seed is 0; inner selection seed is outer seed + fold + 1. Full-training prediction selection uses its supplied seed directly. All random models use the selection seed.
2. In every fit, learn numeric median imputation and scaling, categorical constant imputation and sparse train-only one-hot encoding (max 64 categories, unknown categories ignored). No test/new-row statistics enter fitting.
3. Default candidates in order: prior Dummy; L2 lbfgs Logistic with C=0.1,1,10, tolerance 1e-4 and max_iter=1000; Extra Trees at max_depth=6,12, each 64 trees, max_leaf_nodes=256, min_samples_leaf=2, max_features=sqrt, gini, bootstrap=False. All train with equal row weights, class_weight=None, no resampling. The Dummy prior is training row prevalence, not an optimized equal-group constant.
4. Combine each candidate's inner out-of-fold probabilities. Minimize the mean of group Brier scores, not pooled loss or mean fold scores. Choose the best inside each family and overall, resolving exact ties by declared candidate order. Refit every requested family once; selected reuses the winning family without another fit.
5. Predict untouched outer groups once per entry/row. Outer results are descriptive and must not become another tuning stage. Full-training `fit_predict_binary` is a separate inner-selected refit, not an unbiased performance estimate.

No objective, candidate-grid, class-weight, calibration, or threshold adaptation occurs. Logistic constructor keywords omit penalty/l1_ratio/multi_class/n_jobs for compatibility: finite-C lbfgs retains L2 defaults in inspected sklearn 1.4 and installed 1.8. Dependency docs are not a substitute for an executed version matrix. Any nonconvergence or fit failure aborts, without skipping/retrying candidates or writing a success bundle. A failed estimator refit preserves its previous complete fitted state, never a mixed partial state.

## Metrics and precision

The mathematical binary Brier score is `(p1-y)^2` averaged over rows, on [0,1], not the two-column squared-error sum on [0,2]. `group_brier` averages each group's Brier equally. This is the fixed primary; its mathematical propriety is not a claim of empirical calibration, a population sampling design, or exact float64 ranking.

Secondary `log_loss_clipped` uses a metric-only copy `q=clip(p1, eps, 1-eps)`, with float64 epsilon 2.220446049250313e-16. For label 1 use `-log(q)`; for label 0 use `-log1p(-q)`, with natural logarithms. Report pooled, group-balanced and worst-group versions, clipping counts and raw endpoint counts. Clipping ties distinct extreme probabilities, so this finite diagnostic is not claimed strictly proper across the entire raw interval. Raw probabilities are unchanged in the output.

Guard original model outputs before conversion: exactly two ordered classes [0,1], real finite numeric probabilities in [0,1], row sums within absolute 1e-12 of one (no relative tolerance). Complex, object, Boolean, and wider-than-float64 probability arrays are rejected rather than silently narrowed/repaired. Float16/32 values may be promoted exactly to float64. No probability clipping or renormalization repairs model output.

Reject a nonzero residual whose squared error becomes zero, or any positive loss terms whose pooled/per-group/group-balanced mean becomes zero. The error explains float64 scoring resolution; do not rescale probabilities or floor scores to zero. Representable nonzero subnormal losses and genuine exact-endpoint zeros remain allowed. Final Brier values must be finite in [0,1] with no tolerance/clamping. Ordinary floating-point rounding still applies.

Fixed label predictions use raw p1 >= 0.5, including class 1 on ties. Report class supports, predicted supports, mean probability, prevalence, confusion counts, accuracy, recall_1, specificity_0, precision_1 and balanced_accuracy. A zero denominator yields JSON null, never a fabricated zero: recall needs true 1, specificity needs true 0, precision needs predicted 1, balanced accuracy needs both true classes. Do not drop one-class groups or average threshold metrics over an unstated subset of groups. No ROC/PR-AUC, F1, confidence interval, significance or fairness claim is added.

Show all families and selected, worst groups, and selected-minus-Dummy group Brier deltas and group wins/ties/losses when Dummy was requested. Positive delta means worse. Without Dummy, mark the comparison unavailable and perform no hidden fits.

## Bounds and no-fit plans

Training and new frames each admit up to 20,000 rows and 100 source features. Inner/outer folds are integers from 2 to 10. At most six candidates, three family refits per selection, 1000 Logistic iterations, 64 trees/model and 256 leaves/tree are allowed. Fits/predictions use one thread, including threadpool limits. Encoded width is at most 6,400; no explicit dense expansion is added.

For C candidates, F families, K inner folds and O outer folds, prediction fits C*K+F pipelines; evaluation fits O*(C*K+F). Defaults are 21 prediction and 105 evaluation fits. Maximum admitted counts are 63 and 630. Dummy and preprocessing are counted in each pipeline fit; individual trees/transform steps are not separate fits. Selected has zero extra refits. Plans and execution consume the same actual schedule, including derived seeds and class counts. Plans omit raw group IDs and feature rows.

These are algorithm/input bounds, not hard time/RAM quotas. CSV snapshots, long strings, normalized copies, encoded arrays, model internals and manifests still consume resources. Use trusted local files and OS-level isolation for hard quotas. Per-payload output size is capped at 64 MiB before writing; serialization is not a streaming-memory guarantee.

## Artifacts, schemas and compatibility

Each binary JSON uses schema_version "1.0", task "binary_classification", an explicit label_contract and one schema_id:

- mlcalc.binary.evaluation.summary
- mlcalc.binary.evaluation.manifest
- mlcalc.binary.prediction.manifest
- mlcalc.binary.execution_plan, with mode run/predict

Evaluation protocol: nested-group-binary-brier-v1. Prediction protocol: group-binary-brier-fit-predict-v1. Configuration, normalized/raw-source fingerprints, environment, ordered candidates, seeds, class supports, metric conventions and full split/selection audit accompany results. Cross-version/hardware bitwise equality is not promised.

Evaluation files remain summary.json, manifest.json, predictions.csv, report.html plus checksums.json. CSV begins row_position, group, outer_fold, observed_label, then probability_0_ENTRY, probability_1_ENTRY, predicted_label_ENTRY for requested families and selected. Prediction bundles contain manifest.json, predictions.csv and checksums.json; CSV columns are row_position, probability_0, probability_1, predicted_label. Prediction source-only input_support retains the existing schema and its non-calibration/non-OOD limitations.

Binary output is fresh-only: none of the five known output names may exist or be a symlink. CLI checks before fitting; API writers check before writing. No overwrite option exists. Writes reuse the existing staged atomic per-file no-clobber writer, not a whole-directory transaction. A race/interruption can leave a partial bundle; it must not be called success. Review identifiers/labels/probabilities before sharing; spreadsheet applications may interpret formula-leading identifiers.

The existing byte verifier remains unchanged: it checks the fixed two/four payload sets without parsing content, executing HTML, deserializing models or reading source references. It reports generic evaluation/prediction byte integrity only, not binary schema validity, scientific correctness, authenticity or proof of execution. Existing no-follow/POSIX/size restrictions remain.

All previous regression signatures, defaults, CSV columns and schema versions are unchanged, including regression targets containing 0/1. No executable model is saved/loaded; retain the in-memory estimator or rerun the local fit. Breaking binary schema changes require a major schema version; compatible fields a minor version, with migration notes. Protocol/candidate/metric changes require an explicit config/protocol update, not silent benchmark reuse.

## Scientific boundaries

The included fixture is original invented data with group effects and missing features, not SECOM, proprietary data or a physical semiconductor model. It is generated once from a seed, never rejection-sampled until a model wins or folds pass. Example new rows are reused invented feature rows to illustrate the interface, not independent validation. Synthetic results establish only software behavior.

Probabilities are not guaranteed calibrated failure risk, causal effect, OOD safety or permission for automatic wafer/process disposition. A fixed 0.5 threshold is not cost-sensitive decision analysis. Grouping does not solve crossed/time dependence, proxy features, duplicates or selection bias. Absence of source-feature warning flags is not proof of support or safety. Operational use requires independent domain validation and human decisions.

Primary references: [Brier](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html), [log loss](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html), [Logistic 1.4](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html), [Logistic 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html), [Dummy](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html), [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).
