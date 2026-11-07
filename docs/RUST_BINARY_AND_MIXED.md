# Native binary classification and mixed-feature preprocessing

This extension adds native logistic classification, a training-prevalence baseline,
and explicit mixed CSV preprocessing to the Rust workflow. It is a new documented
protocol, not a bitwise port of sklearn or a claim that the entire legacy product
has been rewritten. No Python subprocess is used at runtime.

## Binary tasks

Use `--model logistic` or `--model majority` with labels exactly 0 and 1.

```sh
clotho evaluate --csv devices.csv --target failed \
  --features temperature,thickness --group wafer --folds 5 \
  --model logistic --alpha 0.1 --threshold 0.5 --out binary-study

clotho fit --csv devices.csv --target failed \
  --features temperature,thickness --model logistic --alpha 0.1 \
  --out binary-model.json

clotho predict --artifact binary-model.json --csv new-devices.csv \
  --threshold 0.5 --out probabilities.csv
```

The target `failed` and group `wafer` are examples of column names, not an included
real dataset. Supply a licensed, scientifically appropriate dataset. Calling an
arbitrary column 'wafer' does not establish independence.

`LogisticClassifier` uses training-only numeric standardization, an unpenalized
intercept and objective mean binary log loss + (alpha/2) × squared standardized
feature coefficients. This alpha convention differs from the ridge SSE objective;
the same numerical alpha does not imply the same penalty strength. Non-convergence,
unsupported numeric ranges, invalid labels and single-class training are errors.
In particular alpha=0 on separable data need not have a finite optimum; increasing
regularization is a model-design choice, not a reason to report a failed fit as
successful. `majority` returns the empirical training positive-label prevalence;
its thresholded decision is the majority class at threshold 0.5 (ties positive).

Prediction writes positive-class probability and a thresholded label, not a
regression value. The fixed threshold defaults to 0.5, using probability >=
threshold as positive. Threshold values must be finite in [0,1]. Choosing a
threshold from held-out results is tuning and needs an independent evaluation.
These probabilities are not automatically calibrated.

Grouped evaluation uses the same group-disjoint deterministic split as regression,
**without class stratification**. Logistic training folds must each contain both
classes; otherwise the study fails without being published. It does not silently
skip inconvenient folds. The prevalence baseline can fit one class.

Binary reports contain accuracy, balanced accuracy, precision, recall, F1,
log loss, Brier score, ROC AUC, class counts and confusion counts. ROC AUC handles
ties with half credit and is null for single-class truth. Undefined ratios are
null, rather than invented zeros. Log loss clips endpoint probabilities using
epsilon 1e-15; it is therefore a finite clipped diagnostic, not infinite exact
log likelihood at impossible events. See the metric API for all denominator rules.

Reports include pooled row-weighted, per-fold and per-group results. Equal-group
mean log loss and accuracy use every group. Mean group balanced accuracy uses
only groups where that metric is defined, and explicitly records its denominator.
AUC is not naively averaged across incomparable groups.

## Explicit mixed preprocessing

The default remains strict numeric input. Enable the mixed path deliberately:

```sh
clotho evaluate --csv devices.csv --target failed \
  --features temperature,thickness,material \
  --preprocess mixed --categorical material \
  --group wafer --folds 5 --model logistic --alpha 0.1 --out mixed-study
```

`--preprocess mixed` also permits numeric-only median imputation without any
categorical columns. The `--categorical` list must be a unique subset of the
explicit feature list. No automatic column type or missing-marker inference occurs.

- Only empty CSV cells are missing. Literal `NA`, `null`, `None` and similar
  categorical strings remain literal categories. Whitespace is significant.
- Numeric missing cells use the **training-fold median**; an entirely missing
  training numeric column is filled with zero. Even medians avoid overflow.
- Each categorical feature gets a lexically sorted training vocabulary and a
  separate missing indicator. Query-only unknown categories map to all zero.
  Missing and unknown are different; neither expands the training vocabulary.
- At most 63 observed category strings plus one reserved missing indicator per
  categorical feature. Exceeding this cap fails explicitly. This differs from
  sklearn's infrequent-category pooling. The preprocessor can produce up to
  6,400 columns, but current estimators accept at most 100 encoded features;
  wider models fail instead of silently dropping categories.
- Feature names are indexed and collision-free; category values are not placed
  in generated output names. Fitted metadata describes source columns,
  missing counts, numeric fills and expansion widths, not source row contents.

Every evaluation fold fits medians and vocabulary on its **training rows only**.
The held-out rows are transformed afterwards. Deployment fits on all supplied
training rows; reload reconstructs the same fitted preprocessing. Reordered query
columns are aligned by original names. No imputation uses the query distribution.
The all-missing-zero fallback is a software convention, not a physical model.

## Persistence, privacy and compatibility

New fits use `clotho.numeric.v3`. This accepts prior numeric-regression v1 and
numeric-model v2 specifications through explicit compatibility paths; it does not
read Python pickle files. Mixed specifications store the selected **raw training
cells**, feature roles and targets so preprocessing can be reconstructed. These
can reveal materials, process values or sensitive category strings. Keep the model
private, as with raw datasets. The existing private Unix modes and no-overwrite
rules apply; Windows uses directory ACLs.

Do not pass new schemas to old binaries or assume that protocol evolution
preserves identical outputs. Ordinary binary64 rounding, optimization tolerances
and each model's documented numeric limits still apply. Numeric JSON parsing uses
roundtrip support; it is not an integrity signature.

## Not included

No random forests, histogram boosting, binary/mixed nested model selection, intervals,
forward-time evaluation, measurement-file adapters, legacy artifact importer,
HTML reporting, automatic causal identification or material-science accuracy
claim is provided by this extension. These remain tracked in the migration
inventory. Current examples and software tests do not substitute for real-data
validation, calibration studies or comparison with strong scientific baselines.
The separate [nested numeric regression path](RUST_NESTED_SELECTION.md) does
not yet accept mixed preprocessing or classification.

## Verification record (2026-11-07)

The workspace passed **167 tests** on Rust 1.70.0 and 1.99.1: 30 CLI tests,
135 core integration tests and two core unit tests. New coverage includes binary
fit/reload/evaluate, median/vocabulary leakage checks, 80 global categories with
40 per training fold, unknown versus missing categories, private artifact errors,
v1/v2 compatibility and encoded-width limits. Stable Clippy with warnings denied
and changed-file formatting checks passed.

Independent review checked five-model fold predictions against separately fitted,
saved and reloaded raw-training-fold models; predictions agreed exactly in the
review fixtures. Independent logistic numerical checks included 2,000 regularized
stress fits and 177 successful SciPy reference comparisons (largest probability
difference 2.08e-8); three unregularized separated cases correctly refused a finite
solution. These are numerical/software checks, not semiconductor accuracy evidence.

The release binary ran all 15 mixed example paths (fit, predict and evaluate for
five models). Example files are `rust/clotho-cli/examples/synthetic_mixed_process.csv`
and `synthetic_mixed_query.csv`; values and labels are explicitly synthetic.
Queries include missing cells and an unseen synthetic category. No real laboratory
measurements, private manuscript data, Python parity, hosted CI or production
certification is claimed.
