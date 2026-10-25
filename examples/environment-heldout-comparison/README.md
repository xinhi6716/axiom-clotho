# Fixed environment-heldout feature comparison

A small **runnable local ML workflow**, separate from the Calculator's existing
nested-CV commands: fit five predeclared pipelines, predict devices from one
held-out environment without accepting their outcomes, then score a separately
provided exact cohort. Compare two fixed feature sets and a training-label-
shuffle diagnostic without calling either feature set causal.

This is a source-checkout Python companion. It reuses the Calculator's existing
numeric preprocessing/model builder and already-declared NumPy, pandas,
scikit-learn and threadpoolctl dependencies. It installs or downloads nothing,
uses one CPU thread per fit, saves no executable model, and sends no data
anywhere. Existing package APIs, schemas, nested selection and default commands
remain unchanged. No new package or binary release is implied.

## Try the invented example

From the repository root in an already provisioned environment:

```sh
PYTHONPATH=src python examples/environment-heldout-comparison/probe.py predict \
  examples/environment-heldout-comparison/fixtures/training.csv \
  examples/environment-heldout-comparison/fixtures/future.csv \
  --spec examples/environment-heldout-comparison/fixtures/spec.json \
  --accept-heldout-contract --output /tmp/invented-environment-predictions.json

# This separate command, not the fitting command, first reads held-out outcomes.
PYTHONPATH=src python examples/environment-heldout-comparison/probe.py score \
  /tmp/invented-environment-predictions.json \
  examples/environment-heldout-comparison/fixtures/outcomes.csv \
  --output /tmp/invented-environment-scores.json

PYTHONPATH=src python -m unittest discover -s tests -p test_environment_feature_probe.py -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

Use new output paths; there is no overwrite or implicit stdout-data mode. Both
commands exit 0 for these fixtures. Their values are invented software examples,
not measured semiconductor data, physical simulations or research findings.

The fixture gives two training environments with eight distinct device IDs
each and one held-out environment with four devices. Invented response obeys
`target = 3 + 0.2 * preprocess_setting`. A proxy follows the target in training
and reverses in the held-out environment. Arm A includes both columns; B uses
only the setting. This intentionally constructed association shift illustrates
a prediction failure mode. It does **not** prove that either feature is causal
or that dropping a feature is generally beneficial.

Analytic checks: the training-mean dummy predicts 3.9. Arm B predicts
`3.9 + 0.2 * (16/17) * (x - 4.5)` under the fixed unit-variance scaling and
alpha 1. Its held-out MAE is about 0.02353 V; the dummy MAE is 0.4 V. Tests also
compare both feature sets and shuffle fits against independent ridge equations.
The actual report retains all five results, including adverse shuffle results.
These fixture numbers are not model-performance claims.

## Freeze the declared question before looking at test outcomes

Edit a private copy of the explicit JSON specification, not the fixtures:

```json
{
  "schema_version": "environment-feature-probe-v1",
  "heldout_environment": "declared_new_environment",
  "features_a": ["pre_outcome_setting", "pre_outcome_proxy"],
  "features_b": ["pre_outcome_setting"],
  "ridge_alpha": 1.0,
  "shuffle_seed": 7,
  "shuffle_scope": "within_training_environment",
  "target_unit": "V"
}
```

All keys are required; unknown/duplicate keys and unsupported versions fail.
Feature lists are unique, nonempty, distinct as sets, and have at most 32 names
in their union. Overlap between A and B is expected and allowed. Metadata and
`target` cannot be selected. Alpha is positive, finite and at most 1e12; seed
is an explicit integer from 0 through 4294967295. Booleans are not numbers for
these declarations. The unit is descriptive: no physical unit validation or
conversion is performed. Freeze features, alpha, seed, environment, dependency
boundary and interpretation before test exposure. A saved hash cannot establish
that this happened.

### Exact CSV inputs and supported question

Training CSV: exactly `device_id,dependency_block,environment,target`, plus the
union of the two feature lists. Future CSV: exactly those three metadata
columns and the selected feature union, **without target**. Column order may
differ; additional/unselected columns are rejected. Outcome CSV for scoring:
exactly `device_id,target`, with the exact future device set, in any order.

The supported question is predictive transfer to new devices in one explicitly
held-out environment, with **no shared declared dependency block** across the
training/future boundary. It is not a general intervention, same-device
longitudinal or repeated-measure evaluation mode.

- Exactly one row per declared physical device in each input; no device may
  cross training/future. Duplicate rows are not averaged, purged or relabeled.
- A dependency block may denote a wafer, lot or a previously justified connected
  component of crossed dependencies. It must not cross the boundary. The tool
  does not construct blocks or verify their truth. Use the existing
  [dependency-group-components utility](https://github.com/xinhi6716/research-workflow-utilities/tree/main/tools/dependency-group-components)
  when multiple declared dependency axes require transitive grouping. Do not
  weaken constraints just to make the comparison run.
- Every future row must name the one held-out environment, which must be absent
  from training. At least two other training environments are required, each
  with at least two device rows and two distinct target values. These are
  supported-input restrictions, **not scientific sample-size sufficiency**.
- IDs and names remain exact strings. No whitespace trimming, Unicode
  normalization, numeric-ID conversion or alias guessing occurs. `001` and `1`
  are different. Strings must be nonempty, unpadded, at most 128 characters,
  and free of C0/C1 controls and embedded BOM. Distinct strings do not prove
  distinct devices, environments or blocks.
- Predictors are numeric only. An empty feature cell is missing and uses a
  training-fitted median. An entirely unobserved training feature is rejected;
  future values cannot supply its definition. Targets must never be missing.
- Numeric tokens are unpadded decimal/scientific strings up to 64 characters;
  finite binary64 conversion is required, without nonzero-to-zero underflow.
  Nonzero absolute input magnitude must lie between 1e-100 and 1e100; zero is
  accepted. This explicit numerical-range restriction avoids unsupported tiny
  target/variance arithmetic. Rescale source units rather than rely on clipping. Categorical features, NaN and infinity tokens
  are unsupported.

CSV is UTF-8 with optional BOM, parsed with Python's strict CSV reader. Standard
quoting and CR/LF/CRLF record conventions follow that parser; this is not the
stricter byte grammar of the separate physical-holdout tool. Blank records,
duplicate/missing/extra headers, ragged widths, invalid UTF-8 and NUL fail. Each
input is capped at 8 MiB and 10,000 rows, with a union of at most 32 predictors.
The parser's field-size limit also applies. These are input/work bounds, not a
hard RAM guarantee. Complete reports must also fit the 8 MiB output/read limit.

## What the five fits do

The existing numeric pipeline median-imputes and standardizes using training
rows only. It then fits:

1. `dummy`: training-target mean.
2. `a`: fixed ridge on feature set A and real training targets.
3. `b`: fixed ridge on feature set B and real training targets.
4. `shuffle_a`: the same A model with shuffled training labels.
5. `shuffle_b`: the same B model with the **same** shuffled labels.

For this dense numeric-only companion, Ridge uses a fixed `svd` solver and the
declared alpha. The shared pipeline builder supplies preprocessing; only this
companion changes its new estimator to `svd`, without editing the core model
builder or its existing `lsqr` behavior. This avoids the tiny-target stopping
arithmetic observed with iterative `lsqr` in boundary tests. There is no hyperparameter selection, inner CV, feature search, refitting
on test rows or outcome-guided retry. All five pipelines include train-only
preprocessing; the dummy's preprocessing does not make its mean feature-dependent.
Expected work is five pipeline-fit attempts, not five cross-validation folds.

For each training environment, device IDs are sorted exactly and the declared
seed permutes its donor positions once. A and B receive that identical mapping.
The report records donor/recipient IDs, changed assignments, changed target
values and no-op indicators. A random permutation can be the identity, and
swapping equal labels can leave values unchanged. Both cases are retained;
there is no redraw until performance degrades.

Within-environment shuffling preserves each environment's target multiset,
but may break block dependence. It is a **training-label-shuffle diagnostic**,
not a causal negative control, exchangeability-preserving randomization,
permutation test, significance calculation or null distribution. One seeded
shuffle cannot establish that a real feature set carries signal or is causal.

### Incomplete execution

A fit or prediction numerical failure marks the entire arm failed, retaining
all future device rows with null values for that arm. Other requested arms are
still attempted. The prediction command emits `INCOMPLETE_PREDICTIONS`, names
the failed stage categorically, and exits 1. There is no zero-loss substitution
or silent arm omission. **Scoring refuses every incomplete artifact**; it never
computes a flattering subset comparison. Fitting warnings are recorded by
category without raw values. Unexpected programming errors are not scientific
results. CLI/schema/I/O errors exit 2, without a completed report.

## Separate, exact-cohort scoring

The prediction report contains no future outcome values and has no argument
for them. Only `score` reads the outcome file. It verifies the artifact schema,
body digest, finite values for all five arms, complete exact-ID cohort, declared
boundary and internal shuffle/fit metadata, including reconstruction of the
seeded donor mapping. Changed-target-value counts remain producer declarations
without the original training labels. Reordering outcomes is safe; missing,
extra or duplicate IDs fail rather than using an intersection join.

Scores are MAE and scaled-arithmetic RMSE in declared target units. A nonzero
aggregate that would underflow to a false zero is rejected, not called a perfect
prediction or tie. Scores have equal weight per supplied
device row. Every row retains its five predictions, target, absolute errors and
B−A absolute-error contrast. The aggregate contrast is the mean of the paired
row contrasts. Negative means B had lower absolute error on this exact supplied
held-out cohort only. No automatic winner, significance, confidence interval,
causal effect or expected future performance is inferred. Multiple devices in a
block are still equally weighted devices, not independent blocks or effective n.

The report preserves source/specification hashes, actual training membership,
explicit dependency counts, shuffle mapping, runtime versions and hashes of
numeric fitted states. There is no executable model file. The body hash is an
accidental-change/internal-consistency check, **not authentication or proof that
the stated run occurred**. A dishonest producer can recompute it. Record any
pre-exposure freeze and source authorization independently.

Python APIs in `probe.py`:

```python
prediction = predict(training_bytes, future_bytes, specification_bytes,
                     contract_declared=True)
scored = score(report_bytes(prediction), outcome_bytes)
```

Only literal Boolean `True` accepts the prediction contract. Scoring performs
no fit and cannot alter the prediction artifact. These example APIs are source
companions, not new exported `groupbench` package interfaces.

## Causal interpretation and safeguards

An environment label, date, device ID, predictive improvement or lower error
after restricting features does not identify a treatment effect. Real causal
inference needs a defensible estimand, exposure/intervention and outcome timing,
assignment/identification assumptions, confounding and selection analysis,
positivity/support, and suitable dependence-aware uncertainty. This tool makes
none of those assumptions true. Likewise a causal negative-control design needs
justified causal relationships; label shuffling does not supply one.

The tool does not discover hidden data leakage, global preprocessing performed
before input, post-outcome features, omitted devices, false physical identities,
future-distribution changes or unmeasured confounding. Its boundary checks are
necessary software constraints for this narrow declared prediction task, not
scientific validation. Do not repeatedly change features or alpha after scoring
and continue calling the same cohort untouched test data.

Reports deliberately disclose device/block/environment identities, predictions
and, in scoring mode, outcomes. Keep them private unless explicitly approved for
sharing. Outputs use exclusive creation with requested POSIX mode 0600, refuse
existing files/directories/live or dangling symlinks, and create no parents.
Serialization and size checks happen before creation. A write failure can leave
a partial new file; discard it. Regular-file reads use nonblocking open so FIFOs
do not hang. Caller-selected input symlinks to regular files are followed.
Diagnostics are categorical and omit raw values and paths. This is not a hostile
concurrent-filesystem sandbox. Windows ACL/runtime behavior and other dependency
versions have not been verified by this addition.
