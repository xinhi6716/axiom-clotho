# Baseline-relative regression skill

`clotho_core::evaluation::regression_mse_skill(observed, predicted, baseline)`
reports `1 - MSE(predicted) / MSE(baseline)` using identical aligned rows.
For example, targets `[1, 3]`, predictions `[1, 2]`, and baseline `[0, 1]`
give `0.8`: 80% lower squared error than that baseline.

- Positive favors the model; zero is a tie; negative scores are retained.
- A perfect baseline returns `None`, even if the model is also perfect.
- Unlike centered R², skill may be defined for constant targets.
- All inputs must be finite, nonempty, and equally sized. Invalid values are
  rejected rather than silently dropped, including with a perfect baseline.
- Raw squared errors are not formed, permitting large or tiny finite errors.
  Unrepresentable RMSE or score returns `EvaluationError::NumericOverflow`.

Choose the baseline before evaluation. Persistence should use only the value
available at the prediction origin, with the same target and horizon as the
model. Fit any learned baseline exclusively on training data. This API checks
numeric inputs, not timestamps, split provenance, physical independence, or
whether the supplied baseline leaks held-out outcomes.

The primitive itself is a core API, not a real-data accuracy claim.
The separate declared-series CLI adapter is documented below.
Compare paired errors on held-out groups and retain negative results. A gain
over noisy persistence alone does not prove physical state prediction or
causality: shared-read mathematical coupling can create apparent improvement.

## Declared ordered cohorts

`clotho_core::saved_comparison::compare_saved_regression` wraps the numeric
primitive for three in-memory `SavedRegressionSeries`: observed, model and
baseline. Each supplies a `CohortContract` (cohort, target, unit, horizon), ordered
opaque `row_ids`, and numeric values. Contracts and complete row rosters must
match literally. Duplicate IDs, reordered rows, missing rows, nonfinite values,
blank/control-containing declarations, more than 20,000 rows or declarations
longer than 256 UTF-8 bytes are rejected. No sorting, joins or normalization occur.

The report retains the declared contract, row count, both RMSEs and MSE skill.
It performs no fitting or selection. For a training-mean baseline, retain the
correct outer-training mean for each held-out row; for RRAM persistence, retain
the origin-available read in the same units and target/horizon. This API cannot
check those conditions or distinguish forged IDs from authentic ones.

This is not the Python `compare_saved_predictions` file protocol: it verifies
no byte checksums, model execution, physical identity, hidden cohort omissions,
historical independence, or units' actual meaning. In particular, swapping
values while falsely keeping the same IDs passes the declared-roster check.
The core API does not provide an uncertainty interval or a causal/accuracy claim.

## Fit-free declared-series CLI

`compare-saved` exposes the declared-cohort API without loading model artifacts,
training, or running prediction. It is separate from `compare` (nested model
selection). Each of the three JSON files must independently supply exactly:

```json
{
  "schema_version": "clotho.declared_regression_series.v1",
  "cohort": "synthetic-1",
  "target": "resistance",
  "unit": "ohm",
  "horizon": "1 write",
  "row_ids": ["a", "b"],
  "values": [1, 3]
}
```

Save that as `observed.json`. Make `model.json` with the same declarations and
keys but values `[1, 2]`, and `baseline.json` with values `[0, 1]`. From `rust/`:

```sh
cargo run --locked -p clotho-cli -- compare-saved \
  --observed observed.json --model model.json --baseline baseline.json \
  --out comparison.json
```

This synthetic example reports MSE skill `0.8`. Positive favors the model, zero
is a tie, negative favors the baseline; a perfect baseline serializes as JSON
`null`. The report includes both RMSEs, row count and declarations, but not the
raw row IDs or values. Unknown/duplicate JSON fields, invalid inputs, unsupported
schemas and unknown options fail before creating output. Inputs are limited to
64 MiB each and the core row/declaration limits above. Output must not already
exist, is owner-only on Unix, and is created only after scoring/serialization.
An I/O failure during writing can still leave a partial file; check exit status.

This format carries caller declarations, NOT verified provenance. The report
explicitly sets `provenance_verified`, `query_bytes_verified` and `fit_performed`
to `false`. There is no query-byte digest or model-output binding. Native
`predict` CSV row indices are not accepted as physical identities; this command
also does not accept Python saved-scoring files. Independently establish actual
row identity, complete held-out coverage, leakage-free baseline construction,
and prediction-time availability before making these JSON files. Matching false
keys with changed/swapped values still pass. No accuracy, causal, uncertainty,
or physical-independence claim follows from a successful comparison.
