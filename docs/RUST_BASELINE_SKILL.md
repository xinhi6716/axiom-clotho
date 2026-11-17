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

The v1 format carries caller declarations, NOT verified provenance. Its report
explicitly sets `provenance_verified`, `query_bytes_verified` and `fit_performed`
to `false`. There is no query-byte digest or model-output binding. Native
`predict` CSV row indices are not accepted as physical identities; this command
also does not accept Python saved-scoring files. Independently establish actual
row identity, complete held-out coverage, leakage-free baseline construction,
and prediction-time availability before making these JSON files. Matching false
keys with changed/swapped values still pass. No accuracy, causal, uncertainty,
or physical-independence claim follows from a successful comparison.

## Opt-in exact query-byte binding (v2)

To check a saved series' declared query digest against a file, use
`clotho.declared_regression_series.v2` in **all three** series files and add a
`query_sha256` field containing exactly 64 lowercase hexadecimal characters.
Supply the query artifact with `--query`:

```sh
cargo run --locked -p clotho-cli -- compare-saved \
  --observed observed-v2.json --model model-v2.json --baseline baseline-v2.json \
  --query query.csv --out comparison-v2.json
```

The command reads the query once (at most 64 MiB) and computes SHA-256 over its
exact bytes, without parsing, sorting, decoding, newline normalization or
rewriting. All three declared digests must match that computed digest. A
one-byte or newline change fails before output creation. The query can be any
byte artifact, including empty; accepting its digest does not validate its
format or establish a relationship between its rows and the scored series.

For a reproducible byte-only fixture, the three literal bytes `abc` have digest
`ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`.
Use `printf abc > query.bin` (no newline), set this digest in v2 copies of the
synthetic series above, and pass `--query query.bin`. This is deliberately not
a real model evaluation: the predictions were not generated from `abc`.

A v2 report has schema `clotho.declared_regression_comparison.v2`, includes
`query_sha256`, and sets `query_bytes_verified` to `true`. That flag means only
that the supplied query bytes matched the three declarations at read time. It
still sets `provenance_verified` and `fit_performed` to `false`. It does not prove
that predictions, observations or baselines were produced from the query,
that row identities are real, or that the dataset is correct. Manually copying
the same digest into all three files still passes. The digest is not a
signature, attestation, privacy guarantee, or protection against later changes
to the file. Reports do not include the query path or contents.

The v1 format and report are unchanged: use no digest field and no `--query`.
Mixed versions, v2 without `--query`, v1 with `--query` or a digest (even null),
missing/null/malformed v2 digests and mismatches are rejected. Existing semantic,
row-order and finite-number checks remain mandatory after byte matching. This
is not the Python saved-scoring protocol and does not load or run any model.

Implementation uses the pinned RustCrypto `sha2` crate with portable software
compression, not a handwritten hash or external executable. The lockfile pins
its transitive packages; no existing dependency version is changed.

## Opt-in declared group summaries

Add `--groups groups.json` to either v1 or v2 `compare-saved` to summarize
paired squared errors by an explicitly supplied group roster:

```json
{
  "schema_version": "clotho.declared_row_groups.v1",
  "cohort": "synthetic-1",
  "row_ids": ["a", "b"],
  "group_ids": ["wafer-A", "wafer-B"]
}
```

The sidecar must be a strict JSON object with exactly these fields. Its cohort
and complete ordered row IDs must match the validated series literally. Missing,
extra, reordered or duplicate row IDs, missing group assignments, unknown or
duplicate fields, blank/control-containing labels and labels exceeding 256 UTF-8
bytes fail before output creation. Input is bounded to 64 MiB. Group labels are
opaque literal strings: no trimming, hierarchical inference, or parsing IDs to
invent group membership. Repeated group labels are expected for multiple rows
in the same declared group; each row has exactly one assignment.

Only opted-in reports add `grouped_summary`, with its own
`clotho.declared_group_comparison.v1` schema. Existing pooled fields and v1/v2
report schemas remain unchanged. The summary contains lexically ordered groups,
each with `group_id`, `n_rows`, `baseline_mse`, `model_mse`,
`mean_squared_error_delta`, and `baseline_perfect`.
The delta is `mean((observed-baseline)^2 - (observed-model)^2)`: positive favors
the model. It is in squared target units, not relative skill or a percentage.
The pooled delta weights each row equally; the macro delta weights each declared
group equally. Counts include all groups: improved, tied, worsened and those
with a perfect baseline. Ties use the computed floating-point zero, no tolerance.

For example, four rows in one group with baseline error 2 and model error 0 give
that group's delta +4. One row in a second group with baseline error 0 and model
error 3 gives -9. The pooled delta is +1.4, but the equal-group macro delta is
-2.5. Neither summary replaces the other. The perfect-baseline group is retained,
not dropped or divided by zero. There is no group-relative ratio, so a tiny
baseline denominator cannot inflate the group delta.

Unlike the scale-safe pooled RMSE/skill, this optional summary requires each
raw squared error to be representable. Overflow, nonzero squares rounded to
zero, or detected unrepresentable scaled means reject the entire requested
report rather than emit misleading numbers. Scaled compensated accumulation
reduces cancellation, but outputs remain binary64 floating-point descriptive
statistics. The paired delta is accumulated before averaging; it may retain a
small gain lost when subtracting the separately rounded MSE fields.

The sidecar does not establish physical provenance, correct grouping, independent
samples, or effective sample size. Reports set `independence_verified` and
`provenance_verified` to false inside this summary. A v2 query digest does not
validate the sidecar's declared grouping. Group labels are included in the
private output; row IDs and raw values are not. No confidence intervals,
significance tests, model fitting, or causal claims are added.
