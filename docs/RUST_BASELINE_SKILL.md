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
keys but values `[1, 2]`, and `baseline.json` with values `[0, 1]`.

Run the shell examples in this guide from the repository root. Their example
inputs and outputs are relative to that directory. Cargo uses the explicit
`rust/Cargo.toml` workspace; see the [native build guide](RUST_NATIVE_WORKFLOW.md#build-and-run)
for dependency setup and offline builds.

```sh
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-cli -- compare-saved \
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
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-cli -- compare-saved \
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

## Generate a v2 series with native regression prediction

Native `predict` can produce the existing v2 saved-series JSON instead of CSV.
Supply **both** `--row-id ID_COLUMN` and `--saved-contract CONTRACT.json`:

```json
{
  "schema_version": "clotho.saved_prediction_contract.v1",
  "cohort": "synthetic-1",
  "target": "resistance",
  "unit": "ohm",
  "horizon": "1 write"
}
```

```sh
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-cli -- predict \
  --artifact model.json --csv query.csv --row-id measurement_id \
  --saved-contract contract.json --out predictions-v2.json
```

The contract is a strict named JSON object. Its target must match the fitted
artifact target exactly; all declarations obey the saved-series text limits.
The selected ID column must exist, be separate from model features and target,
and contain unique, nonblank, control-free IDs of at most 256 UTF-8 bytes. IDs
are copied in query row order without numeric conversion, trimming or sorting.
The output has exactly the established `clotho.declared_regression_series.v2`
fields and can be passed as `--model` or `--baseline` to `compare-saved`.

The query is read once: the exact byte buffer parsed for prediction is also
SHA-256 hashed, so there is no second-read race between prediction and digest.
Before v2 output is published, that buffer must also pass the strict CSV syntax
checks described below for the label-separated workflow. Malformed quoting,
blank records, and unsupported record endings cannot silently become a bound
series. The check does not change the bytes being hashed. A rejected export can
have computed predictions internally; it creates no output.
Only artifact-selected feature columns feed the estimator. Other query columns,
including a target column if present, are not treated as observations or used
for prediction. The digest nevertheless covers the entire file, including those
unused columns, headers and line endings. No observed-series file is fabricated.
Prepare independently justified aligned observations for comparison. A learned
baseline can be exported through its own separately fitted regression artifact;
its training split and leakage-free construction remain the caller's duty.

This path uses the existing native predictor, which **refits the saved model
specification on its stored training data**. It is not the fit-free comparator.
Strict-numeric and explicit mixed regression artifacts use their existing
preprocessing behavior. Binary classifiers, `--threshold`, partial option pairs,
invalid IDs or contracts fail without creating the output. Without these two
options, legacy prediction CSV output is unchanged. Outputs remain new-file-only
and owner-readable/writable on Unix; they contain supplied row IDs and predictions.

The export records what this invocation computed, but it is not signed or
attested. Files can still be edited or their digests copied. It does not establish
real-world identity, truthful unit/horizon declarations, complete cohorts,
training provenance, held-out independence, or correctness of physical data.
`compare-saved` retains its existing explicit verification boundaries.

## Attach separate observed labels without rewriting the query

`prepare-observed` builds a v2 observed-series JSON from a feature-only query and
a separate ID/target CSV. This closes the native label-separated scoring path:

```sh
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-cli -- prepare-observed \
  --query query.csv --observed labels.csv --row-id measurement_id \
  --saved-contract contract.json --out observed-v2.json
```

Use the same `clotho.saved_prediction_contract.v1` contract as prediction. The
query must contain the explicitly named ID column and **must not contain the
contract's target column**. The labels CSV must contain exactly two headers: that
ID name and that target name, in either order. Other columns, duplicate headers,
missing target/ID headers and a target equal to the ID name are rejected.

For example, a query roster `a,b` and labels `b,3` then `a,1` produces ordered
IDs `["a","b"]` and values `[1,3]`. This is a strict one-to-one alignment, not an
inner join: both rosters must be unique and contain exactly the same literal IDs.
Missing, extra, repeated, blank, overlong or control-containing IDs fail before
output creation. IDs are never trimmed, numerically coerced, synthesized or
inferred from row positions. Labels may be shuffled; the query order is retained.

Targets must be explicit finite binary64 numbers. Empty/invalid values, NaN,
infinity, overflow and nonzero decimal inputs rounded down to zero are rejected,
not dropped or imputed. Ordinary binary64 rounding still applies; this is not an
arbitrary-precision decimal format. Representable subnormal values and signed
zero are retained. Shared input limits remain 64 MiB per file and 20,000 CSV rows.

The exact query buffer is read once for parsing and SHA-256, is never rewritten,
and supplies the output's `query_sha256`. Labels are read separately and their
bytes do not alter that digest. Existing bound predictions for those exact query
bytes can now be scored with `compare-saved --query query.csv --observed
observed-v2.json ...`. A later change to query bytes still invalidates comparison.
The output is a new private file containing IDs and observed values; pre-existing
files are never overwritten. All parsing, matching and serialization precede
creation. No model is loaded, trained or executed by `prepare-observed`.

This checks declared ID correspondence and byte identity, not label authenticity,
truthful units/horizon, real-world identity or correct outcomes. A target-free
header does not prove that remaining columns are leakage-free. It cannot establish
when labels became available, prior query immutability, blinding or custody. The
labels file is not authenticated or hashed into a new provenance protocol. The
existing v2 format, comparison boundaries and predict/compare defaults remain
unchanged.

For this command, both CSV byte buffers undergo strict lexical validation
before the existing parser runs. Inputs must be valid UTF-8; one leading UTF-8
BOM is accepted as the parser's header marker, but remains included in the query
digest. Record endings must be LF or CRLF (an EOF without a final terminator is
allowed). Blank physical records outside quoted fields are rejected rather than
silently skipped. Quotes may start only at the beginning of a field; embedded
quotes must be doubled, and a closing quote must be followed immediately by a
comma, record ending or EOF. Quoted commas and quoted UTF-8 IDs are supported.
Quoted field content is decoded by the existing CSV parser; embedded line breaks
remain content and are subsequently rejected in IDs by the control-character
rule. No whitespace is trimmed. The v2 prediction producer uses the same syntax
check before publishing; legacy prediction CSV and byte-only comparison are unchanged.

## Pool caller-supplied scalar comparison reports

`pool-saved` reduces a complete, explicitly listed set of ungrouped v2
`compare-saved` reports without refitting or averaging their skill scores:

```json
{
  "schema_version": "clotho.saved_report_manifest.v1",
  "model_id": "ridge-fixed",
  "baseline_id": "training-mean",
  "reports": ["fold-a.json", "fold-b.json"]
}
```

```sh
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-cli -- pool-saved \
  --manifest reports.json --out pooled.json
```

Relative report paths resolve against the manifest's directory, not the working
directory. Absolute paths are also accepted. The manifest must list 1–64 reports;
each manifest/report file is limited to 64 KiB. Inputs must be strict named JSON
objects with no unknown or duplicate fields. This first interface intentionally
accepts only the exact ungrouped `clotho.declared_regression_comparison.v2`
shape; v1, already pooled and `--groups` reports are rejected rather than silently
ignoring information. No wildcard discovery, missing-file skipping or inferred
fold list is performed. The caller must independently decide the complete list.

Each report must declare 1–20,000 rows (at most 1,280,000 in the pool), a unique
cohort label and a unique lowercase query digest, with identical target, unit
and horizon across reports. Report-supplied verification flags and validation
text must match the existing v2 shape. These strings and flags are **untrusted
input declarations**, not evidence that this command verified any query bytes.
RMSEs must be finite and nonnegative, and skill must exactly agree with their
binary64 ratio (null for a perfect baseline). Nonzero decimal tokens that would
underflow to zero are rejected. Editing or rounding report numbers may cause
consistency checks to fail; use the unchanged native report files.

For report row counts `n_i`, pooled model MSE is
`sum(n_i * model_rmse_i^2) / sum(n_i)`, with the equivalent baseline formula.
The command uses scaled, compensated accumulation to avoid raw-square overflow,
then reports both pooled RMSEs and `1 - pooled_model_MSE / pooled_baseline_MSE`.
A perfect pooled baseline produces null skill, even if the model is perfect.
Perfect-baseline individual reports still contribute their model error and rows.
There is no epsilon denominator or clipping: negative scores remain negative,
and an unrepresentable ratio or detected normalization underflow fails before
output creation. Ordinary binary64 rounding remains; original per-row residuals
cannot be reconstructed from rounded RMSE scalars.

For example, a 9-row report with model RMSE 0 and baseline RMSE 2, plus a 1-row
report with model RMSE 3 and baseline RMSE 0, gives pooled model MSE 0.9, baseline
MSE 3.6 and skill 0.75. Averaging the individual report skills would be undefined
and would incorrectly omit the second report's error.

The new `clotho.declared_report_pool.v1` output retains the declared source
cohort, digest, row count and RMSEs in manifest order, not file paths. It sets
`query_bytes_verified`, `roster_completeness_verified`, `disjointness_verified`,
`provenance_verified` and `fit_performed` to false. **Scalar reports have no row
rosters. Different digests/cohort labels do not prove disjoint samples**, correct
row counts, complete fold coverage, genuine report origin, common baseline
construction, independence or held-out evaluation. The command cannot detect
shared rows under different IDs/files or invented scalars and digests. Its
output is a conditional descriptive aggregation of supplied reports, not an
uncertainty estimate, causal claim, signed receipt or cross-validation audit.
Existing predict/compare defaults are unchanged; output is a new private file.

The manifest also requires distinct `model_id` and `baseline_id` declarations
(nonblank, no control characters, at most 256 UTF-8 bytes each). Choose these
before pooling to explicitly state which pair the complete list is intended to
represent. They are echoed with `identities_verified: false` and
`membership_verified: false`: v2 scalar reports contain no originating model or
baseline IDs, so this command cannot check that a report actually belongs to
that declared pair. A common unit/target/horizon is not evidence of a common
model or baseline. Distinct query digests also permit the same roster to have
been re-encoded or reordered; those declarations do not establish disjointness.

### Subnormal RMSE ratio boundary

The native skill calculation rejects nonzero subnormal RMSE values when both
model and baseline have nonzero error. Their binary64 quantization can materially
change the MSE ratio: residuals `[min_subnormal, 0]` and
`[min_subnormal, min_subnormal]` both round to the same RMSE, which would falsely
report skill zero instead of 0.5. Reversing them would similarly hide skill -1.
The function returns the existing numeric-range error, and `compare-saved` does
not create an output. Normal RMSE calculations keep the existing scalar formula
and pooling compatibility. A perfect baseline is still undefined (`None`), and
a perfect model against a nonperfect baseline still has skill 1. This is a
conservative range restriction, not arbitrary-precision scoring; even some
exactly representable subnormal ties are rejected. Choosing appropriate units
before producing predictions can avoid this range, but no inputs are silently
rescaled. Scalar reports cannot reconstruct precision already lost in their
RMSE fields. `pool-saved` applies the same restriction to both imported scalar
reports and the pooled result, preventing a caller-supplied report from bypassing
the range check. Formats and grouped calculations are unchanged.
