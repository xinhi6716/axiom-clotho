# Optional saved prediction consistency checks

`mlcalc validate-prediction-bundle RUN_DIR` and Python
`groupbench.validate_prediction_bundle_schema(RUN_DIR)` check a bounded subset of
saved **regression prediction** metadata and CSV relationships. They do not fit,
load models, follow source-file paths, render HTML, or contact a service.

This is separate from `mlcalc verify` / `verify_bundle`, whose existing contract
is unchanged: checksum byte integrity only. A byte-valid bundle can fail these
optional consistency checks. Neither command authenticates the bundle's origin.

## Supported combinations

| Outer mode | Outer schema | Protocol | Input support |
| --- | --- | --- | --- |
| `fit_predict` | `1.0` | absent | must be absent; reported unavailable |
| `fit_predict` | `1.1` | absent | requires nested version `1.0` |
| `fit_predict` | `1.2` | `group-fit-predict-distinct-v2` | requires nested version `1.0` |

Schema 1.2 also requires `config.categorical_encoding: distinct-v2`.
Legacy schemas must not contain an encoding override. Binary `schema_id`
formats, evaluation bundles, plans and unknown schemas/protocols fail closed.
They are not guessed from column names. This does not make old bundles invalid
for the byte-only verifier; it limits this explicitly selected consumer.

## What is checked

- Checksum file lists exactly `manifest.json` and `predictions.csv`. The hashes
  match the same byte buffers subsequently parsed, not files reopened later.
- Positive integer counts, never JSON booleans: training/prediction rows at most
  20,000, and training groups no greater than training rows.
- One to 100 ordered, unique feature names with nonempty dtype labels; names
  exactly match ordered `config.features`. Dtype is a recorded label, **not** an
  independently inferred type, unit or feature-semantic validation.
- CSV header exactly `row_position,prediction`, two cells per data row, no blank
  records. Row positions are exactly `0,1,...,n_predictions-1` in that order:
  `+0`, `00`, `0.0`, whitespace or Unicode digit substitutes are rejected.
- Predictions use at most 64 ASCII decimal characters, optionally signed and
  exponent-formatted, and convert to finite numbers. Finite underflow is not a
  numerical accuracy assessment. UTF-8, CSV quoting and NUL checks are explicit.
- For input support 1.0: fixed reference/action, feature names/order, counts
  within their populations, all-missing-source relationships, and necessary
  aggregate union bounds. Numeric below/above counts are mutually exclusive
  and disjoint from missingness; unrelated flags can overlap. An aggregate
  union is **not** assumed to equal the sum of its component counts.

Unknown extra manifest/feature metadata is allowed but **not schema validated**.
All JSON still rejects duplicate keys, nonfinite values, or numeric tokens longer
than 100 characters. This is a documented metadata subset, not a complete
manifest validator. Selection records, model parameters, environment, timestamps,
source hashes, training-group identifiers and unchecked metadata are not verified.
The optional current-source `package_source` record is also allowed but not
validated or authenticated; see [observed package-source metadata](PACKAGE_SOURCE_IDENTITY.md).
`PredictionSchemaResult.checked_scope` and `.unchecked_metadata` state the scope;
`.contract_version` is `1.0` for this consumer, not a new writer schema.

## Caller-declared source digests at creation

Current-source regression and binary benchmark, prediction, and corresponding
planning APIs reject malformed `source_sha256` and `prediction_source_sha256`
arguments before preparing data, scheduling or fitting. Each argument must be
`None` (not supplied) or exactly 64 ASCII hexadecimal characters. Uppercase and
lowercase are accepted and retained; whitespace, prefixes, non-text values and
wrong lengths are rejected rather than repaired. Signatures and writer schemas
are unchanged; this source change is not a new packaged release.

This is **syntax validation only**. The API does not read or hash a source file,
compare the declaration to input bytes, or authenticate provenance. Even a
well-formed digest may be wrong. CLI-computed file digests and independently
observed normalized-data fingerprints retain their existing meanings. This
creation-time guard does not expand the saved-bundle consumer's checked subset
above or retroactively validate historical source hashes.

## Bounds and filesystem safety

The parser accepts checksums up to 16 KiB, manifest up to 1 MiB, and prediction CSV
up to 4 MiB. These opt-in caps are intentionally smaller than the byte verifier's
64 MiB per payload; exceeding them is an unsupported size, not proof of bad science.
The CSV and manifest are collected into bounded buffers; this is not a streaming
constant-memory validator. Python parsing adds memory overhead beyond file sizes.

The POSIX no-follow, directory-relative regular-file checks used by the byte
verifier are reused. Symlink components, special files, `..` paths and detectable
replacements/mutations fail. Unsupported platforms fail closed. All files are
read relative to one pinned directory descriptor, and final signatures are
checked after parsing. This is **not a lock or filesystem-atomic snapshot**;
validate an idle directory. No guarantee prevents modification after return.

Unexpected unrelated files remain unread and unverified. `report.html` or
`summary.json` alongside a prediction bundle are rejected as stale evaluation
artifacts, without opening them. No executable model deserialization occurs.

## Interpretation and migration

A successful check establishes limited internal consistency of the saved bytes.
It does not prove predictions came from the stated model or inputs, recalculate
input-support flags from training values, validate uncertainty, scientific
accuracy, causal claims, physical units or deployment safety. No rows are changed,
removed or newly predicted. Older schema 1.0 absence is **unavailable**, never
zero flags. Do not rewrite historical bundles just to obtain a newer schema.

Existing writers, writer schemas, numerical algorithms and `verify` behavior are
unchanged. Select this additive API/CLI only when the documented subset is useful.
