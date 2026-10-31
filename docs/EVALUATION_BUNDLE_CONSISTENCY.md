# Optional saved regression evaluation consistency

Current source provides `mlcalc validate-evaluation-bundle RUN_DIR` and Python
`groupbench.validate_evaluation_bundle(RUN_DIR)`. This is an opt-in, read-only
consumer, not a new writer format or packaged release. Existing `verify` remains
byte-integrity-only. A rehashed bundle with incorrect metrics can pass `verify`
and fail this check.

## Supported scope

Only regression evaluation schema `1.0` / protocol `nested-group-kfold-v1` and
schema `1.1` / `nested-group-kfold-distinct-v2` are supported. The latter requires
`config.categorical_encoding: distinct-v2`. Binary, prediction-only, unknown
versions and unknown model families fail closed. Existing writers are unchanged.

Checks cover:

- Exact evaluation checksum file set and the bytes used for parsing. HTML is
  hashed but never parsed, rendered or checked against the numerical summary.
- Configured baseline families plus `selected`, exact CSV columns, sequential
  zero-based row positions, finite observed/predicted values and row counts.
- Group labels as literal CSV text: `001`, `NA`, `nan` and `Inf` are identities,
  not numbers or missing values. Quoted commas/newlines retain CSV semantics.
- Group counts, complete fold coverage, one held-out fold per group, and each
  outer audit's train/test group lists, row counts, group counts and zero overlap.
- Summary schema, selection objective and primary entry consistency.
- All overall, per-group and per-fold metrics, recomputed with unchanged
  `regression_metrics` arithmetic and group aggregation parity-tested against the
  writer's `grouped_metrics`. Group ordering follows first occurrence;
  folds follow writer order. Counts, identities, field sets and null R² are exact.
  Float scores use `math.isclose(rel_tol=1e-10, abs_tol=1e-12)` to accommodate CSV
  float round trips. This tolerance is not a scientific error bound, unit inference
  or guarantee of sensitivity to smaller edits. Constant observed targets require
  null R², not sklearn's substituted score. Overflow/underflow failures propagate.

## Bounds and failure behavior

POSIX no-follow/directory-relative access is required. Limits are 16 KiB checksums,
16 MiB each for manifest, summary and CSV, 64 MiB HTML, 20,000 rows and at most
20,000 groups/folds. CSV numeric tokens are at most 64 characters; JSON numeric
tokens are at most 100. JSON duplicate keys and nonfinite values are rejected.
These are input bounds, not a latency guarantee. The consumer builds stable
row/group/fold partitions once, retaining within-partition input order. It calls
the unchanged regression metric arithmetic for each group and fold, reusing group
scores across their single held-out fold. Group-weighted reductions retain the
writer's first-occurrence order, metric definitions and float64 arithmetic.
This removes repeated full-row scans per group/fold without changing writers.
The explicit train-group lists in every fold audit still require work proportional
to their saved size, potentially groups times folds; this is not a claim of
universally linear runtime or qualification of every maximum-size bundle.
The bounded JSON parser is shared with the prediction checker, so its parse
errors may carry the `Prediction bundle schema` prefix. No source paths are read.

The same buffered bytes are checked and parsed, and file signatures are checked
again after computation. Symlinks, mismatched checksums, oversized inputs and
observed concurrent changes fail. There is no lock or filesystem-atomic snapshot;
use an idle directory. Unlisted files are not read. API failures raise ValueError
(or filesystem errors); CLI failures return status 2 and success returns 0.
The result names the checked scope and unvalidated metadata.

## What a pass does not establish

This only establishes internal consistency of retained outer-validation rows and
metrics within the declared tolerance. It does not authenticate origin, prove
that fitting happened or validate scientific independence, model quality or
causal claims. An invented but internally consistent bundle can pass. Features,
source digests, environment metadata, model parameters and inner-selection records
are not reconstructed or validated. No models are loaded/refit, no network calls
are made and no artifacts are written. Inner-validation predictions are not
recovered. Synthetic tests establish plumbing, not real-data performance.
