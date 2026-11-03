# Save and validate fixed temporal evaluations

**Current, unreleased source only.** This does not change a frozen download,
package version, legacy result protocol, or CLI modeling route.

The Python API persists an already completed
[`GroupForwardEvaluationResult`](GROUP_FORWARD_BASELINE_EVALUATION.md) and checks
saved bytes and declared-result consistency without rerunning models. It retains
the original `partial_predeclared` coverage: warmup, crossing, and late groups can
remain untested. No full-OOF conversion, selected model, estimator serialization,
result-to-model loader, prediction/refit API, search, HTML report, or upload is
added.

## Python API

```python
from groupbench import (
    GroupForwardBundleError,
    GroupForwardBundleValidation,
    write_group_forward_evaluation_bundle,
    validate_group_forward_evaluation_bundle,
)

# result is an already completed exact GroupForwardEvaluationResult.
# The parent directory must already exist; the output leaf must not exist.
payload_path = write_group_forward_evaluation_bundle(result, "runs/temporal-001")
checked = validate_group_forward_evaluation_bundle(payload_path.parent)
assert checked.validation_fits_performed == 0
assert checked.declared_pipeline_fits == 2 * checked.n_windows
```

Neither call fits a model or proves that one was fitted. The writer accepts the
exact result class, not a preview, dictionary, subclass, or executable model.
It checks a bounded detached projection of every retained field before creating
an output directory. Being a frozen dataclass alone is not evidence of valid
content. For writer admission, an exact `mappingproxy` type alone is also
insufficient: before performing any operation on a retained proxy, the writer
requires its GC referents to expose exactly one directly inspectable exact
built-in `dict`. Unsupported interpreter layouts reject safely rather than
invoking arbitrary backing-mapping hooks. This is a narrow, writer-only
frozen-result admission boundary; the existing result and its `to_dict()`
behavior remain unchanged. The reader returns a frozen
`GroupForwardBundleValidation` summary, not a reconstructed executable result.

`GroupForwardBundleError` identifies malformed, inconsistent, resource-exceeding,
or unsupported safe-reading content. Public content errors are categorical and
omit original values, group labels, paths, dependency text, and chained parser
or model exceptions. An occupied writer destination raises `FileExistsError`;
actual writer I/O failures remain `OSError`. No partial validation summary is
returned on failure.

## Exact two-file bundle

The writer owns only:

```text
temporal_result.json
checksums.json
```

The payload is an envelope with exactly `schema_id`, `schema_version`, and
`result`. Its identity is `mlcalc.group_forward_baseline_bundle`, integer
version `1`. The inner result remains
`mlcalc.group_forward_baseline_evaluation`, integer version `1`, protocol
`group-forward-fixed-regression-v1`, and coverage mode `partial_predeclared`.
Envelope and result versions are separate; the existing result is not relabeled.
Predictions, plan, metrics, and environment are stored once in the inner result.

Checksums use the existing format, with exactly one payload entry:

```json
{"algorithm":"sha256","files":{"temporal_result.json":"<64 lowercase hex>"}}
```

Payload serialization is UTF-8 of
`json.dumps(envelope, sort_keys=True, ensure_ascii=True, allow_nan=False,
separators=(",", ":")) + "\n"`. Meaningful whitespace in group IDs, integer `1`,
string `"1"`, string `"01"`, exact signed-64-bit integer identities, Unicode
normalization distinctions, escaped NUL/lone surrogates, and signed binary64 zero
are preserved. There is no CSV conversion, ID stringification, or normalization.
A reader can accept other JSON whitespace/key ordering; the checksum covers the
exact retained bytes, not a reserialization.

## What the semantic reader checks

1. **Retained bytes and strict typed schema.** Open/pin a supported direct local
   directory, read checksums and payload once each, and hash and parse the same
   retained payload bytes. Reject unknown/missing fields, duplicate keys at any
   depth (including escaped-key equivalence), wrong exact types, invalid UTF-8,
   BOM, trailing documents, and nonfinite numbers. Observations, predictions,
   and scores require JSON floating tokens, not integer/bool substitutes.
2. **The complete declared temporal plan.** Rebuild it from typed row groups,
   spans, windows, clock/unit, and gap using the pure planner. Compare every
   field, order, fixed marker, exclusion, and membership. Establish exhaustive
   train/test/unused partitions, disjoint testing across windows, the exact
   tested union, and its untested complement. Repeated training and later
   training of earlier test groups remain valid when the declared cutoff permits.
3. **Roles, fixed recipes, resources, and declared fit counts.** Check feature
   dtype/name and metadata-role structure, the exact ordered `dummy_mean` and
   `ridge_alpha_1` declarations, train-only preprocessing declarations, one-thread
   policy, Ridge cap, and the exact per-window and total fit budgets. Recompute
   matrix-cell visits as `2 * P * sum_w(n_train[w] + n_test[w])`. Other operation
   counts, including retry, selection, bootstrap and full-data refit, are zero.
4. **Tested records.** Require exactly one record for every globally ordered
   tested row, with its exact typed group and unique rebuilt window index. Check
   finite observed/predicted binary64 values and bitwise-constant Dummy
   predictions within each window. Untested predictions are prohibited.
5. **Metrics from retained values.** Recompute pooled metrics directly across
   tested rows, per-window scores in window order, and per-group scores in first
   tested-row encounter order, preserving typed IDs. Group RMSE is the arithmetic
   mean of group RMSEs, not pooled RMSE. Constant/singleton target partitions have
   null R². Counts, identities, order, keys, and nulls match exactly; finite score
   comparisons use `math.isclose(rel_tol=1e-10, abs_tol=1e-12)`. Checked arithmetic
   overflow/underflow rejects. The tolerance is not a confidence interval or a
   guarantee of detecting every edit.
6. **Internal declaration bindings.** Recompute `plan_sha256` and
   `execution_sha256` with the unchanged typed binding encoding, and recompute
   the saved source-map aggregate digest. Environment/version strings are inert
   bounded declarations. Source filenames are bounded relative POSIX `.py`
   records, never opened, resolved, imported, or compared with current installed
   files. A supported old 37-file source map remains readable after the current
   inventory grows to 38. Current dependency versions need not match saved ones.

The reader rechecks retained file signatures and known stale legacy basenames
after semantics. It does not construct estimators, fit, predict, observe installed
source, resolve paths declared inside JSON, contact a network, or write files.
The package's existing imports can transitively import estimator and
source-observation modules; this is not import isolation or a dependency-free
reader.

## What remains unchecked

The summary's `checked_scope` and `unchecked_scope` are explicit. Validation does
**not** establish:

- Authenticity, source truth/timing, or the original row/index mapping.
- Original features, their missingness, full targets, or reconstruction of
  `input_sha256`. Only that opaque digest's syntax is checked. Structurally valid
  feature/role declarations cannot be linked back to the absent original inputs.
- Undeclared aliases, copied targets, feature availability, or historical
  predeclaration/blinding.
- Actual training, isolation of fitted preprocessing, solver state, fit-count
  execution, or which code ran. A constant Dummy prediction is only a necessary
  consequence of its recipe, not proof that it equals an unavailable training mean.
- Causality, group independence, model quality, deployment generalization,
  scientific validity, cryptographic authentication, or an atomic snapshot.

An invented but internally consistent result can pass. Predictions changed along
with consistent metrics and checksums can pass if the fixed recipe consequences
still hold. A replacement opaque input digest can pass if the execution binding
is recomputed. An internally consistent source map need not describe any actual
installation. All hashes are unauthenticated declarations, not provenance proof.

`GroupForwardBundleValidation` reports bundle/result versions, protocol, coverage
mode, total/tested/untested row and group counts, window/feature/recipe counts,
`declared_pipeline_fits`, `validation_fits_performed`, `payload_bytes`,
`temporal_result_sha256`, and both scopes. Payload bytes exclude `checksums.json`.
The digest is for the exact consumed payload; declared fits are not observed fits.

## Bounded local transport

- Checksum metadata: 16 KiB. Payload: 64 MiB, including its final newline.
- Before parsing: a string/escape-aware scan bounds payload container depth at 16
  and JSON values at 2,000,000 (containers and values, not object-key strings).
- Integer tokens: at most 19 digits after a minus sign and within signed-64-bit
  range. Float tokens: at most 64 characters, finite binary64, with nonzero
  decimal tokens that round to zero rejected. `NaN`/`Infinity` are rejected.
- Existing evaluation bounds: 1–20,000 rows, 1–100 features, 1–32 windows,
  `W * (N + G) <= 250,000`, conservative complete-plan bound 16 MiB, and at most
  20,000,000 matrix-cell visits.
- Source map: 1–256 filename records of at most 256 characters. Version strings:
  nonempty, unpadded built-in strings of at most 256 characters.

These persistence admission bounds can reject unusually verbose, hand-forged
in-memory results. They are allocation/scheduling limits, not hard memory,
wall-clock, or filesystem-latency guarantees.

Both safe paths require POSIX no-follow directory-relative primitives. Paths
must be direct local relative/absolute paths without `..`, network-style `//`
roots, or observed symlink components. Checksum/payload files must be bounded
regular files, not symlinks, FIFOs, directories, or devices. Unsupported platforms
fail closed. No source path or URL stored in JSON becomes an I/O input.

Temporal bundles reject `manifest.json`, `summary.json`, `predictions.csv`, and
`report.html`, including broken symlinks/directories at those fixed basenames.
This prevents visibly mixed temporal/legacy output. Other unlisted files are
neither opened nor claimed verified. Use trusted, idle local storage: detected
mutation/replacement fails, but there is no lock, authenticated origin, atomic
directory snapshot, or protection against undetectable/reverted hostile changes.

## Fresh-only publication and failure

The writer fully validates, serializes, bounds, and strictly decodes/checks its
payload before any directory creation. The output parent must already exist;
the leaf must be absent. An existing empty directory, file, symlink, broken link,
FIFO, or device is occupied. There is no overwrite argument or automatic retry
in place. A racing `mkdir` failure never authorizes reuse.

After exclusive leaf creation, the existing generic publisher stages exact UTF-8
bytes, publishes payloads using atomic per-file no-clobber links, and publishes
checksums last. Staging failure publishes no payloads. Link/publication failure
can leave the new directory or an incomplete file set; it is not a completed
bundle and missing payload/checksums fail reading. The writer does not remove
another writer's files, silently roll back, or reuse the failed directory.
Choose a fresh destination for a later authorized retry.

Parent preflight plus path-based publication is not a pinned-path transaction.
Another process can change ancestors or the destination after checks. There is
no whole-directory transaction, race-freedom promise, or fsync/power-loss
persistence guarantee. A returned path means both owned files were published,
not that hostile concurrent edits or later storage failures are impossible.

## Byte verification and legacy separation

`mlcalc verify runs/temporal-001` recognizes the exact single-payload inventory
as `temporal_evaluation`. It still only checks opaque bytes and file safety. Even
non-JSON payload bytes pass if their checksum matches and filesystem checks pass.
It does not silently call semantic validation. Use the Python semantic API for
schema/plan/metric checks; there is no new temporal model or semantic-check CLI.

Existing regression/binary evaluation, prediction, delayed-outcome scoring,
budget comparison, and interval validators do not accept this inventory or gain
a temporal mode. Repacking temporal content under legacy filenames does not make
it a legacy result. Legacy writer file sets, schemas, selection semantics, and
unlisted-file behavior stay unchanged.

## Privacy

Saved JSON retains all declared group IDs and timing metadata, tested observations
and predictions, names, and environment/source descriptors. Raw features and
untested outcomes are omitted. Review this content before sharing. Writing a
local bundle does not authorize publication, external review, or upload.
