# Evaluate saved binary probabilities without retraining

The current-source API
`groupbench.binary.saved_evaluation.evaluate_saved_binary_probabilities(document)`
scores an externally prepared, frozen prediction cohort without fitting a model.
It requires explicit expected row IDs and status for every row. Missing, extra,
duplicate, failed or invalid rows reject the entire document; no rows are dropped.
It does not change training workflows or existing evaluation bundles. The
`mlcalc evaluate-binary` command below provides read-only JSON input and stdout
output through the same strict loader and evaluator.

## Input contract

Provide a plain Python dictionary with exactly these fields:

- `schema_version`: the integer `1`, not a Boolean or a string.
- `metadata`: six nonempty declarations, each trimmed, no control characters,
  at most 2048 characters: `target_definition`, `positive_class_definition`,
  `negative_class_definition`, `prediction_definition`, `group_definition`,
  `cohort_definition`.
- `expected_row_ids`: a plain list of 1–20000 unique strings. Row IDs are trimmed,
  nonempty, at most 256 characters and contain no control characters.
- `rows`: a plain list with exactly one matching row for every expected ID.
  Each row has exactly `row_id`, `group`, `observed_label`, `probability_1`, `status`.
  Status must be the string `"ok"`.

Labels and probabilities must be plain Python `int`/`float` scalars. Labels are
explicit 0 or 1; probabilities are finite values in [0,1] for the declared class 1.
Boolean, missing, text, Decimal, NumPy scalar/array and custom conversion types
are rejected rather than coerced. Numeric 0 and 1 endpoints remain supported.
This intentionally narrower document contract is not a change to array metrics.

Groups are plain strings or int/float values, using the shared exact group-identity
validator. Numeric and text groups cannot be mixed; 1 and 1.0 identify the same
group, while adjacent large integers remain distinct. Text groups retain their
identity, including meaningful whitespace, and have at most 1024 characters.
Blank/NUL groups reject. Integers have at most 100 decimal digits. Use text for
identifiers whose numeric identity was not preserved at their source.

The low-level evaluator does not parse text or read a file. If you construct the
dictionary with an ordinary JSON parser, that parser may already have rounded
decimals or discarded duplicate object keys. Neither loss can be recovered here.
Prefer the strict `load_saved_binary_document` helper below for text input.
Source authenticity and exact decimal arithmetic are not established by either
path. A document schema check is not a source audit.

## Example with invented data

```python
from groupbench.binary.saved_evaluation import evaluate_saved_binary_probabilities

saved = {
    "schema_version": 1,
    "metadata": {
        "target_definition": "Invented component outcome",
        "positive_class_definition": "1 means fail",
        "negative_class_definition": "0 means pass",
        "prediction_definition": "Previously frozen invented probabilities",
        "group_definition": "Declared manufacturing batch",
        "cohort_definition": "Both expected invented observations",
    },
    "expected_row_ids": ["r1", "r2"],
    "rows": [
        {"row_id": "r1", "group": "A", "observed_label": 0,
         "probability_1": 0.25, "status": "ok"},
        {"row_id": "r2", "group": "B", "observed_label": 1,
         "probability_1": 0.75, "status": "ok"},
    ],
}
result = evaluate_saved_binary_probabilities(saved)
assert result["metrics"]["brier"] == 0.0625
assert result["cohort"]["omitted_rows"] == 0
```

Only explicitly supplied observations are scored. The function does not obtain
labels, construct probabilities, infer the positive class or choose a threshold.
It cannot verify that predictions were frozen before outcomes became known.

## Result and interpretation

The result has its own schema ID `mlcalc.saved_binary_probability_evaluation`
and version `"1.0"`. It is a standalone descriptive object, not a loadable
training/prediction bundle. It makes no file writes.

- `cohort` gives expected, input, scored and omitted counts. Omitted is always zero
  for accepted input. Rows omitted from both expected IDs and supplied rows remain
  undetectable; preserve an independently frozen complete manifest.
- `metrics` and `groups` use the existing [binary metric definitions](BINARY_PROTOCOL.md):
  row-averaged Brier on [0,1], clipped natural-log loss, threshold-0.5 confusion
  and rates, equal-group mean losses and worst-group losses. Groups receive equal
  weight only in the named group metrics. No new metric or rounding scheme is used.
- `observations` are ordered by row ID, with normalized group, observed label,
  probability of class 1 and fixed-threshold predicted label. Sorting precedes
  scoring so row permutations preserve floating summation order and alignment.
- `support` counts groups with both classes or only one class. Single-class cohorts
  and groups are retained. Undefined rates remain null; they are not zero, dropped
  groups or evidence that the evaluation failed. Group counts do not prove
  independent devices, studies or sources.
- `metadata`, `label_contract` and `metric_contract` carry definitions;
  `provenance_verified` remains false. Declarations are copied, not authenticated.

Existing positive-loss underflow rejection still applies. Probabilities already
rounded by a caller are evaluated as supplied; unsupported source precision is
not reconstructed. Diagnostic mean-probability rounding follows float64 and the
[controlled arithmetic policy](BINARY_ARITHMETIC_ERROR_POLICY.md).

No calibration fit, AUC, threshold tuning, confidence interval, independence test,
causal inference or future-performance guarantee is added. Do not optimize on
this evaluated cohort or remove inconvenient rows after seeing scores. Compare
results only when target/class meaning, cohort and weighting definitions match.
The returned object contains row IDs, group IDs, labels, probabilities and metadata;
review these before sharing. The examples and tests are invented arithmetic checks,
not evidence of real-device performance.


## Strict JSON text loader and offline commands

```sh
mlcalc evaluate-binary frozen-binary.json
groupbench evaluate-binary frozen-binary.json
clotho tabular evaluate-binary frozen-binary.json
# Equivalent canonical module entry point:
python -m groupbench.binary.saved_evaluation frozen-binary.json
```

The command prints a complete JSON report to stdout only. It does not create or
modify files, train a model, read paths declared in metadata or contact a service.
It has no output-path or overwrite flag. Shell redirection is controlled by the
shell and may overwrite a file; this command does not protect shell redirection.
All entry points delegate to the same parser; command help and usage errors show
the canonical `python -m groupbench.binary.saved_evaluation` invocation.
This is distinct from `mlcalc binary run`, which trains/evaluates models, and
`mlcalc score-predictions`, which scores saved regression predictions.

Python callers can use the same strict text loader without file I/O:

```python
from groupbench.binary.saved_evaluation import (
    load_saved_binary_document, evaluate_saved_binary_probabilities,
)
parsed = load_saved_binary_document(saved_json_text)
result = evaluate_saved_binary_probabilities(parsed)
```

The loader checks JSON syntax, duplicate keys, container shape and numeric
conversion. The evaluator then applies all metadata, type and full-cohort checks.
Always call the evaluator; a successful parse alone is not a valid evaluation.
Neither step silently repairs a failed row or substitutes a value.

### Explicit numeric conversion policy

JSON decimal/scientific tokens are first parsed as `Decimal`, before conversion
to the evaluator's plain Python numeric scalars. This is **not** exact-decimal
metric arithmetic:

- Labels must be exactly decimal 0 or 1. For example, `1.0000000000000000000001`
  is rejected, never rounded into class 1.
- Probabilities must lie in [0,1] before rounding. Ordinary interior decimal
  values use explicit float64 rounding (`0.1` has its ordinary binary64 value).
  Overflow, nonzero-to-zero and interior-to-one endpoint collapse reject rather
  than silently creating a perfect endpoint forecast. `1e-400` and
  `0.9999999999999999999999` therefore reject; explicit `0.0` and `1.0` work.
- Numeric group identity must be preserved. Integral decimal groups become
  exact bounded integers, so `9007199254740993.0` retains that distinct ID.
  A fractional decimal group is accepted only when exactly representable as
  float64 (for example `1.5`). Use a string such as `"0.1"` when a decimal ID
  would otherwise round. Distinct source IDs are never merged by this loader.
- Integer tokens remain Python integers. Schema version still requires integer
  syntax `1`, not `1.0`; row IDs and metadata still require strings.
- Booleans, null or quoted numbers are not mapped into numeric values. They may
  be syntactically valid JSON, but the full evaluator rejects them where numeric
  observations/probabilities are required.

Previously lost source precision still cannot be recovered when callers bypass
this loader or supply already rounded Python floats. Neither route authenticates
predictions or establishes historical freezing. Use the same documented parser
and scoring policy when comparing methods.

### Bounds and failure behavior

Accepted JSON text is at most 4 MiB after UTF-8 encoding. Nesting is limited to
16 containers outside strings, integer tokens to 100 digits, and decimal tokens
to 64 characters with at most 64 coefficient digits and exponent magnitude at
most 400. Existing 20000-row, identifier and declaration limits still apply.
Duplicate object keys at any level and nonfinite JSON constants are rejected.

The module reuses the saved-interval utility's regular-local-file reader, which
allows explicitly supplied local symlinks, checks the opened descriptor is a
regular file and reads at most 8,000,001 Unicode characters. The loader then
applies the stricter 4 MiB accepted UTF-8 limit. **The 4 MiB acceptance bound is
not a 4 MiB read-I/O bound.** This is not an atomic filesystem snapshot or a
network-filesystem deadline. FIFOs/directories are not valid input files.

All parsing, cohort scoring and JSON serialization finish before stdout is
written. Success exits 0; invalid input or unavailable files exit 2 with a
categorical stderr message and no partial score JSON. Supplied values and paths
are not echoed in errors. The successful report intentionally includes declared
metadata, row/group IDs, labels and probabilities; review it before sharing.
