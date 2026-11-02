# Evaluate saved binary probabilities without retraining

The current-source API
`groupbench.binary.saved_evaluation.evaluate_saved_binary_probabilities(document)`
scores an externally prepared, frozen prediction cohort without fitting a model.
It requires explicit expected row IDs and status for every row. Missing, extra,
duplicate, failed or invalid rows reject the entire document; no rows are dropped.
It does not change training workflows, existing evaluation bundles or the CLI.

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

The API does not parse text or read a file. If you construct the dictionary with
a JSON parser, that parser may already have rounded decimals or discarded duplicate
object keys. Neither loss can be recovered here. Source authenticity and exact
decimal arithmetic are not established. A document schema check is not a source audit.

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
