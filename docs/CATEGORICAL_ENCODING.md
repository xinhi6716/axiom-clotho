# Explicit categorical encoding protocols

Use `categorical_encoding="distinct-v2"` in either estimator or any regression/binary
run, prediction, or no-fit planning API to distinguish a real missing-marker string
from a missing source value. The matching CLI option is
`--categorical-encoding distinct-v2`, including `demo`, `plan run`, `plan predict`,
and the `binary` namespace.

```bash
mlcalc run data.csv --target response --group batch --categorical-encoding distinct-v2 --out runs/v2
mlcalc binary plan predict labeled.csv new.csv --target fail --group lot --categorical-encoding distinct-v2
```

## Why this is opt-in

The default is `legacy-v1`, exactly the previously shipped preprocessing. It
replaces categorical missing values with `__GROUPBENCH_MISSING__` before one-hot
encoding. This has a known collision: a real source string with that exact value
and true missing values receive the same internal key. The collision can occur
when both appear in training, when an unseen literal arrives after missing was
seen in training, or when missing arrives after the literal was seen in training.
Input-support diagnostics distinguish those source values, but do not repair the
legacy fitted representation. The default remains affected; explicitly select
`distinct-v2` for the corrected key mapping.

Switching silently to native-NaN one-hot encoding is not equivalent. NaN sorts
after source strings, which can change equal-frequency cutoff membership under
the 64-category cap even if the real marker string never occurs. The v2 mapping
below retains the old lexical boundary instead.

## Exact distinct-v2 mapping

Existing input validation still normalizes categorical non-missing values to
strings and missing values to NaN. It does not change case, trim whitespace, or
restore distinctions already lost during source normalization.

Let `M` be `__GROUPBENCH_MISSING__`. Inside each categorical pipeline:

1. Map a missing value to `M`.
2. For every real string starting with `M`, insert one colon immediately after
   that prefix: `M + suffix` becomes `M + ":" + suffix`.
3. Leave every other real string unchanged.

The entire prefix domain is escaped, not just the exact marker. For example,
real `M`, `M + ":"`, and `M + "::"` become `M + ":"`, `M + "::"`, and
`M + ":::"` respectively. No real source string maps to the bare missing key.
Escaped keys cannot collide with unchanged strings, because unchanged strings
do not start with `M`. Inserting the same character at the same prefix boundary
preserves the order of all real source strings, including Unicode, control
characters, and whitespace accepted by the Python API. The CLI's separate CSV
NUL rejection remains unchanged.

This mapping is stateless and is applied once per pipeline transform to raw
normalized values. It learns no vocabulary or statistics from prediction rows.
Only the existing train-fitted one-hot encoder learns categories/frequencies.
Its `handle_unknown="ignore"`, `max_categories=64`, sparse representation, and
all existing numeric preprocessing are unchanged. Numeric-only inputs are valid
under either explicit protocol.

Distinct source keys do **not** guarantee distinct final feature vectors:

- Two categories absent from fitting can both map to the all-zero unknown vector.
- Missing and a real marker can both enter the same infrequent-category bucket
  under the existing cap.
- Equal predictions can also arise from model parameters or other features.

These are unchanged encoder/model behaviors, not the fixed literal-token collision.
No global vocabulary, extra missing indicator, extra fitted model, or wider
encoded-feature bound is introduced. Prefix escaping adds one character only to
matching non-missing values; this is not a hard memory/time guarantee.

## Compatibility and provenance

Omitting the option and explicitly selecting `legacy-v1` retain the previous
predictions, candidate selection, split schedule, fit counts, schemas, protocol
strings, configuration fields, and evaluation run-key calculation. In legacy
artifacts, absence of `config.categorical_encoding` means legacy-v1. No new warning
or metadata field is added to those existing formats.

With `distinct-v2`, plans and manifests add `config.categorical_encoding` and use
these explicit protocol identities and compatible schema minors:

| Workflow | Protocol | Schema version |
| --- | --- | --- |
| Regression evaluation and its plan | `nested-group-kfold-distinct-v2` | 1.1 |
| Regression prediction and its plan | `group-fit-predict-distinct-v2` | 1.2 |
| Binary evaluation and its plan | `nested-group-binary-brier-distinct-v2` | 1.1 |
| Binary prediction and its plan | `group-binary-brier-fit-predict-distinct-v2` | 1.1 |

Evaluation summaries also identify the v2 protocol and advance to schema 1.1.
Existing binary schema IDs, CSV columns, payload file sets, and checksum verifier
are unchanged. Evaluation run keys incorporate the explicit encoding option,
so v1/v2 evaluations cannot share a run key just because their predictions match.
Schema minors describe compatible fields; the protocol identity describes the
changed scientific procedure. Prediction mode still is not held-out evaluation.

For data without the exact literal marker, the order and frequency of categorical
keys, encoded matrices, and model calculations remain equivalent under the same
normalized input, seed, and dependency versions, including category-cap ties.
When the literal appears in training or prediction, differences are intentional.
Changing dependencies/hardware still has no bitwise reproducibility guarantee.

The fitted pipeline retains its chosen encoding. Changing an estimator constructor
parameter with `set_params` does not change an already-fitted model's predictions;
a successful refit replaces the fitted state, and a failed refit preserves it.
`categorical_encoding_` records the successfully fitted choice. No executable
model save/load format or migration is added. To adopt v2, explicitly rerun
selection/evaluation and full-training fitting, retaining old results separately.
Do not relabel old v1 results or treat synthetic regression tests as evidence of
better real-data accuracy, calibration, or scientific validity.
