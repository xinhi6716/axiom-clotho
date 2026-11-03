# Compare two saved regression prediction bundles (no fitting)

This current-source API bridges existing strict delayed-outcome scoring and exact
paired absolute-error arithmetic. It does not fit/select/load models, write files,
perform significance tests, or infer future performance. The stdout-only CLI
below uses this same API. Existing score commands and report schemas are unchanged.

```python
from groupbench.saved_scoring import compare_saved_predictions

report = compare_saved_predictions(
    "runs/model-a-predictions", "runs/model-b-predictions",
    "original-new-features.csv", "later-outcomes.csv",
    {
        "method_a": "reference", "method_b": "candidate",
        "target_column": "response",
        "target_definition": "Define precisely what response measures",
        "target_unit": "State the same outcome/prediction unit for both models",
        "cohort_definition": "Define the entire original query cohort",
        "row_mapping_definition": "Original query position mapping retained; neither producer filtered or reordered rows",
        "group_definition": "Define the literal supplied scoring groups",
    },
)
```

The example declarations are placeholders to replace, not proof of those facts.
This requires Clotho's existing dependencies and supported local bundle access.
`SavedScoringError` rejects the entire comparison; no partial result is returned.
The function returns a JSON-ready dictionary without printing or writing it.

## Pairing contract: same bytes, not authenticated physical identity

Supply two supported regression prediction bundles (schemas 1.0–1.3), the **same
original query CSV** used by both producers, and one complete outcomes CSV with
exact header `row_position,group,observed`. Each position 0 through n−1 must occur
exactly once. Outcomes may be reordered; positions, not CSV order, perform the
join. Groups remain literal text, including distinctions such as `01` versus `1`.

Both retained manifests must contain a valid `prediction_source_file_sha256`
matching the exact consumed original query bytes. Both prediction counts must
match its parsed row count. Missing hashes, reordered/reformatted source bytes,
missing declared feature columns and duplicate CSV headers reject comparison.
There is no hash-bypass flag. This deliberately rejects byte-different files even
if a human considers them equivalent; no remapping or intersection join occurs.
The query parser inherits the existing 64 MiB limit; outcomes inherit 4 MiB and
bundles retain their existing limits. At most 20,000 positions are supported.

The contract must be a plain dictionary with exactly the eight shown fields.
Values are nonblank strings without NUL, at most 2048 characters (method labels:
256). Method labels must differ. Both manifest target columns must exactly match
`target_column`; target meaning, units, cohort/mapping and group semantics remain
caller declarations. No unit conversion, normalization or identity inference is
performed. If the query contains a target-named column, it is never substituted
for the explicit outcomes file. Historical target leakage is not audited.

Feature subsets, training sources and model protocols may differ. This supports
descriptive comparison, not a controlled or causal interpretation. The same
bundle may be supplied twice for a zero-contrast check; `identical_bundle_hashes`
reports that equality, not evidence of a second independent run.

A matching source hash links supplied bytes to producer declarations; it cannot
authenticate how a model used those rows. A dishonest producer can rewrite its
manifest and checksums. `physical_identity_verified` and `provenance_verified`
are therefore always false. Preserve and audit physical identity and pre-outcome
freezing externally. Sequential reads are not a joint atomic filesystem snapshot.

## Arithmetic and report

For each accepted row, the contrast is **B absolute error minus A absolute error**.
Negative means B has lower absolute error on this supplied cohort only. The shared
primitive computes exactly on the scoring inputs' binary64 values, then rounds
once for each displayed contrast. It cannot recover decimal digits lost during
conversion. Both predictions and outcomes use the stricter saved-scoring numeric
checks, including integer exact representability and nonzero-underflow rejection.

Two explicitly different summaries are reported:

- `row_mean_absolute_error_b_minus_a`: equal weight per supplied row.
- `equal_group_mean_absolute_error_b_minus_a`: first average rows within each
  supplied group, then give groups equal weight. This does not certify independence.

Group and aggregate means use unrounded rational row contrasts, not rounded
individual outputs. With differences [0,2,4] and groups [A,B,B], row mean is 2 and
equal-group mean is 1.5. With y=1e20, A=1 and B=2, the contrast is −1 even if both
rounded absolute errors display as 1e20. Thus subtracting displayed model MAEs
may differ from the paired contrast. A nonzero contrast that would round to zero,
or one outside finite float range, rejects the whole comparison.

`model_a` and `model_b` preserve the existing [saved scoring](SAVED_PREDICTION_SCORING.md)
report format and grouped metrics. `paired_rows` and `per_group` retain complete
paired contrasts. There is no paired RMSE shortcut, automatic winner, tie-tolerance
rule, p-value, confidence interval, independence claim or estimated population
effect. Repeatedly comparing models against these outcomes does not create fresh
validation evidence. Complete provenance includes both bundle-file hash sets,
common query/outcome hashes and the supplied semantic contract.

Output contains supplied declarations, groups, targets and predictions. Review it
before sharing. If serializing or redirecting output yourself, protect your input
paths: this API does not manage or safeguard caller-controlled file writes.

The older [two-budget utility](SAVED_BUDGET_COMPARISON.md) has a different contract
(two methods × two budgets with matched repeats). Do not invent a budget axis to
compare two models. The fixed environment-heldout example retains its own strict
five-arm artifact contract and now delegates to the same paired arithmetic.


## Unified stdout-only command

```sh
mlcalc compare-predictions --run-a runs/model-a-predictions \
  --run-b runs/model-b-predictions --query original-new-features.csv \
  --outcomes later-outcomes.csv --contract comparison-contract.json
```

`groupbench compare-predictions` and `clotho tabular compare-predictions` delegate
to the same parser/API. All five paths are explicit and required; there are no
`--out`, `--overwrite`, sampling, unit-conversion or source-hash-bypass flags.
Success prints complete JSON and exits 0. Input/contract/comparison failures exit 2
with categorical stderr, no raw source values/paths and no partial stdout report.
Help uses the canonical `mlcalc compare-predictions` command.

The contract file is strict UTF-8 JSON with exactly the eight API declaration
fields. Duplicate keys, unknown fields, nonstring values, malformed JSON and
nonfinite constants reject. No numeric coercion is used. At most **64 KiB of
contract bytes is accepted**. The reused regular-file reader has a **4 MiB read
bound** (with at most one extra detection byte), not a 64 KiB I/O maximum; it
checks descriptor/path signatures and allows
explicit local symlinks to regular files. It rejects nonregular files. These
checks are not an atomic snapshot or a storage-access timeout.

The complete serialized report, including its newline, must fit 64 MiB; otherwise
the command rejects without printing partial JSON. This is an output limit, not
a claim about peak memory. Nothing is written to disk by the command. Shell
redirection can overwrite an input before Python starts: choose safe destinations
yourself. Output includes private values if you supply them; review before sharing.

### Tiny invented self-comparison sanity check

First complete the [optional-model walkthrough](../examples/optional-model-delayed-scoring/README.md)
([繁體中文](../examples/optional-model-delayed-scoring/README.zh-TW.md)) to create its
prediction bundle. Then, from the repository root:

```sh
clotho tabular compare-predictions \
  --run-a runs/optional-later-prediction --run-b runs/optional-later-prediction \
  --query examples/optional-model-delayed-scoring/new_features.csv \
  --outcomes examples/optional-model-delayed-scoring/later_outcomes.csv \
  --contract examples/optional-model-delayed-scoring/comparison_contract.json
```

This intentionally supplies the **same bundle twice**: expect
`identical_bundle_hashes: true` and zero paired contrasts. Distinct display labels
do not imply two different models or independent runs. This only checks command
plumbing on invented arbitrary-unit data; it does not establish scientific
performance. For a real two-model comparison, provide both original compatible
bundles and replace all declarations with your actual common-cohort contract.
This command performs no fits; creating the walkthrough bundle earlier does.
