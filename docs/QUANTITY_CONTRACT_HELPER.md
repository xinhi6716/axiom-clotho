# Opt-in electrical quantity helper

`groupbench.quantity_contract` is a pure, no-fit helper. It is **not wired into
models, CLI commands, feature manifests or saved-scoring schemas** and does not
protect an existing workflow merely by being installed. Legacy version-1 feature
manifests still select names without checking scientific unit compatibility.
Nothing here is a universal research-admission rule.

```python
from groupbench.quantity_contract import (
    QuantityDeclaration, compare_declarations,
    normalize_decimal_tokens, decimal_to_float64,
)
train = QuantityDeclaration("current", "terminal_current", "median",
                            "protocol-v1", "electric_current", "A")
query = QuantityDeclaration("current", "terminal_current", "median",
                            "protocol-v1", "electric_current", "nA")
canonical = compare_declarations(train, query, mode="normalize")
a = normalize_decimal_tokens(["1e-9"], train)
b = normalize_decimal_tokens(["1"], query)
assert a == b
assert decimal_to_float64(a) == decimal_to_float64(b)
```

Default comparison mode `validate` requires identical units. Explicit
`normalize` comparison permits approved same-dimension units and returns a
canonical declaration; it does not convert any table. Explicit
`normalize_decimal_tokens` always converts to the declared dimension's canonical
unit. Match declarations before separately normalizing training and query inputs.
The helper neither reads files nor constructs estimators.

## Closed registry

| Dimension | Accepted units | Canonical |
|---|---|---|
| electric_current | A, mA, uA, nA, pA | A |
| electric_potential | V, mV, uV | V |
| resistance | ohm, kohm, Mohm | ohm |
| electric_conductance | S, mS, uS, nS, pS | S |
| elapsed_time | s, ms, us, ns, ks | s |
| dimensionless | 1 | 1 |

Registry version is `electrical-decimal-v2`. Spelling/case are exact; no aliases,
unit suffix parsing, magnitude inference, reciprocal slopes, log transforms,
area normalization, affine temperatures or percentage conversion are supported.
`dimensionless` is a dimension, not an accepted unit spelling; use `1` explicitly.
Unknown, arbitrary, blank and mixed labels are rejected, not guessed.
Version 2 adds conductance and elapsed-time scales without changing version 1
unit behavior. A conductance unit does not define read bias or authorize inverse
resistance conversion. An elapsed-time unit does not identify the reference
event, distinguish pulse-on duration from wall time, or align different clocks;
use the declared quantity and condition IDs to represent those distinctions.

Column, quantity ID, statistic ID, condition ID and dimension must agree exactly.
Thus mean versus median, different protocols, or two different ratio statistics
remain incompatible even when their units match. IDs are bounded opaque caller
assertions, not ontology-resolved or scientifically verified definitions. A
per-column declaration **cannot detect falsely declared or secretly mixed-unit
rows**. Separate and explicitly normalize such sources before pooling them;
never infer a majority unit. No claim of physical comparability, independence,
predictive accuracy or causal validity follows from passing these checks.

## Numeric boundary

Input is a nonempty list/tuple of at most 20,000 strict decimal strings. Each has
at most 256 characters, 64 coefficient digits (including written leading zeros),
and explicit/effective decimal exponent magnitude at most 200. Whitespace,
missing values, nonfinite tokens, Booleans, typed numeric objects, unit suffixes,
and unsupported containers reject. Normalization uses exact Decimal tuple
exponent shifting, independent of global decimal precision and observed values.
It copies values and returns an immutable tuple; it retains no raw-input log or
provenance by default.

`decimal_to_float64` is a separate explicit **approximate** boundary accepting a
bounded nonempty Decimal list/tuple. It rejects nonfinite values, overflow,
nonzero underflow, integral values not exactly representable in binary64, and
distinct decimals collapsing to one float within the supplied batch. Fractional
rounding such as Decimal("0.1") to binary64 is allowed and is not certified
lossless. This conservative integral rule may reject large whole-number decimal
values. Signed zeros are retained as floats but are not treated as distinct
numeric quantities for collision checks.

Collision checks cover only the submitted batch, not separately called train and
query batches. A caller requiring a joint collision check must submit their
combined canonical values to this boundary before separating them again; this
is deterministic validation, not distribution-based scale learning. Earlier
precision loss from CSV/float conversion cannot be recovered. No values are
clipped, filled, dropped or automatically rescaled to evade an error.

Tests use invented values only, with no model fit or external data. Future
workflow integration must explicitly thread declarations through planning,
training, querying and artifact versioning; this helper alone is not that work.

For a declared pair entering the same binary64 comparison,
`normalize_comparable_pair_to_float64(expected, expected_tokens, supplied,
supplied_tokens)` checks matching measurement meaning first, normalizes both
batches to one canonical unit, and rejects distinct decimal values that would
collapse to the same float across either batch. It returns the canonical
declaration and one float tuple per batch, in the original order. The 20,000
value limit applies across both batches together. This is an explicit optional
boundary, not an automatic ML ingestion step or evidence of physical
comparability.


### CLI: exact mirror-pair audit

```sh
mlcalc audit-polarity-mirror readings.json frozen-pairs.json
clotho tabular audit-polarity-mirror readings.json frozen-pairs.json
```

Readings use `{"schema_version":1,"rows":[["block",0.1,0.4,-0.1,-0.2]]}`
where each row is block, positive voltage/current, negative voltage/current.
The separate roster uses
`{"schema_version":1,"expected_pairs":[["block",0.1]]}`. Freeze the roster
independently of observed outcomes. The tool does not infer matches or convert
units. Both inputs must be bounded local UTF-8 JSON; exact JSON decimals are
used, with at most 20,000 rows per file. Success emits aggregate JSON with
rational strings for exact metrics (`null` for an undefined relative metric).
Rejection exits 2 without a report. It does not fit or write files, and the
diagnostic describes the supplied pairs rather than proving physical mechanisms.


### CLI: reserved-standard affine readout audit

```sh
mlcalc audit-affine-readouts comparison.json
clotho tabular audit-affine-readouts comparison.json
```

The input is a local UTF-8 JSON object, for example:

```json
{"schema_version":1,"standards":[0,1,3],"reference":[0,1,3],
 "comparison":[0.01,1.01,3.01],"max_interior_error":0,"max_path_difference":0.01}
```

Each triple is in low/interior/high order. Only the endpoints define each
line; the reserved interior standard provides a separate discrepancy check.
Predeclare both inclusive tolerances in common response units. JSON decimals
are exact, and output deviations are rational strings. Exit 0 means an
audit was computed, not that its flags passed: inspect `interior_pass` and
`shared_affine_pass`. Malformed or unavailable inputs exit 2 without a
report. This audit does not certify calibration or rule out unsampled
nonlinearity, and writes no files.


### CLI: compare matched anchor changes against both controls

```sh
mlcalc audit-anchor-change-controls readings.json frozen-blocks.json
clotho tabular audit-anchor-change-controls readings.json frozen-blocks.json
```

Use `{"schema_version":1,"rows":[["block",0,0.3,0,0.1,0,0.2]]}`
for a reading document. Each row gives a block ID and the measured slot 0
and slot 1 values for candidate, sham and independent reference in that
order. A separate `{"schema_version":1,"expected_blocks":["block"]}` declares
the exact frozen roster independently of the observed changes. Both files
are bounded local UTF-8 JSON, each with at most 20,000 entries. Missing,
extra or duplicate blocks reject the audit. Success reports only block count
and three exact contrasts as rational strings; invalid input exits 2 with
no report. The tool does not check the actual instrument/units/dependencies,
estimate reliability, fit models or write files.


### CLI: compare within-group elapsed lag support

```sh
mlcalc compare-elapsed-lag-support reference.json candidate.json associations.json
clotho tabular compare-elapsed-lag-support reference.json candidate.json associations.json
```

Each schedule declares `{"schema_version":1,"time_unit":"s","rows":[["a","group",0.1],["b","group",0.2]]}`;
rows supply ID, group and exact JSON timestamp in the agreed time unit.
Freeze `{"schema_version":1,"associations":[["a","group"],["b","group"]]}`
independently of timestamps. Both schedules must match every association
and the time-unit declaration. Row order may differ; duplicate row IDs
reject. Distinct rows with the same timestamp contribute zero lags.
Only unordered pairs within the same group count; multiplicities and
group boundaries matter. Output contains aggregate support counts, not
identifiers, timestamps or measurements. Empty pair support or more than
100,000 pairs per schedule rejects without a report. Inputs are local
UTF-8 JSON with at most 20,000 associations per file. Exact lag equality
does not imply a unique schedule, verified clock, stationarity, common
measurement history or scientific comparability. No files are written.


### CLI: fixed versus updated control availability

```sh
mlcalc compare-control-availability paired-schedule.json associations.json
clotho tabular compare-control-availability paired-schedule.json associations.json
```

Use `{"schema_version":1,"time_unit":"s","rows":[["a","group",0,2,3,1]]}`:
row ID, group, fixed cutoff, updated cutoff, forecast issue, and the
one shared control-release timestamp. The independent roster uses
`{"schema_version":1,"associations":[["a","group"]]}`. Both parties use the
same declared time unit and clock; that common clock is not verified.
Each row must satisfy fixed cutoff ≤ updated cutoff ≤ forecast issue.
The fixed cutoff cannot vary within a group. Updated cutoffs cannot move
backward as issue time advances, or differ for the same group/issue time.
A control must be released strictly before a cutoff to count; a tie does
not establish event order. JSON decimals are compared exactly.

The command describes the same complete declared roster under both policies.
It returns counts only, not row IDs, timestamps or measurements. Unknown or
inconsistent declarations exit 2 without a report. Scope is at most 20,000
associations per document. It does not score outcomes, verify actual control
availability or show that an updated policy improves forecasts.
