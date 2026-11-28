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
