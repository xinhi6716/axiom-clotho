# Evaluate saved prediction intervals (descriptive only)

This local calculator scores **already saved** observations and lower/upper
bounds. It does not generate intervals, fit or load models, calibrate anything,
contact a service, or change existing evaluation/prediction bundles. This is a
general-purpose ML Super Calculator utility, not a semiconductor-specific model.

```sh
PYTHONPATH=src python -m groupbench.interval_evaluation \
  examples/saved-interval-evaluation/invented_voltage.json
```

The example is wholly invented voltage data, not measurements or evidence of
materials/semiconductor performance. Output is JSON on stdout only; rejection
produces stderr and exit code 2, without a partial report. No output file is
opened. If redirecting stdout, choose a new path: a shell can truncate an input
file before Python starts. After installation, omit `PYTHONPATH=src`.

The Python API is deliberately separate from training:

```python
from pathlib import Path
from groupbench.interval_evaluation import load_document, evaluate_saved_intervals

document = load_document(Path("saved_intervals.json").read_text(encoding="utf-8"))
report = evaluate_saved_intervals(document)
```

Use `load_document` for JSON, rather than first parsing decimal tokens to float.
The CLI reuses the saved-budget utility's bounded regular-local-file reader:
explicit symlinks to regular files are supported, but directories, FIFOs and
devices are rejected. It checks file type before opening and on the descriptor;
where supported, nonblocking open guards a substituted FIFO. Use trusted, idle
files. These checks are not an atomic snapshot, hostile-filesystem sandbox or
storage deadline. UTF-8 decoding and the 8-million-**character** limit apply.

## One explicit cohort, no dropped failures

Version 1 requires exactly the following document keys:

- `schema_version`: integer `1`, not boolean.
- `metadata`: exactly `target_definition`, `value_unit`, `group_definition`,
  `cohort_definition`, each nonempty trimmed text of at most 2048 characters,
  without C0 control characters (U+0000–U+001F). State the common target, physical/reporting unit,
  what a group represents, and which observations belong in the cohort.
- `expected_row_ids`: the complete declared cohort's unique text identifiers.
- `alpha`: `null`, or **one fixed miscoverage** strictly between 0 and 1 for
  every row. For example, `0.2` declares nominal coverage `0.8`, not `0.2`.
- `rows`: each row has exactly `row_id`, `group`, `observed`, `lower`, `upper`,
  `status`. The only accepted status is `"ok"`.

Both lists must contain 1–20,000 entries. Row identifiers are case-sensitive,
nonempty trimmed text, at most 256 characters, without C0 control characters (U+0000–U+001F).
The rows must match `expected_row_ids` **exactly once**, regardless of order.
Missing, extra or duplicate rows, non-`ok` status, missing/nonfinite values,
unknown keys and reversed bounds reject the entire cohort. A point interval
`lower == upper` is permitted. Every summary uses the same validated rows; an
error does not produce coverage from one subset and widths from another.

Do not remove failed rows before preparing input. A row omitted from *both*
lists cannot be discovered. Units and target definitions are shared **user
declarations**, not verified facts: the utility cannot detect secretly mixed
units, different target definitions, bad joins or leakage from numeric values.
It does no unit conversion or normalization. A report is not provenance proof.

## Coverage, width and interval score

For row `i`, let observed value be `y_i`, bounds `l_i <= u_i`, and let `g(i)`
be its declared group. Endpoints are **closed**:

```text
c_i = 1 if l_i <= y_i <= u_i, otherwise 0
w_i = u_i - l_i
s_i = w_i + (2/alpha) * [(l_i-y_i) 1{y_i<l_i} + (y_i-u_i) 1{y_i>u_i}]
```

The interval score is the negatively oriented score in Gneiting and Raftery
(2007), §6.2, p.370, Eq.(43), for **central equal-tailed** interval forecasts
with quantile levels `alpha/2` and `1-alpha/2`. This is symmetry of probability
tails, not necessarily equal numeric distances from a median. Providing alpha
declares this interpretation; the utility does not verify it.
Primary source: [author-hosted JASA paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf),
[DOI 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).

Coverage is dimensionless. Width and interval score are in the same units as
the target. At the **same alpha, target, units, cohort and weighting definition**,
smaller interval score is preferred. Do not sum or rank scores across different
datasets, units or alpha levels. Narrow width alone cannot select a better model.
With `alpha: null`, coverage and width remain available, but interval score and
nominal coverage are `null`; no alpha is inferred from observations or bounds.

For any row quantity `z_i` in `(c_i, w_i, s_i)`:

```text
row-weighted mean = (1/N) sum_i z_i
group mean z_g = (1/n_g) sum_{i:g(i)=g} z_i
equal-group mean = (1/G) sum_g z_g
```

The report contains `row_weighted`, `equal_group`, each `groups` summary, and
all `observations`, plus exact cohort counts and zero omitted rows. Groups are
not automatically independent units. Repeating all rows of one group with new
row IDs changes its pooled row weight but not its equal-group mean or weight.

### Hand-check the invented example

With alpha `0.2`, group A has observations `0, 1, 3` and bounds `[0,2]` for
each; group B has observation `10` and bounds `[0,0]`.

| Quantity | Four row values | Row-weighted | Equal-group |
|---|---|---:|---:|
| Covered | 1, 1, 0, 0 | 1/2 | 1/3 |
| Width (V) | 2, 2, 2, 0 | 3/2 | 1 |
| Interval score (V) | 2, 2, 12, 100 | 29 | 158/3 |

Nominal coverage is `4/5`, **not** either measured coverage. Both `y=0` and
`y=2` are covered by `[0,2]`; `[1,1]` covers `y=1` with width and score zero,
but misses `y=0`, giving score `10` at alpha `0.2`. `[2,0]` is rejected, not
swapped. None of these calculations establishes future coverage or calibration.

## Exact input semantics; approximate reported numbers

`numeric_semantics` is included in each report. This utility does not silently
turn integer or decimal inputs into binary64 before comparing interval bounds:

- JSON integers remain exact integers; decimal tokens become exact `Decimal`.
- Python API numeric values may be built-in `int`, `Decimal` or finite built-in
  `float`. A float means its **exact already-stored binary64 ratio**, not the
  decimal spelling of its display. Booleans, numeric strings, complex numbers,
  NumPy target/bound scalars and other numeric classes are rejected explicitly.
- Comparisons, widths, penalties and both levels of averaging use standard
  library `Fraction` arithmetic. Group means are never averaged after rounding.
- All metric names end in `_approx`: final values are rounded to binary64.
  Distinct exact scores can round to a tie. Any nonfinite result, overflow or
  nonzero value rounded to zero rejects the whole document, including every
  numeric input (also alpha) or a within-group mean. Zero itself is valid where
  the field permits it. Rescale into practical units rather than clipping.
- The report retains `observed_exact`, `lower_exact`, `upper_exact` as canonical
  rational strings (`"1/10"`, for example), not rounded endpoints. Likewise,
  `alpha_exact` and `nominal_coverage_exact` accompany their `_approx` fields.
  Tiny positive alpha can make nominal coverage *round* to `1`; alpha very
  close to one can itself round to `1`. Consult the exact companions, not an
  apparent rounded 100% label. These fields do not represent guarantees.

For example, integer observation `2**60+1` misses point interval
`[2**60,2**60]`. If the caller already converted all three to binary64, the
observation has been rounded down and is now covered. Similarly, exact JSON
decimal `0.1` differs from Python float `0.1`. The exact report strings make
these differences inspectable; rounding them for display can give an apparent
coverage/boundary contradiction. The calculator cannot recover precision lost
before the call. Preserve the original saved input alongside the report.

The float output range is checked, but intermediate rational arithmetic need
not fit float: alpha `1e-310`, zero-width `[0,0]`, and observation `1e-310`
produce finite score `2`. Conversely, finite endpoints `[-1e308,1e308]` have
unreportable width and reject. An input such as `1e-400`, or a nonzero average
below the output range, rejects rather than appearing as zero.

### Bounds apply to the Python API too

- Built-in integers: absolute value `< 10**100` (at most 100 decimal digits).
- Decimal: finite, at most 100 coefficient digits and absolute tuple exponent
  at most 400, checked **before** expanding a rational or integral group ID.
  This is a representation bound: excessive trailing digits/exponents are not
  silently normalized. JSON integer tokens are length-checked before `int`;
  JSON decimal tokens are limited to 1024 characters before `Decimal` parsing.
- Finite built-in float numeric values use their stored binary64 ratios; the
  decimal coefficient-digit cap is not applied to that binary representation.
- These bounds also bound the exact rational source strings in the report.
  They are resource limits, not a physical unit or scientific validity check.

## Group identity is separate from score arithmetic

Raw group IDs reuse the same `validate_groups` validator as ordinary evaluation,
before any NumPy numeric promotion. Distinct integer IDs `2**60` and `2**60+1`
remain distinct. Equivalent stored numeric IDs `1` and `1.0` share a group.
Numeric and text identifiers cannot be mixed. Missing, boolean, nonfinite,
container, blank and NUL-containing identifiers reject the entire document.
Text IDs preserve whitespace and Unicode identity; `"1"`, `"01"`, `" 1"`
and `"1 "` are different (row IDs have the stricter trimmed-text rule).

For bounded document input, text groups are at most 1024 characters and integer
groups at most 100 decimal digits. API floating group IDs supported by the shared
validator must have numerator and denominator of their exact ratios each at
most 4096 bits. Unlike score values, the shared validator supports NumPy numeric
group scalars within those bounds. Decimal group IDs must be bounded exact
integers, converted to `int` without loss. For fractional decimal identifiers,
use text IDs consistently; no decimal-to-float group conversion is performed.

No grouping rule proves exchangeability or independence. This utility is not a
conformal predictor or calibration algorithm, and supplies no new confidence
intervals, population inference, future coverage guarantees or automatic
physical/process decisions.
