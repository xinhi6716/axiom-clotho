# Typed regression-target precision

Regression preparation and `GroupBenchRegressor.fit` normalize accepted targets
to float64. A downstream scorer cannot recover values that were already rounded
at this boundary. The typed-numeric checks therefore run before the final target
conversion, split scheduling, model construction or artifact writing.

## Typed integers and floating-point values

- Python and NumPy integer scalars must be exactly representable in float64.
  `2**53`, `2**53 + 2` and `2**60` remain valid; `2**53 + 1` is rejected.
  Signed and unsigned integers use the same exact-value rule, not a blanket
  magnitude cutoff. Overflow during conversion is a `ValueError`.
- Mixed Python lists/tuples and object arrays are checked using their original
  scalar types before a common NumPy numeric dtype can round their integers.
- Python floats and NumPy float16, float32 and float64 retain their already
  stored values. NumPy floating dtypes wider than float64 are rejected, including
  values that happen to fit. Wider scalar values inside object containers are
  rejected too. This matches the typed-numeric rule in the
  [direct regression scorer](REGRESSION_METRIC_INPUTS.md).
- An all-false mask or `np.ma.nomask` does not bypass these precision checks.
  Actively masked targets and missing/nonfinite outcomes remain invalid; no
  observations are filled, dropped or silently repaired.
- Complex values remain unsupported, including Python/NumPy complex scalars
  inside object arrays or Series and values whose imaginary part is zero.
  They are rejected before conversion can discard an imaginary component or
  emit a warning in place of enforcing the existing target contract.

For example, the int64 targets `[2**53, 2**53 + 1]` used to become two identical
float64 values. A constant prediction of `2**53` then appeared to have MAE zero,
although its exact error against the supplied integer values is `0.5`. The raw
metric API already rejects those typed targets; preparation now rejects them
before that information disappears. Metric formulas and output fields have not
changed, and neither this synthetic arithmetic example nor the validation guard
establishes real-data predictive performance.

## Legacy interpretation is not a lossless parser

The target-ingestion contract is intentionally broader than the direct scoring
API. Existing numeric-string and Boolean parsing remains unchanged: `"1.25"`
becomes `1.25`, and `False`/`True` become `0.0`/`1.0`. Numeric text is still parsed
into float64, so `"9007199254740993"` can round to `9007199254740992.0`. This guard
does **not** certify lossless text or CSV parsing.

Existing object conversions also remain unchanged. Decimal and Fraction values
are converted through their float interpretation; `Decimal("0.1")` and
`Fraction(1, 3)` are accepted with ordinary float64 approximation. Large values
such as `Decimal(2**53 + 1)` or `Fraction(2**53 + 1, 1)` can still lose a unit.
This compatibility behavior is a limitation, not permission to claim their
precision was retained. Unsupported objects and overflowing object conversions
raise `ValueError`; they are not silently dropped. The direct metric APIs have
a narrower contract and reject Decimal/Fraction, text and Boolean inputs.

Precision already lost when the caller constructed a NumPy array, pandas table,
CSV, or explicitly converted a value cannot be reconstructed. An explicitly
converted float64 vector is evaluated as the supplied float64 values, not as an
unavailable earlier representation. Feature-column precision is a separate
boundary; this change does not certify every numerical input in the workflow.

If conversion is unsuitable, choose a scientifically meaningful unit or a fixed
reference before conversion, review representability, and preserve the endpoint
meaning and missingness. No automatic centering, unit inference, target remapping
or learned target transformation is introduced here. Any learned transformation
must still respect training-partition boundaries. Datetime/timedelta endpoints
require an explicit caller-declared numeric-unit conversion, not an inferred
causal interpretation of time.
