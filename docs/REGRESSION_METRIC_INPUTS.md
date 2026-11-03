# Regression metric input precision

`groupbench.metrics.regression_metrics` and `grouped_metrics` score supplied
observations and point predictions in float64. They do not fit, calibrate or
construct uncertainty intervals. The input contract is the same for both
functions:

- Supply matching, nonempty one-dimensional vectors of finite real numbers.
- Python and NumPy integer values must be **exactly representable** in float64.
  Exact large integers remain supported; for example, `2**60` is accepted while
  `2**60 + 1` is rejected. This check occurs before Python sequences are promoted
  to a common NumPy dtype, including mixed integer/float sequences.
- Python floats and NumPy float16, float32 and float64 values are supported.
  Wider floating dtypes are rejected rather than silently narrowed, even when a
  particular value would fit. An explicit caller conversion is a decision to
  accept the resulting float64 values, not a promise that precision is retained.
- Numeric lists, tuples, NumPy arrays and pandas Series are supported, including
  object containers whose individual values meet the rules above.
- Complex numbers, booleans, numeric strings, Decimal values, missing/nonfinite
  entries and masked arrays are rejected with `ValueError`. Masked arrays are
  rejected even with an all-false mask: explicitly provide the intended complete
  unmasked vector rather than relying on implicit mask removal. No observations
  or failed groups are silently dropped.

For example, the integer targets `[2**53, 2**53 + 1]` and predictions
`[2**53, 2**53]` must not acquire a false zero error by an implicit float64
conversion. These inputs now reject. Changing units or explicitly centering
both vectors before conversion can be appropriate, but the caller must preserve
the scientific target definition and assess representability. The scorer cannot
recover precision already lost upstream, including a prior array, table or CSV
conversion. This guard applies to the two metric APIs; it does not certify
lossless precision at every training-data ingestion boundary.

The metric definitions and output fields are unchanged. MAE, RMSE, R² and signed
bias retain their existing arithmetic; R² is null for constant targets. The
existing arithmetic guard rejects overflow, underflow and invalid intermediate
calculations. It does not detect every finite rounding or cancellation error. `grouped_metrics` retains equal weighting of declared groups for `group_mae`
and `group_rmse` and the existing exact group-identifier contract. Overall bias
is row-weighted; per-group biases are reported separately. A row count is not a count of
independent sources, and declared groups do not establish independence.

Shape and value checks run before grouped indexing, so scalar, mismatched and
multidimensional vectors consistently raise `ValueError`, not an incidental
indexing exception. Errors do not echo the rejected source values. Neither
accepted inputs nor synthetic regression tests establish prediction reliability,
calibration, future coverage or scientific predictive performance.

## Signed bias: cancellation limitation

Bias is currently computed as the float64 mean of the float64 residuals
`prediction - observation`. In data with a very large spread of residuals,
cancellation can make its magnitude and even its sign depend on row order,
without triggering the arithmetic guard. For example, the following outputs
were observed with NumPy 2.3.5 and Python 3.12.14; results can vary with the
platform and NumPy implementation:

```python
from groupbench.metrics import regression_metrics
y = [0.0] * 4
regression_metrics(y, [1e16, 1.0, -1e16, -0.5])["bias"]  # -0.125
regression_metrics(y, [1e16, -1e16, 1.0, -0.5])["bias"]  # +0.125
```

For these stored float64 values, the exact mean signed error is +0.125 in
both orders. This limitation applies to pooled and per-group bias, including
outputs that reuse these metrics. It does not by itself imply an error in MAE
or the model-selection objective. Preserve the observations, predictions,
row/group alignment and enough precision to reconstruct the scored float64
operands, along with the original metric outputs. When bias matters,
independently recompute it using higher-precision subtraction of each
prediction and observation before summation; summing already rounded float64
residuals can retain subtraction errors. Review the numerical scale as well. Stable bias calculation
would require a versioned metric/reader strategy so historical outputs are not
silently reinterpreted. The other metrics are not certified exact by this note.
