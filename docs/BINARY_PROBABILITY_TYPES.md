# Explicit numeric probabilities, not Boolean labels

Binary probability inputs must be real numeric probabilities. Python `bool` and
NumPy `bool_` are not probability scalar types. They reject with `ValueError`
**before** a mixed list or tuple can promote them into numeric 0/1 values.
This applies to probability vectors, either column of two-column probability
matrices and the estimator probability-return boundary.

For example, both `[False, 1.0]` and `[[True, 0.0], [0.0, 1.0]]` now reject.
Previously, those mixed sequences could become floats and pass, while an
all-Boolean array or list already rejected. The new check closes that
container-dependent inconsistency; it is a compatibility change for previously
accepted mixed Boolean/numeric inputs. It never converts labels into
probabilities or drops an observation to obtain a valid input.

## Supported numeric values remain unchanged

Use explicit numeric `0`, `1`, `0.0` or `1.0` when endpoint probabilities are
intended. Ordinary supported integer/float arrays and numeric lists or tuples
retain their existing scores. This is not a ban on endpoint probabilities.
The shape, probability range, sum-to-one tolerance, class order, supported
float precision, metric arithmetic, row/equal-group weighting and threshold
rules are unchanged. Object, complex, missing and other unsupported probability
inputs remain rejected under their existing contracts.

The source-type preflight traverses built-in list/tuple storage and recognizes
Boolean NumPy arrays, pandas Series/Index/extension arrays and Boolean memoryview
buffers, including those used as a row inside a numeric matrix list. It does not
consume generators or arbitrary iterables. Existing active-mask validation runs
first, preserving the [binary mask contract](BINARY_MASKED_INPUTS.md), including
nested-mask rejection and all-false/nomask numeric compatibility. It does not
change the separate binary-label validation API.

## Information already lost cannot be recovered

If the caller first builds `np.asarray([False, 1.0])`, the stored result is a
numeric float array. These APIs cannot recover the original Boolean scalar and
continue to score the numeric values supplied. The same limitation applies to
other caller-side coercions and custom conversion protocols. Inspect source
semantics before converting; the validator is not evidence that values were
probabilities at their origin.

Rejection does not modify source values or echo private payloads. The tests use
invented arrays without model fitting or scientific data. This input-contract
repair makes no real-data accuracy, calibration, uncertainty, future-coverage
or causal claim. A complete row count is not an independent-source count.
