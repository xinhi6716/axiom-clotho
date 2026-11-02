# Binary labels and probabilities: masked-array handling

Binary validation and scoring reject an **active NumPy mask** before converting
the supplied array or built-in list/tuple sequence. This includes masked scalar
entries and masked row arrays nested inside probability-matrix lists or tuples.
Hidden backing values are not observed labels or
probabilities, even if they happen to be valid numbers or would give a perfect
score. The entire input fails with `ValueError`; the code does not fill hidden
entries, discard rows, choose a different cohort or resplit the data.

This guard covers:

- `validate_binary_target`, including classifier training labels and scoring
  labels when a single observed class is allowed;
- `binary_metrics` and `grouped_binary_metrics` probability vectors;
- `checked_probabilities` model-class labels and either column of its two-column
  probability matrix, including the estimator probability-return boundary.

For example, `np.ma.array([0., 1.], mask=[False, True])` is not a complete pair
of probabilities. It now rejects rather than silently becoming `[0., 1.]` and
possibly producing a false perfect score or inaccurate class/row counts.
Changing the hidden backing value does not make the input acceptable. Errors do
not echo that value, and rejection does not modify the data or mask.

## Compatibility and limits

Ordinary arrays retain their previous output. A numeric masked array with
`np.ma.nomask` or an all-false mask also retains its previous binary behavior:
there are no hidden entries, so its explicit values pass through the existing
label, shape, class-order, range, dtype and float64 arithmetic checks. This does
not make Boolean/object/string labels or invalid probability dtypes valid.

Do not assume all APIs accept the same containers. The direct regression metric
contract in [Regression metric input precision](REGRESSION_METRIC_INPUTS.md)
is deliberately stricter: it rejects **every masked-array container**, including
an all-false mask, as well as lossy integer conversion and wider floating types.
The binary guard does not relax that regression contract. The active-mask change
here is a compatibility break for previously accepted arrays with hidden data;
those old results should not be treated as complete-cohort scores.

Once a caller has already discarded a mask, these APIs cannot reconstruct it.
The mask preflight inspects NumPy masked arrays and built-in list/tuple storage;
it does not consume generators or arbitrary iterables. Cyclic or more than
64-level-deep sequence containers reject with `ValueError` before NumPy conversion.
Repeated references to the same legitimate row container remain supported.
Object/structured dtypes retain their existing rejection; this is not an
expansion of the supported probability dtype contract.

Pandas or another caller-side conversion may instead represent missing entries
as missing values, which existing validation rejects; a conversion that exposes
only finite backing values loses the missingness information. No mask semantics
are inferred from a plain numeric array. Group-identifier validation is separate
and is not changed by this binary patch.

No metric definitions, row/equal-group weighting, threshold (`p1 >= 0.5`),
calibration procedure or report schema are changed. Declared groups do not prove
independence, and a row count is not an independent-source count. These synthetic
input-validation tests establish no real-data accuracy, calibrated uncertainty
or future-coverage guarantee.
