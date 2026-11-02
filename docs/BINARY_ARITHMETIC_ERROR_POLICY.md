# Binary arithmetic and caller error settings

Binary metric arithmetic uses its own bounded NumPy error policy rather than
inheriting the application's global `numpy.seterr` settings. The caller's
settings are restored on return or failure.

This includes the `mean_probability_1` diagnostic. Previously, valid subnormal
probabilities such as `[3 * np.nextafter(0., 1.), 0.]` with labels `[1, 1]` could
produce finite scores under NumPy's default policy but raise a raw
`FloatingPointError` when the caller enabled underflow exceptions. Computing
that diagnostic inside the existing arithmetic scope removes the discrepancy.

Diagnostic means retain ordinary float64 rounding, including rounding an
unrepresentably small mean probability to zero. The separate exact endpoint
counts describe the supplied probabilities, not the rounded mean. The repair
does not change Brier or clipped log-loss definitions, clipping epsilon,
thresholds, class coverage, group weighting, supported input values or report
fields. In particular, a positive probability loss that rounds to zero still
raises `ValueError`; diagnostic probability rounding does not relax that guard.

These checks use synthetic arrays without training. No model-quality,
calibration, independent-source or causal improvement is established. AUC is
not a reported metric in the present binary API and is not introduced here.
