# Compare two readout chains before sharing a correction

Use the source Python API when two routes, loads or instrument configurations
may give different responses to the same input. Run known low, interior and
high standards through **each** route. Keep their identities, units, selected
range, connection, bias and measurement state. The function does not operate
instruments or infer that the standards match the device load.

```python
from groupbench.quantity_contract import compare_affine_readouts

check = compare_affine_readouts(
    [2, 6, 10], [3.5, 8.5, 13.5], [3.5, 9.5, 13.5],
    max_interior_error=0.25, max_path_difference=0.25,
)
assert check.max_affine_path_difference == 0  # identical endpoint lines
assert check.comparison_interior_error == 1  # reserved middle check disagrees
assert not check.shared_affine_pass
```

These values are invented. The two endpoint points define each affine line.
The signed interior residual is observed minus that line. The largest absolute
difference between two affine lines **on the bracketed interval** is attained
at one of the endpoints. Both interior checks and that endpoint comparison must
pass their inclusive thresholds. Supplied numbers are compared in exact ratio
arithmetic; use Decimal where tolerances are intended as exact decimals.

The result is a deterministic threshold check, not a statistical equivalence
test, instrument calibration, or proof of transfer. Use externally chosen error
budgets, calibration and interleaved repeats if appropriate. Do not substitute a
different load/path or extrapolate between instruments. Additional check levels
may reveal curvature undetected by one interior point. [NIST calibration check
standards](https://itl.nist.gov/div898/handbook/mpc/section3/mpc37.htm).
