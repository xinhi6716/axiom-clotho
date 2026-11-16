# Baseline-relative regression skill

`clotho_core::evaluation::regression_mse_skill(observed, predicted, baseline)`
reports `1 - MSE(predicted) / MSE(baseline)` using identical aligned rows.
For example, targets `[1, 3]`, predictions `[1, 2]`, and baseline `[0, 1]`
give `0.8`: 80% lower squared error than that baseline.

- Positive favors the model; zero is a tie; negative scores are retained.
- A perfect baseline returns `None`, even if the model is also perfect.
- Unlike centered R², skill may be defined for constant targets.
- All inputs must be finite, nonempty, and equally sized. Invalid values are
  rejected rather than silently dropped, including with a perfect baseline.
- Raw squared errors are not formed, permitting large or tiny finite errors.
  Unrepresentable RMSE or score returns `EvaluationError::NumericOverflow`.

Choose the baseline before evaluation. Persistence should use only the value
available at the prediction origin, with the same target and horizon as the
model. Fit any learned baseline exclusively on training data. This API checks
numeric inputs, not timestamps, split provenance, physical independence, or
whether the supplied baseline leaks held-out outcomes.

This is a core API addition, not a CLI option or a real-data accuracy claim.
Compare paired errors on held-out groups and retain negative results. A gain
over noisy persistence alone does not prove physical state prediction or
causality: shared-read mathematical coupling can create apparent improvement.

## Declared ordered cohorts

`clotho_core::saved_comparison::compare_saved_regression` wraps the numeric
primitive for three in-memory `SavedRegressionSeries`: observed, model and
baseline. Each supplies a `CohortContract` (cohort, target, unit, horizon), ordered
opaque `row_ids`, and numeric values. Contracts and complete row rosters must
match literally. Duplicate IDs, reordered rows, missing rows, nonfinite values,
blank/control-containing declarations, more than 20,000 rows or declarations
longer than 256 UTF-8 bytes are rejected. No sorting, joins or normalization occur.

The report retains the declared contract, row count, both RMSEs and MSE skill.
It performs no fitting or selection. For a training-mean baseline, retain the
correct outer-training mean for each held-out row; for RRAM persistence, retain
the origin-available read in the same units and target/horizon. This API cannot
check those conditions or distinguish forged IDs from authentic ones.

This is not the Python `compare_saved_predictions` file protocol: it verifies
no byte checksums, model execution, physical identity, hidden cohort omissions,
historical independence, or units' actual meaning. In particular, swapping
values while falsely keeping the same IDs passes the declared-roster check.
It is not a CLI, an uncertainty interval, or a causal/accuracy claim.
