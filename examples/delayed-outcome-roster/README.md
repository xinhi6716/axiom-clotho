# Delayed labels on a frozen prediction roster

Run from the checkout root with Clotho's Python dependencies:

```sh
PYTHONPATH=src python examples/delayed-outcome-roster/run.py
```

Six invented rows have both forecasts fixed before any scoring; the forecasts
are never updated. This toy reveals outcomes only if their declared arrival
tick is strictly before the scoring cutoff. In an actual evaluation, record
the roster and forecast artifact first and have a separate evaluation process
control access to outcomes. A supplied tick is not proof of real availability.

At cutoffs 2 and 6, four labels are available. The available-row B-minus-A MAE
is +1, but on the original six rows its sharp missing-target bounds are
[−1/3, 5/3]. The available-row ranking cannot determine the roster ranking.
The complete-roster interval comparison rejects both partial snapshots instead
of scoring a smaller roster. At cutoff 7 both late labels are available: the
full MAE difference is −1/3 and the paired interval score difference is zero
(a distinct measure of interval width and misses, not a point forecast score).

The bounds describe the fixed roster with unrestricted missing labels; they
are not imputation, a confidence interval, or evidence of calibration. The
artificial labels are visible in this demonstration; no fitting or writes run.
