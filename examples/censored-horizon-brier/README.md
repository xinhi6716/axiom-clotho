# Early censoring is not survival to the target horizon

Run `python examples/censored-horizon-brier/run.py`. The outcome is whether
a first failure occurred by a fixed horizon. An observed failure before
the horizon resolves the label to 1; a confirmed failure-free assessment
through the horizon resolves it to 0. Loss of follow-up earlier, with no
event yet, leaves the horizon label unresolved. This is not the same
question as a continuous measurement of conductance at the horizon.

On this invented two-unit roster a constant baseline forecasts 1/2 and the
candidate 3/4 on both units. The known event favors the candidate by
−3/16 (B minus A). Retaining the unresolved unit gives the exact paired
full-roster interval [−3/16, 1/16]. Each unresolved unit uses the **same**
binary label for both forecasts. Without additional constraints the
choices can be attained jointly across units, so the bounds are sharp.
Replacing the missing label by 0 or excluding it chooses a different
analysis and can lock in opposite winners.

[Graf et al. (1999), original survival-score research](https://pubmed.ncbi.nlm.nih.gov/10474158/)
motivates the problem of scoring censored probability forecasts. This
example is not an inverse-probability-of-censoring estimator, a confidence
interval or evidence of calibration. Freeze the roster, forecasts,
horizon and event definition; record each unit's censoring status. Without
an independently supported censoring model, hold a rank claim if the range
crosses zero and seek adjudication on the unresolved physical units.
