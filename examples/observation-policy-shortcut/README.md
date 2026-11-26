# A timely read-count feature can learn the observation policy

Run `python examples/observation-policy-shortcut/run.py` from the checkout.
In this invented example an unpublished pre-origin flag determines each
unit's eventual binary outcome. In development and one new independent
cohort, the measurement system uses that flag to decide whether to add
an early read. The model uses only the recorded presence of that read,
which is known at prediction time. Its value is unused and the read has
no effect on the outcome by construction.

A model frozen on development obtains Brier loss 0 on a held cohort
under the same observation policy, versus 1/4 for the train-only pooled
forecast. In a second held cohort with early reads assigned independently
of the flag and the same outcome mix, losses become 1/2 vs 1/4. No model
is refit, no future fields appear, and the change is in the read policy,
not the device. The example enumerates support with equal probability
in each cohort, rather than claiming precision from two or four devices.

[Lin, Scharfstein and Rosenheck (2004), original JRSS B article](https://doi.org/10.1111/j.1467-9868.2004.b5543.x)
studies outcome-dependent observation times. The example shows a separate
prediction transport issue and does not implement their weighting
estimator. Before claiming cross-policy transport, freeze the same-roster
schedule-free baseline and candidate and compare on a held policy that
changes observation scheduling while preserving the target and time origin.
Report the policy shift separately from ordinary held-parent validation.
If reads can disturb the devices or change missing endpoint capture,
their effect must be addressed separately; this toy cannot identify it.
