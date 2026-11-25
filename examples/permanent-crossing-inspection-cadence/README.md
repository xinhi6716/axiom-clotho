# Inspection cadence can reverse an apparent retention ranking

```bash
python examples/permanent-crossing-inspection-cadence/run.py
```

Two invented devices per arm are followed to tick 10. One in each arm is
verified stable through 10. For the other, qualified reads bracket a
permanent threshold crossing in continuous time: A in (0, 8] and B in (4, 6]. Take the mean
of the individual restricted times min(T, 10) within each arm.

Using the right endpoint (first failed read, or the horizon if stable) gives A − B = 1.
Using bracket midpoints gives −1/2. Both ranks are imputations: the same
reads allow actual contrasts −5/2 (crossings 1 and 6) and +3/2 (8 and 5).
With no restrictions coupling the devices, the exact finite-roster
contrast is in (−3, 2), with both ends open. Add only a verified stable
read for A at tick 5 and the lower endpoint becomes −1/2. Opening a
bracket endpoint is different from dropping an observation.

This is a synthetic identification calculation, not a confidence interval
or a fitted survival model. It assumes qualified readings of one permanent
transition; transient or reversible excursions between reads need a
different endpoint. Detection time under a specified inspection policy can
itself be a legitimate outcome, but it is not the time of latent failure.
See [Turnbull's original interval-censoring paper](https://doi.org/10.1111/j.2517-6161.1976.tb01597.x)
for the distribution-estimation setting; this example only computes the
finite-roster contrast support.
