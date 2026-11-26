# Retention timing: same timer number, different evidence

A timer can use the same display name after a reset. An export can also preserve
a nominal read label even when the measurement time or origin is unknown.
Subtracting the two displayed numbers alone does not establish elapsed time.

Run from the repository root:

```sh
python examples/retention-clock-identity/run.py
python -m unittest discover -s examples/retention-clock-identity
```

The invented rule admits a scheduled terminal read only when the two timestamps
share a documented clock and uninterrupted epoch, and their entire elapsed-time
range falls within the closed 80,000–120,000 s window. If the two timestamp
errors are bounded by `e_origin` and `e_terminal`, the conservative elapsed
interval is their displayed difference ± `(e_origin + e_terminal)`. This does not
assume independent errors or convert them into a confidence interval. A
verified clock transformation could justify another method, but this example
does not infer one or treat a missing origin as zero.

The linked run passes. A reset, different clock, missing origin, or uncertainty
across a boundary leaves the observed terminal read unscored for this endpoint.
A proven outside-window read has a separate reason. Retain those records and
their reasons in the attempted roster rather than treating them as device
failures or silently discarding them. This is a synthetic timing lesson, not a
model fit, real device result, or full eligibility audit.
