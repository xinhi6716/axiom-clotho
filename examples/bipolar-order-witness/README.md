# Equal per-sign exposure can hide a different sequence

```sh
python examples/bipolar-order-witness/run.py
```

[Shim et al., *Investigation of Read Disturb and Bipolar Read Scheme*](https://par.nsf.gov/servlets/purl/10181610)
measured read-disturb behavior and evaluated bipolar reads on a specific
HfO₂ RRAM platform. That motivates keeping signed command histories;
it does not establish the invented dynamics here.

Each command in the fixture has unit magnitude and duration. Two ordered
records (++−− and −−++) have identical signed sum, identical per-sign
exposure, and identical elapsed duration. The toy starts at state 0;
at each command it adds the signed increment and clips to [0,2]. Final
states are 0 and 2 respectively. An additive unsaturated comparator
would give 0 for both. The difference follows from order under this
nonlinear rule, not from different duty, magnitudes, or per-sign totals.

A missing sign in (+, −, unknown) prevents computing the signed summaries
or replaying a single state. Conditional on the unknown command being
either unit positive or unit negative, enumeration gives states {0,1}.
That conditional set is not permission to fill missing polarity with zero,
infer the actual command, or assign probabilities. On real traces preserve
the event order, polarity reference and provenance, and distinguish
command labels from voltage at the device. Even exact exposure metadata
alone do not measure physical read disturbance.
