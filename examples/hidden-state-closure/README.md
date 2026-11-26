# Same I–V reading, different future reliability

```sh
python examples/hidden-state-closure/run.py
```

Two invented filament states have identical present readings but different
probabilities of remaining conductive for one time step. Holding the
continuation protocol fixed, those probabilities are 1/2 and 1/4. A model
that sees only their identical present state must return the same value for
both. Its smallest possible worst-state error at one step is 1/8.

| Conditional survival model | One step | Two steps |
| --- | ---: | ---: |
| Hidden state A | 1/2 | 1/4 |
| Hidden state B | 1/4 | 1/16 |
| Mixture initially half A, half B | 3/8 | 5/32 |
| Single constant rate matched at step one | 3/8 | 9/64 |

The mixture is not a single constant rate because the slower-decaying state
becomes more common among survivors. The one-rate match misses the next
step by 1/64. The example does not rule out fitting one specified mixture
by an effective time-varying hazard; it rules out recovering both
hidden-state forecasts from the same present coordinate.

To check a simulated compact state, generate separate histories and
match **all** its pre-prediction state coordinates, within tolerances set in
advance. Compare future transition distributions at the same controls.
Compare a state-only and history-augmented model with disclosed information;
if a hidden-state oracle is included, report it separately. If the matched
strata are empty, the comparison does not test closure. A matching simulation
does not establish that a compact state is sufficient on physical devices.
See [Buchholz on Markov aggregation](https://doi.org/10.1017/S0021900200107338).
