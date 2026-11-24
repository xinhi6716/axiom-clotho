# Feature support does not establish outcome coverage

Run from the repository root with the current checkout and its dependencies:

```sh
PYTHONPATH=src python examples/support-is-not-coverage/run.py
```

Both invented worlds have the same four query rows, the same device roster
and the same preassigned intervals [0, 2]. No query rows trigger the
training-feature support checks in either world. Shifting only the outcomes
by three changes observed whole-device coverage from 2/2 to 0/2. Both rows
of a device must be covered for that device to count.

Feature-support flags are useful diagnostics for the declared predictors.
Their absence cannot establish stable conditional outcomes or calibrated
intervals. To check observed coverage, separately collect outcomes and audit
the full declared roster. These counts are descriptive; no models were
fitted and neither world establishes prospective coverage.
