# Score the same forecasts under a fixed measurement schedule

With the current source checkout and Clotho dependencies, run from the repository root:

```sh
PYTHONPATH=src python examples/scheduled-phase-scoring/run.py
```

The two devices have different numbers of early and late observations. All
values are invented. Model A makes a single error of five on D2's late row;
model B makes error one on every row.

| Weighting | A MAE | B MAE | Lower MAE |
| --- | ---: | ---: | --- |
| Every supplied row equally | 0.625 | 1 | A |
| Every device equally, retaining its observed phase mix | 0.5 | 1 | A |
| Every device equally, with declared 50:50 early/late mix | 1.25 | 1 | B |

The last column changes because the estimand changes, not because either
forecast changes. Choose phase weights to match your study question before
scoring. The script separately declares expected rows and verifies that
omitting one row fails even when every device–phase cell still has data.
Neither the declaration nor the score verifies that a real protocol was
preregistered or that devices are physically independent. No models are fitted.
