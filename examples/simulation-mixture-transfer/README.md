# An improved mixed-source score can conceal worse measured predictions

```sh
python examples/simulation-mixture-transfer/run.py
```

Two fixed mean predictors learn from invented training data: one uses
measured rows only; one gives equal weight to each measured and simulated
row. Each is then scored on the same two reserved measured rows and six
reserved simulated rows. The mixed score has more simulation than
measurement by design.

| Training data | Constant forecast | Mixed MAE | Measured MAE | Simulated MAE |
| --- | ---: | ---: | ---: | ---: |
| Measured only | 0 | 6 | 0 | 8 |
| Equal-row hybrid | 6 | 3 | 6 | 2 |

The hybrid halves the mixed score while worsening the target measured
score. This is an exact toy, not evidence that simulated data generally
hurt. The choice of target domain and comparison weights is part of the
question, not a correction to make after seeing which model won.

For physical transfer, reserve complete measured fabrication parents
before any simulation calibration, feature engineering or model selection.
Track measured and simulated source families and seed/parent links;
simulation calibrated with a held-out experiment can disclose information
about it even if the displayed row IDs differ. Compare experimental-only
and augmented models on the same held measured roster and report
per-parent errors. Treat a separate simulation score as a diagnostic,
not as a substitute for the measured endpoint.
