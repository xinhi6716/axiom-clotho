# Decide which calibration offsets can move together

From the repository root, with Clotho's Python dependencies installed:

```sh
PYTHONPATH=src python examples/calibration-groups-mae/run.py
```

Four invented terminal log targets are zero. The two frozen forecasts
swap their predictions between paths A and B. Give each path the same
weight and consider an illustrative offset in [−1, 1] log units.

| Dependence specified before scoring | Paired MAE range (B minus A) |
| --- | ---: |
| Both paths share one offset | [0, 0] |
| Each path has an independently variable offset | [−2, +2] |

At any common offset the two paths' contrasts cancel. With independent
axes, one can move up while the other moves down, so cancellation need
not hold. Independence here means all combinations in the two declared
intervals are allowed; it is not inferred from the predictions or from
having two file names. For genuinely independent groups, minimize and
maximize each group's contrast, then apply the fixed group weights.
For one shared reference, send all relevant pairs into one sweep with the
desired row weights. If calibration shows a different cross-path
constraint, neither of these two worked cases automatically applies.

Keep the actual expected row IDs, same immutable forecasts, target
units, reference IDs and epochs, and a justified sensitivity interval
separate from the result. This is neither a confidence interval nor a
model selection step. The toy fits nothing and has no instrument data.
See [paired MAE contract](../../docs/REGRESSION_METRIC_INPUTS.md#one-shared-calibration-offset).
