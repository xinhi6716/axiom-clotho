# Pooled pair ranking can hide within-stratum failure

```sh
python examples/stratified-concordance/run.py
```

Six invented devices belong to two strata available at prediction time.
The candidate ranks all nine cross-stratum pairs correctly but reverses
all six within-stratum pairs. Its pooled concordance is 3/5, above a
constant baseline's 1/2, but the candidate scores zero within strata.
The frozen origin-stratum baseline uses only the same starting
information and predicts 2 and 11 in the two strata. It gets 4/5 pooled
and 1/2 within strata, because all six within-stratum predictions tie.

Count pairs with identical outcomes separately, and exclude them from
the scoring denominator. For unequal outcomes a prediction tie is half
a correct pair; if no eligible pair remains the result is undefined.
This is a descriptive pair accuracy, not Kendall's tau or a
confidence interval. Pairs share devices; nine pairs do not prove nine
independent units. Compare point error separately and predeclare strata
based on information available before outcomes. The example uses
no fit or measured data.

[van Klaveren et al.](https://doi.org/10.1002/sim.6997) discusses case mix
and concordance in external validation; this example uses a different,
simpler fixed finite calculation, not that paper's estimator.
