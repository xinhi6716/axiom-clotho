# Target weights are not sampling probabilities

```sh
python examples/unequal-block-sampling/run.py
```

Two invented manufacturing blocks contain one and three devices, respectively.
Every device in A has loss 0, and every device in B has loss 1. One complete
block is drawn; A is drawn with probability 3/4 and B with 1/4. The expected
observed block mean is 1/4, irrespective of the desired target. Those fixed
targets are 1/2 if blocks contribute equally and 3/4 if devices do.

The Horvitz–Thompson contribution is the drawn block's target mass times its
mean loss divided by its draw probability. Its design expectation equals
the fixed target. The equal-device estimator returns 0 or 3 in one draw,
with oracle design variance 27/16. Clipping the inverse probability at
two changes the equal-device expectation to 3/8. This is complete
enumeration of an invented population, not an estimated variance from one
draw. An estimate above 1 does not change the underlying loss range.

Use known acquisition probabilities, frozen target masses and observed
outcomes on sampled units. A positive-mass block with no chance of being
sampled cannot be recovered by weighting. The example does not repair
observational selection or justify a small-sample confidence interval.
[Horvitz and Thompson, original sampling design](https://doi.org/10.1080/01621459.1952.10483446).
