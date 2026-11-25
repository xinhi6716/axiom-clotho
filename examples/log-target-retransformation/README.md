# The inverse of a log-scale mean need not be the arithmetic mean

```sh
python examples/log-target-retransformation/run.py
```

Two equally weighted, origin-known synthetic strata have positive
outcomes. In the quiet stratum they are always 1. In the variable
stratum they are either 1/4 or 4 with equal probability. The mean log₂
target is zero in both strata, so direct inversion predicts 1 for both.
The arithmetic means instead are 1 and 17/8.

For a fixed log₂ predictor m, the original-scale mean is
2ᵐ × E[2^(log₂Y−m) | origin information]. The example fixes m=0
and calculates residual factors using only the development roster.
The evaluation roster separately enumerates its declared synthetic outcome
distribution. No outcomes from it enter the prediction calculation.

| Rule | Quiet prediction | Variable prediction | Original-scale MSE |
| --- | ---: | ---: | ---: |
| Direct inverse | 1 | 1 | 153/64 |
| Pooled smearing | 25/16 | 25/16 | 531/256 |
| Stratum smearing | 1 | 17/8 | 225/128 |

The pooled factor recovers the overall mean under this balanced mix
but biases both stratum means. This is a synthetic squared-loss
example, not a universal recommendation to stratify. Estimating
conditional factors requires enough development support and evaluation
on new data. A log-scale objective or an original-scale median is a
different prediction target; zeros cannot be silently logged.

[Duan's original smearing paper](https://doi.org/10.1080/01621459.1983.10478017)
describes residual-based retransformation. The example fits no model,
uses no measurements, and does not assume multiplicative errors for real devices.
