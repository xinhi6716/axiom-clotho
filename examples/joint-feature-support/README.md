# In-range features need not be jointly supported

```sh
python examples/joint-feature-support/run.py
```

An invented two-variable process design contains (0,0), (1,0),
and (0,1). Each input independently ranges from 0 to 1. The proposed
setting (3/4,3/4) passes both range checks but lies outside the joint
triangle: its barycentric coefficient on (0,0) is −1/2. The setting
(1/4,1/4) is inside that triangle.

Neither label proves predictive accuracy. Two possible response rules,
zero and 4 max(0,x+y−1) + 16xy, agree at all three design vertices
but differ by 11 at the first proposed setting and by 1 at the second.
These rules are invented; they do not describe a physical process.
Check joint geometry using training features only, without treating
hull membership as a density, calibration or validity certificate.
The fixture is for one fixed two-dimensional design, not a general
hull library, physical feasibility check, or measured dataset.

[Balestriero, Pesenti and LeCun](https://arxiv.org/abs/2110.09485)
use convex-hull membership to define geometric interpolation and
extrapolation and discuss how common the latter is in high dimensions.
This example only shows the distinction on a small declared design.
