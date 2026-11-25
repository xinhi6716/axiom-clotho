# A peak of the mean profile is not a mean of section peaks

```sh
python examples/profile-aggregation-order/run.py
```

Each invented track has two cross-section profiles on the same unit grid
and above the same zero reference. A peaks at different positions in
the two sections; B has the same shape in both. A/B have peaks of their
pointwise mean profiles of 2 and 3, but mean section
peaks of 4 and 3. Both mean profiles have trapezoid
area 4. Equal area and matched grid do not make different peak
aggregation orders interchangeable.

[Jardon et al., Figs. 9–11](https://doi.org/10.2351/7.0000516)
motivates tracking the protocol: their height is extracted after
averaging registered section profiles. This example has no measured
profiles and no inference about another source's actual method.
Before scoring across sources, record sampled span, common reference,
profile registration and which operation comes first. Do not pool
different height endpoints based on a shared unit or column name.
