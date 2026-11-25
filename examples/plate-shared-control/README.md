# Two contrasts can share a control even when plates are independent

```sh
python examples/plate-shared-control/run.py
```

Two invented independent plates each contain one control and two
treatment-arm observations. All six outcomes independently take −1
or +1 with equal probability; arms have no effect in this toy.
For each treatment, average the two plate-specific treatment-minus-
control contrasts. These two averages reuse both controls.

Enumerating all 64 worlds gives variance 1 for each contrast and
covariance 1/2. The variance of their equally weighted mean is 3/4.
If the two contrast estimates were independent, the same component
variances would yield 1/2. Independent plates do not make the
overlapping contrasts independent. These are known synthetic variances,
not estimates from one two-plate experiment.

Preserve plate, arm and shared observation identities when reporting
overlapping contrasts. A joint analysis needs to preserve their
dependence and use the actual design; a larger count of contrasts
does not add independent plates. This example neither performs a
multiple-comparison test nor supplies a confidence interval.

[Dunnett's original shared-control paper](https://doi.org/10.1080/01621459.1955.10501294)
studies comparisons of several treatments with one control. This
example is separate exact variance arithmetic, not a Dunnett procedure
or evidence about measured materials.
