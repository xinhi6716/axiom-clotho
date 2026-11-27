# Repeat reads do not create more randomized chips

```bash
python examples/chip-assignment-randomization/run.py
```

Six invented chips each supply two identical reads. Exactly three chips are
randomly assigned to each arm, with every balanced chip allocation equally
likely. Under the sharp null, their fixed readouts regardless of assignment
are (0,0,0,1,1,1) by chip. In the displayed assignment the three 1 chips are
treated. The absolute difference of arm means is 1 whether computed by chip
or read, because the repeat count is the same everywhere.

At the chip level, 2 of 20 assignments are at least as extreme (two-sided
p = 1/10). Shuffling individual read labels creates 924 assignments, many
splitting a chip across arms; only two are as extreme (p = 1/462).
Enumerating the **true** twenty sharp-null allocations at nominal 5% gives
0/20 rejections with the valid test and 2/20 with the wrong one. This is
conditional on the invented fixed outcomes and actual allocation scheme,
not a universal false-positive rate.

[Randomization-based confidence intervals for cluster randomized trials](https://doi.org/10.1093/biostatistics/kxaa007)
explains restricting reassignments to the original cluster-level design.
Use the recorded assignment unit and allowed allocation scheme, not the
number of rows or just the names of physical devices. The example does not
infer a weak average-effect null or a prediction error confidence interval.
