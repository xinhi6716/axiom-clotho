# Choosing a narrower band on its own calibration devices can under-cover

```sh
python examples/calibration-width-selection/run.py
```

Each invented device has an independent Uniform(0,1) value, repeated
at every scheduled read on that device. The largest absolute residual
over the complete device is therefore the same as for its single value.
The two frozen predictors have constant centers 0 and 1. At 95% with
19 calibration devices, the split conformal rank is 19, the maximum
device residual. Each fixed-center construction has exact marginal
coverage 19/20. Choosing the narrower band on those same calibration
outcomes has coverage 37/40, even with the device grouping intact.

For calibration minimum L and maximum M, the two widths before doubling
are M and 1−L. The selected band's conditional coverage on Uniform(0,1)
is their minimum. Its expected failure probability is the mean of
max(1−M,L). Each boundary gap has mean 1/(n+1), and their minimum has
mean 1/[2(n+1)] by integrating (1−2t)^n from 0 to 1/2. Hence the selected
failure probability is 3/[2(n+1)] at the maximum-residual rank. This
is marginal over calibration and a new device, not a tolerance guarantee.

In the deterministic illustration, the two half-widths are 49/100 and
69/100. The narrower center-0 band misses 3/5; the center-1 band covers
it. Mirroring calibration outcomes swaps the widths. A prespecified
center and score stay fixed if calibration outcomes change; scoring new
outcomes does not change the emitted band.

See [Liang, Zhu and Barber, §2.3](https://arxiv.org/html/2408.07066v4#S2.SS3)
on selection and calibration on the same outcomes. This synthetic
example neither implements their general correction nor validates any
real device or schedule. Distinct device IDs alone do not establish
exchangeability; this example assumes independent, identically
distributed complete devices with a common fixed read schedule.
