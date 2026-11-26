# Pooled hull, source-group hulls and nearby trials

```sh
PYTHONPATH=src python examples/pooled-hull-and-local-support/run.py
```

The exact two-feature hull audit answers a geometric question, not whether
a recipe was tested nearby. Use a fixed physical metric and distinguish
pooled containment, the number of individual group hulls that contain the
query, survival after each leave-one-group-out operation, and the number
of groups with an observed setting within a predeclared radius.

Four invented groups form two copies of each vertical edge of the square
`[0,2] × [0,2]`. The center lies inside the pooled hull after dropping
any one group, yet inside none of the four individual hulls. Keeping only
one group per edge leaves the pooled hull unchanged and loses resilience.
Duplicating rows within a group adds no independent support.

In another invented panel, three groups each have points on both sides of
zero, at respective distances 10, 20 and 30. All contain zero in their
own hull and the pooled hull survives every one-group deletion. None has
an observation within radius 1. Bringing the points to 1/8, 1/4 and 1/2
keeps these hull findings identical but gives three nearby groups. On
the far panel, a flat zero response and the 1-Lipschitz response
`max(0, 1-|x|)` even agree at every measured point but differ at zero.

Freeze units, distance scale, group hierarchy, and radius from training or
engineering information before using them. Group labels are not evidence
of independence. Keep the primary held denominator unchanged; use the
diagnostics as context rather than filtering harder cases after scoring.
The examples are synthetic, not support findings for a physical printer
or validated process.

[Donoho and Gasko](https://doi.org/10.1214/aos/1176348890) study point
halfspace depth; the source-group counts here answer a different question.
