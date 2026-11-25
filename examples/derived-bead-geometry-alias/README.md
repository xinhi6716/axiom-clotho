# Derived deposition geometry does not determine the measured bead-height trend

```sh
python examples/derived-bead-geometry-alias/run.py
```

[Ye et al., §2.2, Eqs. 2–4](https://doi.org/10.1016/j.jmapro.2025.11.055)
distinguish bead height H and melt depth D from derived dilution d,
hatch spacing h and layer thickness t. For width W, clad area A and
melted area B, their relationships are d=B/(A+B),
h=W√(1−H/(H+D)), and t=A/h.

This exact invented example fixes H+D=200 and reports (d,h,t) of
(7/20,150,65) and (1/5,100,85) for two conditions. Choose measured
heights 72 then 128 or reverse them. For each height, the observed
summary is recovered with W=h/√(1−H/200), A=th, B=Ad/(1−d).
The roots are rational 4/5 and 3/5. In the two worlds, every reported
summary agrees but ΔH is +56 versus −56. All areas are positive
and pass elementary enclosing rectangle bounds; that is not a
certification of actual deposited geometry or manufacturability.

The script checks h² and t² in exact arithmetic; positive h and t
make this equivalent to checking the values. It uses no measured
rows. Do not replace a missing measured-height label with an
increase in designed layer thickness, or treat a derived quantity
that uses an outcome as an input available before measurement.
