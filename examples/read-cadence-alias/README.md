# A fixed read cadence cannot separate elapsed time and read stress

```sh
python examples/read-cadence-alias/run.py
```

At age t, let n count intermediate reads strictly before the common
terminal measurement. In two invented worlds, loss is t or n,
respectively. When every terminal measurement has n=t, every observation
agrees across worlds. A physically different held chip following that
same cadence cannot distinguish these worlds either.

Hold terminal age at 4 and compare the same four intermediate reads
with no intermediate reads. In the time-only world, both losses are 4.
In the read-only world, they are 4 and 0. The linear two-column design
rows (t,n) are collinear for the original cadence (determinant zero);
the two matched-age rows have determinant −16. This rank claim assumes
the toy's known zero baseline and no intercept. A real design would
need additional ages and controls for an unknown intercept, interactions
and nonadditive effects. The numbers are abstract, not test settings.

Do not infer unobserved retention or support a new read schedule just
because a model transfers across chips on one fixed schedule. If read
exposure is the scientific question, predeclare intermediate histories
and use separately assigned cells, a common terminal read procedure,
and preserved cell/chip identities. Equal counts do not establish
equal read stress: pulse shape, timing and history can matter.

[Primary HfO₂ RRAM read-disturb experiment](https://www.sciencedirect.com/science/article/abs/pii/S0026271413002187)
used constant voltage stress, not this example's synthetic counts. It
motivates checking disturbance, not an assumed damage law. No fit,
hardware, or measured data are used here.
