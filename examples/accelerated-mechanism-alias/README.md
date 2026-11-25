# Two accelerated conditions need not identify use-level survival

```sh
python examples/accelerated-mechanism-alias/run.py
```

Consider an invented scalar stress coordinate x. One world has a
single exponential failure clock with hazard (25/3)(6/5)^x.
The other fails at the earlier of two independent exponential clocks
with hazards 8 and 2^x. Without mode labels the latter's survival is
also exponential, with total hazard 8 + 2^x.

At the two accelerated coordinates x=1 and x=2, both worlds have
hazards 10 and 12, respectively. They agree on the entire survival curve
at each, not just one quantile. At the invented use coordinate x=0,
the hazards differ: 25/3 versus 9. More devices at the same two
accelerated conditions cannot distinguish these particular worlds.
A matched third stress condition or qualified mode labels can
distinguish them in principle; neither guarantees an adequate real
experiment or proves a mechanism. Mode labels must not be inferred
from a fitted all-cause rate alone.

The [NIST/SEMATECH reliability handbook](https://www.itl.nist.gov/div898/handbook/apr/section1/apr14.htm)
explains that physical acceleration factors generally depend on failure
mode. This exact example uses no RRAM data, operating stress, or actual
acceleration factor. It demonstrates nonidentification under its
specified exponential models, not a lifetime prediction.
