# A sparse cadence cannot supply an earlier native read

```sh
python examples/early-anchor-rate-envelope/run.py
```

[Huang et al., IEEE Access, Fig. 5](https://ieeexplore.ieee.org/document/10974951/)
is the motivation for checking the reported retention-read schedule rather
than assuming that a dense grid is available. This example has no
measured rows and uses no rate or waveform from that paper.

An invented sparse protocol logs (time,value) at (10,10) and (20,8);
the requested early anchor is at time 1, strictly **before** both reads.
It is not an interpolation problem. Continuing the fitted straight line
backward gives 59/5. But even if the readout is monotone and its drop rate
is bounded between 0 and 1, the sharp identified interval is [10,19].
Both endpoints are attained by piecewise-linear histories that exactly
match both logged reads. The claim concerns a numerical rate-cap
assumption, not the field measurement protocol.

A hypothetical independent certificate for the *same read history* bounds
the drop rate throughout the early interval and through the known reads
between 1/5 and 1/4. It narrows the sharp bounds to [59/5,49/4], still not
a point. A proposed certificate with minimum rate 1/3 contradicts the
observed 1/5 decline and is rejected. A compatible later slope does not,
by itself, validate a certificate for the unobserved early interval.
Changing cadence may change the trajectory, so a certificate from another
schedule requires separate justification.

An envelope is a sensitivity result, not a native measurement. When the
scientific gate requires a logged anchor, hold out the unresolved
endpoint rather than treating extrapolation or a bound as a read.
