# Matching read count does not match spacing

Run `python examples/read-spacing-recovery/run.py`.
All invented policies share the same endpoint and identical command
width and amplitude. A dense policy commands more intermediate reads,
so it does not match nominal exposure even when voltage and elapsed
time match. Two other policies both command two reads but differ in
their spacing from the endpoint. A toy state halves at every tick and
increases by one immediately after each intermediate read. At terminal
tick 6 the early pair (1,2) contributes 3/32 and the late pair (4,5)
contributes 3/4. An elapsed-only baseline and a count-only mechanism
both predict no difference between the two pairs. The dense policy
gives 31/32; the no-intermediate-read reference gives zero. The common
terminal observation is not counted as prior exposure.

[Choi et al., original crossbar paper, Fig. 2i](https://www.nature.com/articles/s41467-025-63831-2.pdf)
is a precedent for retention measured under repeated pulsed reads,
not for this decay rule or a schedule-independent effect. The example
does not use the article's waveform values or resolve exact timing
from its figure. Command count and nominal energized duration do not
measure delivered current, energy or physical disturbance.

A prospective same-age and same-terminal-read comparison can test
whether count alone suffices: vary spacing with count, waveform and
handling matched, and include a no-intermediate-read control. Record
actual acquisition timing and enforce assignment at the electrical
or thermal interference domain. If the control, timing or isolation
cannot be verified, report only the tested policy, not a read-free
retention estimate. No hardware settings are recommended here.
