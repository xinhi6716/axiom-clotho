# A retention schedule is not a target distribution over time

Run `python examples/retention-time-measure/run.py` from the checkout.
One idealized baseline (A) and candidate (B) have known piecewise-constant
absolute losses on [1,10) and [10,100) seconds. A gives losses (0,2) and
B gives (3,0). With equal probability per decade, A wins (1 vs 3/2).
With equal probability per elapsed second, B wins (20/11 vs 3/11).
The predictions and error traces are unchanged. A missing inter-read
trajectory is not a piecewise-constant loss curve by default.

[Frascaroli, primary experimental thesis, §3.4.2](https://air.unimi.it/bitstream/2434/356780/2/phd_unimi_R10355.pdf)
describes logarithmically spaced resistance reads in a hafnia retention
experiment. That is motivation only; it does not prescribe a scoring
distribution. The example uses no source data and does not substitute
nearby reads for nominal anchors.

Prespecify whether the objective is one scheduled anchor, a weighted set
of scheduled reads, or an integral over a declared workload time law.
The last objective needs an independently justified within-interval
measurement, model, or bound. If none is available, report only the
scheduled-read score. Comparing absolute loss by target time without
changing the model or sampling cohort falsifies an assertion that its
ranking is insensitive to that choice; score each deployment question
against its own prespecified weighting, not whichever makes B win.
