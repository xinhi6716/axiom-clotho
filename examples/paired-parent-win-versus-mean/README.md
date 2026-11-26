# A parent win count is not mean improvement

Run `python examples/paired-parent-win-versus-mean/run.py` from the checkout.
This example uses invented paired losses with one difference per parent.
Positive gain means that the candidate has lower loss than the baseline.

Seven gains of +1 and one of −8 give seven wins, but an equal-parent mean
gain of −1/8. The one-sided binomial tail under independent fair signs
is 9/256. This calculation tests fair signs, not zero mean gain. To see
why the distinction matters, an independent population with gains +1
and −7 at probabilities 7/8 and 1/8 has mean zero. A test using the
5% fair-sign cutoff would reject in over 73% of eight-parent samples
from that population. That is not a failure of a correctly specified
sign test: its fair-sign null is false there.

Use the binomial calculation as in the [NIST Dataplot sign-test
reference](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm).
The example makes no symmetry or mean inference. A physical benchmark
should prespecify the paired loss, parent weighting, and direction;
report wins, losses, ties, magnitude, and the worst parent separately.
Do not apply an independent-parent sign test to repeated rows or to
overlapping cross-validation folds just because fold IDs are distinct.
If the goal is mean improvement, evaluate that estimand directly and
choose inference justified by the physical sampling design. A significant
win count does not clear a mean-gain or worst-parent gate.
