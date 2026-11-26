# Disjoint test parents can still give dependent fold losses

Run `python examples/disjoint-fold-loss-dependence/run.py` from the checkout.
This invented example evaluates the train-only mean under leave-one-parent-out
squared loss. There is one observation per independently sampled parent,
equally likely to be −1 or +1. For each sample size the script enumerates all
possible datasets, with each parent used once for testing and never in its
own fit. It compares the exact sampling variance of the average cross-validated
loss to the variance obtained by omitting the between-fold covariance terms
while keeping the correct marginal fold variances.

| Parents | Actual variance | Independence shortcut | Pairwise fold covariance |
| ---: | ---: | ---: | ---: |
| 2 | 4 | 2 | 4 |
| 3 | 3/4 | 3/4 | 0 |
| 4 | 8/27 | 10/27 | −8/81 |

At two parents, each test value trains the other fold and the two losses are
identical, even though the training sets themselves do not overlap. At four
parents the covariance changes sign, so ignoring it is not predictably
conservative. Zero covariance at three does not establish independence.

[Bengio and Grandvalet (2004), original JMLR paper](https://jmlr.csail.mit.edu/papers/v5/grandvalet04a.html)
analyze K-fold error correlations and show that no universal unbiased estimator
of K-fold variance exists. This example computes a sampling variance only
because its data-generating distribution is known by construction.
It is neither a variance estimator for observed scores nor a critique of
the point estimate or the group split. For a physical transfer claim, freeze
the trained comparison before testing it on separately sampled untouched
parents and justify independence at the actual fabrication/acquisition level.
Otherwise keep fold variation descriptive rather than treating the folds as
independent replicates.
