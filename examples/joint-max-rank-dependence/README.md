# Marginal ranks do not determine the rank of a joint maximum

```sh
python examples/joint-max-rank-dependence/run.py
```

Two fixed component scores are already on a common synthetic scale;
larger means less typical. Nineteen calibration rows have one high
value in each column. In one world the high values coincide; in the
other they occur in different rows. For the same query (1,1), the
component tail ranks both equal 2/20 in both worlds. The joint maximum
rank instead changes from 2/20 to 3/20 because the occupancy of the
maximum's upper tail changes. A flag using p ≤ 1/10 flips. Reordering
intact rows is a negative control and leaves all scores unchanged.

The discrete conservative rank is
`(1 + number of calibration scores at least the query score)/(n + 1)`.
It includes ties and the query. Enumerating each member of the 20-score
pooled multiset as the query gives exactly 2 or 0 flags, respectively.
This is a conditional rank argument when the query role is uniform
under exchangeability, not a coverage guarantee under arbitrary chip
shift. Component score definitions and the max aggregator must be
frozen beforehand or learned from independent training data. Rows
must be the relevant exchangeable unit; replicates do not add units.

[Shafer and Vovk’s conformal prediction tutorial](https://www.jmlr.org/papers/v9/shafer08a.html)
motivates the exchangeability and tie convention. Here the score ranks
unlabeled inputs for screening only. A non-flag does not establish
feature support, prediction accuracy, or adequate dependence coverage.
