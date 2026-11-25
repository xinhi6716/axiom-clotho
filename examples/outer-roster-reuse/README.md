# Reusing an evaluated outer roster for selection

```sh
python examples/outer-roster-reuse/run.py
```

Two disjoint invented rosters each have four fair, independent binary
outcomes. All candidate prediction vectors are fixed before any outcomes,
and use the same four label-free coordinates on both rosters. The example
selects a candidate using only the previous roster's outcomes, then scores
it on that roster and on a fresh roster independent of the selection.
It enumerates all 256 equally likely pairs of outcome vectors.

| Candidate family | Previous roster, reused after selection | Fresh roster |
| --- | ---: | ---: |
| Fixed all-zero comparator (no choice) | 1/2 | 1/2 |
| Choose between all-zero and all-one | 5/16 | 1/2 |
| Choose among all 16 hard-probability vectors | 0 | 1/2 |

Scores are exact expected Brier losses over invented label worlds. Even
choosing between just two constants improves the apparent reused score,
not the fresh one. The earlier roster can still be used as development
evidence. Once its results guide a later choice it is not an untouched
test of that choice, even if the earlier run used a valid nested split.
This is not a bias correction, a provenance guard, or measured performance.

See [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)
for model-selection overfitting and evaluation bias; this example only
illustrates the cross-run reuse of a previously evaluated roster.
