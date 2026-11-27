# Conditioning on a post-assignment acceptance flag

```sh
python examples/post-assignment-acceptance/run.py
```

Four invented units have a hidden binary robustness U and fixed measured
endpoint Y=U, with values 0,0,1,1. Treat exactly two units uniformly at
random. Assignment T does not change Y for anyone. A separate acceptance
flag passes if T=1 or U=1. The endpoint is recorded for all units, even
when their flag fails. Selection here is an analyst's filter, not lost
follow-up, censoring, or terminal imputation.

Enumerate all six assignments. The attempted-roster treatment-minus-control
gap averages zero under the true design. Filtering to accepted units leaves
one assignment without any accepted control; among the other five, every
gap is negative and the mean is −3/5. No causal model fit can make that
accepted contrast a test of the original sharp null merely by retaining
the randomized assignment column.

[Elwert and Winship, Endogenous Selection Bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC6089543/)
gives the collider logic: T and U both affect acceptance, while U affects Y.
Checking assignment balance before the filter does not justify conditioning
on the flag. Keep the full attempted cohort for the assignment estimand and
report acceptances as a separate operational endpoint. If accepted-arm or
causal principal-stratum estimands are wanted, define and justify them
separately. The example is not evidence about a real device.
