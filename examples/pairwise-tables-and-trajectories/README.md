# Even matching pair tables can hide different three-read histories

```sh
python examples/pairwise-tables-and-trajectories/run.py
```

In two invented rosters, four devices each have three binary readings.
One roster contains the four histories with an even number of ones;
the other contains the four with an odd number. Every two-read table
contains each binary pair exactly once, even for the first and last
reads. All single-read distributions therefore agree as well.

Among devices that start at 0 and then reach 1, one device qualifies
in each roster. Its final state is 1 in the even roster and 0 in the
odd roster. The conditional return to state 0 is therefore 0 versus 1.
Do not reconstruct a three-read trajectory from separate pair tables
unless additional assumptions are justified. Use a stable cell and
episode key across all three reads, secure their temporal order, and
report the denominator for the history-defined endpoint. This example
assumes complete histories; it does not resolve missing reads.

[Wiefels, Fig. 5.3(c–d)](https://publications.rwth-aachen.de/record/817653/)
uses paired reads to reveal cell-level variation despite a stable
distribution. This invented example makes a different, three-read
point; it does not reproduce that experiment or infer device recovery.
