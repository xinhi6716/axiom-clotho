# A scheduled endpoint is not the first read after a clock cutoff

```sh
python examples/scheduled-anchor-vs-first-after/run.py
```

The invented roster names reads at nominal 0 s, 1 s, and 100 ks.
Each record separately reports when the reading was actually captured.
The scheduled 100 ks row arrives at 100011 s with value 3. The first
read strictly after 100000 s actually belongs to an older nominal slot:
it arrives at 100001 s with value 10. These answer different questions.
From the 1 s value 7, the two endpoint choices even imply opposite
directions (−4 and +3).

Remove the scheduled terminal row. The first-after statistic is still
available, but the exact scheduled endpoint is incomplete. Do not
replace it with a nominally different record, choose between duplicate
nominal matches, or silently reorder inconsistent actual anchor times.
The fixture does not specify how far an actual reading can slip before
becoming ineligible; freeze such a tolerance separately if the protocol
needs it. This example does not infer time labels from observed values
or claim the schedule represents any real measurement.

The [CDISC SDTM time convention](https://www.cdisc.org/standards/foundational/sdtm/sdtm-v1-6)
illustrates the general metadata distinction between planned elapsed
time and observation date/time. This example is not a CDISC interface.
