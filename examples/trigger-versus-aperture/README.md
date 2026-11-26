# A trigger timestamp is not a read aperture

```sh
python examples/trigger-versus-aperture/run.py
```

Compare the read aperture's start with the delivered program's end, not
one issued command with another. With timestamps expressed on a verified
common timebase, let `delta` be the recorded trigger difference and let
`P=[p0,p1]` and `A=[a0,a1]` bound delays from the respective triggers to
program end and aperture start. The conservative elapsed interval is
`[delta+a0-p1, delta+a1-p0]`. The same additive clock offset cancels; a
clock scale error, differential path delay, and configuration-dependent
arming do not. The example assumes any scale uncertainty is already resolved.
The bounds are not standard errors or confidence intervals.

| Invented scenario | Physical elapsed, µs | In (0, 1,020,000]? |
| --- | ---: | --- |
| 1,005,000 µs trigger difference; 25,000 µs read delay | [1,030,000, 1,030,000] | No |
| 1,018,000 µs trigger difference; both path delays in [2,000, 7,000] µs | [1,013,000, 1,023,000] | Ambiguous |
| The upper edge equals the inclusive cutoff | [1,018,000, 1,020,000] | Yes |
| The interval can include zero | [0, 100] | Ambiguous |

These are synthetic counterexamples, not timing limits for a specific
instrument. A measurement protocol can qualify a declared trigger/aperture
mapping with a shared timebase and a separately timed reference fixture;
verify the actual routes, configuration, and program-to-cell event
identity. Where aperture or differential timing is unresolved, do not
promote the trigger timestamp to an acquisition timestamp.

The [Keysight 3458A guide, p. 133](https://www.keysight.com/us/en/assets/9018-01343/user-manuals/9018-01343.pdf)
describes configuration-dependent trigger timing for that instrument only.
[JCGM 100:2008, §5.2](https://doi.org/10.59161/JCGM100-2008E) treats
correlation between standard uncertainties; the deterministic intervals
above address a different question.
