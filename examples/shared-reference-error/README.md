# Repeated agreement does not rule out shared reference error

```sh
python examples/shared-reference-error/run.py
```

Enumerate eight equally likely invented cases. A signal and two
acquisition offsets independently take values −1 and +1. The physical
target is 4 plus the signal. Two frozen rules have the same origin
information and either keep or ignore the origin-path offset. Neither
reads a later label. One later reference path shares the origin offset;
a separate reference path uses the other offset.

| Scoring target | Keep-offset MSE | Ignore-offset MSE |
| --- | ---: | ---: |
| Shared reference path | 0 | 1 |
| Separate reference path | 2 | 1 |
| Physical target (known only in the toy) | 1 | 0 |

The shared path's two duplicate readings agree exactly in all eight
cases even though both carry the offset. Zero disagreement cannot
establish validity against the physical target. The separate reference
reverses the ranking in this toy but still adds noise to both errors.

For a received label R = Y + e, the paired squared-error contrast of
predictions p and q differs from its physical-target contrast by
−2 E[e(p−q)]. Mean-zero error alone does not make that term vanish.
Decide whether the intended target is a reading on a declared chain
or a physical quantity before scoring. Different files or channel IDs
do not establish truly separate errors.

See [Bland and Altman on measurement agreement](https://pubmed.ncbi.nlm.nih.gov/2868172/)
and [JCGM 100:2008, §5.2 on correlated inputs](https://www.bipm.org/en/committees/jc/jcgm/publications).
This example fits nothing, does not model real instruments, and neither
estimates physical error from repeats nor corrects a measured score.
