# Held devices are not held protocols

Five distinct batteries were cycled to failure per selected validation protocol
in [Attia et al., Nature 2020, Fig. 4](https://doi.org/10.1038/s41586-020-1994-5).
That distinction motivates this example; the script contains no measurements
and does not replay the paper.

```sh
python examples/device-versus-protocol/run.py
```

The fixture has 20 invented devices and four protocols. Each physical device
appears in exactly one held fold per regime. Both rules see the same training
labels and protocol keys: one predicts the training mean, the other uses the
mean of training siblings under that exact protocol, falling back to the same
training mean when no sibling is available.

| Held unit | Held devices with protocol seen in train | Lookup MAE | Training mean MAE |
| --- | ---: | ---: | ---: |
| Device, within known protocols | 20/20 | 0 | 1 |
| Complete protocol | 0/20 | 4/3 | 4/3 |

The comparison is paired within each regime. The two regimes target different
questions, have different training rosters (16 vs. 15 devices per fold), and
the fixture provides no basis for inferring which rule transfers to a novel
real protocol. Group episodes from one cell by physical identity first; a
distinct protocol key should represent the full, ordered experimental recipe,
not a file, device name, or outcome. This example uses only the standard library.
