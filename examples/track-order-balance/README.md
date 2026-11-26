# Balance run order and check for carryover

```sh
python examples/track-order-balance/run.py
```

Two invented tracks A and B are measured within each block in the shown
order. In arbitrary units B's direct effect is +2, and position two adds
+4 regardless of the recipe. Without carryover, always putting B second
returns +6. Balance AB and BA across blocks, and the average paired B−A
contrast returns +2 because the common position shift cancels.

| Assumption | Always AB | Always BA | Equal mix |
| --- | ---: | ---: | ---: |
| No carryover | +6 | −2 | +2 |
| B changes the next track by +6 | +6 | −8 | −1 |

Recipe-dependent carryover remains in the balanced average and can flip
its sign. For real deposition, retain block/plate ID, planned and actual
order, prior attempts, cooldown and measured pre-track plate temperature.
A pre-track reading after an earlier randomized recipe may be affected by
that prior assignment; simply conditioning on it is not a design fix.
Use fresh, appropriately comparable sites or prespecified recovery checks
as justified by the measurement protocol, and count all attempted tracks.
The example tests no device and establishes neither a physical effect nor
an approved process. See the distinct [bonded-track summary](../../docs/BONDED_TRACK_ATTEMPTS.md)
for accounting for failed or unmeasured tracks.
