# Audit scheduled retention anchors before exposing outcomes

From a source checkout:

```sh
python tools/audit_rram_retention_anchors.py \
  --frozen-roster attempted.csv --capture timestamp_metadata.csv
```

The roster contains exactly `lot,wafer,die,cell,episode`, one row per
intended attempt. Freeze it independently of outcomes. The capture contains
those same identity columns and `nominal_us,clock,epoch,tick_us,error_us`.
Column order may differ. Conductance, resistance, and prediction columns
are rejected. Clock and epoch must be stable identifiers, not display names
reused across resets. Ticks and absolute error bounds are unsigned integral
microseconds on the declared clock; a blank clock, epoch, tick, or bound
means unknown. Supply `0` only for a known zero tick or justified zero bound.

| Nominal anchor | Admission |
| --- | --- |
| 0 µs | Exactly one same-episode origin |
| 1,000,000 µs | Exactly one read; elapsed interval within (0, 1,020,000] µs |
| 100,000,000,000 µs | Exactly one read; elapsed interval within [80,000,000,000, 120,000,000,000] µs |

For either elapsed interval, use the displayed read minus origin tick and
expand by the sum of the two absolute timestamp error bounds. Exactly named
slots only: a delayed 95 ks read at 100 ks does not supply a missing terminal
row. Do not match across episodes, combine different epochs, or select a
preferred duplicate. A proven outside interval is distinct from an interval
that straddles a boundary.

The output is aggregate timing eligibility, not a measurement or model
result. Exit 0 requires all frozen episodes eligible and no unexpected
capture episode; exit 1 emits counts for missing, ineligible or unexpected
data; exit 2 denotes malformed input and emits no JSON. Off-anchor rows in
frozen episodes are counted but do not determine eligibility. The tool
does not verify that the roster was frozen, a pulse was delivered, clocks
are calibrated, or a read is physically comparable. A confirmed origin and
calibrated acquisition metadata must come from the instrument and protocol.


To check the roster against a receipt archived independently before outcome
access, supply `--frozen-roster-sha256` with its 64-character lowercase SHA-256.
The check binds the exact bytes subsequently parsed, including an optional BOM,
line endings, and row order. A mismatch stops before opening the capture. The
aggregate report records that verification passed, without printing the digest
or roster. A hash calculated from the current roster is not evidence of when
that roster was created or whether it was changed before archiving.

For troubleshooting, add `--diagnose-anchors`. It reports a disjoint set
of status counts for each nominal slot on the same frozen-episode denominator,
including absent captures. A terminal clock or boundary problem remains visible
even if an earlier issue already explains the episode's failure. A read with
reported timing but an absent, duplicate, or unknown origin is marked
`unusable_origin`, not `different_clock_or_epoch` or `within_window`. The
diagnostic never makes an individual read scorable and does not change
whole-episode eligibility, exit status, or the baseline report fields.

For a frozen group denominator, add `--diagnose-dies`. On the declared
`(lot,wafer,die)` keys, the report counts dies with all, some but not all, or
none of their frozen episodes timing-eligible. An entirely missing die stays
in the denominator and an unexpected die does not enter it. The maximum
eligible-episode count on any one die makes concentration visible without
emitting identities. These counts describe timing availability, not independent
samples, valid electrical readout, or cross-chip model performance. They do
not alter the exit status.

If the protocol declares acquisition blocks independently of the capture,
pass `--acquisition-block-map blocks.csv`. The map has exactly
`lot,wafer,die,cell,episode,acquisition_block` and assigns one opaque block
label to each frozen episode. Missing, duplicate, or out-of-roster entries
fail before loading capture metadata. The aggregate distinguishes blocks in
the plan from those with eligible episodes, and counts blocks spanning
multiple declared dies versus dies repeated across blocks. Neither clocks nor
die labels substitute for independently documented block assignments. The
report is descriptive support, not a claim of independent blocks or permission
to count repeated dies as new physical units. Timing eligibility
and the exit status for a valid mapping are unchanged.

To replay an independently archived block assignment, also supply
`--acquisition-block-map-sha256` with its 64-character lowercase SHA-256.
This checks the exact map bytes before capture access; changed assignments or
row order fail without a report even if the map still covers the roster. A
successful check adds a boolean only, not the digest or any identities.
This receipt is separate from the frozen-roster receipt and does not prove
when the block assignments were made or establish independence.

To audit whether timing-eligible episodes also have electrical stress history,
supply `--stress-history history.csv`. Its exact columns are
`lot,wafer,die,cell,episode,bias_0_to_1s,bias_1s_to_100ks`. For each interval
between nominal anchors, use `off_schedule_bias_observed` for recorded extra
electrical bias, `no_off_schedule_bias_documented` only when a complete,
linked record supports its absence, or `unknown` otherwise. The scheduled
anchor reads themselves are not extra bias. Do not infer the category from
the schedule or a missing instrument log. A missing episode declaration is
counted separately from an explicit unknown, and unrecognized categories,
duplicates, or episodes outside the roster are rejected.

The report separates the full frozen and timing-eligible episode counts for
each interval status. It changes no timing eligibility or success criterion,
and does not certify physical exposure, equal read bias or stress dose,
or mechanism. Those need separately validated measurement evidence.

The whole-episode summary retains the link between the two interval
declarations. A documented absence of extra bias across the episode requires
that status in both intervals of the same episode. Extra bias observed in
either interval takes precedence over an unknown in the other; unknown without
an observed extra bias and absent declarations remain separate. Equal
per-interval counts can conceal different whole-episode support: two episodes
with opposite extra-bias intervals have no fully documented unbiased episode,
even though the marginals match a plan with one clean and one biased episode.
These are declared stress histories, not instrument validation.

With a stress history table, `--diagnose-die-stress` also reports how many
distinct declared `(lot,wafer,die)` keys have at least one episode in each
whole-episode stress category. It reports the frozen die denominator and
the die denominator with at least one timing-eligible episode, separately.
Categories can overlap: one die can have both a documented extra-bias
episode and a documented no-extra-bias episode, so do not sum the category
counts. Matching episode marginals can hide concentration of the clean
episodes on one die. Reported counts do not establish independence or
comparability between dies. This option changes no eligibility or exit rule.

To investigate dropped or repeated exports, add `--diagnose-duplicate-anchors`.
For each nominal slot, this counts frozen episodes having multiple rows and
distinguishes identical usable timestamps, conflicting usable timestamps, a
mixture of usable and unresolved metadata, and entirely unresolved metadata.
Conflicts include differences in clock, epoch, tick, or error bound; examples
with both a conflict and an unresolved row are counted as conflicts. Equivalent
integer encodings are identical timestamp metadata, not necessarily identical
raw rows or electrical measurements. The tool never selects a preferred row
or repairs a duplicate. Every such episode remains timing-ineligible.
