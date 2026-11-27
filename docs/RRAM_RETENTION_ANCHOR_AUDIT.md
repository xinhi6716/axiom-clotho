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

Use `--capture-sha256` with an independently retained SHA-256 receipt to
replay exactly the same metadata-only timestamp capture. The roster and
acquisition-block receipts do not pin these actual timestamps. The command
checks the precise bytes it parses; altered timestamps, row order, or
encoding fail even when they would describe a syntactically valid capture.
It prints a verification boolean, not the digest or any timestamps. This
verifies a supplied reference, not the instrument source, lineage, or when
the referenced capture was archived. Omit the option to audit a new export.

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

Use `--diagnose-terminal-alternatives` to see whether frozen episodes
without any nominal 100 ks row nevertheless contain off-slot data near
the target time. Only rows at other, non-anchor nominal slots on the same
episode are checked. The report distinguishes no off-slot rows, unusable
origin, at least one proven in-window row, boundary uncertainty, unresolved
clock or timestamp, and all rows proven outside. That order resolves mixed
cases. It never selects an off-slot read or adds it to the eligible
endpoint. A duplicated or unresolved nominal terminal row is not an absent
one; use the other diagnostics for those cases.

For observed readout-path provenance, add `--readout-paths paths.csv` with
exact columns `lot,wafer,die,cell,episode,nominal_us,readout_path`. Use
one opaque path label per uniquely captured frozen anchor, or leave the
path blank when unknown. The label can represent the instrument channel,
gain/range, and read configuration as one documented identity; do not
infer one from a shared clock or nominal setting. A row must refer to
one captured 0, 1 s, or 100 ks slot; duplicate map rows or ambiguous or
absent capture bindings are rejected. The report distinguishes observed
changes, all three paths declared identical, missing/unknown paths, and
absent/ambiguous capture, separately on the attempted and timing-eligible
denominators. Matching labels do not establish calibration or equivalent
electrical response, and no timing eligibility or exit rule changes.

Add `--read-bias bias.csv` for independently observed read bias, not a
commanded setpoint. Its exact columns are
`lot,wafer,die,cell,episode,nominal_us,observed_read_bias_uV,read_bias_error_uV`.
Each row refers to one unambiguous captured frozen anchor. Signed observed
bias and nonnegative absolute error bounds are integral microvolts, at most
18 digits each; a blank field leaves the interval unknown. Matching clock or
readout-path labels do not fill missing bias evidence. Disjoint intervals rule
out a common bias under the reported bounds, even if one interval is unknown.
A nonempty common intersection is not proof of equal bias, calibration or
acceptable mismatch. The diagnostic reports attempted and timing-eligible
counts without printing bias values or modifying eligibility.

With documented integration offsets, `--read-apertures apertures.csv` checks
whether the full early or terminal read aperture fits its elapsed-time window.
The exact columns are `lot,wafer,die,cell,episode,nominal_us,aperture_open_offset_us,aperture_close_offset_us`.
Offsets are signed integral microseconds relative to the captured anchor tick.
Supply one row per uniquely captured early/terminal anchor; leave unknown
offsets blank. Use this diagnostic only with known offsets on the capture clock,
not a programmed aperture setting standing in for the actual acquisition.
The two aperture endpoints shift together under the sum of the origin and read
timestamp error bounds. The report separates aperture fit for every admissible
shift, fit for some but not all shifts, and impossible full fit. Missing
metadata, unusable timestamps, and clock changes stay distinct. This diagnostic
does not alter the scheduled timestamp eligibility or certify ADC calibration.

Add `--diagnose-joint-aperture` with that table to check whether both full
apertures can fit with one shared origin correction. This is conditional
on an error box: one bounded origin correction and separate per-read
bounded corrections whose values may be chosen independently. It is not
an assumption of probabilistic independence. If read errors are coupled,
possible fit under the box is only a necessary condition and not proof
of feasibility under the actual constraints. Two possible marginal fits
can disagree on the shared origin. The diagnostic distinguishes no fit,
some fit, and all-error fit under the stated box on both denominators.
It still does not change eligibility or certify the stated bounds.

For a separate sensitivity check, `--diagnose-common-read-error-aperture`
assumes the two read corrections are exactly equal and satisfy both reported
bounds. The common correction is then bounded by the smaller read bound;
the same shared origin correction applies as before. This is an explicit
alternative to separate bounded read corrections, not a setting inferred
from a common instrument, path or clock. It can make some individually
possible corrections jointly impossible, or make a fit robust on its
narrower error set. Use alongside `--diagnose-joint-aperture` to see both
models; neither asserts the actual error coupling or alters the gate.

Add `--commanded-schedule schedule.csv` to compare capture slots against an
independently retained plan. The exact columns are
`lot,wafer,die,cell,episode,nominal_us`. Every frozen episode must have at
least one unique commanded slot; include planned dense or off-anchor slots
explicitly. Do not recreate the plan from the observed row count. Each
configured opportunity is counted as missing, observed once, or observed
more than once. Uncommanded slots and rows are reported separately. An
episode with an equal row total can still have wrong slot coverage, such as
two early reads and no terminal read. This reports nominal-slot structure
only: a captured row can have unknown timing or unusable measurements.
It does not establish that commands were issued, repair repeats, or modify
the primary timestamp gate.

For replay against a previously frozen plan, add
`--commanded-schedule-sha256 <trusted-lowercase-hex-digest>`. Supply the
receipt from the independently saved pre-capture plan, not a new digest of
the current file. Exact byte verification precedes capture parsing; it
rejects edited plans even when the same frozen episodes still join. For
example, removing the terminal slot after acquisition can turn a missing
command into apparent exact nominal coverage without changing the capture.
The digest flag cannot prove pre-capture custody or actual command issuance.

If a controller or independent run log records how attempts ended, add
`--termination-dispositions terminations.csv` with exact fields
`lot,wafer,die,cell,episode,termination_disposition`. Acceptable declared
dispositions are `completed`, `value_threshold_stop`, `instrument_abort`,
`operator_abort`, and `unknown`. Unreported episodes remain in their
own category. Do not infer a disposition from the final value or the
presence or absence of a CSV row. Per disposition, the audit counts frozen
attempts, any capture, nominal terminal rows, verified time-window metadata
for the terminal reading, and complete timing eligibility separately.
The controller saying completed does not establish endpoint observability,
and a stop value is not a substitute for a scheduled terminal observation.
No safety setting, timing gate, or missing-outcome rule changes.

When both optional tables are supplied, the audit also reports commanded
nominal-slot coverage by declared termination disposition. Each group
includes its frozen-attempt denominator and the same slot counters as the
unstratified report. Missing declarations and declared unknowns are
distinct. This distinguishes equal-count slot mismatches in purportedly
completed attempts from missing slots in declared threshold stops, without
assuming whether either explains the missing rows or how often commands
were actually issued.
