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
