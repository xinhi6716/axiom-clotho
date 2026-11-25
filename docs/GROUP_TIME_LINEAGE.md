# Audit declared as-of feature ancestry (Python)

A group-forward split can be chronologically correct and still use a
measurement corrected after the forecast. The metadata-only Python
`groupbench.group_time_lineage.audit_group_time_lineage` audit checks the
specific revisions declared for each feature. It has no CLI, file formats or
I/O. It takes an existing `GroupForwardWindowPlan` and does not alter any
planner, evaluator, prediction or saved-result schema or default.

## Declarations

- Supply an ordered independent `ForecastRow` roster covering every plan row,
  including unused rows. A `LineageKey` includes source namespace, declared
  physical group, episode and row/event ID. String and integer group IDs are
  distinct. The plan supplies ordered **group IDs**, not physical row IDs:
  the audit detects a mismatch between roster and lineage, but cannot detect
  the same substitution applied to both.
- Supply `RowLineage` in that order, with one named `FeatureLineage` root per
  declared feature in the declared order. Names and version pointers describe
  consumption; they are not proof of what an estimator actually consumed.
- Supply observed, derived and deleted `EventRevision` records with explicit
  logical event keys and revision ordinals. `measured_at` describes an event's
  effective time and stays fixed across revisions. `available_at` is the first
  tick the forecasting system could actually read that revision (for derived
  features, publication time, not the timestamp assigned to a backfill).
  This time must not precede the measured time and must increase with revision
  ordinal for that event. Gaps in ordinal numbering are permitted.
- All ticks are exact built-in signed 64-bit integers on one caller-declared
  clock and unit that must match the plan. Every forecast origin must be
  within its group's inclusive observation span. Only the most recent
  revision **strictly before** origin is eligible; an event needed by the
  audit with a version at the origin is ambiguous. Equal availability for
  different revisions of a single event is always rejected; this mode has
  no independent clock synchronization or tie sequencer. A latest deleted
  version cannot be used or replaced by an earlier live one.
- A derived revision requires explicit parents. All used parents must be
  within the same source, physical group and episode and strictly precede
  the child's publication. Every used ancestor must also be the latest
  version visible at the row's cutoff. Consequently, a cached derivative
  built from a superseded parent is rejected even if the old calculation
  was valid when it ran. Cross-group aggregates, explicit staleness policies,
  multiple clocks and post-span origins are not supported.

There are bounds of 20,000 rows, 64 windows, 64 features, 100,000 revisions
and 200,000 supplied parent links. Planner reconstruction, feature cells
and the ancestry walk also have work limits. Failure raises a
`GroupTimeLineageError` (a `ValueError`), not an audit result.

## Tiny example

```python
from groupbench.group_time_lineage import (
    EventRevision, FeatureLineage, ForecastRow, LineageKey, RevisionRef, RowLineage,
    audit_group_time_lineage,
)
from groupbench.group_time_plan import GroupSpan, TimeWindow, plan_group_forward_windows

plan = plan_group_forward_windows(
    ["previous", "held"],
    group_spans=[GroupSpan("previous", 0, 2, 3), GroupSpan("held", 8, 10, 14)],
    windows=[TimeWindow(8, 12)], clock_id="capture", time_unit="step",
)
previous, held = [
    LineageKey("experiment", group, "episode-1", "row-1") for group in plan.row_groups
]
first = RevisionRef(LineageKey("experiment", "previous", "episode-1", "read"), 0)
original = RevisionRef(LineageKey("experiment", "held", "episode-1", "read"), 0)
corrected_later = RevisionRef(original.event, 1)
result = audit_group_time_lineage(
    plan, clock_id="capture", time_unit="step", features=("x",),
    rows=(ForecastRow(previous, 2), ForecastRow(held, 10)),
    revisions=(EventRevision(first, 0, 1, "observed"),
               EventRevision(original, 7, 9, "observed"),
               EventRevision(corrected_later, 7, 12, "observed")),
    lineage=(RowLineage(previous, (FeatureLineage("x", first),)),
             RowLineage(held, (FeatureLineage("x", original),))),
)
assert result.n_row_revisions_checked == 2
```

The same declaration with `corrected_later` as the held feature would fail,
though both versions have measured tick 7. The result records counts of
declared rows, feature names, revisions, and used distinct revisions per row
(including ancestors). It does not replay features or authenticate source
completeness, values, independently observed clocks, physical identity, or
lineage truth. The availability check cannot detect a missing revision absent
from the declaration or omitted ancestry that was never declared. Labels,
training behavior, and broader leakage are not audited.

## Separate pre-run and post-process inputs

A planned setting and a same-track measurement need different provenance even
when both are in millimetres. For example, Ancalmo and Narra's
[data article, Table 1 and §4.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12351335/)
lists laser spot diameter as a process parameter but extracts track width
from offline optical profilometry. For a pre-run predictor, audit the nominal
setting at the original decision origin. Audit any separate post-process
estimator at its own declared origin and give its baseline the same
information budget. A later origin can make a feature temporally eligible;
it does not establish that the feature and target are scientifically
independent (e.g. both may come from the same measurement).
