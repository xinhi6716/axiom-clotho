# Whole-group forward-window planning

**Current source only, not the frozen b7 download.** This standalone Python API
plans memberships from declared group timing metadata. It does not fit a model,
read a feature table or target vector, produce predictions, or write results.
It is not connected to the CLI, `SelectionPlan`, `BenchmarkResult`, existing
schemas, or default split behavior.

## Small explicit example

```python
from groupbench.group_time_plan import GroupSpan, TimeWindow, plan_group_forward_windows

plan = plan_group_forward_windows(
    ["warmup", "early", "crossing", "later", "early"],
    group_spans=[
        GroupSpan("warmup", start=0, end=1, label_available_at=1),
        GroupSpan("early", start=10, end=12, label_available_at=12),
        GroupSpan("crossing", start=8, end=12, label_available_at=12),
        GroupSpan("later", start=20, end=22, label_available_at=22),
    ],
    windows=[TimeWindow(10, 15), TimeWindow(20, 25)],
    time_unit="day",
    clock_id="declared-study-clock",
    gap=2,
)
assert plan.folds[0].train.rows == (0,)
assert plan.folds[0].test.rows == (1, 4)
assert plan.folds[0].unused.rows == (2, 3)
assert plan.folds[1].train.rows == (0, 1, 2, 4)
assert plan.folds[1].test.rows == (3,)
assert plan.tested.groups == ("early", "later")
assert plan.untested.groups == ("warmup", "crossing")
```

This invented metadata example makes no performance claim. The crossing group
is never tested, yet becomes eligible for training in the second window. A
warmup group can train without ever being tested.

## Input contract

- `row_groups`, `group_spans` and `windows` must be nonempty built-in lists or
  tuples. There is no automatic DataFrame, CSV, dictionary, datetime, generator
  or array parsing. Explicitly construct `GroupSpan` and `TimeWindow` records.
- Row positions are zero-based positions in the supplied `row_groups`. Repeated
  group identities are allowed; all rows in a group stay together. Identifiers
  are nonblank built-in strings or built-in integers in `[-2**63, 2**63 - 1]`.
  Booleans, floats, missing values, NumPy scalars and subclasses are rejected.
  Strings retain their exact spelling, including leading zeros and surrounding
  spaces: `1`, `"1"`, `"01"`, `" 1"` and `"NA"` are distinct.
- Each distinct row group has exactly one `GroupSpan`; missing, extra and
  duplicate identities are errors. Span-table order does not affect output.
  Spans may overlap or tie across groups. Each span must satisfy
  `start <= end <= label_available_at`; zero-duration spans are permitted.
- Every timing value, including `gap`, must be a built-in signed-64-bit integer
  tick. Negative clock ticks are allowed, but negative gaps are not. The sum
  `label_available_at + gap` must also fit this range for every group. No
  rounding, clipping, unit conversion or integer wrapping is performed.
- `time_unit` and `clock_id` are required nonempty built-in string declarations
  without surrounding whitespace. Custom labels are accepted, not interpreted.
  All spans, window bounds and the gap must use that same declared clock and
  unit. These labels are not a clock registry or a synchronization check.
- `TimeWindow(lower, upper)` represents `[lower, upper)`, with `lower < upper`.
  Windows must arrive in increasing, nonoverlapping order. Adjacent windows and
  gaps between windows are allowed. The planner never sorts or invents windows.

## Membership and failure rules

For a window `[L, U)`, each entire group belongs to exactly one partition:

1. **Train:** `label_available_at + gap < L`, strictly. Equality is excluded.
2. **Test:** `start >= L and end < U`. The observation span is inclusive of
   `end`, so a group ending exactly at `U` is excluded from this test window.
   Test labels may become available after the test window.
3. **Unused:** every other group. Crossing groups are never truncated.

The span-order rule makes training and testing mutually exclusive. A delayed
label can prevent training even if all observations ended long ago. A group can
train in multiple later folds, but the nonoverlapping windows prevent it from
being tested twice.

Invalid inputs raise `ValueError` before returning a plan. Each fold must have
at least one training group and one test group. An empty-fold error identifies
its zero-based index, boundaries, and `empty_train`, `empty_test`, or both. The
API does not skip an invalid fold or return a partially successful plan.

## Immutable result and coverage

`GroupForwardWindowPlan` and its nested records are frozen dataclasses with
slots; collections are tuples. Caller lists are snapshotted, never mutated.
Group ordering is first occurrence in `row_groups`; row ordering is original
position, including for the combined coverage memberships.

- `protocol` is `group-forward-window-plan-v1`, `fits_performed` is `0`, and
  `coverage_mode` is `partial_predeclared`.
- `time_unit`, `clock_id`, `gap`, `row_groups` and `group_spans` retain the
  validated declaration. `group_spans` is normalized to first-occurrence order.
- Each `folds` entry has `index`, `window`, and `train`, `test`, `unused`
  memberships. Each membership contains `rows` and unique `groups`, with
  `n_rows` and `n_groups` properties. Every input row appears exactly once per
  fold across those three memberships.
- `total`, `tested` and `untested` are memberships for all input rows, the union
  of test rows, and its complement. Untested includes warmup, crossing, late,
  delayed-label and other never-tested groups, whether or not they train later.
- Each fold's `exclusions` has one `GroupExclusion` per group. `train` and `test`
  are `None` exactly when the group is included in that partition. Training
  reasons are `label_not_available_before_window` or
  `gap_not_elapsed_before_window`. Test reasons are `before_window`,
  `at_or_after_window`, `crosses_lower_boundary`, `crosses_upper_boundary` or
  `crosses_both_boundaries`. An unused group has both reasons.

Coverage is partial by design. Do not describe these memberships as full
out-of-fold coverage or silently calculate a full-dataset score from them.

## What declarations do not prove

Timing declarations do not prove causality, source truth, clock synchronization,
independence across groups, or that features and labels were available when
claimed. Group-span metadata does not reveal relationships among separate
identities or the row-level timing inside a declared span.

**This metadata-only API does not enforce downstream feature exclusion or
train-only preprocessing.** Any separately implemented modeling workflow must
use an explicit feature allowlist that excludes identifiers, timing metadata,
labels and post-outcome information. Fit every learned transformation and
estimator only on that fold's training membership, with any selection protocol
separately reviewed. This planner supplies no temporal nested-selection,
benchmark, scoring, or end-to-end temporal-ML guarantee.
