# Whole-group forward-window planning

**Current source only, not the frozen b7 download.** A standalone typed Python
API and a strict metadata JSON CLI plan memberships from declared group timing
metadata. They do not fit a model, read a feature table or target vector,
produce predictions, or score results. The CLI prints one plan to stdout and
does not write output files. `SelectionPlan`, `BenchmarkResult`, evaluation and
prediction schemas, and default split behavior are unchanged.

## Strict JSON command

Run from the source checkout with the current package available:

```bash
mlcalc plan-group-forward-windows examples/group_forward_window_metadata.json
```

The `groupbench` alias, `python -m groupbench plan-group-forward-windows`, and
`clotho tabular plan-group-forward-windows` expose the same command. This is
separate from regression `mlcalc plan run` / `plan predict` and from measurement
`clotho plan`; none of those routes is extended or replaced.

### Import boundary

The metadata command and its `--help`, including the aliases above and
`python -m groupbench.group_time_json`, use only the Python standard library.
They do not import NumPy, pandas, SciPy, scikit-learn, or model/evaluation code.
Plain `import groupbench` and importing the JSON adapter are also light.
The bounded single-open metadata reader and its file-safety checks are unchanged.

Package public APIs resolve on first access to their original function/class
objects and are cached. The ordered `__all__`, `dir`, explicit imports and
`from groupbench import *` remain supported. Star import intentionally loads
all exported APIs, including the model dependencies. Existing incidental package
submodule attributes remain available through a bounded compatibility map.

This does not make model dependencies optional installation extras. General
`groupbench --help` / `clotho tabular --help`, legacy CSV fit-budget planning,
scoring and model APIs can still require them. No estimator, default, dependency
version, saved-result format or package-source inventory is changed. A light
metadata import is not a claim that every zero-fit command has light imports.

The [tiny handwritten fixture](../examples/group_forward_window_metadata.json)
contains only invented metadata:

```json
{
  "schema_id": "mlcalc.group_forward_window_metadata",
  "schema_version": 1,
  "time_unit": "day",
  "clock_id": "invented-study-clock",
  "gap": 2,
  "row_groups": [1, "1", "01", 1],
  "group_spans": [
    {"group": 1, "start": 0, "end": 1, "label_available_at": 1},
    {"group": "1", "start": 10, "end": 12, "label_available_at": 12},
    {"group": "01", "start": 8, "end": 12, "label_available_at": 12}
  ],
  "windows": [{"lower": 10, "upper": 15}]
}
```

The single fold has train rows `[0, 3]`, test rows `[1]`, and unused rows `[2]`.
Tested groups are `["1"]`; untested groups are `[1, "01"]`. Integer `1`, string
`"1"`, and string `"01"` are different group identities. This is a membership
example, not measurements or performance evidence.

### JSON input contract v1

- The root is exactly one object with the eight keys shown above. Require
  `schema_id: "mlcalc.group_forward_window_metadata"` and integer
  `schema_version: 1`. All declarations, including `gap`, are required. Missing,
  unknown and duplicate keys at any object level are errors, including keys
  with different JSON escapes that decode to the same name.
- `row_groups`, `group_spans` and `windows` are arrays. Every span object has
  exactly `group`, `start`, `end`, `label_available_at`; every window object has
  exactly `lower`, `upper`. Group identities never become JSON object keys.
  A field cannot request reading another file. Features, targets, data paths,
  URLs and model settings are not supported fields.
- Group IDs are nonblank strings or signed-64 integers. Strings retain decoded
  spelling exactly: `1`, `"1"`, `"01"`, `" 1"`, `"1 "` and `"NA"` remain
  distinct. No trimming, Unicode normalization, missing-value inference,
  stringification or sorting occurs. `"a"` and `"\u0061"` decode to the same
  identity, so declaring separate spans for them is a duplicate.
- All numeric tokens must be integers in `[-2**63, 2**63 - 1]`, with at most
  19 decimal digits excluding a minus sign. Decimal and exponent syntax,
  including `1.0` and `1e0`, is rejected rather than converted. `NaN` and
  infinities are rejected. Boolean, null and container values are not IDs or
  ticks; `true` is not schema version 1. JSON `-0` means integer zero.
- The unchanged core applies the [typed input](#typed-input-contract) and
  [membership rules](#membership-and-failure-rules) below. It rejects invalid
  spans, duplicate/missing/extra groups, arithmetic overflow, unordered or
  overlapping windows, and any fold with empty training or testing membership.
  No fold is skipped and no crossing group is truncated.

### Bounded parsing and planning

These are JSON adapter resource limits, not scientific eligibility criteria or
limits added to the typed Python API:

- At most **4 MiB (4,194,304 bytes)** of UTF-8 input, with no BOM, trailing
  content or second JSON document.
- At most **16 nested containers**, checked before recursive JSON decoding.
  Brackets and escaped quotes inside strings do not count as containers.
- **1–20,000** row groups, **1–20,000** span records, and **1–1,000** windows.
- With `R` row groups, `G` span records and `W` windows, require
  **`W * (R + G) <= 250_000`** before calling the planner or constructing folds.
  Invalid groups are not deduplicated to make an input fit.
- An additional conservative estimate must fit **16 MiB (16,777,216 bytes)**
  before planning. Let `rbytes` be the sum of compact ASCII JSON lengths of all
  row group entries, `gbytes` the corresponding sum of span group IDs, and
  `dbytes` the compact ASCII JSON lengths of `time_unit` plus `clock_id`:

  ```text
  8192 + 32*R + 256*G + rbytes + 4*gbytes + dbytes
       + W*(512 + 16*R + 256*G + 2*gbytes)
  ```

  This bounds accepted work and repeated identifier bytes, not the exact
  eventual file size. A document whose real output would be smaller can still
  be rejected by this conservative bound. The complete serialized output,
  including its final newline, is checked against 16 MiB again before stdout.

Parsing, validation, preflight, core planning and serialization finish before
any output is emitted. Success returns exit 0 and exactly one JSON document;
validation or serialization failure returns exit 2 and empty stdout. CLI errors
use fixed categories instead of exposing source paths, group IDs or timing
values. A physical stdout write failure can still leave partial bytes in an
external destination; stdout delivery is not a filesystem transaction.

### JSON output contract v1

The output has exactly `schema_id`, `schema_version`, `protocol`,
`fits_performed`, `coverage_mode`, `time_unit`, `clock_id`, `gap`, `row_groups`,
`group_spans`, `folds`, `total`, `tested`, `untested` and `limitations`.

- `schema_id` is `"mlcalc.group_forward_window_plan"`; `schema_version` is
  integer `1`. The plan is not an evaluation, prediction or fit-budget schema.
- `protocol` is `"group-forward-window-plan-v1"`, `fits_performed` is `0`, and
  `coverage_mode` is always `"partial_predeclared"`, even if an input happens
  to test every supplied group.
- `row_groups` preserves repeated identities in original row order.
  `group_spans` contains the same four-field records as the input, normalized
  to first-occurrence group order.
- Each fold has exactly `index`, `window`, `train`, `test`, `unused` and
  `exclusions`. Each window has `lower` and `upper`. Every membership has
  `rows`, `groups`, `n_rows` and `n_groups`. Rows are zero-based positions in
  the caller's `row_groups`, not a source-table join or persistent row identity.
- Exclusions are ordered records with `group`, `train`, `test`, retaining the
  core reason strings below and JSON `null` for inclusion. Unused groups and
  their two exclusion reasons remain visible.
- `total`, `tested` and `untested` use the same membership representation.
  `tested` is the union of all test memberships; `untested` is its complement
  across all supplied rows and groups, not the last fold's unused membership.
- `limitations` always records that this is zero-fit metadata-only planning;
  timing/source/identity relationships are unverified declarations; test
  coverage is partial; downstream feature exclusion and train-only
  preprocessing need a separate workflow; and output retains identities and
  timing metadata whose row positions need the caller's retained mapping.

Serialization is deterministic compact JSON with sorted object keys, preserved
array order, ASCII string escaping, no nonfinite numbers and a final newline.
Counts, ticks and integer identities stay JSON integers. **Use an
integer-preserving JSON decoder:** signed-64 values can exceed JavaScript's
safe integer range, and ordinary `JSON.parse`/Number decoding may lose exact
identity or tick values. ASCII escaping changes spelling, not decoded identity.

There are no metrics, predictions or selected models. The serializer adds no
timestamps, source paths or execution attestations. Existing saved-result
validators do not accept this as an evaluation/prediction bundle; no saved-plan
evaluation or reload-and-trust route is added.

### Local reads, output safety and privacy

The CLI accepts one explicitly named regular local metadata file. Explicit
symlinks to regular files are supported. Missing files, directories, FIFOs,
devices, oversized content, invalid UTF-8 and detected replacements are rejected.
The bounded retained bytes are parsed without reopening the file for inference.
No CSV parser, features, targets, estimators, preprocessing, RNG, network,
datetime parsing, unit conversion or implicit path discovery is involved. A
URL-like path is not fetched. Existing package imports still require the
current ML dependencies; this is not a dependency-free distribution.

Use trusted, idle local files. Pre/post descriptor and path checks can detect
many concurrent changes; they do not authenticate the input, establish an
atomic snapshot or impose a timeout on network-mounted storage.

The only arguments are the metadata filename and help. `--out`, `--output`,
`--overwrite`, feature/model options and option abbreviations are rejected
before reading input. The command never creates directories, opens output
paths or calls result writers. **Shell redirection is outside this protection:**
`> metadata.json` can truncate the input before Python starts, and the shell
can overwrite other files. The basic example intentionally prints to stdout.
If you save output yourself, use a distinct fresh destination and manage its
permissions and overwrite safety yourself.

Output retains supplied group IDs, spans, labels and gap. Review it before
sharing; do not treat it as anonymous merely because it contains no targets.
Keep the row mapping separately if you intend to use the positions later.

### Pure text-to-JSON Python convenience function

`groupbench.group_time_json.plan_group_forward_windows_json(text: str) -> str`
applies the same strict parsing and adapter resource limits, calls the typed
planner, and returns the complete JSON document plus a final newline. It does
no file I/O. Callers choose how to supply text and whether to save the result.
Use the typed API below when immutable dataclass records are preferable.

<a id="small-explicit-example"></a>

## Typed Python example

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

<a id="input-contract"></a>

## Typed input contract

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

**This metadata-only planner does not enforce downstream feature exclusion or
train-only preprocessing.** Any separately implemented modeling workflow must
use an explicit feature allowlist that excludes identifiers, timing metadata,
labels and post-outcome information. Fit every learned transformation and
estimator only on that fold's training membership, with any selection protocol
separately reviewed. This planner supplies no temporal nested-selection,
benchmark, scoring, or end-to-end temporal-ML guarantee.
