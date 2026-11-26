"""Audit declared as-of feature lineage, without values, fitting or I/O."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

from .group_time_plan import (
    ForwardWindowFold, GroupForwardWindowPlan, GroupIdentity, GroupSpan,
    MAX_TICK, MIN_TICK, plan_group_forward_windows,
)

_MAX_ROWS = 20_000
_MAX_WINDOWS = 64
_MAX_FEATURES = 64
_MAX_REVISIONS = 100_000
_MAX_EDGES = 200_000
_MAX_CHECKS = 1_000_000


class GroupTimeLineageError(ValueError):
    """An incomplete, inconsistent or unsupported declaration."""


@dataclass(frozen=True, slots=True)
class LineageKey:
    """Caller-declared source, physical group, episode, and row/event ID."""

    source: str
    group: GroupIdentity
    episode: str
    id: str


@dataclass(frozen=True, slots=True)
class RevisionRef:
    event: LineageKey
    ordinal: int


@dataclass(frozen=True, slots=True)
class EventRevision:
    ref: RevisionRef
    measured_at: int
    available_at: int
    status: Literal["observed", "derived", "deleted"]
    parents: tuple[RevisionRef, ...] = ()


@dataclass(frozen=True, slots=True)
class ForecastRow:
    key: LineageKey
    origin: int


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    feature: str
    root: RevisionRef


@dataclass(frozen=True, slots=True)
class RowLineage:
    key: LineageKey
    roots: tuple[FeatureLineage, ...]


@dataclass(frozen=True, slots=True)
class GroupTimeLineageAudit:
    protocol: Literal["group-forward-as-of-lineage-v1"]
    n_rows: int
    n_features: int
    n_revisions_declared: int
    n_row_revisions_checked: int
    frozen_timing_roster_matched: bool = False


def _fail(message):
    raise GroupTimeLineageError(message)


def _items(value, *, name, maximum, minimum=1):
    if type(value) not in (tuple, list) or not minimum <= len(value) <= maximum:
        _fail(f"{name} requires an explicit bounded list or tuple.")
    return tuple(value)


def _name(value, name):
    if type(value) is not str or not 1 <= len(value) <= 256 or not value.strip() or value.strip() != value:
        _fail(f"{name} requires a nonblank unpadded string of at most 256 characters.")


def _tick(value):
    if type(value) is not int or not MIN_TICK <= value <= MAX_TICK:
        _fail("Lineage clock coordinates require built-in signed-64-bit integer ticks.")


def _key(value):
    if type(value) is not LineageKey:
        _fail("Lineage keys require a LineageKey.")
    for name in ("source", "episode", "id"):
        _name(getattr(value, name), f"Lineage key {name}")
    _group(value.group)


def _group(value):
    if type(value) is str:
        if not 1 <= len(value) <= 256 or not value.strip():
            _fail("String physical-group identities require nonblank strings of at most 256 characters.")
    elif type(value) is not int or not MIN_TICK <= value <= MAX_TICK:
        _fail("Physical-group identities require a built-in string or signed-64-bit integer.")


def _ref(value):
    if type(value) is not RevisionRef:
        _fail("Lineage links require a RevisionRef.")
    _key(value.event)
    if type(value.ordinal) is not int or not 0 <= value.ordinal <= MAX_TICK:
        _fail("Revision ordinals require nonnegative built-in signed-64-bit integers.")


def _scope(key):
    return key.source, key.group, key.episode


def _same_plan_fields(left, right):
    if type(left) is not type(right):
        return False
    if is_dataclass(right):
        return all(_same_plan_fields(getattr(left, field.name), getattr(right, field.name))
                   for field in fields(right))
    if type(right) is tuple:
        return len(left) == len(right) and all(_same_plan_fields(a, b) for a, b in zip(left, right))
    return left == right


def _checked_plan(plan):
    if type(plan) is not GroupForwardWindowPlan:
        _fail("Expected a whole-group forward window plan.")
    groups = _items(plan.row_groups, name="Plan row groups", maximum=_MAX_ROWS)
    spans = _items(plan.group_spans, name="Plan spans", maximum=len(groups))
    folds = _items(plan.folds, name="Plan folds", maximum=_MAX_WINDOWS)
    if (any(type(value) is not tuple for value in (plan.row_groups, plan.group_spans, plan.folds))
            or len(folds) * (len(groups) + len(spans)) > _MAX_CHECKS):
        _fail("Plan containers or window-by-row/group work are unsupported.")
    _name(plan.clock_id, "Plan clock ID")
    _name(plan.time_unit, "Plan time unit")
    for group in groups:
        _group(group)
    for span in spans:
        if type(span) is not GroupSpan:
            _fail("Expected a GroupSpan.")
        _group(span.group)
    if any(type(fold) is not ForwardWindowFold for fold in folds):
        _fail("Invalid forward-window fold.")
    try:
        # Memberships in a user-constructed dataclass are not trusted.
        rebuilt = plan_group_forward_windows(
            plan.row_groups, group_spans=plan.group_spans,
            windows=tuple(fold.window for fold in folds), gap=plan.gap,
            time_unit=plan.time_unit, clock_id=plan.clock_id,
        )
    except ValueError:
        _fail("The forward-window plan cannot be reconstructed.")
    if not _same_plan_fields(plan, rebuilt):
        _fail("The plan differs from canonical forward-window reconstruction.")
    return rebuilt


def _catalog(revisions, scopes):
    by_ref = {}
    by_event = {}
    edges = 0
    for item in revisions:
        if type(item) is not EventRevision:
            _fail("Evidence records require EventRevision.")
        _ref(item.ref)
        if _scope(item.ref.event) not in scopes:
            _fail("Evidence contains an undeclared source, physical group or episode.")
        _tick(item.measured_at)
        _tick(item.available_at)
        if item.measured_at > item.available_at:
            _fail("A revision cannot become available before it was measured.")
        if type(item.status) is not str or item.status not in ("observed", "derived", "deleted"):
            _fail("A revision has an unsupported status.")
        parents = _items(item.parents, name="Parent revisions", maximum=_MAX_EDGES - edges, minimum=0)
        edges += len(parents)
        if (item.status == "derived") != bool(parents):
            _fail("Derived revisions require parents; observed and deleted revisions cannot have parents.")
        for parent in parents:
            _ref(parent)
        if len(set(parents)) != len(parents):
            _fail("A derived revision repeats a parent.")
        if item.ref in by_ref:
            _fail("Duplicate event revision.")
        by_ref[item.ref] = item, parents
        by_event.setdefault(item.ref.event, []).append(item)

    timelines = {}
    for event, versions in by_event.items():
        versions.sort(key=lambda item: item.ref.ordinal)
        for prev, current in zip(versions, versions[1:]):
            if prev.measured_at != current.measured_at or prev.available_at >= current.available_at:
                _fail("An event requires one measured tick and strictly increasing revision availability.")
        timelines[event] = ([item.available_at for item in versions], versions)
    return by_ref, timelines


def _check_frozen_revision_timing(roster, by_ref):
    roster = _items(roster, name="Frozen revision timing roster", maximum=_MAX_REVISIONS)
    expected = {}
    for entry in roster:
        if type(entry) is not tuple or len(entry) != 3:
            _fail("Frozen timing entries require (RevisionRef, measured tick, available tick).")
        ref, measured, available = entry
        _ref(ref)
        _tick(measured)
        _tick(available)
        if measured > available:
            _fail("A frozen revision cannot become available before it was measured.")
        if ref in expected:
            _fail("Duplicate revision in the frozen timing roster.")
        expected[ref] = (measured, available)
    if expected.keys() != by_ref.keys():
        _fail("Declared revisions must exactly match the frozen timing roster.")
    for ref, ticks in expected.items():
        revision, _ = by_ref[ref]
        if ticks != (revision.measured_at, revision.available_at):
            _fail("Revision clocks differ from the frozen timing roster.")


def audit_group_time_lineage(
    plan: GroupForwardWindowPlan, *, clock_id: str, time_unit: str,
    features: tuple[str, ...], rows: tuple[ForecastRow, ...],
    lineage: tuple[RowLineage, ...], revisions: tuple[EventRevision, ...],
    expected_revision_timing: tuple[tuple[RevisionRef, int, int], ...] | None = None,
) -> GroupTimeLineageAudit:
    """Check declared selected revisions against their row's historical snapshot.

    Every plan row, ordered independently of the evidence, must have one root
    per declared feature, with cutoff inside its plan group span. A selected
    root or ancestor must be the last version strictly before that cutoff;
    a revision at the cutoff is unresolved. Each used parent must strictly
    precede its child's availability and share source, group, and episode.
    Optionally bind the entire catalog to an independently frozen sequence
    of (ref, measured_at, available_at) tuples. This detects omissions and
    clock changes against that declaration, not independent clock provenance,
    status or ancestry authenticity. Without this input, an omitted correction
    is not detectable. No sync between independent clocks, labels, values,
    fit or I/O.
    """
    plan = _checked_plan(plan)
    _name(clock_id, "Clock ID")
    _name(time_unit, "Time unit")
    if clock_id != plan.clock_id or time_unit != plan.time_unit:
        _fail("Lineage clock and unit must match the plan.")
    features = _items(features, name="Features", maximum=_MAX_FEATURES)
    for feature in features:
        _name(feature, "Feature names")
    if len(set(features)) != len(features):
        _fail("Feature names must be unique.")
    size = len(plan.row_groups)
    rows = _items(rows, name="Rows", minimum=size, maximum=size)
    lineage = _items(lineage, name="Row lineage", minimum=size, maximum=size)
    revisions = _items(revisions, name="Event revisions", maximum=_MAX_REVISIONS)
    if size * len(features) > _MAX_CHECKS:
        _fail("Declared feature cells exceed the audit work bound.")

    spans = {span.group: span for span in plan.group_spans}
    keys = set()
    roots_by_row = []
    for group, row, binding in zip(plan.row_groups, rows, lineage):
        if type(row) is not ForecastRow or type(binding) is not RowLineage:
            _fail("Each row requires ForecastRow and RowLineage records.")
        _key(row.key)
        _key(binding.key)
        if row.key != binding.key or row.key.group != group or row.key in keys:
            _fail("Duplicate, out-of-order or mismatched forecast row.")
        keys.add(row.key)
        _tick(row.origin)
        if not spans[group].start <= row.origin <= spans[group].end:
            _fail("Forecast origin must lie inside its declared group span.")
        roots = _items(binding.roots, name="Feature roots", minimum=len(features), maximum=len(features))
        for feature, root in zip(features, roots):
            if type(root) is not FeatureLineage:
                _fail("Feature roots require named FeatureLineage records.")
            _name(root.feature, "Feature lineage name")
            if root.feature != feature:
                _fail("Feature lineage names do not match the ordered feature list.")
            _ref(root.root)
        roots_by_row.append(tuple(root.root for root in roots))

    by_ref, timelines = _catalog(revisions, {_scope(key) for key in keys})
    if expected_revision_timing is not None:
        _check_frozen_revision_timing(expected_revision_timing, by_ref)
    checks = visited_count = 0
    for row, roots in zip(rows, roots_by_row):
        pending = list(roots)
        seen = set()
        while pending:
            ref = pending.pop()
            checks += 1
            if checks > _MAX_CHECKS:
                _fail("Lineage ancestry exceeds the audit work bound.")
            if ref in seen:
                continue
            if _scope(ref.event) != _scope(row.key):
                _fail("A feature refers outside its source, physical group or episode.")
            if ref not in by_ref:
                _fail("A feature or ancestor revision is missing from the declaration.")
            item, parents = by_ref[ref]
            ticks, versions = timelines[ref.event]
            i = bisect_left(ticks, row.origin)
            if i < len(ticks) and ticks[i] == row.origin:
                _fail("An event revision at the forecast cutoff has unresolved order.")
            if i == 0 or versions[i - 1].ref != ref:
                _fail("A feature or ancestor is not the latest revision visible before its cutoff.")
            if item.status == "deleted":
                _fail("A feature or ancestor uses a deleted event.")
            for parent in parents:
                if (parent not in by_ref
                        or by_ref[parent][0].available_at >= item.available_at):
                    _fail("A derived feature requires parents available strictly before its publication.")
            pending.extend(parents)
            seen.add(ref)
            visited_count += 1

    return GroupTimeLineageAudit(
        "group-forward-as-of-lineage-v1", size, len(features), len(revisions), visited_count,
        frozen_timing_roster_matched=expected_revision_timing is not None,
    )
