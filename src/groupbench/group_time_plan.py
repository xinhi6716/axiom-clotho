"""Deterministic metadata-only whole-group forward windows; no fitting.

Times are caller-declared signed-64-bit integer ticks on one named clock and
unit. Declarations do not establish causality, truth or clock synchronization.
This module does not inspect features or enforce downstream feature exclusion
or train-only preprocessing. See docs/GROUP_FORWARD_WINDOW_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal


MIN_TICK = -(2**63)
MAX_TICK = 2**63 - 1
GroupIdentity = str | int


@dataclass(frozen=True, slots=True)
class GroupSpan:
    """One group's inclusive observation span and label-availability tick."""

    group: GroupIdentity
    start: int
    end: int
    label_available_at: int


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A predeclared half-open test window [lower, upper)."""

    lower: int
    upper: int


@dataclass(frozen=True, slots=True)
class Membership:
    """Zero-based row positions and unique groups, both in input order."""

    rows: tuple[int, ...]
    groups: tuple[GroupIdentity, ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_groups(self) -> int:
        return len(self.groups)


@dataclass(frozen=True, slots=True)
class GroupExclusion:
    """Per-partition reasons for one group; None means included there."""

    group: GroupIdentity
    train: str | None
    test: str | None


@dataclass(frozen=True, slots=True)
class ForwardWindowFold:
    index: int
    window: TimeWindow
    train: Membership
    test: Membership
    unused: Membership
    exclusions: tuple[GroupExclusion, ...]


@dataclass(frozen=True, slots=True)
class GroupForwardWindowPlan:
    time_unit: str
    clock_id: str
    gap: int
    row_groups: tuple[GroupIdentity, ...]
    group_spans: tuple[GroupSpan, ...]
    folds: tuple[ForwardWindowFold, ...]
    total: Membership
    tested: Membership
    untested: Membership
    protocol: Literal["group-forward-window-plan-v1"] = field(
        default="group-forward-window-plan-v1", init=False)
    fits_performed: Literal[0] = field(default=0, init=False)
    coverage_mode: Literal["partial_predeclared"] = field(
        default="partial_predeclared", init=False)


def _ordered(value, name):
    # Deliberately no dataframe, iterator, set, mapping or scalar coercion.
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an explicit list or tuple.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return tuple(value)


def _tick(value, name):
    if type(value) is not int or not MIN_TICK <= value <= MAX_TICK:
        raise ValueError(f"{name} must be a built-in signed-64-bit integer tick.")


def _identity(value, name):
    if type(value) is str and value.strip():
        return
    if type(value) is int and MIN_TICK <= value <= MAX_TICK:
        return
    raise ValueError(f"{name} must be a nonblank built-in string or signed-64-bit integer identity.")


def _declaration(value, name):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be an explicit nonblank string without surrounding whitespace.")


def _test_exclusion(span, window):
    if span.end < window.lower:
        return "before_window"
    if span.start >= window.upper:
        return "at_or_after_window"
    if span.start < window.lower:
        if span.end >= window.upper:
            return "crosses_both_boundaries"
        return "crosses_lower_boundary"
    if span.end >= window.upper:
        return "crosses_upper_boundary"
    return None


def _membership(row_groups, group_order, included):
    return Membership(
        rows=tuple(i for i, group in enumerate(row_groups) if group in included),
        groups=tuple(group for group in group_order if group in included),
    )


def plan_group_forward_windows(
    row_groups: list[GroupIdentity] | tuple[GroupIdentity, ...],
    *,
    group_spans: list[GroupSpan] | tuple[GroupSpan, ...],
    windows: list[TimeWindow] | tuple[TimeWindow, ...],
    time_unit: str,
    clock_id: str,
    gap: int = 0,
) -> GroupForwardWindowPlan:
    """Plan whole-group membership without features, targets, RNG or fitting.

    All containers must be explicit lists or tuples. Group identities are exact
    built-in nonblank strings or signed-64-bit integers (1 and "1" differ).
    Spans cover distinct row groups exactly once. All ticks and gap are exact
    built-in signed-64-bit integers, with no bool, float or datetime conversion.
    Unit/clock labels are required, not inferred or independently verified.

    A group trains iff label_available_at + gap < lower. It tests iff
    start >= lower and end < upper; every other group is unused in that fold.
    Each fold must have nonempty train and test memberships. Invalid inputs
    raise ValueError; empty-fold errors identify the zero-based fold and cause.
    A successful plan and all nested records/collections are immutable.
    """
    _declaration(time_unit, "time_unit")
    _declaration(clock_id, "clock_id")
    _tick(gap, "gap")
    if gap < 0:
        raise ValueError("gap must be nonnegative.")
    rows = _ordered(row_groups, "row_groups")
    for i, group in enumerate(rows):
        _identity(group, f"row_groups[{i}]")
    group_order = tuple(dict.fromkeys(rows))
    row_identities = set(group_order)

    spans = {}
    for i, span in enumerate(_ordered(group_spans, "group_spans")):
        if type(span) is not GroupSpan:
            raise ValueError(f"group_spans[{i}] must be a GroupSpan.")
        _identity(span.group, f"group_spans[{i}].group")
        if span.group in spans:
            raise ValueError(f"group_spans[{i}] duplicates a group identity.")
        for name in ("start", "end", "label_available_at"):
            _tick(getattr(span, name), f"group_spans[{i}].{name}")
        if not span.start <= span.end <= span.label_available_at:
            raise ValueError(f"group_spans[{i}] requires start <= end <= label_available_at.")
        if span.label_available_at > MAX_TICK - gap:
            raise ValueError(f"group_spans[{i}].label_available_at + gap exceeds signed-64-bit ticks.")
        spans[span.group] = span
    if set(spans) != row_identities:
        missing = len(row_identities - set(spans))
        extra = len(set(spans) - row_identities)
        raise ValueError(f"group_spans must cover row identities exactly once: {missing} missing, {extra} extra.")

    bounds = _ordered(windows, "windows")
    for i, window in enumerate(bounds):
        if type(window) is not TimeWindow:
            raise ValueError(f"windows[{i}] must be a TimeWindow.")
        _tick(window.lower, f"windows[{i}].lower")
        _tick(window.upper, f"windows[{i}].upper")
        if window.lower >= window.upper:
            raise ValueError(f"windows[{i}] requires lower < upper.")
        if i and window.lower < bounds[i - 1].upper:
            raise ValueError(f"windows[{i}] must increase without overlapping the preceding window.")

    folds = []
    tested = set()
    for index, window in enumerate(bounds):
        partitions = {}
        partition_groups = ([], [], [])
        exclusions = []
        for group in group_order:
            span = spans[group]
            if span.label_available_at + gap < window.lower:
                train_reason = None
            elif span.label_available_at >= window.lower:
                train_reason = "label_not_available_before_window"
            else:
                train_reason = "gap_not_elapsed_before_window"
            test_reason = _test_exclusion(span, window)
            role = 0 if train_reason is None else (1 if test_reason is None else 2)
            partitions[group] = role
            partition_groups[role].append(group)
            exclusions.append(GroupExclusion(group, train_reason, test_reason))
        empty = tuple(name for name, members in (("empty_train", partition_groups[0]),
                                             ("empty_test", partition_groups[1]))
                      if not members)
        if empty:
            raise ValueError(f"fold {index} [{window.lower}, {window.upper}): {', '.join(empty)}.")
        # Assign each row once, preserving original order within each partition.
        partition_rows = ([], [], [])
        for position, group in enumerate(rows):
            partition_rows[partitions[group]].append(position)
        train, test, unused = (Membership(tuple(positions), tuple(groups))
                               for positions, groups in zip(partition_rows, partition_groups))
        folds.append(ForwardWindowFold(
            index=index, window=window,
            train=train, test=test, unused=unused,
            exclusions=tuple(exclusions),
        ))
        tested.update(test.groups)
    return GroupForwardWindowPlan(
        time_unit=time_unit, clock_id=clock_id, gap=gap, row_groups=rows,
        group_spans=tuple(spans[group] for group in group_order), folds=tuple(folds),
        total=_membership(rows, group_order, row_identities),
        tested=_membership(rows, group_order, tested),
        untested=_membership(rows, group_order, row_identities - tested),
    )


_LABEL_AUDIT_MAX_ROWS = 20_000
_LABEL_AUDIT_MAX_WINDOWS = 1_000
_LABEL_AUDIT_MAX_WORK = 250_000


@dataclass(frozen=True, slots=True)
class ForecastLabelTiming:
    """One plan row's declared origin, target measurement and label arrival."""

    row_position: int
    group: GroupIdentity
    forecast_origin: int
    measured_at: int
    available_at: int


@dataclass(frozen=True, slots=True)
class ForwardLabelTimingAudit:
    """Counts from a metadata-only fixed-horizon audit."""

    n_rows: int
    n_tested_labels: int
    fold_train_label_counts: tuple[int, ...]
    fold_test_label_counts: tuple[int, ...]


def _same_label_plan_fields(left, right):
    # Dataclass equality alone accepts bool for int and 1.0 for 1.
    if type(left) is not type(right):
        return False
    if is_dataclass(right):
        return all(_same_label_plan_fields(getattr(left, f.name), getattr(right, f.name)) for f in fields(right))
    if type(right) is tuple:
        return len(left) == len(right) and all(_same_label_plan_fields(a, b) for a, b in zip(left, right))
    return left == right


def _canonical_label_plan(plan):
    if type(plan) is not GroupForwardWindowPlan:
        raise ValueError("Expected a GroupForwardWindowPlan.")
    if any(type(items) is not tuple for items in (plan.row_groups, plan.group_spans, plan.folds)):
        raise ValueError("Expected canonical tuple containers in the plan.")
    rows, groups, windows = len(plan.row_groups), len(plan.group_spans), len(plan.folds)
    if (not 1 <= groups <= rows <= _LABEL_AUDIT_MAX_ROWS
            or not 1 <= windows <= _LABEL_AUDIT_MAX_WINDOWS
            or windows * (rows + groups) > _LABEL_AUDIT_MAX_WORK):
        raise ValueError("Label audit exceeds the supported plan work bound.")
    if any(type(fold) is not ForwardWindowFold for fold in plan.folds):
        raise ValueError("Expected ForwardWindowFold records.")
    if any(type(span) is not GroupSpan for span in plan.group_spans):
        raise ValueError("Expected GroupSpan records.")
    for value in (plan.time_unit, plan.clock_id, *plan.row_groups, *(span.group for span in plan.group_spans)):
        if type(value) is str and len(value) > 256:
            raise ValueError("Label audit plan strings must have at most 256 characters.")
    rebuilt = plan_group_forward_windows(
        plan.row_groups, group_spans=plan.group_spans,
        windows=tuple(fold.window for fold in plan.folds), clock_id=plan.clock_id,
        time_unit=plan.time_unit, gap=plan.gap,
    )
    if not _same_label_plan_fields(plan, rebuilt):
        raise ValueError("Plan does not match its canonical membership.")
    return rebuilt


def audit_group_forward_label_timing(
    plan: GroupForwardWindowPlan, *, clock_id: str, time_unit: str, horizon: int,
    labels: list[ForecastLabelTiming] | tuple[ForecastLabelTiming, ...],
) -> ForwardLabelTimingAudit:
    """Check declared label ticks against a fixed horizon and forward plan.

    Provide exactly one row record per plan position (including unused rows)
    in row order, with its matching typed group. Origin is inside the inclusive
    group span. Measured tick is origin plus the positive horizon; available
    tick is no earlier than measurement. Across every training fold, label
    availability plus the plan's gap must strictly precede the window start.
    The plan's group availability bound must also cover every declared row.
    Test labels may arrive after their forecast window; scoring is not checked.
    """
    plan = _canonical_label_plan(plan)
    if any(type(label) is not str or len(label) > 256 for label in (clock_id, time_unit)):
        raise ValueError("Label audit clock and time unit must be bounded strings.")
    _declaration(clock_id, "clock_id")
    _declaration(time_unit, "time_unit")
    if clock_id != plan.clock_id or time_unit != plan.time_unit:
        raise ValueError("Label audit clock and time unit must match the plan.")
    _tick(horizon, "horizon")
    if horizon <= 0:
        raise ValueError("Label audit horizon must be positive.")
    if type(labels) not in (list, tuple) or len(labels) != len(plan.row_groups):
        raise ValueError("Label timing must cover every plan row exactly in order.")
    labels = tuple(labels)
    spans = {span.group: span for span in plan.group_spans}
    for i, (item, group) in enumerate(zip(labels, plan.row_groups)):
        if type(item) is not ForecastLabelTiming:
            raise ValueError("Expected ForecastLabelTiming records.")
        if (type(item.row_position) is not int or item.row_position != i
                or type(item.group) is not type(group) or item.group != group):
            raise ValueError("Label timing row positions and typed groups must match the plan.")
        for name in ("forecast_origin", "measured_at", "available_at"):
            _tick(getattr(item, name), name)
        span = spans[group]
        if not span.start <= item.forecast_origin <= span.end:
            raise ValueError("Forecast origin must lie inside the declared group span.")
        if item.forecast_origin > MAX_TICK - horizon:
            raise ValueError("Forecast origin plus horizon exceeds signed-64-bit ticks.")
        if item.measured_at != item.forecast_origin + horizon:
            raise ValueError("Target measurement must match the fixed horizon.")
        if item.available_at < item.measured_at:
            raise ValueError("A label cannot become available before it was measured.")

    for fold in plan.folds:
        # Python arithmetic here is exact; equality after the embargo fails.
        for position in fold.train.rows:
            if labels[position].available_at + plan.gap >= fold.window.lower:
                raise ValueError(
                    f"Training label at row {position} in fold {fold.index} must be available "
                    "with the plan gap strictly before the window starts."
                )

    if any(item.available_at > spans[item.group].label_available_at for item in labels):
        raise ValueError("A row label exceeds its declared group availability bound.")
    return ForwardLabelTimingAudit(
        n_rows=len(labels), n_tested_labels=plan.tested.n_rows,
        fold_train_label_counts=tuple(fold.train.n_rows for fold in plan.folds),
        fold_test_label_counts=tuple(fold.test.n_rows for fold in plan.folds),
    )
