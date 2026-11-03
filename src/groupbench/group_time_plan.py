"""Deterministic metadata-only whole-group forward windows; no fitting.

Times are caller-declared signed-64-bit integer ticks on one named clock and
unit. Declarations do not establish causality, truth or clock synchronization.
This module does not inspect features or enforce downstream feature exclusion
or train-only preprocessing. See docs/GROUP_FORWARD_WINDOW_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        train, test = set(), set()
        exclusions = []
        for group in group_order:
            span = spans[group]
            if span.label_available_at + gap < window.lower:
                train.add(group)
                train_reason = None
            elif span.label_available_at >= window.lower:
                train_reason = "label_not_available_before_window"
            else:
                train_reason = "gap_not_elapsed_before_window"
            test_reason = _test_exclusion(span, window)
            if test_reason is None:
                test.add(group)
            exclusions.append(GroupExclusion(group, train_reason, test_reason))
        empty = tuple(name for name, members in (("empty_train", train), ("empty_test", test))
                      if not members)
        if empty:
            raise ValueError(f"fold {index} [{window.lower}, {window.upper}): {', '.join(empty)}.")
        folds.append(ForwardWindowFold(
            index=index, window=window,
            train=_membership(rows, group_order, train),
            test=_membership(rows, group_order, test),
            unused=_membership(rows, group_order, row_identities - train - test),
            exclusions=tuple(exclusions),
        ))
        tested.update(test)
    return GroupForwardWindowPlan(
        time_unit=time_unit, clock_id=clock_id, gap=gap, row_groups=rows,
        group_spans=tuple(spans[group] for group in group_order), folds=tuple(folds),
        total=_membership(rows, group_order, row_identities),
        tested=_membership(rows, group_order, tested),
        untested=_membership(rows, group_order, row_identities - tested),
    )
