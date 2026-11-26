"""Deterministic group-disjoint, approximately row-balanced folds."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import unicodedata

import numpy as np
import pandas as pd

from .data import MAX_ROWS, validate_groups, validate_seed


class GroupKFold:
    """Assign whole groups to folds, balancing row counts without seeing targets.

    Largest groups are assigned first; ties are shuffled using random_state.
    The same row order, group labels, and seed produce the same assignment.
    Supplied X and y must match the number of group rows. Only lengths are
    checked: target values and pandas index labels never determine the folds.
    This is not a temporal, stratified, or multi-axis holdout splitter.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required; row-random fallback is never used.")
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, Integral) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two.")
        g = validate_groups(groups, len(groups))
        if X is not None and len(X) != len(g):
            raise ValueError("X and groups must contain the same number of rows.")
        if y is not None and len(y) != len(g):
            raise ValueError("y and groups must contain the same number of rows.")
        ids, unique = pd.factorize(g, sort=False)
        if len(unique) < self.n_splits:
            raise ValueError(f"Need at least {self.n_splits} distinct groups; found {len(unique)}.")
        rng = np.random.default_rng(validate_seed(self.random_state))
        counts = np.bincount(ids)
        shuffled = rng.permutation(len(unique))
        ordered = shuffled[np.argsort(-counts[shuffled], kind="stable")]
        loads = np.zeros(self.n_splits, dtype=int)
        assignments = np.empty(len(unique), dtype=int)
        for group_id in ordered:
            candidates = np.flatnonzero(loads == loads.min())
            # Uniform choice over an array draws an integer offset. Skip
            # the singleton (which consumes no RNG state), and retain the
            # exact candidate order and random draws on ties.
            fold = int(candidates[0] if candidates.size == 1 else
                       candidates[rng.integers(candidates.size)])
            assignments[group_id] = fold
            loads[fold] += counts[group_id]
        fold_ids = assignments[ids]
        for fold in range(self.n_splits):
            test = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _audit_positions(indices, n_rows: int, name: str):
    # NumPy supports boolean masks and negative indexing; neither describes the
    # literal row positions declared by a split. Reject them before indexing.
    if np.ma.isMaskedArray(indices) or (
        isinstance(indices, (list, tuple))
        and any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in indices)
    ):
        raise ValueError(f"Invalid split: {name} requires one-dimensional integer row positions.")
    positions = np.asarray(indices)
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError(f"Invalid split: {name} requires one-dimensional integer row positions.")
    if not positions.size:
        raise ValueError(f"Invalid split: {name} requires a nonempty partition.")
    # Check the original unsigned/signed values before converting to np.intp;
    # a too-large uint64 otherwise wraps to a negative NumPy index.
    if np.any(positions < 0) or np.any(positions >= n_rows):
        raise ValueError(f"Invalid split: {name} positions outside group row bounds.")
    positions = positions.astype(np.intp, copy=False)
    roster = set(positions)
    if len(roster) != positions.size:
        raise ValueError(f"Invalid split: {name} contains duplicate row positions.")
    return positions, roster


def audit_split(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> dict:
    """Report disjoint positional subsets of a non-missing group roster; coverage may be partial."""
    if np.ma.isMaskedArray(groups) and np.ma.getmaskarray(groups).any():
        raise ValueError("Invalid split: provide one non-missing group identifier per row.")
    # Keep the declared NumPy labels; coercing mixed identities can merge them.
    if not isinstance(groups, np.ndarray) or groups.ndim != 1 or pd.isna(groups).any():
        raise ValueError("Invalid split: provide one non-missing group identifier per row.")
    train, train_roster = _audit_positions(train, len(groups), "training")
    test, test_roster = _audit_positions(test, len(groups), "validation")
    train_groups = set(groups[train])
    test_groups = set(groups[test])
    overlap = train_groups & test_groups
    if overlap or train_roster & test_roster:
        raise ValueError("Invalid split: training and validation observations overlap.")
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_group_count": len(train_groups), "test_group_count": len(test_groups),
        "overlap_count": len(overlap),
        "train_groups": sorted(train_groups), "test_groups": sorted(test_groups),
    }


def audit_split_row_eligibility(train, test, groups, *, eligibility) -> dict:
    """Describe declared eligibility against an unchanged grouped split.

    Give one literal status per original row, including unused rows:
    "eligible", "ineligible", or "unknown". Unknown is kept apart from a
    declared failure. The caller freezes the roster and eligibility criterion
    independently of outcomes and verifies provenance. This audit does not
    choose rows, repair labels, resplit, fit, or access any outcomes.
    """
    statuses = ("eligible", "ineligible", "unknown")
    if (type(eligibility) not in (list, tuple) or not 1 <= len(eligibility) <= MAX_ROWS
            or any(type(s) is not str or s not in statuses for s in eligibility)):
        raise ValueError("Row eligibility requires an ordered roster of explicit eligibility statuses.")
    groups = validate_groups(groups, len(eligibility))
    train, train_rows = _device_partition(train, len(groups), "training")
    test, test_rows = _device_partition(test, len(groups), "validation")
    if train_rows & test_rows or set(groups[train]) & set(groups[test]):
        raise ValueError("Invalid split: training and validation observations overlap.")

    def summarize(positions):
        row_counts = dict.fromkeys(statuses, 0)
        group_statuses = {}
        for pos in positions:
            status = eligibility[pos]
            row_counts[status] += 1
            group_statuses.setdefault(groups[pos], set()).add(status)
        state_sets = group_statuses.values()
        return {
            "n_declared_rows": sum(row_counts.values()), "n_declared_groups": len(group_statuses),
            "eligible_rows": row_counts["eligible"], "ineligible_rows": row_counts["ineligible"],
            "unknown_rows": row_counts["unknown"],
            "groups_with_eligible_rows": sum("eligible" in state for state in state_sets),
            "groups_with_all_rows_eligible": sum(state == {"eligible"} for state in state_sets),
            "groups_with_no_eligible_rows_but_unknown": sum(
                "eligible" not in state and "unknown" in state for state in state_sets),
            "groups_with_only_ineligible_rows": sum(state == {"ineligible"} for state in state_sets),
        }

    used = train_rows | test_rows
    return {"train": summarize(train), "test": summarize(test),
            "unused": summarize(i for i in range(len(groups)) if i not in used)}


@dataclass(frozen=True, slots=True)
class DeviceAlias:
    """A row's declared device label in its original source namespace."""

    source: str
    alias: str | int


@dataclass(frozen=True, slots=True)
class PhysicalDevice:
    """A device identity within one declared canonical identity namespace."""

    namespace: str
    device: str | int


@dataclass(frozen=True, slots=True)
class DeviceAliasBinding:
    """Resolve one source alias to its declared physical device."""

    alias: DeviceAlias
    physical: PhysicalDevice


def _device_text(value):
    return type(value) is str and 0 < len(value) <= 256 and value == value.strip()


def _device_identifier(value):
    return _device_text(value) or (type(value) is int and -(2**63) <= value < 2**63)


def _check_device_alias(value):
    if (type(value) is not DeviceAlias or not _device_text(value.source)
            or not _device_identifier(value.alias)):
        raise ValueError("Device aliases require an exact scoped label.")


def _check_physical_device(value):
    if (type(value) is not PhysicalDevice or not _device_text(value.namespace)
            or not _device_identifier(value.device)):
        raise ValueError("Physical devices require an exact scoped label.")


def _device_partition(indices, size, name):
    # Bound work before NumPy creates a positional array. Preserve the existing
    # splitter's output arrays and forward-plan membership tuples.
    if (type(indices) not in (list, tuple, np.ndarray)
            or (type(indices) is np.ndarray and indices.ndim != 1)):
        raise ValueError(f"Invalid split: {name} requires one-dimensional integer row positions.")
    if len(indices) > size:
        raise ValueError(f"Invalid split: {name} exceeds the row roster size.")
    return _audit_positions(indices, size, name)


def _resolve_device_split(train, test, row_aliases, bindings):
    if type(row_aliases) not in (list, tuple) or not 1 <= len(row_aliases) <= MAX_ROWS:
        raise ValueError(f"Device roster requires an explicit list or tuple of 1 to {MAX_ROWS} rows.")
    row_aliases = tuple(row_aliases)
    train, train_roster = _device_partition(train, len(row_aliases), "training")
    test, test_roster = _device_partition(test, len(row_aliases), "validation")
    if train_roster & test_roster:
        raise ValueError("Invalid split: training and validation row positions overlap.")

    for alias in row_aliases:
        _check_device_alias(alias)
    aliases = set(row_aliases)
    if type(bindings) not in (list, tuple) or not 1 <= len(bindings) <= MAX_ROWS:
        raise ValueError(f"Device bindings require an explicit list or tuple of 1 to {MAX_ROWS} records.")
    by_alias = {}
    for binding in tuple(bindings):
        if type(binding) is not DeviceAliasBinding:
            raise ValueError("Device bindings require DeviceAliasBinding records.")
        _check_device_alias(binding.alias)
        _check_physical_device(binding.physical)
        if binding.alias in by_alias:
            raise ValueError("Device bindings repeat a source alias.")
        by_alias[binding.alias] = binding.physical
    if by_alias.keys() != aliases:
        raise ValueError("Device bindings must cover every roster alias exactly, with no extras.")

    train_aliases, test_aliases = ({row_aliases[i] for i in positions} for positions in (train, test))
    train_devices = {by_alias[alias] for alias in train_aliases}
    test_devices = {by_alias[alias] for alias in test_aliases}
    return (train, test, train_aliases, test_aliases, train_devices, test_devices,
            by_alias.values(), len(row_aliases))


def audit_physical_device_split(train, test, row_aliases, *, bindings) -> dict[str, int]:
    """Reject alias-mediated device overlap in one declared positional split.

    Supply a metadata-only roster in table order and one binding per distinct
    alias across the entire roster. Many source aliases can map to one physical
    device. Only train vs test are compared; unused positions do not count.
    Namespaces are nonblank unpadded built-in strings, and IDs are such strings
    or signed-64-bit built-in integers (1 and "1" differ). No data access, fit,
    device identity verification or wafer-level inference is performed.
    """
    (train, test, train_aliases, test_aliases, train_devices, test_devices,
     _, n_rows) = _resolve_device_split(train, test, row_aliases, bindings)
    overlap = len(train_devices & test_devices)
    if overlap:
        raise ValueError(f"Invalid split: {overlap} declared physical device(s) overlap training and validation.")
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_alias_count": len(train_aliases), "test_alias_count": len(test_aliases),
        "train_physical_device_count": len(train_devices), "test_physical_device_count": len(test_devices),
        "unused_rows": n_rows - len(train) - len(test),
        "physical_device_overlap_count": overlap,
    }


@dataclass(frozen=True, slots=True)
class PhysicalParent:
    """Parent identity in one declared namespace, e.g. a named lot or wafer."""

    namespace: str
    parent: str | int


@dataclass(frozen=True, slots=True)
class DeviceParentBinding:
    physical: PhysicalDevice
    parent: PhysicalParent


def audit_physical_parent_split(train, test, row_aliases, *, bindings, parent_bindings,
                                parent_level: str) -> dict[str, int]:
    """Audit disjointness at a predeclared physical-parent level.

    Resolve every source alias, then require one parent for every distinct
    physical device in the full roster, including unused rows. Use the same
    declared parent level and canonical namespace semantics throughout
    (for example, a wafer identity qualified by its fabrication lot).
    Apart from checking the level declaration and rejecting common unresolved
    parent identifiers, this audit does not infer or authenticate physical genealogy.
    Only counts are returned. Unused rows do not create train/test overlap.
    """
    if not _crossed_key(parent_level) or parent_level.casefold() in _UNRESOLVED_DEVICE_IDS:
        raise ValueError("Physical parent level must be explicit and resolved.")
    (train, test, _, _, train_devices, test_devices, devices, n_rows) = _resolve_device_split(
        train, test, row_aliases, bindings)
    if type(parent_bindings) not in (list, tuple) or not 1 <= len(parent_bindings) <= MAX_ROWS:
        raise ValueError("Physical parent bindings require an explicit bounded list or tuple.")
    by_device = {}
    for record in parent_bindings:
        if type(record) is not DeviceParentBinding or type(record.parent) is not PhysicalParent:
            raise ValueError("Physical parent bindings require DeviceParentBinding records.")
        _check_physical_device(record.physical)
        parent = record.parent
        if (not _crossed_key(parent.namespace) or not _resolved_identifier(parent.namespace)
                or not _resolved_identifier(parent.parent)):
            raise ValueError("Physical parent identifiers must be exact and resolved.")
        if record.physical in by_device:
            raise ValueError("Physical parent bindings repeat a declared physical device.")
        by_device[record.physical] = parent
    if by_device.keys() != set(devices):
        raise ValueError("Physical parent bindings must cover every roster device exactly, with no extras.")
    train_parents, test_parents = ({by_device[device] for device in subset}
                                  for subset in (train_devices, test_devices))
    overlap = len(train_parents & test_parents)
    if overlap:
        raise ValueError(f"Invalid split: {overlap} declared physical parent(s) overlap training and validation.")
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_physical_device_count": len(train_devices),
        "test_physical_device_count": len(test_devices),
        "train_physical_parent_count": len(train_parents),
        "test_physical_parent_count": len(test_parents),
        "unused_rows": n_rows - len(train) - len(test),
        "physical_parent_overlap_count": overlap,
    }


@dataclass(frozen=True)
class CrossedHoldout:
    fit: tuple[str, ...]
    new_recipe: tuple[str, ...]
    new_device: tuple[str, ...]
    both: tuple[str, ...]


def _crossed_key(x):
    return (type(x) is str and 0 < len(x) <= 256 and x == x.strip()
            and not any(unicodedata.category(c) in ("Cc", "Cs") for c in x))


def plan_crossed_holdout(rows, held_device, held_recipe):
    """Partition (row_id, device_id, recipe_id) triples without outcomes.

    The input is a bounded list or tuple of triples; IDs are opaque and scoped
    by their role. Preserve input order within each quadrant. A nonempty
    quadrant does not authenticate physical identity, support or independence.
    Neither held device nor held recipe occurs in the fit quadrant.
    """
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Rows must be a nonempty bounded roster.")
    if not _crossed_key(held_device) or not _crossed_key(held_recipe):
        raise ValueError("Held identities must be valid text.")
    quadrants = ([], [], [], [])
    row_ids = set()
    for row in rows:
        if type(row) not in (list, tuple) or len(row) != 3 or not all(map(_crossed_key, row)):
            raise ValueError("Rows must contain three valid identity fields.")
        row_id, device, recipe = row
        if row_id in row_ids:
            raise ValueError("Declared row identities must be unique.")
        row_ids.add(row_id)
        index = 2 * (device == held_device) + (recipe == held_recipe)
        quadrants[index].append(row_id)
    if not all(quadrants):
        raise ValueError("Each crossed quadrant must be nonempty; check held identities and support.")
    return CrossedHoldout(*(tuple(ids) for ids in quadrants))


@dataclass(frozen=True, slots=True)
class RunDeviceAlias:
    """A locally named device in one source run (acquisition episode)."""

    source: str
    run: str
    raw_device: str | int


@dataclass(frozen=True, slots=True)
class SourceReadKey:
    """A read identity in its original acquisition, not a physical identity."""

    alias: RunDeviceAlias
    observation: str | int


@dataclass(frozen=True, slots=True)
class RunDeviceBinding:
    alias: RunDeviceAlias
    physical: PhysicalDevice


@dataclass(frozen=True, slots=True)
class ResolvedDeviceRead:
    key: SourceReadKey
    physical: PhysicalDevice


@dataclass(frozen=True, slots=True)
class DeviceRolePreflight:
    train: tuple[ResolvedDeviceRead, ...]
    calibration: tuple[ResolvedDeviceRead, ...]
    evaluation: tuple[ResolvedDeviceRead, ...]


_UNRESOLVED_DEVICE_IDS = frozenset({"unknown", "unreported", "not_reported"})


def _run_identifier(value):
    return _crossed_key(value) or (type(value) is int and -(2**63) <= value < 2**63)


def _resolved_identifier(value):
    return _run_identifier(value) and (
        type(value) is int or value.casefold() not in _UNRESOLVED_DEVICE_IDS
    )


def _run_alias(alias):
    if (type(alias) is not RunDeviceAlias or not _crossed_key(alias.source)
            or not _crossed_key(alias.run) or not _resolved_identifier(alias.raw_device)):
        raise ValueError("Run device alias requires an exact source, run and resolved local device ID.")


def preflight_device_roles(*, train, calibration, evaluation, bindings) -> DeviceRolePreflight:
    """Resolve and keep every original read while requiring disjoint roles.

    Every observed source/run/local-device alias requires one explicit binding
    to a canonical physical device; distinct aliases can resolve to the same
    device within a role. Identical full source read keys are not allowed. Run
    labels distinguish episodes and need not be globally unique. This is a
    metadata check only: no measuring, fitting, ranking, or identity inference.
    """
    roles = (("train", train), ("calibration", calibration), ("evaluation", evaluation))
    if any(type(records) not in (list, tuple) or not records for _, records in roles):
        raise ValueError("Each device role requires a nonempty explicit list or tuple of reads.")
    size = sum(len(records) for _, records in roles)
    if size > MAX_ROWS:
        raise ValueError(f"Device-role read roster exceeds the {MAX_ROWS}-row limit.")
    if type(bindings) not in (list, tuple) or not 1 <= len(bindings) <= size:
        raise ValueError("Run device bindings require a nonempty bounded list or tuple.")

    by_alias = {}
    for binding in tuple(bindings):
        if type(binding) is not RunDeviceBinding:
            raise ValueError("Run device bindings require RunDeviceBinding records.")
        _run_alias(binding.alias)
        device = binding.physical
        if (type(device) is not PhysicalDevice or not _crossed_key(device.namespace)
                or not _resolved_identifier(device.device)):
            raise ValueError("A binding requires an explicit resolved canonical physical device.")
        if binding.alias in by_alias:
            raise ValueError("Run device bindings repeat a source/run/device alias.")
        by_alias[binding.alias] = device

    seen_reads, seen_aliases, device_role = set(), set(), {}
    outputs = []
    for role, records in roles:
        output = []
        for key in tuple(records):
            if type(key) is not SourceReadKey:
                raise ValueError("Device-role reads require SourceReadKey records.")
            _run_alias(key.alias)
            if not _run_identifier(key.observation):
                raise ValueError("A source read requires an exact local observation ID.")
            device = by_alias.get(key.alias)
            if device is None:
                raise ValueError("A declared source/run/device alias has no physical binding.")
            previous = device_role.setdefault(device, role)
            if previous != role:
                raise ValueError(f"A declared physical device overlaps the {previous} and {role} roles.")
            if key in seen_reads:
                raise ValueError("A compound source read key is repeated.")
            seen_reads.add(key)
            seen_aliases.add(key.alias)
            output.append(ResolvedDeviceRead(key, device))
        outputs.append(tuple(output))
    if seen_aliases != by_alias.keys():
        raise ValueError("Run device bindings contain aliases absent from the read roster.")
    return DeviceRolePreflight(*outputs)
