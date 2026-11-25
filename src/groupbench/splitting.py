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
            fold = int(rng.choice(candidates))
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
    """Report counts for disjoint positional subsets without requiring full coverage."""
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


def audit_physical_device_split(train, test, row_aliases, *, bindings) -> dict[str, int]:
    """Reject alias-mediated device overlap in one declared positional split.

    Supply a metadata-only roster in table order and one binding per distinct
    alias across the entire roster. Many source aliases can map to one physical
    device. Only train vs test are compared; unused positions do not count.
    Namespaces are nonblank unpadded built-in strings, and IDs are such strings
    or signed-64-bit built-in integers (1 and "1" differ). No data access, fit,
    device identity verification or wafer-level inference is performed.
    """
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
    overlap = len(train_devices & test_devices)
    if overlap:
        raise ValueError(f"Invalid split: {overlap} declared physical device(s) overlap training and validation.")
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_alias_count": len(train_aliases), "test_alias_count": len(test_aliases),
        "train_physical_device_count": len(train_devices), "test_physical_device_count": len(test_devices),
        "unused_rows": len(row_aliases) - len(train) - len(test),
        "physical_device_overlap_count": overlap,
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
