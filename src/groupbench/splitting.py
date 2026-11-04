"""Deterministic group-disjoint, approximately row-balanced folds."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from .data import validate_groups, validate_seed


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


def audit_split(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> dict:
    train_groups = set(groups[train])
    test_groups = set(groups[test])
    overlap = train_groups & test_groups
    if overlap or set(train) & set(test):
        raise ValueError("Invalid split: training and validation observations overlap.")
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_group_count": len(train_groups), "test_group_count": len(test_groups),
        "overlap_count": len(overlap),
        "train_groups": sorted(train_groups), "test_groups": sorted(test_groups),
    }
