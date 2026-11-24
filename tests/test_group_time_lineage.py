"""Synthetic one-clock lineage fixtures; no features/labels loaded."""

from dataclasses import FrozenInstanceError, replace
from unittest import TestCase
from unittest.mock import patch

from groupbench.group_time_lineage import (
    EventRevision, FeatureLineage, ForecastRow, GroupTimeLineageError, LineageKey,
    RevisionRef, RowLineage, audit_group_time_lineage,
)
from groupbench.group_time_plan import GroupSpan, TimeWindow, plan_group_forward_windows


class GroupTimeLineageTests(TestCase):
    def setUp(self):
        self.plan = plan_group_forward_windows(
            [1, "1"], group_spans=[GroupSpan(1, 1, 3, 5), GroupSpan("1", 8, 11, 15)],
            windows=[TimeWindow(8, 20)], clock_id="acquisition", time_unit="step",
        )
        self.rows = (
            ForecastRow(self.key(1, "row-a"), 3),
            ForecastRow(self.key("1", "row-b"), 10),
        )
        self.raw_train = self.event(1, "read", 0, 1, 2)
        self.raw_test = self.event("1", "read", 0, 4, 5)
        self.derived = self.event("1", "published", 0, 6, 7, parents=(self.raw_test.ref,))
        self.roots = (
            RowLineage(self.rows[0].key, (FeatureLineage("x", self.raw_train.ref),)),
            RowLineage(self.rows[1].key, (FeatureLineage("x", self.derived.ref),)),
        )
        self.revisions = (self.raw_train, self.raw_test, self.derived)

    @staticmethod
    def key(group, name):
        return LineageKey("lab", group, "episode", name)

    def event(self, group, name, ordinal, measured, available, *, status="observed", parents=()):
        return EventRevision(
            RevisionRef(self.key(group, name), ordinal), measured, available,
            "derived" if parents else status, parents,
        )

    def audit(self, **changes):
        kwargs = dict(clock_id="acquisition", time_unit="step", features=("x",),
                      rows=self.rows, lineage=self.roots, revisions=self.revisions)
        kwargs.update(changes)
        return audit_group_time_lineage(self.plan, **kwargs)

    def with_root(self, ref):
        return (self.roots[0], replace(self.roots[1], roots=(FeatureLineage("x", ref),)))

    def test_clean_declared_ancestry_is_immutable_and_has_typed_group_identity(self):
        audit = self.audit(revisions=self.revisions[::-1])
        self.assertEqual((audit.n_rows, audit.n_features, audit.n_row_revisions_checked), (2, 1, 3))
        self.assertEqual(audit.protocol, "group-forward-as-of-lineage-v1")
        with self.assertRaises(FrozenInstanceError):
            audit.n_rows = 0

    def test_retroactive_poison_or_deletion_strictly_after_origin_does_not_change_selection(self):
        late = self.event("1", "read", 1, 4, 13)
        deleted = self.event("1", "read", 2, 4, 14, status="deleted")
        audit = self.audit(revisions=self.revisions + (late, deleted))
        self.assertEqual(audit.n_row_revisions_checked, self.audit().n_row_revisions_checked)
        self.assertEqual(audit.n_revisions_declared, 5)
        with self.assertRaisesRegex(GroupTimeLineageError, "not the latest revision"):
            self.audit(revisions=self.revisions + (late,), lineage=self.with_root(late.ref))

    def test_retroactive_preorigin_revision_invalidates_stale_ancestry_but_new_chain_passes(self):
        newer = self.event("1", "read", 2, 4, 6)
        rebuilt = self.event("1", "published", 1, 6, 8, parents=(newer.ref,))
        with self.assertRaisesRegex(GroupTimeLineageError, "not the latest revision"):
            self.audit(revisions=self.revisions + (newer,))
        self.assertEqual(
            self.audit(revisions=self.revisions + (newer, rebuilt),
                       lineage=self.with_root(rebuilt.ref)).n_row_revisions_checked,
            3,
        )

    def test_two_hop_cached_ancestry_is_also_checked(self):
        top = self.event("1", "aggregate", 0, 7, 8, parents=(self.derived.ref,))
        config = dict(lineage=self.with_root(top.ref))
        self.assertEqual(self.audit(revisions=self.revisions + (top,), **config).n_row_revisions_checked, 4)
        # This source correction arrives after both derivations, but before the forecast.
        after_compute = self.event("1", "read", 1, 4, 9)
        with self.assertRaisesRegex(GroupTimeLineageError, "not the latest revision"):
            self.audit(revisions=self.revisions + (top, after_compute), **config)
        # Both reported derivatives preceded an ancestor's actual availability.
        future_ancestor = replace(self.raw_test, available_at=11)
        with self.assertRaisesRegex(GroupTimeLineageError, "strictly before its publication"):
            self.audit(revisions=(self.raw_train, future_ancestor, self.derived, top), **config)

    def test_deletion_before_cutoff_never_falls_back_to_previous_value(self):
        dropped = self.event("1", "read", 1, 4, 8, status="deleted")
        with self.assertRaisesRegex(GroupTimeLineageError, "not the latest revision"):
            self.audit(revisions=self.revisions + (dropped,))
        with self.assertRaisesRegex(GroupTimeLineageError, "uses a deleted event"):
            self.audit(revisions=self.revisions + (dropped,), lineage=self.with_root(dropped.ref))

    def test_origin_and_same_event_arrival_ties_are_fail_closed(self):
        at_cutoff = self.event("1", "read", 1, 4, 10)
        same_arrival = self.event("1", "read", 1, 4, 5)
        with self.assertRaisesRegex(GroupTimeLineageError, "unresolved order"):
            self.audit(revisions=self.revisions + (at_cutoff,))
        with self.assertRaisesRegex(GroupTimeLineageError, "strictly increasing"):
            self.audit(revisions=self.revisions + (same_arrival,))

    def test_forged_or_future_parent_time_fails_even_if_before_origin(self):
        for arrival in (7, 8):
            with self.subTest(arrival=arrival):
                impossible = replace(self.raw_test, available_at=arrival)
                with self.assertRaisesRegex(GroupTimeLineageError, "strictly before its publication"):
                    self.audit(revisions=(self.raw_train, impossible, self.derived))
        with self.assertRaisesRegex(GroupTimeLineageError, "before it was measured"):
            self.audit(revisions=(self.raw_train, replace(self.raw_test, measured_at=6), self.derived))

    def test_missing_foreign_duplicate_and_disguised_parents_are_refused(self):
        with self.assertRaisesRegex(GroupTimeLineageError, "outside its source, physical group or episode"):
            self.audit(lineage=self.with_root(self.raw_train.ref))
        for parent, message in (
            (self.raw_train.ref, "outside its source, physical group or episode"),
            (RevisionRef(self.key("1", "missing"), 0), "requires parents available"),
            (self.derived.ref, "requires parents available"),
        ):
            with self.subTest(parent=parent), self.assertRaisesRegex(GroupTimeLineageError, message):
                self.audit(revisions=(*self.revisions[:2], replace(self.derived, parents=(parent,))))
        with self.assertRaisesRegex(GroupTimeLineageError, "Duplicate event revision"):
            self.audit(revisions=(*self.revisions, self.raw_test))

    def test_ordered_complete_bindings_and_origin_in_span(self):
        with self.assertRaisesRegex(GroupTimeLineageError, "out-of-order"):
            self.audit(lineage=self.roots[::-1])
        with self.assertRaisesRegex(GroupTimeLineageError, "bounded list or tuple"):
            self.audit(lineage=self.roots[:1])
        with self.assertRaisesRegex(GroupTimeLineageError, "ordered feature list"):
            self.audit(features=("another-feature",))
        with self.assertRaisesRegex(GroupTimeLineageError, "inside its declared group span"):
            self.audit(rows=(self.rows[0], replace(self.rows[1], origin=13)))
        with self.assertRaisesRegex(GroupTimeLineageError, "signed-64-bit"):
            self.audit(rows=(replace(self.rows[0], origin=True), self.rows[1]))

    def test_wrong_clock_and_forged_fold_membership_fail(self):
        with self.assertRaisesRegex(GroupTimeLineageError, "must match the plan"):
            self.audit(clock_id="wall-time")
        fold = self.plan.folds[0]
        forged = replace(fold, train=replace(fold.train, rows=(False,)))
        self.plan = replace(self.plan, folds=(forged,))
        with self.assertRaisesRegex(GroupTimeLineageError, "canonical"):
            self.audit()

    def test_walk_checks_work_bound_before_duplicate_elision(self):
        nodes = (self.raw_train, self.raw_test, self.derived)
        with patch("groupbench.group_time_lineage._MAX_CHECKS", 4):
            self.assertEqual(self.audit().n_row_revisions_checked, 3)
            with self.assertRaisesRegex(GroupTimeLineageError, "ancestry exceeds"):
                self.audit(
                    features=("a", "b"),
                    lineage=tuple(RowLineage(row.key, (FeatureLineage("a", roots.roots[0].root),
                                                       FeatureLineage("b", roots.roots[0].root)))
                                  for row, roots in zip(self.rows, self.roots)),
                    revisions=nodes,
                )
