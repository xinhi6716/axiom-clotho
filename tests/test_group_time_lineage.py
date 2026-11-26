"""Synthetic one-clock lineage fixtures; no features/labels loaded."""

from dataclasses import FrozenInstanceError, replace
from unittest import TestCase
from unittest.mock import patch

from groupbench.group_time_lineage import (
    EventRevision, FeatureLineage, ForecastRow, GroupTimeLineageError, LineageKey,
    RevisionRef, RowLineage, audit_group_time_lineage, audit_joint_forecast_timing,
)
from groupbench.group_time_plan import (
    ForecastLabelTiming, GroupSpan, TimeWindow, audit_group_forward_label_timing,
    plan_group_forward_windows,
)


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

    def test_nominal_and_postprocess_feature_budgets_require_different_origins(self):
        # Retain one event archive and one row roster for both declarations.
        plan = plan_group_forward_windows(
            [1, "1"], group_spans=[GroupSpan(1, 1, 4, 5), GroupSpan("1", 8, 14, 15)],
            windows=[TimeWindow(8, 16)], clock_id="acquisition", time_unit="step",
        )
        nominal = tuple(self.event(group, "nominal-spot", 0, -2, -1)
                        for group in (1, "1"))
        raw = (self.event(1, "width-read", 0, 2, 2),
               self.event("1", "width-read", 0, 11, 11))
        width = (self.event(1, "processed-width", 0, 2, 3, parents=(raw[0].ref,)),
                 self.event("1", "processed-width", 0, 11, 12, parents=(raw[1].ref,)))
        revisions = nominal + raw + width
        nominal_only = tuple(RowLineage(row.key, (FeatureLineage("spot", spot.ref),))
                             for row, spot in zip(self.rows, nominal))
        augmented = tuple(replace(binding, roots=(*binding.roots, FeatureLineage("width", w.ref)))
                          for binding, w in zip(nominal_only, width))
        before = (replace(self.rows[0], origin=1), replace(self.rows[1], origin=9))
        after = (replace(self.rows[0], origin=4), replace(self.rows[1], origin=13))

        def audit(rows, names, bindings):
            return audit_group_time_lineage(
                plan, clock_id="acquisition", time_unit="step", features=names,
                rows=rows, revisions=revisions, lineage=bindings,
            ).n_row_revisions_checked

        self.assertEqual(audit(before, ("spot",), nominal_only), 2)
        with self.assertRaisesRegex(GroupTimeLineageError, "not the latest revision"):
            audit(before, ("spot", "width"), augmented)
        self.assertEqual(audit(after, ("spot", "width"), augmented), 6)

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

    def test_frozen_timing_detects_an_omitted_newer_correction(self):
        correction = replace(self.raw_test, ref=replace(self.raw_test.ref, ordinal=1),
                             available_at=9)
        roster = [(item.ref, item.measured_at, item.available_at)
                  for item in (*self.revisions, correction)]
        self.assertFalse(self.audit().frozen_timing_roster_matched)
        with self.assertRaisesRegex(GroupTimeLineageError, 'exactly match the frozen'):
            self.audit(expected_revision_timing=roster)
        with self.assertRaisesRegex(GroupTimeLineageError, 'latest revision'):
            self.audit(revisions=(*self.revisions, correction), expected_revision_timing=roster)

    def test_frozen_timing_detects_retiming_even_when_ancestry_still_passes(self):
        roster = [(x.ref, x.measured_at, x.available_at) for x in self.revisions]
        valid = self.audit(expected_revision_timing=roster[::-1], revisions=self.revisions[::-1])
        self.assertTrue(valid.frozen_timing_roster_matched)
        for measured, available in ((self.raw_test.measured_at - 1, self.raw_test.available_at),
                                    (self.raw_test.measured_at, self.raw_test.available_at + 1)):
            changed = replace(self.raw_test, measured_at=measured, available_at=available)
            self.assertEqual(self.audit(revisions=(self.raw_train, changed, self.derived)),
                             self.audit())
            with self.assertRaisesRegex(GroupTimeLineageError, 'clocks differ'):
                self.audit(revisions=(self.raw_train, changed, self.derived),
                           expected_revision_timing=roster)

    def test_frozen_timing_validates_exact_typed_keys_and_ticks(self):
        roster = [(x.ref, x.measured_at, x.available_at) for x in self.revisions]
        entry = roster[0]
        for bad in ([], roster + [entry], [list(entry), *roster[1:]],
                    [(entry[0], True, entry[2]), *roster[1:]],
                    [(entry[0], entry[2]+1, entry[2]), *roster[1:]],
                    [(replace(entry[0], event=replace(entry[0].event, group='1')),
                      *entry[1:]), *roster[1:]],
                    {entry}):
            with self.subTest(roster=bad), self.assertRaises(GroupTimeLineageError):
                self.audit(expected_revision_timing=bad)


    def joint(self, labels, **overrides):
        kwargs = dict(clock_id="acquisition", time_unit="step", horizon=1,
                      features=("x",), rows=self.rows, lineage=self.roots,
                      revisions=self.revisions, labels=labels)
        kwargs.update(overrides)
        return audit_joint_forecast_timing(self.plan, **kwargs)

    def test_joint_uses_identical_origins_and_preserves_component_diagnostics(self):
        labels = [ForecastLabelTiming(0, 1, 3, 4, 5),
                  ForecastLabelTiming(1, "1", 10, 11, 15)]
        roster = tuple((x.ref, x.measured_at, x.available_at) for x in self.revisions)
        got = self.joint(labels, expected_revision_timing=roster)
        self.assertEqual(got.feature_lineage, self.audit(expected_revision_timing=roster))
        self.assertEqual(got.label_timing, audit_group_forward_label_timing(
            self.plan, clock_id="acquisition", time_unit="step", horizon=1, labels=labels))
        self.assertEqual(got.label_timing.n_tested_labels, 1)
        self.assertTrue(got.feature_lineage.frozen_timing_roster_matched)

    def test_separately_valid_metadata_can_incorporate_post_origin_features(self):
        later = self.event("1", "late-measurement", 0, 9, 9)
        roots = self.with_root(later.ref)
        revisions = self.revisions + (later,)
        labels = [ForecastLabelTiming(0, 1, 3, 4, 5),
                  ForecastLabelTiming(1, "1", 8, 9, 15)]
        self.assertEqual(self.audit(lineage=roots, revisions=revisions).n_rows, 2)
        self.assertEqual(audit_group_forward_label_timing(
            self.plan, clock_id="acquisition", time_unit="step", horizon=1, labels=labels).n_rows, 2)
        with self.assertRaisesRegex(GroupTimeLineageError, 'origins? differ.*position 1'):
            self.joint(labels, lineage=roots, revisions=revisions)

    def test_joint_does_not_bypass_clock_row_and_catalog_rules(self):
        labels = [ForecastLabelTiming(0, 1, 3, 4, 5),
                  ForecastLabelTiming(1, "1", 10, 11, 15)]
        with self.assertRaisesRegex(ValueError, 'typed groups'):
            self.joint([labels[0], replace(labels[1], group=1)])
        with self.assertRaisesRegex(ValueError, 'every plan row'):
            self.joint(labels[:1])
        with self.assertRaisesRegex(GroupTimeLineageError, 'clock and unit'):
            self.joint(labels, clock_id="different")
        with self.assertRaisesRegex(GroupTimeLineageError, 'exactly match'):
            self.joint(labels, expected_revision_timing=[
                (self.raw_train.ref, self.raw_train.measured_at, self.raw_train.available_at)])
