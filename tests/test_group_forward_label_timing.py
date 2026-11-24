"""Forecast-origin and label-availability tests using invented integer ticks."""

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from groupbench import group_time_plan as temporal
from groupbench.group_time_plan import GroupSpan, MAX_TICK, TimeWindow, plan_group_forward_windows


class GroupForwardLabelTimingTests(TestCase):
    def setUp(self):
        self.plan = plan_group_forward_windows(
            [1, "1", 1, "later", "unused"],
            group_spans=[GroupSpan(1, 0, 3, 7), GroupSpan("1", 10, 11, 16),
                         GroupSpan("later", 20, 22, 23), GroupSpan("unused", 12, 17, 21)],
            windows=[TimeWindow(10, 15), TimeWindow(20, 24)],
            clock_id="bench", time_unit="ticks", gap=2,
        )

    def rows(self):
        row = temporal.ForecastLabelTiming
        return (
            row(0, 1, 0, 1, 3),
            row(1, "1", 11, 12, 16),
            row(2, 1, 3, 4, 7),
            row(3, "later", 22, 23, 23),
            row(4, "unused", 14, 15, 18),
        )

    def audit(self, rows=None, **overrides):
        kwargs = dict(clock_id="bench", time_unit="ticks", horizon=1,
                      labels=self.rows() if rows is None else rows)
        kwargs.update(overrides)
        return temporal.audit_group_forward_label_timing(self.plan, **kwargs)

    def edited(self, i, **fields):
        rows = list(self.rows())
        rows[i] = replace(rows[i], **fields)
        return rows

    def test_valid_packet_and_delayed_test_scoring(self):
        result = self.audit()
        self.assertEqual(result.n_rows, 5)
        self.assertEqual(result.fold_train_label_counts, (2, 3))
        self.assertEqual(result.fold_test_label_counts, (1, 1))
        self.assertEqual(result.n_tested_labels, 2)

    def test_group_aggregate_can_accept_train_label_wrongly(self):
        # Old planner sees only aggregate label ready at 7, then 7 + 2 < 10.
        self.assertEqual(self.plan.folds[0].train.rows, (0, 2))
        for late in (8, 9, 10):
            # Tick 8 is itself before origin; with the embargo it is too late.
            with self.subTest(late=late), self.assertRaisesRegex(ValueError, r"fold 0.*strictly before"):
                self.audit(self.edited(2, available_at=late))

    def test_previous_test_group_can_fail_when_reused_in_later_train(self):
        self.assertEqual(self.plan.folds[0].test.rows, (1,))
        self.assertIn(1, self.plan.folds[1].train.rows)
        with self.assertRaisesRegex(ValueError, r"fold 1.*strictly before"):
            self.audit(self.edited(1, available_at=18))

    def test_understated_group_envelope_detected_even_without_train_violation(self):
        # Still safe for any training fold, but disagrees with the plan's
        # declared upper bound. Last row is never train or test.
        with self.assertRaisesRegex(ValueError, "group availability bound"):
            self.audit(self.edited(1, available_at=17))
        with self.assertRaisesRegex(ValueError, "group availability bound"):
            self.audit(self.edited(4, available_at=22))

    def test_time_of_measurement_is_not_time_of_arrival_and_horizon_is_fixed(self):
        with self.assertRaisesRegex(ValueError, "before it was measured"):
            self.audit(self.edited(0, available_at=0))
        for tick in (0, 2):
            with self.subTest(tick=tick), self.assertRaisesRegex(ValueError, "fixed horizon"):
                self.audit(self.edited(0, measured_at=tick))

    def test_position_type_group_identity_and_roster_are_exact(self):
        for rows in (self.edited(0, row_position=True), self.edited(0, group=True),
                     self.edited(0, group="1"), tuple(reversed(self.rows()))):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "positions and typed groups"):
                self.audit(rows)
        with self.assertRaisesRegex(ValueError, "cover every plan row"):
            self.audit(self.rows()[:-1])

    def test_clock_tick_origin_and_arithmetic_boundaries(self):
        for overrides in (dict(clock_id="UTC"), dict(time_unit="seconds")):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "match the plan"):
                self.audit(**overrides)
        for value in (False, 1.0, MAX_TICK + 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "built-in signed-64-bit"):
                self.audit(self.edited(0, forecast_origin=value))
        with self.assertRaisesRegex(ValueError, "origin.*span"):
            self.audit(self.edited(0, forecast_origin=4))
        with self.assertRaisesRegex(ValueError, "positive"):
            self.audit(horizon=0)
        with self.assertRaisesRegex(ValueError, "origin plus horizon"):
            self.audit(self.edited(0, forecast_origin=1), horizon=MAX_TICK)

    def test_forged_membership_and_bounded_reconstruction(self):
        first, second = self.plan.folds
        self.plan = replace(self.plan, folds=(replace(first, train=replace(first.train, rows=(False, 2))), second))
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.audit()
        with patch.object(temporal, "_LABEL_AUDIT_MAX_WORK", 8), patch.object(
            temporal, "plan_group_forward_windows", side_effect=AssertionError("Bound before rebuilding")
        ):
            with self.assertRaisesRegex(ValueError, "plan work bound"):
                self.audit()
