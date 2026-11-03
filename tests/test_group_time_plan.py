"""Handwritten metadata fixtures for the standalone no-fit time-window plan."""

from dataclasses import FrozenInstanceError, fields
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch
import copy
import unittest

import numpy as np

from groupbench.group_time_plan import (
    MAX_TICK, MIN_TICK, GroupSpan, TimeWindow, Membership, GroupExclusion,
    ForwardWindowFold, GroupForwardWindowPlan, plan_group_forward_windows,
)


_UNSET = object()


def plan(rows=_UNSET, spans=_UNSET, windows=_UNSET, **changes):
    options = dict(
        row_groups=["warm", "test"] if rows is _UNSET else rows,
        group_spans=[GroupSpan("warm", 0, 1, 1), GroupSpan("test", 10, 12, 12)]
        if spans is _UNSET else spans,
        windows=[TimeWindow(10, 15)] if windows is _UNSET else windows,
        time_unit="day", clock_id="handwritten-clock", gap=0,
    )
    options.update(changes)
    return plan_group_forward_windows(**options)


def mixed_fixture():
    rows = ["warm", "a", "cross", "warm", "b", "delay", "c", "late", "bridge", "never", "a"]
    spans = [GroupSpan("warm", 0, 1, 1), GroupSpan("a", 10, 12, 12),
             GroupSpan("cross", 8, 12, 12), GroupSpan("b", 20, 21, 23),
             GroupSpan("delay", 2, 3, 25), GroupSpan("c", 30, 31, 31),
             GroupSpan("late", 35, 36, 36), GroupSpan("bridge", 15, 25, 25),
             GroupSpan("never", 45, 46, 46)]
    return rows, spans, [TimeWindow(10, 15), TimeWindow(20, 25), TimeWindow(30, 35)]


class ForwardWindowMembershipTests(unittest.TestCase):
    def test_exact_partial_memberships_and_coverage(self):
        rows, spans, windows = mixed_fixture()
        result = plan(rows, spans, windows, gap=2)
        expected = [((0, 3), (1, 10), (2, 4, 5, 6, 7, 8, 9)),
                    ((0, 1, 2, 3, 10), (4,), (5, 6, 7, 8, 9)),
                    ((0, 1, 2, 3, 4, 5, 8, 10), (6,), (7, 9))]
        for fold, (train, test, unused) in zip(result.folds, expected):
            self.assertEqual((fold.train.rows, fold.test.rows, fold.unused.rows), (train, test, unused))
        self.assertEqual(result.total, Membership(tuple(range(11)), tuple(dict.fromkeys(rows))))
        self.assertEqual(result.tested, Membership((1, 4, 6, 10), ("a", "b", "c")))
        self.assertEqual(result.untested, Membership((0, 2, 3, 5, 7, 8, 9),
                                                   ("warm", "cross", "delay", "late", "bridge", "never")))
        self.assertEqual((result.total.n_rows, result.total.n_groups), (11, 9))
        self.assertEqual((result.tested.n_rows, result.tested.n_groups), (4, 3))
        self.assertEqual((result.untested.n_rows, result.untested.n_groups), (7, 6))
        self.assertEqual(result.row_groups, tuple(rows))
        self.assertEqual(result.protocol, "group-forward-window-plan-v1")
        self.assertEqual(result.coverage_mode, "partial_predeclared")
        self.assertEqual(result.fits_performed, 0)
        self.assertEqual((result.time_unit, result.clock_id, result.gap), ("day", "handwritten-clock", 2))

    def test_every_row_and_group_accounted_once_per_fold(self):
        rows, spans, windows = mixed_fixture()
        result = plan(rows, spans, windows, gap=2)
        tested = []
        for index, fold in enumerate(result.folds):
            self.assertEqual((fold.index, fold.window), (index, windows[index]))
            memberships = (fold.train, fold.test, fold.unused)
            self.assertEqual(sorted(i for m in memberships for i in m.rows), list(range(len(rows))))
            self.assertEqual(sorted(g for m in memberships for g in m.groups), sorted(set(rows)))
            for membership in memberships:
                self.assertEqual(membership.rows, tuple(i for i, g in enumerate(rows) if g in membership.groups))
            self.assertEqual(tuple(e.group for e in fold.exclusions), result.total.groups)
            for exclusion in fold.exclusions:
                self.assertEqual(exclusion.train is None, exclusion.group in fold.train.groups)
                self.assertEqual(exclusion.test is None, exclusion.group in fold.test.groups)
            tested.extend(fold.test.rows)
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(sorted(result.tested.rows + result.untested.rows), list(range(len(rows))))

    def test_label_cutoff_is_strict_including_zero_gap(self):
        rows = ["past", "equal", "after", "test"]
        spans = [GroupSpan("past", 0, 1, 9), GroupSpan("equal", 0, 1, 10),
                 GroupSpan("after", 0, 1, 11), GroupSpan("test", 10, 12, 12)]
        fold = plan(rows, spans).folds[0]
        self.assertEqual(fold.train.groups, ("past",))
        self.assertEqual(fold.unused.groups, ("equal", "after"))
        self.assertEqual(fold.exclusions[1].train, "label_not_available_before_window")

    def test_positive_gap_equality_stays_unused(self):
        rows = ["past", "equal", "test"]
        spans = [GroupSpan("past", 0, 1, 7), GroupSpan("equal", 0, 1, 8),
                 GroupSpan("test", 10, 12, 12)]
        fold = plan(rows, spans, gap=2).folds[0]
        self.assertEqual(fold.train.groups, ("past",))
        self.assertEqual(fold.unused.groups, ("equal",))
        self.assertEqual(fold.exclusions[1].train, "gap_not_elapsed_before_window")

    def test_overlapping_and_tied_group_spans_are_not_an_ordering_error(self):
        rows = ["warm", "one", "two", "overlap", "two"]
        spans = [GroupSpan("warm", 0, 1, 1), GroupSpan("one", 10, 12, 12),
                 GroupSpan("two", 10, 12, 12), GroupSpan("overlap", 11, 14, 14)]
        fold = plan(rows, spans).folds[0]
        self.assertEqual(fold.test.groups, ("one", "two", "overlap"))
        self.assertEqual(fold.test.rows, (1, 2, 3, 4))

    def test_crossing_groups_are_wholly_unused_not_truncated(self):
        rows = ["warm", "test", "lower", "upper", "both", "touch_lower", "at_upper", "old"]
        spans = [GroupSpan("warm", 0, 1, 1), GroupSpan("test", 10, 14, 99),
                 GroupSpan("lower", 9, 11, 11), GroupSpan("upper", 11, 15, 15),
                 GroupSpan("both", 9, 15, 15), GroupSpan("touch_lower", 9, 10, 10),
                 GroupSpan("at_upper", 15, 15, 15), GroupSpan("old", 0, 1, 99)]
        fold = plan(rows, spans).folds[0]
        self.assertEqual(fold.test.groups, ("test",))  # Test labels need not be available yet.
        self.assertEqual(fold.unused.groups, tuple(rows[2:]))
        self.assertEqual(tuple(e.test for e in fold.exclusions),
                         ("before_window", None, "crosses_lower_boundary", "crosses_upper_boundary",
                          "crosses_both_boundaries", "crosses_lower_boundary", "at_or_after_window", "before_window"))

    def test_adjacent_windows_allow_zero_length_span_at_next_lower(self):
        rows = ["warm", "a", "boundary"]
        spans = [GroupSpan("warm", 0, 1, 1), GroupSpan("a", 10, 10, 10),
                 GroupSpan("boundary", 15, 15, 15)]
        result = plan(rows, spans, [TimeWindow(10, 15), TimeWindow(15, 20)])
        self.assertEqual(result.folds[0].test.groups, ("a",))
        self.assertEqual(result.folds[1].test.groups, ("boundary",))
        self.assertEqual(result.folds[1].train.groups, ("warm", "a"))

    def test_never_tested_group_can_train_later_and_warmup_is_untested(self):
        rows, spans, windows = mixed_fixture()
        result = plan(rows, spans, windows, gap=2)
        self.assertIn("warm", result.untested.groups)
        self.assertIn("cross", result.untested.groups)
        self.assertIn("cross", result.folds[0].unused.groups)
        self.assertIn("cross", result.folds[1].train.groups)
        self.assertIn("delay", result.folds[1].unused.groups)
        self.assertIn("delay", result.folds[2].train.groups)
        self.assertNotIn("delay", result.tested.groups)

    def test_original_row_order_not_sorted_by_identity_or_time(self):
        rows = ["test", "warm", "test", "warm"]
        result = plan(rows)
        self.assertEqual(result.total.groups, ("test", "warm"))
        self.assertEqual(result.folds[0].test.rows, (0, 2))
        self.assertEqual(result.folds[0].train.rows, (1, 3))
        self.assertEqual(tuple(s.group for s in result.group_spans), ("test", "warm"))

    def test_span_table_order_invariance_and_determinism(self):
        rows, spans, windows = mixed_fixture()
        expected = plan(rows, spans, windows, gap=2)
        self.assertEqual(expected, plan(rows, list(reversed(spans)), windows, gap=2))
        self.assertEqual(expected, plan(tuple(rows), tuple(spans), tuple(windows), gap=2))
        self.assertEqual(expected, plan(rows, spans, windows, gap=2))

    def test_nonmutation_and_later_input_list_edits_do_not_change_plan(self):
        rows, spans, windows = mixed_fixture()
        before = copy.deepcopy((rows, spans, windows))
        result = plan(rows, spans, windows, gap=2)
        self.assertEqual((rows, spans, windows), before)
        expected = copy.deepcopy(result)
        rows[0] = "replacement"
        spans.reverse()
        windows.clear()
        self.assertEqual(result, expected)

    def test_outputs_recursively_immutable_and_hashable(self):
        result = plan()
        def inspect(value):
            if isinstance(value, (GroupSpan, TimeWindow, Membership, GroupExclusion,
                                  ForwardWindowFold, GroupForwardWindowPlan)):
                for member in fields(value):
                    with self.assertRaises(FrozenInstanceError):
                        setattr(value, member.name, None)
                    inspect(getattr(value, member.name))
                self.assertFalse(hasattr(value, "__dict__"))
            elif isinstance(value, tuple):
                for item in value:
                    inspect(item)
            else:
                self.assertIn(type(value), (str, int, type(None)))
        inspect(result)
        self.assertEqual(hash(result), hash(plan()))

    def test_no_fitting_preprocessing_or_randomness(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("no fitting")), \
             patch("numpy.random.default_rng", side_effect=AssertionError("no RNG")), \
             patch("random.random", side_effect=AssertionError("no RNG")):
            result = plan()
        self.assertEqual(result.fits_performed, 0)

    def test_exact_integer_string_and_whitespace_identities_are_preserved(self):
        rows = [1, "1", " 1", "001", "01", "NA", "1 "]
        spans = [GroupSpan(1, 0, 1, 1)] + [GroupSpan(g, 10, 11, 11) for g in rows[1:]]
        result = plan(rows, spans)
        self.assertEqual(result.total.groups, tuple(rows))
        self.assertEqual(result.folds[0].train.groups, (1,))
        self.assertEqual(result.folds[0].test.groups, tuple(rows[1:]))

    def test_negative_ticks_and_signed_integer_endpoints_are_supported(self):
        result = plan([MIN_TICK, MAX_TICK],
                      [GroupSpan(MIN_TICK, MIN_TICK, MIN_TICK, MIN_TICK),
                       GroupSpan(MAX_TICK, MAX_TICK - 1, MAX_TICK - 1, MAX_TICK)],
                      [TimeWindow(MAX_TICK - 1, MAX_TICK)])
        self.assertEqual(result.folds[0].train.groups, (MIN_TICK,))
        self.assertEqual(result.folds[0].test.groups, (MAX_TICK,))
        negative = plan(["warm", "test"],
                        [GroupSpan("warm", -20, -15, -11), GroupSpan("test", -10, -9, -9)],
                        [TimeWindow(-10, -5)])
        self.assertEqual(negative.folds[0].train.rows, (0,))


class ForwardWindowValidationTests(unittest.TestCase):
    def test_missing_required_declarations_are_python_errors(self):
        for omitted in ("time_unit", "clock_id"):
            kwargs = dict(group_spans=[GroupSpan("a", 0, 1, 1)], windows=[TimeWindow(2, 3)],
                          time_unit="day", clock_id="clock")
            del kwargs[omitted]
            with self.assertRaises(TypeError):
                plan_group_forward_windows(["a"], **kwargs)

    def test_blank_or_coerced_clock_and_unit_are_rejected(self):
        for name in ("time_unit", "clock_id"):
            for value in (None, "", " ", " day", "clock ", 1, True, [], np.str_("clock")):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    plan(**{name: value})

    def test_explicit_custom_unit_label_is_not_parsed_or_converted(self):
        result = plan(time_unit="instrument-cycle", clock_id="local-oscillator-7")
        self.assertEqual((result.time_unit, result.clock_id), ("instrument-cycle", "local-oscillator-7"))

    def test_only_explicit_nonempty_lists_or_tuples_are_accepted(self):
        for name in ("row_groups", "group_spans", "windows"):
            for value in (None, [], (), "warm", {}, {"warm"}, iter(["warm"]), np.array([1])):
                with self.subTest(name=name, value=repr(value)), self.assertRaisesRegex(ValueError, name):
                    plan(**{name: value})

    def test_unsupported_group_identities_are_rejected_without_coercion(self):
        class IntegerSubclass(int):
            pass
        class StringSubclass(str):
            pass
        invalid = (True, False, 1.0, float("nan"), None, "", " \t", b"a", ("a",), [], {},
                   np.int64(1), np.str_("a"), IntegerSubclass(1), StringSubclass("a"),
                   MIN_TICK - 1, MAX_TICK + 1)
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, r"row_groups\[0\]"):
                plan(rows=[value, "test"])
            with self.subTest(span_value=repr(value)), self.assertRaisesRegex(ValueError, r"group_spans\[0\].group"):
                plan(spans=[GroupSpan(value, 0, 1, 1), GroupSpan("test", 10, 12, 12)])

    def test_span_records_and_windows_are_typed_not_automatically_parsed(self):
        for value in (None, ("warm", 0, 1, 1), {"group": "warm", "start": 0, "end": 1, "label_available_at": 1}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a GroupSpan"):
                plan(spans=[value, GroupSpan("test", 10, 12, 12)])
        for value in (None, (10, 15), {"lower": 10, "upper": 15}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a TimeWindow"):
                plan(windows=[value])

    def test_missing_extra_and_duplicate_span_identities(self):
        cases = (([GroupSpan("warm", 0, 1, 1)], "1 missing, 0 extra"),
                 ([GroupSpan("warm", 0, 1, 1), GroupSpan("test", 10, 12, 12), GroupSpan("extra", 0, 1, 1)], "0 missing, 1 extra"),
                 ([GroupSpan("warm", 0, 1, 1), GroupSpan("other", 10, 12, 12)], "1 missing, 1 extra"),
                 ([GroupSpan("warm", 0, 1, 1), GroupSpan("warm", 10, 12, 12)], "duplicates a group identity"))
        for spans, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan(spans=spans)
        with self.assertRaisesRegex(ValueError, "1 missing, 1 extra"):
            plan(rows=[1, "test"], spans=[GroupSpan("1", 0, 1, 1), GroupSpan("test", 10, 12, 12)])

    def test_span_order_requires_start_end_label_order(self):
        for span in (GroupSpan("warm", 2, 1, 3), GroupSpan("warm", 0, 2, 1)):
            with self.subTest(span=span), self.assertRaisesRegex(ValueError, "start <= end <= label_available_at"):
                plan(spans=[span, GroupSpan("test", 10, 12, 12)])

    def test_bool_float_missing_datetime_and_nonbuiltin_ticks_are_rejected(self):
        class IntegerSubclass(int):
            pass
        invalid = (True, False, None, 1.0, float("nan"), float("inf"), "1", Decimal(1),
                   datetime(2020, 1, 1), np.datetime64("2020-01-01"), np.int64(1), IntegerSubclass(1),
                   MIN_TICK - 1, MAX_TICK + 1)
        for value in invalid:
            for field_name in ("start", "end", "label_available_at"):
                kwargs = dict(group="warm", start=0, end=1, label_available_at=1)
                kwargs[field_name] = value
                with self.subTest(value=repr(value), field=field_name), self.assertRaisesRegex(ValueError, "integer tick"):
                    plan(spans=[GroupSpan(**kwargs), GroupSpan("test", 10, 12, 12)])
            for window in (TimeWindow(value, 15), TimeWindow(10, value)):
                with self.subTest(window=window), self.assertRaisesRegex(ValueError, "integer tick"):
                    plan(windows=[window])
            with self.subTest(gap=repr(value)), self.assertRaisesRegex(ValueError, "integer tick"):
                plan(gap=value)

    def test_negative_gap_and_addition_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "gap must be nonnegative"):
            plan(gap=-1)
        for gap, available in ((1, MAX_TICK), (MAX_TICK, 1)):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, r"label_available_at \+ gap exceeds"):
                plan(spans=[GroupSpan("warm", 0, 1, available), GroupSpan("test", 10, 12, 12)], gap=gap)

    def test_label_plus_gap_exactly_maximum_does_not_overflow(self):
        result = plan(spans=[GroupSpan("warm", MIN_TICK, MIN_TICK, MIN_TICK),
                            GroupSpan("test", 10, 12, MAX_TICK - 1)], gap=1)
        self.assertEqual(result.folds[0].train.groups, ("warm",))
        self.assertEqual(result.folds[0].test.groups, ("test",))

    def test_malformed_overlapping_tied_reversed_windows_rejected(self):
        cases = (([TimeWindow(10, 10)], "lower < upper"),
                 ([TimeWindow(15, 10)], "lower < upper"),
                 ([TimeWindow(10, 15), TimeWindow(14, 20)], "increase without overlapping"),
                 ([TimeWindow(10, 15), TimeWindow(10, 15)], "increase without overlapping"),
                 ([TimeWindow(10, 15), TimeWindow(0, 5)], "increase without overlapping"))
        for windows, message in cases:
            with self.subTest(windows=windows), self.assertRaisesRegex(ValueError, message):
                plan(windows=windows)

    def test_empty_train_test_and_both_report_exact_fold(self):
        cases = ((["test"], [GroupSpan("test", 10, 12, 12)], "empty_train"),
                 (["warm"], [GroupSpan("warm", 0, 1, 1)], "empty_test"),
                 (["future"], [GroupSpan("future", 30, 31, 31)], "empty_train, empty_test"))
        for rows, spans, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, rf"fold 0 \[10, 15\): {reason}"):
                plan(rows, spans)
        with self.assertRaisesRegex(ValueError, r"fold 1 \[20, 25\): empty_test"):
            plan(windows=[TimeWindow(10, 15), TimeWindow(20, 25)])

    def test_invalid_input_does_not_mutate_containers(self):
        rows = ["warm", "test"]
        spans = [GroupSpan("warm", 0, 1, 1)]
        windows = [TimeWindow(10, 15)]
        before = copy.deepcopy((rows, spans, windows))
        with self.assertRaises(ValueError):
            plan(rows, spans, windows)
        self.assertEqual((rows, spans, windows), before)


if __name__ == "__main__":
    unittest.main()
