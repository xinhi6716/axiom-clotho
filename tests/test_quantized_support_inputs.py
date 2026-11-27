from decimal import Decimal
from fractions import Fraction as F
import unittest

from groupbench.input_support import quantize_positive_operand as quantize


class QuantizedSupportInputsTest(unittest.TestCase):
    def test_decimal_halfway_and_zero_separate(self):
        step = F(Decimal("0.02"))
        self.assertIsNone(quantize(F("0.009999999999999999999999999999"), step))
        self.assertEqual(quantize(F("0.01"), step), F("0.02"))
        self.assertEqual(quantize(F("0.03"), step), F("0.04"))
        self.assertEqual(quantize(F("0.029999999999999999999999999999"), step), F("0.02"))

    def test_bounded_exhaustive_integer_grid_matches_nearest_distance(self):
        for denom in (3, 7, 21):
            for num in range(1, 7):
                step = F(num, denom)
                orig = [step * F(i, 14) for i in range(1, 171)]
                snapped = []
                for value in orig:
                    oracle = min((k * step for k in range(14)), key=lambda x: (abs(x - value), -x))
                    out = quantize(value, step)
                    self.assertEqual(out, oracle or None)
                    snapped.append(out if out is not None else F(0))
                self.assertEqual(snapped, sorted(snapped))

    def test_rescaling_and_domain(self):
        factor = F(10) ** 80
        for i in range(1, 30):
            x, step = F(i, 5), F(2)
            out = quantize(x, step)
            self.assertEqual(quantize(factor * x, factor * step), None if out is None else factor * out)
        for bad in (True, 1, 0.5, float("inf"), "0.1", Decimal("0.1")):
            with self.assertRaises(TypeError):
                quantize(bad, F(1))
            with self.assertRaises(TypeError):
                quantize(F(1), bad)
        for bad in (F(0), F(-1)):
            with self.assertRaises(ValueError):
                quantize(bad, F(1))
            with self.assertRaises(ValueError):
                quantize(F(1), bad)


class QuantizedChangeSignsTests(unittest.TestCase):
    def test_touching_cells_exclude_ties_and_mixed_grid_can_reverse(self):
        from groupbench.input_support import positive_quantized_change_signs as signs
        self.assertEqual(signs(F(1), F(2), first_step=F(1), second_step=F(1)), (1,))
        self.assertEqual(signs(F(2), F(1), first_step=F(1), second_step=F(1)), (-1,))
        self.assertEqual(signs(F(1), F(2), first_step=F(1), second_step=F(2)), (-1, 0, 1))
        self.assertEqual(signs(F(1), F(1), first_step=F(1), second_step=F(1)), (-1, 0, 1))

    def test_matches_direct_forward_quantizer_over_exact_rational_witnesses(self):
        from groupbench.input_support import positive_quantized_change_signs as signs
        from itertools import product

        steps = (F(1, 3), F(1, 2), F(1), F(2), F(3))
        values = [F(i, 48) for i in range(1, 720)]
        encoded = {
            step: {step * k: [x for x in values if quantize(x, step) == step * k]
                   for k in range(1, 5)}
            for step in steps
        }
        for first_step, second_step in product(steps, repeat=2):
            for (first, a), (second, b) in product(
                    encoded[first_step].items(), encoded[second_step].items()):
                with self.subTest(first=first, second=second, steps=(first_step, second_step)):
                    expected = tuple(sorted({(y > x) - (y < x) for x, y in product(a, b)}))
                    self.assertEqual(signs(first, second, first_step=first_step,
                                           second_step=second_step), expected)

    def test_unscorable_is_separate_and_doesnt_mask_malformed_other_read(self):
        from groupbench.input_support import positive_quantized_change_signs as signs
        self.assertIsNone(signs(None, F(1), first_step=F(1), second_step=F(1)))
        for center, error in ((0, TypeError), (F(0), ValueError), (F(1, 2), ValueError),
                              (F(-1), ValueError), (1.0, TypeError)):
            with self.subTest(center=center), self.assertRaises(error):
                signs(None, center, first_step=F(1), second_step=F(1))
        for step, error in ((1, TypeError), (F(0), ValueError), (F(-1), ValueError)):
            with self.subTest(step=step), self.assertRaises(error):
                signs(None, None, first_step=F(1), second_step=step)


class IntervalGroupSupportTests(unittest.TestCase):
    def test_disjoint_range_touches_are_not_simultaneous(self):
        from groupbench.input_support import audit_numeric_interval_group_support as audit
        result = audit([0, 1, 2, 3], [(0, 3), (1, 2)], training_groups=[0, 0, 1, 1],
                       minimum_training_groups=2)
        self.assertEqual(result["n_queries_meeting_threshold_nowhere"], 2)
        result = audit([0, 1, 1, 3], [(0, 3), (1, 1)], training_groups=[0, 0, 1, 1],
                       minimum_training_groups=2)
        self.assertEqual(result["n_queries_meeting_threshold_only_somewhere"], 1)
        self.assertEqual(result["n_queries_meeting_threshold_everywhere"], 1)

    def test_sweep_matches_direct_point_and_gap_witnesses(self):
        from itertools import combinations_with_replacement, product
        from groupbench.input_support import audit_numeric_interval_group_support as audit
        groups = [0, 0, 1, 1, 2, 2]
        source = tuple(combinations_with_replacement(range(3), 2))
        queries = tuple(combinations_with_replacement((-1., 0., .5, 1., 2., 3.), 2))
        keys = ["n_queries_meeting_threshold_" + s for s in ("everywhere", "only_somewhere", "nowhere")]
        for roster in product(source, repeat=3):
            training = [x for interval in roster for x in interval]
            for k in (1, 2, 3, 4):
                expected = [0, 0, 0]
                for lo, hi in queries:
                    knots = sorted({lo, hi, *(x for x in training if lo <= x <= hi)})
                    points = knots + [(a + b) / 2 for a, b in zip(knots, knots[1:])]
                    hit = [sum(a <= x <= b for a, b in roster) >= k for x in points]
                    expected[0 if all(hit) else 1 if any(hit) else 2] += 1
                result = audit(training, queries, training_groups=groups, minimum_training_groups=k)
                self.assertEqual([result[field] for field in keys], expected)

    def test_adjacent_binary64_singletons_do_not_cover_the_open_interval(self):
        from math import nextafter
        from groupbench.input_support import audit_numeric_interval_group_support as audit
        b = nextafter(1., 2.)
        result = audit([1., b], [(1., b), (1., 1.)], training_groups=[0, 1],
                       minimum_training_groups=1)
        self.assertEqual(result["n_queries_meeting_threshold_only_somewhere"], 1)
        self.assertEqual(result["n_queries_meeting_threshold_everywhere"], 1)
        for interval in ((2., 1.), (False, 1.), [0., 1.], (float("nan"), 1.)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                audit([1.], [interval], training_groups=[0], minimum_training_groups=1)


class ConformalScoreTieAuditTests(unittest.TestCase):
    def test_matched_discrete_encoder_changes_attainable_five_percent_boundary(self):
        from groupbench.input_support import audit_conformal_score_ties as audit
        raw, encoded = 0, 0
        incorrect_extra = 0
        pooled = tuple(range(40))
        for held in pooled:
            train = [x for x in pooled if x != held]
            old = audit(train, [held], level=F(1, 20))
            new = audit([4 * (x // 4) for x in train], [4 * (held // 4)], level=F(1, 20))
            raw += old["n_conservative_alerts"]
            encoded += new["n_conservative_alerts"]
            incorrect_extra += new["n_extra_alerts_if_ties_wrongly_dropped"]
            self.assertEqual(new["n_calibration_scores"], 39)
            self.assertEqual(new["max_calibration_ties_for_one_query"], 3)
        self.assertEqual((raw, encoded, incorrect_extra), (2, 0, 4))

    def test_exact_inclusive_ranks_and_multiplicity_match_direct_fraction_counts(self):
        from itertools import product
        from groupbench.input_support import audit_conformal_score_ties as audit
        queries = [-2., -1., 0., 1., 2.]
        for calibration in product((-1., 0., 1.), repeat=4):
            for level in (F(1, 10), F(1, 5), F(2, 5), F(4, 5)):
                n = len(calibration) + 1
                outcomes = []
                for score in queries:
                    tie = sum(x == score for x in calibration)
                    inclusive = F(1 + sum(x >= score for x in calibration), n) <= level
                    strict = F(1 + sum(x > score for x in calibration), n) <= level
                    outcomes.append((tie, inclusive, strict and not inclusive))
                actual = audit(calibration, queries, level=level)
                self.assertEqual(actual["any_alert_attainable_at_level"], F(1, n) <= level)
                self.assertEqual(actual["n_queries_tied_with_calibration"], sum(bool(a) for a, _, _ in outcomes))
                self.assertEqual(actual["n_calibration_query_equal_pairs"], sum(a for a, _, _ in outcomes))
                self.assertEqual(actual["max_calibration_ties_for_one_query"], max(a for a, _, _ in outcomes))
                self.assertEqual(actual["n_conservative_alerts"], sum(b for _, b, _ in outcomes))
                self.assertEqual(actual["n_extra_alerts_if_ties_wrongly_dropped"], sum(c for _, _, c in outcomes))
                self.assertEqual(audit(calibration[::-1], queries[::-1], level=level), actual)

    def test_rejects_bad_boundaries_and_never_deduplicates_scores(self):
        from groupbench.input_support import audit_conformal_score_ties as audit
        actual = audit([-0., 0., 0.], [0., -0., 1.], level=F(1, 4))
        self.assertEqual((actual["n_queries_tied_with_calibration"], actual["n_calibration_query_equal_pairs"],
                          actual["n_conservative_alerts"], actual["n_extra_alerts_if_ties_wrongly_dropped"]),
                         (2, 6, 1, 2))
        for level in (F(0), F(1), F(2)):
            with self.assertRaises(ValueError):
                audit([0.], [0.], level=level)
        with self.assertRaises(TypeError):
            audit([0.], [0.], level=.05)
        for bad in (float("nan"), float("inf"), True, 2**53 + 1):
            with self.assertRaises(ValueError):
                audit([0.], [bad], level=F(1, 2))
