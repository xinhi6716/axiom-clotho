"""Synthetic resumed-device role checks; no observations or model fits."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from math import ceil
from unittest import TestCase
from unittest.mock import patch

from groupbench import splitting
from groupbench.splitting import (
    PhysicalDevice, RunDeviceAlias, RunDeviceBinding, SourceReadKey, preflight_device_roles,
)


class DeviceRolePreflightTests(TestCase):
    def setUp(self):
        self.aliases = (
            RunDeviceAlias("lab", "run-one", 1),
            RunDeviceAlias("lab", "run-two", "resumed"),
            RunDeviceAlias("lab", "run-two", 1),
            RunDeviceAlias("lab-2", "run-two", 1),
        )
        self.reads = tuple(SourceReadKey(alias, "t1") for alias in self.aliases)
        self.devices = tuple(PhysicalDevice("bench-ledger", i) for i in (1, 1, 2, 3))
        self.bindings = tuple(map(RunDeviceBinding, self.aliases, self.devices))

    def case(self, **overrides):
        kwargs = dict(train=(self.reads[2],), calibration=self.reads[:2],
                      evaluation=(self.reads[3],), bindings=self.bindings)
        return preflight_device_roles(**(kwargs | overrides))

    def test_all_roles_retain_read_identity_original_order_and_many_to_one_mapping(self):
        result = self.case(calibration=self.reads[1::-1], bindings=self.bindings[::-1])
        self.assertEqual(tuple(x.key for x in result.calibration), self.reads[1::-1])
        self.assertEqual(tuple(x.physical for x in result.calibration), (self.devices[0],) * 2)
        self.assertEqual(result.train[0].key, self.reads[2])
        self.assertEqual(result.evaluation[0].key, self.reads[3])
        # Same local observation ID exists in distinct source/run/device keys.
        self.assertEqual(len({x.key for x in (*result.train, *result.calibration, *result.evaluation)}), 4)
        with self.assertRaises(FrozenInstanceError):
            result.train[0].physical = self.devices[0]

    def test_same_run_same_device_different_reads_and_same_run_local_devices_are_legal(self):
        another_read = replace(self.reads[1], observation="t2")
        result = self.case(calibration=(self.reads[1], another_read, self.reads[0]))
        self.assertEqual(tuple(x.key.observation for x in result.calibration), ("t1", "t2", "t1"))
        self.assertEqual(len({x.physical for x in result.calibration}), 1)
        self.assertNotEqual(result.train[0].physical, result.calibration[0].physical)

    def test_every_role_pair_rejects_renamed_device_with_no_literal_overlap(self):
        for pair, role_data in (
            ("train and calibration", ((0, 2), (1,), (3,))),
            ("train and evaluation", ((0,), (2,), (1, 3))),
            ("calibration and evaluation", ((2,), (0,), (1, 3))),
        ):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, pair) as error:
                self.case(**dict(zip(("train", "calibration", "evaluation"),
                                    (tuple(self.reads[i] for i in row) for row in role_data))))
            self.assertNotIn("resumed", str(error.exception))
            self.assertNotIn("bench-ledger", str(error.exception))

    def test_duplicate_read_in_same_role_fails_instead_of_being_dropped(self):
        with self.assertRaisesRegex(ValueError, "source read key is repeated"):
            self.case(calibration=(*self.reads[:2], self.reads[0]))

    def test_binding_coverage_duplicate_conflicting_and_foreign(self):
        with self.assertRaisesRegex(ValueError, "has no physical binding"):
            self.case(bindings=self.bindings[1:])
        for second in (self.bindings[0], replace(self.bindings[0], physical=self.devices[2])):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "bindings repeat"):
                self.case(bindings=(self.bindings[0], second, *self.bindings[2:]),
                          calibration=(*self.reads[:2], replace(self.reads[0], observation=2)))
        extra = RunDeviceBinding(RunDeviceAlias("lab", "unused", 9), self.devices[0])
        with self.assertRaisesRegex(ValueError, "absent from the read roster"):
            self.case(bindings=(*self.bindings, extra),
                      calibration=(*self.reads[:2], replace(self.reads[0], observation=2)))

    def test_unknown_ids_and_typed_lookalikes_do_not_resolve(self):
        for bad in ("UNKNOWN", "unreported", None, True, 1.0, " x", "x\0y", 2**63):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "resolved canonical"):
                first = replace(self.bindings[0], physical=PhysicalDevice("ledger", bad))
                self.case(bindings=(first, *self.bindings[1:]))
        train_key = self.reads[2]
        with self.assertRaisesRegex(ValueError, "has no physical binding"):
            self.case(train=(replace(train_key, alias=replace(train_key.alias, raw_device="1")),))
        self.assertEqual(self.case(bindings=(self.bindings[0], self.bindings[1],
                                              replace(self.bindings[2], physical=PhysicalDevice("bench-ledger", "1")),
                                              self.bindings[3])).train[0].physical.device, "1")

    def test_roster_checks_bounds_before_work_and_no_implicit_iterator(self):
        for bad in (iter(self.reads), [], {"name": "record"}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "explicit list or tuple"):
                self.case(train=bad)
        with patch.object(splitting, "MAX_ROWS", 3), patch.object(
            splitting, "_run_alias", side_effect=AssertionError("Must bound roster first")
        ):
            with self.assertRaisesRegex(ValueError, "row limit"):
                self.case()

    def test_resumed_reads_change_group_max_support_not_observation_count(self):
        # Small independent exact-rank oracle; this does not call a calibrator.
        for n, resumed_score, expected in ((18, 2, None), (38, 100, 100)):
            with self.subTest(n=n):
                initial = tuple(SourceReadKey(RunDeviceAlias("lab", "initial", i), "t1")
                                for i in range(1, n + 1))
                resumed = SourceReadKey(RunDeviceAlias("lab", "resume", "renamed"), "t1")
                extra = tuple(RunDeviceBinding(key.alias, PhysicalDevice("bench-ledger", i))
                              for i, key in enumerate(initial, 1))
                result = self.case(calibration=(*initial, resumed), bindings=(
                    *extra, RunDeviceBinding(resumed.alias, extra[0].physical),
                    replace(self.bindings[2], physical=PhysicalDevice("bench-ledger", n + 1)),
                    replace(self.bindings[3], physical=PhysicalDevice("bench-ledger", n + 2)),
                ))
                maxima = {}
                for mapped, score in zip(result.calibration, (*range(1, n + 1), resumed_score)):
                    maxima[mapped.physical] = max(score, maxima.get(mapped.physical, 0))
                self.assertEqual(len(result.calibration), n + 1)
                self.assertEqual(len(maxima), n)
                rank = ceil(Fraction(19, 20) * (len(maxima) + 1))
                self.assertEqual(rank, 19 if n == 18 else 38)
                self.assertEqual(sorted(maxima.values())[rank - 1] if rank <= n else None, expected)
