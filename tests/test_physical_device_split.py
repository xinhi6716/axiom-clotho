"""Declared device aliases in split diagnostics, independent of features and targets."""

from unittest import TestCase

import numpy as np

from groupbench.splitting import (
    DeviceAlias, DeviceAliasBinding, PhysicalDevice, audit_physical_device_split, audit_split,
    DeviceParentBinding, PhysicalParent, audit_physical_parent_split,
)


class PhysicalDeviceSplitTests(TestCase):
    def setUp(self):
        self.original = DeviceAlias("lab-A", "read-1")
        self.renamed = DeviceAlias("lab-B", "read-renamed")
        self.held = DeviceAlias("lab-B", "read-1")  # same literal label as original, other source
        self.a = PhysicalDevice("device-registry", 1)
        self.b = PhysicalDevice("device-registry", "1")  # distinct from integer 1
        self.rows = [self.original, self.held, self.renamed, self.original]
        self.bindings = [
            DeviceAliasBinding(self.original, self.a),
            DeviceAliasBinding(self.held, self.b),
            DeviceAliasBinding(self.renamed, self.a),
        ]

    def audit(self, train=(0, 3), test=(1,), **overrides):
        row_aliases = overrides.pop("row_aliases", self.rows)
        bindings = overrides.pop("bindings", self.bindings)
        return audit_physical_device_split(train, test, row_aliases, bindings=bindings, **overrides)

    def test_alias_rename_can_pass_ordinary_group_check_but_is_not_a_held_device(self):
        train, test = np.array([0, 3]), np.array([2])
        ordinary = audit_split(train, test, np.array(["original", "true-held", "renamed", "original"]))
        self.assertEqual(ordinary["overlap_count"], 0)
        with self.assertRaisesRegex(ValueError, "1 declared physical device"):
            self.audit(train, test)

    def test_same_literal_alias_different_source_and_typed_devices_do_not_merge(self):
        expected = {
            "train_rows": 2, "test_rows": 1, "train_alias_count": 1, "test_alias_count": 1,
            "train_physical_device_count": 1, "test_physical_device_count": 1,
            "unused_rows": 1, "physical_device_overlap_count": 0,
        }
        self.assertEqual(self.audit(), expected)
        self.assertEqual(self.audit(bindings=tuple(reversed(self.bindings))), expected)

    def test_multiple_aliases_on_same_side_count_as_one_device(self):
        result = self.audit(train=(3, 2, 0), test=(1,))
        self.assertEqual(result["train_rows"], 3)
        self.assertEqual(result["train_alias_count"], 2)
        self.assertEqual(result["train_physical_device_count"], 1)
        self.assertEqual(result["unused_rows"], 0)

    def test_unused_alias_binding_still_required_and_extra_refused(self):
        for bindings in (self.bindings[:2],
                         self.bindings + [DeviceAliasBinding(DeviceAlias("lab-B", "extra"), self.a)]):
            with self.subTest(bindings=bindings), self.assertRaisesRegex(ValueError, "exactly, with no extras"):
                self.audit(bindings=bindings)

    def test_duplicate_and_conflicting_bindings_are_both_rejected(self):
        for repeat in (self.bindings[0], DeviceAliasBinding(self.original, self.b)):
            with self.subTest(repeat=repeat), self.assertRaisesRegex(ValueError, "repeat a source alias"):
                self.audit(bindings=self.bindings + [repeat])

    def test_compound_key_does_not_delimiter_collide_and_identity_namespace_is_literal(self):
        roster = (DeviceAlias("source:one", "two"), DeviceAlias("source", "one:two"))
        bindings = (
            DeviceAliasBinding(roster[0], PhysicalDevice("registry-1", "01")),
            DeviceAliasBinding(roster[1], PhysicalDevice("registry-2", "01")),
        )
        self.assertEqual(self.audit((0,), (1,), row_aliases=roster, bindings=bindings)["physical_device_overlap_count"], 0)

    def test_integers_and_strings_distinct_for_aliases_and_boolean_cannot_masquerade(self):
        roster = (DeviceAlias("source", 1), DeviceAlias("source", "1"))
        bindings = (DeviceAliasBinding(roster[0], self.a), DeviceAliasBinding(roster[1], self.b))
        self.assertEqual(self.audit((0,), (1,), row_aliases=roster, bindings=bindings)["train_alias_count"], 1)
        with self.assertRaisesRegex(ValueError, "aliases require"):
            self.audit(row_aliases=(DeviceAlias("source", True), *self.rows[1:]))
        with self.assertRaisesRegex(ValueError, "devices require"):
            self.audit(bindings=[DeviceAliasBinding(self.original, PhysicalDevice("registry", True)), *self.bindings[1:]])

    def test_invalid_types_whitespace_and_bounds_do_not_coerce_into_valid_identity(self):
        for bad in ("", " source", "source ", "   ", "x" * 257, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "aliases require"):
                self.audit(row_aliases=(DeviceAlias(bad, "id"), *self.rows[1:]))
        for bad in ("", " ", 2**63, -(2**63)-1, 1.0, np.int64(1)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "devices require"):
                self.audit(bindings=[DeviceAliasBinding(self.original, PhysicalDevice("registry", bad)), *self.bindings[1:]])
        with self.assertRaisesRegex(ValueError, "explicit list or tuple"):
            self.audit(bindings=dict((b.alias, b.physical) for b in self.bindings))

    def test_positional_validation_bounds_before_array_conversion(self):
        class PretendVector:
            def __array__(self):
                raise AssertionError("Duck-typed conversion is out of scope")

        for indices in (PretendVector(), np.array(0), np.empty((0, 20))):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "integer row positions"):
                self.audit(train=indices)
        with self.assertRaisesRegex(ValueError, "exceeds the row roster size"):
            self.audit(train=(0, 0, 0, 0, 0))

    def test_positional_validation_rejects_ambiguous_overlap_and_no_implicit_index_wrap(self):
        for train, diagnostic in (
            ((True,), "integer row positions"),
            (np.ma.array([0], mask=[True]), "integer row positions"),
            (np.array([2**64-1], dtype=np.uint64), "outside group row bounds"),
            ((0, 0), "duplicate row positions"),
            ((-1,), "outside group row bounds"),
            ((1,), "row positions overlap"),
        ):
            with self.subTest(train=train), self.assertRaisesRegex(ValueError, diagnostic):
                self.audit(train=train)


class PhysicalParentSplitTests(TestCase):
    def setUp(self):
        self.aliases = [DeviceAlias("lab", i) for i in range(4)]
        self.devices = [PhysicalDevice("devices", i) for i in range(3)]
        self.bindings = [DeviceAliasBinding(a, self.devices[i])
                         for a, i in zip(self.aliases, [0, 1, 0, 2])]
        self.parents = [DeviceParentBinding(d, PhysicalParent("lot-9", p))
                        for d, p in zip(self.devices, [1, 1, '1'])]

    def audit(self, train=(0, 2), test=(3,), parent_bindings=None, parent_level="wafer"):
        return audit_physical_parent_split(
            train, test, self.aliases, bindings=self.bindings, parent_level=parent_level,
            parent_bindings=self.parents if parent_bindings is None else parent_bindings)

    def test_distinct_devices_can_share_a_parent_and_leak(self):
        old = audit_physical_device_split((0, 2), (1,), self.aliases, bindings=self.bindings)
        self.assertEqual(old["physical_device_overlap_count"], 0)
        with self.assertRaisesRegex(ValueError, '1 declared physical parent'):
            self.audit(test=(1,))
        with self.assertRaisesRegex(ValueError, '1 declared physical parent'):
            self.audit(train=(0,), test=(2,))

    def test_typed_scoped_parent_ids_and_unused_rows(self):
        self.assertEqual(self.audit(), {
            'train_rows': 2, 'test_rows': 1, 'train_physical_device_count': 1,
            'test_physical_device_count': 1, 'train_physical_parent_count': 1,
            'test_physical_parent_count': 1, 'unused_rows': 1,
            'physical_parent_overlap_count': 0,
        })
        other = DeviceParentBinding(self.devices[1], PhysicalParent("lot-10", 1))
        self.assertEqual(self.audit(test=(1,), parent_bindings=[*self.parents[:1], other, self.parents[2]])
                         ['test_physical_parent_count'], 1)

    def test_full_exact_bindings_and_explicit_parent_level(self):
        for bindings in (self.parents[1:], [self.parents[0], self.parents[2]], [*self.parents, self.parents[0]],
                         [*self.parents, DeviceParentBinding(PhysicalDevice("other", 2),
                                                            PhysicalParent("lot-9", 1))],
                         [DeviceParentBinding(self.devices[0], PhysicalParent("lot-9", "unknown")),
                          *self.parents[1:]],
                         [DeviceParentBinding(self.devices[0], PhysicalParent(42, 1)),
                          *self.parents[1:]]):
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                self.audit(parent_bindings=bindings)
        for level in (None, "", "unknown", "wafer "):
            with self.assertRaisesRegex(ValueError, 'level'):
                self.audit(parent_level=level)
