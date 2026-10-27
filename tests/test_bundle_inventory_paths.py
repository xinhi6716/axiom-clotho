"""Bundle identities use portable slash keys, not host-native separators.

The Windows-flavour tests isolate pathlib's relative-path serialization. They
exercise real local bundle I/O, but are not a Windows operating-system run.
"""
import os
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest.mock import patch

from mlworkflow import contracts as c


class _WindowsInventoryEntry:
    def __init__(self, path):
        self.path = path

    def is_symlink(self):
        return self.path.is_symlink()

    def is_dir(self):
        return self.path.is_dir()

    def relative_to(self, root):
        return PureWindowsPath(*self.path.relative_to(root).parts)


class BundleInventoryPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, kind):
        path = self.root / kind
        payloads = {name: (name + "\n").encode("utf-8") for name in c.INVENTORIES[kind]}
        c.write_bundle(path, kind, payloads)
        return path, payloads

    def windows_entries(self):
        original = Path.rglob

        def rglob(path, pattern):
            for item in original(path, pattern):
                yield _WindowsInventoryEntry(item)

        return patch.object(Path, "rglob", rglob)

    def test_native_inventory_roundtrip_all_bundle_kinds(self):
        for kind in c.INVENTORIES:
            with self.subTest(kind=kind):
                path, expected = self.bundle(kind)
                actual, receipt_hash = c.load_bundle(path, kind)
                self.assertEqual(actual, expected)
                self.assertEqual(len(receipt_hash), 64)

    def test_windows_relative_path_inventory_roundtrip_all_bundle_kinds(self):
        for kind in c.INVENTORIES:
            with self.subTest(kind=kind):
                path, expected = self.bundle(kind)
                with self.windows_entries():
                    actual, receipt_hash = c.load_bundle(path, kind)
                self.assertEqual(actual, expected)
                self.assertEqual(len(receipt_hash), 64)

    def test_windows_relative_path_inventory_still_rejects_extra_file(self):
        path, _ = self.bundle("prepared")
        (path / "features" / "unexpected.csv").write_bytes(b"not in receipt\n")
        with self.windows_entries(), self.assertRaisesRegex(c.WorkflowError, "unexpected files"):
            c.load_bundle(path, "prepared")

    @unittest.skipUnless(os.name == "posix", "literal backslash is a POSIX filename")
    def test_posix_literal_backslash_is_not_normalized_into_directory(self):
        path, _ = self.bundle("prepared")
        (path / "features" / "features.csv").rename(path / "features\\features.csv")
        with self.assertRaisesRegex(c.WorkflowError, "unexpected files"):
            c.load_bundle(path, "prepared")


if __name__ == "__main__":
    unittest.main()
