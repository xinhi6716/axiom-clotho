"""Versioned FET inventories retain fail-closed and portable bundle semantics."""
import copy
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
import unittest
from unittest.mock import patch

from mlworkflow import contracts as c
from mlworkflow.adapters import FET_INVENTORIES, write_bundle, load_bundle
from mlworkflow.components import RESOURCE_ROOT, ComponentError, verify_components


class _WindowsInventoryEntry:
    def __init__(self, path):
        self.path = path
    def is_symlink(self):
        return self.path.is_symlink()
    def is_dir(self):
        return self.path.is_dir()
    def relative_to(self, root):
        return PureWindowsPath(*self.path.relative_to(root).parts)


class FetBundleVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, kind):
        path = self.root / kind
        data = {name: (name + "\n").encode() for name in FET_INVENTORIES[kind]}
        write_bundle(path, kind, data)
        return path, data

    def test_all_v2_bundle_inventories_roundtrip_without_v1_relaxation(self):
        for kind in FET_INVENTORIES:
            path, expected = self.bundle(kind)
            actual, digest = load_bundle(path, kind)
            self.assertEqual(actual, expected)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(c.WorkflowError):
                c.load_bundle(path, kind)

    def test_windows_flavour_receipt_slashes_all_kinds(self):
        rglob = Path.rglob
        def windows(path, pattern):
            return (_WindowsInventoryEntry(item) for item in rglob(path, pattern))
        for kind in FET_INVENTORIES:
            path, expected = self.bundle(kind)
            with patch.object(Path, "rglob", windows):
                self.assertEqual(load_bundle(path, kind)[0], expected)

    def test_v1_bundle_is_not_reinterpreted_as_v2(self):
        path = self.root / "legacy"
        c.write_bundle(path, "score", {n: b"legacy" for n in c.INVENTORIES["score"]})
        with self.assertRaises(c.WorkflowError):
            load_bundle(path, "score")
        self.assertEqual(set(c.load_bundle(path, "score")[0]), c.INVENTORIES["score"])

    def test_incomplete_shortened_and_unknown_adapter_receipts_reject(self):
        path, _ = self.bundle("prepared")
        original = c.parse_json((path / "COMPLETE.json").read_bytes())
        for modify in (lambda x: x.update(adapter="unknown"), lambda x: x.update(files_sha256={}),
                       lambda x: x.update(kind="run"), lambda x: x.update(integrity_not_authenticity=False)):
            value = copy.deepcopy(original)
            modify(value)
            (path / "COMPLETE.json").write_bytes(c.json_bytes(value))
            with self.assertRaises(c.WorkflowError):
                load_bundle(path, "prepared")
        (path / "COMPLETE.json").write_bytes(c.json_bytes(original))
        (path / "INCOMPLETE.json").write_bytes(b"{}")
        with self.assertRaises(c.WorkflowError):
            load_bundle(path, "prepared")

    def test_new_fet_vendor_file_missing_changed_or_manifest_shortened_rejects(self):
        resources = self.root / "vendor"
        shutil.copytree(RESOURCE_ROOT, resources)
        path = resources / "semiconductor/tools/transfer-log-slope/transfer_log_slope.py"
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaises(ComponentError):
            verify_components(resources)
        path.unlink()
        with self.assertRaises(ComponentError):
            verify_components(resources)
        path.write_bytes(original)
        manifest = c.parse_json((resources / "manifest.json").read_bytes())
        manifest["components"][0]["files"] = [f for f in manifest["components"][0]["files"] if f["path"] != "tools/transfer-log-slope/transfer_log_slope.py"]
        (resources / "manifest.json").write_bytes(c.json_bytes(manifest))
        with self.assertRaises(ComponentError):
            verify_components(resources)

    @unittest.skipUnless(os.name == "posix", "literal backslash is a POSIX filename")
    def test_literal_backslash_cannot_alias_feature_subdirectory(self):
        path, _ = self.bundle("prepared")
        (path / "features/features.csv").rename(path / "features\\features.csv")
        with self.assertRaises(c.WorkflowError):
            load_bundle(path, "prepared")


if __name__ == "__main__":
    unittest.main()
