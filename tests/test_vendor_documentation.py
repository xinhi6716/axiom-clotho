"""Pinned documentation completeness without changing upstream content."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from mlworkflow.components import ComponentError, RESOURCE_ROOT, verify_components


PIN = "4c0ec2ae3eaa4e0b7afc9aaf8f9bada544e7c0de"
DOCUMENTS = {
    "tools/iv-readpoint/README.md": "802cc8857b47bf1f1ba03cb2b03566d3d7313891",
    "tools/rram-window/README.md": "4770619c4db142248d6500eb3defb1b247b5f458",
    "docs/RRAM_DEVICE_CONDITION.md": "27ba38dd697eeb141dbe89daf95085d20db82116",
}
MEASUREMENT_README = "examples/measurement-to-features/README.md"


def git_blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


class VendorDocumentationTests(unittest.TestCase):
    def test_original_document_bytes_and_pin_are_preserved(self):
        component = next(c for c in verify_components()["components"] if c["component"] == "semiconductor")
        self.assertEqual(component["commit"], PIN)
        entries = {e["path"]: e for e in component["files"]}
        for path, expected in DOCUMENTS.items():
            with self.subTest(path=path):
                raw = (RESOURCE_ROOT / "semiconductor" / path).read_bytes()
                self.assertEqual(git_blob(raw), expected)
                self.assertEqual(entries[path]["git_blob_sha1"], expected)

    def test_existing_upstream_license_and_measurement_readme_are_unchanged(self):
        root = RESOURCE_ROOT / "semiconductor"
        self.assertEqual(git_blob((root / "LICENSE").read_bytes()),
                         "e30de29cdc65af0bc8a3d676c05262ddd60bff9b")
        self.assertEqual(git_blob((root / MEASUREMENT_README).read_bytes()),
                         "156634373c7d58055bee422500ef7d012c372555")

    def test_affected_upstream_relative_links_resolve(self):
        root = RESOURCE_ROOT / "semiconductor"
        checked = 0
        for name in [MEASUREMENT_README, *DOCUMENTS]:
            source = root / name
            text = re.sub(r"(?ms)^```.*?^```\s*$", "", source.read_text(encoding="utf-8"))
            for url in re.findall(r"\[[^\]]*\]\(([^\s)]+)", text):
                target = urlsplit(url)
                if target.scheme or target.netloc or not target.path:
                    continue
                with self.subTest(source=name, url=url):
                    self.assertTrue((source.parent / unquote(target.path)).is_file())
                checked += 1
        self.assertEqual(checked, 4)

    def test_first_party_reference_uses_bundled_output_contract(self):
        source = Path(__file__).resolve().parents[1] / "docs/USAGE_REFERENCE.md"
        text = source.read_text(encoding="utf-8")
        link = "../src/mlworkflow/_vendor/semiconductor/examples/measurement-to-features/README.md#outputs-and-failure-behavior"
        self.assertIn(
            "[bundled synthetic RRAM measurement-to-features example](" + link + ")", text)
        self.assertNotIn("semiconductor-machine-learning-toolbox/blob/b485b", text)
        target = (RESOURCE_ROOT / "semiconductor" / MEASUREMENT_README).read_text(encoding="utf-8")
        self.assertIn("## Outputs and failure behavior", target)
        self.assertIn("`feature_schema.json`", target)
        self.assertIn("empty target list", target)

    def test_missing_new_document_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "resources"
            shutil.copytree(RESOURCE_ROOT, root)
            for name in DOCUMENTS:
                path = root / "semiconductor" / name
                raw = path.read_bytes()
                path.unlink()
                with self.subTest(path=name), self.assertRaises(ComponentError):
                    verify_components(root)
                path.write_bytes(raw)

    def test_omitting_new_manifest_entry_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "resources"
            shutil.copytree(RESOURCE_ROOT, root)
            original = verify_components(root)
            for name in DOCUMENTS:
                manifest = copy.deepcopy(original)
                component = next(c for c in manifest["components"] if c["component"] == "semiconductor")
                component["files"] = [e for e in component["files"] if e["path"] != name]
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(path=name), self.assertRaises(ComponentError):
                    verify_components(root)


if __name__ == "__main__":
    unittest.main()
