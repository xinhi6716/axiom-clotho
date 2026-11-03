"""Package-source observations: bounded local bytes, never an attestation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench import benchmark, benchmark_binary, fit_predict, fit_predict_binary, validate_evaluation_bundle
from groupbench import package_source as source
from groupbench.prediction_schema import validate_prediction_bundle_schema
from groupbench.verification import verify_bundle
from test_report_reading_path import fixture as regression_fixture
from test_binary_report_reading_path import fixture as binary_fixture
from groupbench.report import render_html
from groupbench.binary.report import render_binary_html


@unittest.skipUnless(source._SAFE_OPEN_SUPPORTED, "Requires POSIX no-follow source observation")
class PackageSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "package"
        self._populate(self.root)
        self.patch = patch.object(source, "_PACKAGE_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def _populate(self, root):
        for name in source.SOURCE_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("# synthetic owned source: " + name + "\n").encode())

    def assert_unavailable(self, observed):
        self.assertEqual(observed["status"], "unavailable")
        self.assertNotIn("files", observed)
        self.assertNotIn("sha256", observed)
        self.assertNotIn(str(self.root), json.dumps(observed))

    def test_fixed_inventory_covers_owned_python_files_including_self(self):
        installed = Path(source.__file__).parent
        self.assertEqual(tuple(sorted(source.SOURCE_FILES)), source.SOURCE_FILES)
        self.assertEqual(set(source.SOURCE_FILES), {p.relative_to(installed).as_posix() for p in installed.rglob("*.py")})
        self.assertEqual(len(set(source.SOURCE_FILES)), 37)
        self.assertIn("baselines.py", source.SOURCE_FILES)
        self.assertIn("kernel.py", source.SOURCE_FILES)
        self.assertIn("binary/saved_evaluation.py", source.SOURCE_FILES)
        self.assertIn("saved_scoring.py", source.SOURCE_FILES)
        self.assertIn("quantity_contract.py", source.SOURCE_FILES)
        self.assertIn("domain_selection.py", source.SOURCE_FILES)
        self.assertIn("group_time_evaluation.py", source.SOURCE_FILES)
        self.assertIn("group_time_json.py", source.SOURCE_FILES)
        self.assertIn("group_time_plan.py", source.SOURCE_FILES)

    def test_canonical_digest_is_explicit_and_stable(self):
        observed = source.observe_package_source()
        self.assertEqual(observed["status"], "available")
        expected_files = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in source.SOURCE_FILES}
        expected_record = {"schema_version": "1.0", "algorithm": "sha256",
                           "scope": "fixed_groupbench_python_sources_v1", "files": expected_files}
        canonical = json.dumps(expected_record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self.assertEqual(observed["sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(observed["files"], expected_files)
        self.assertEqual(source.observe_package_source(), observed)
        self.assertNotIn(str(self.root), json.dumps(observed))

    def test_copied_package_and_different_cwd_keep_identity_without_scan(self):
        expected = source.observe_package_source()
        other = Path(self.temp.name) / "unrelated-cwd"
        other.mkdir()
        (other / ".git").mkdir()
        (other / "private.csv").write_text("must not be read")
        cwd = Path.cwd()
        try:
            os.chdir(other)
            with patch.object(Path, "rglob", side_effect=AssertionError("production scan forbidden")), \
                 patch.object(Path, "glob", side_effect=AssertionError("production scan forbidden")):
                self.assertEqual(source.observe_package_source(), expected)
        finally:
            os.chdir(cwd)
        copied = Path(self.temp.name) / "copied-package"
        self._populate(copied)
        with patch.object(source, "_PACKAGE_ROOT", copied):
            self.assertEqual(source.observe_package_source(), expected)

    def test_one_source_change_changes_digest_and_unlisted_files_are_not_read(self):
        before = source.observe_package_source()
        (self.root / "unlisted.py").symlink_to(self.root / "missing-secret")
        (self.root / "data.csv").write_text("unrelated data")
        self.assertEqual(source.observe_package_source(), before)
        (self.root / "binary/report.py").write_bytes(b"# modified synthetic renderer\n")
        after = source.observe_package_source()
        self.assertNotEqual(after["sha256"], before["sha256"])
        self.assertEqual([n for n in source.SOURCE_FILES if before["files"][n] != after["files"][n]], ["binary/report.py"])

    def test_missing_oversized_and_symlink_source_are_unavailable(self):
        path = self.root / "__init__.py"
        original = path.read_bytes()
        path.unlink()
        self.assert_unavailable(source.observe_package_source())
        with path.open("wb") as stream:
            stream.truncate(source.MAX_SOURCE_FILE_BYTES + 1)
        self.assert_unavailable(source.observe_package_source())
        path.unlink()
        outside = Path(self.temp.name) / "outside-private.py"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with patch.object(source, "_read_file", wraps=source._read_file) as reader:
            self.assert_unavailable(source.observe_package_source())
            self.assertEqual(reader.call_args.args[1], "__init__.py")

    def test_symlink_package_or_nested_directory_never_followed(self):
        alias = Path(self.temp.name) / "package-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with patch.object(source, "_PACKAGE_ROOT", alias):
            self.assert_unavailable(source.observe_package_source())
        (self.root / "binary").rename(self.root / "old-binary")
        (self.root / "binary").symlink_to(self.root / "old-binary", target_is_directory=True)
        self.assert_unavailable(source.observe_package_source())

    def test_special_file_permission_error_and_change_are_unavailable(self):
        path = self.root / "__init__.py"
        path.unlink()
        os.mkfifo(path)
        self.assert_unavailable(source.observe_package_source())
        path.unlink()
        path.write_bytes(b"# synthetic\n")
        with patch.object(source, "_read_file", side_effect=PermissionError(str(self.root / "private"))):
            self.assert_unavailable(source.observe_package_source())
        real_read = source._read_file
        def read_and_change(fd, name, limit):
            result = real_read(fd, name, limit)
            if name == "__init__.py":
                path.write_bytes(b"# synthetic changed after reading\n")
            return result
        with patch.object(source, "_read_file", side_effect=read_and_change):
            self.assert_unavailable(source.observe_package_source())

    def test_read_limits_and_no_source_content_collection(self):
        with patch.object(source, "_read_file", wraps=source._read_file) as reader:
            self.assertEqual(source.observe_package_source()["status"], "available")
        self.assertEqual(reader.call_count, len(source.SOURCE_FILES))
        self.assertTrue(all(call.args[2] == source.MAX_SOURCE_FILE_BYTES and not call.kwargs for call in reader.call_args_list))

    def _observe_with_directory_swap(self, *, replace_root=False, symlink_child=False):
        real_read = source._read_file
        seen = 0
        def read_and_swap(fd, name, limit):
            nonlocal seen
            result = real_read(fd, name, limit)
            if name == "__init__.py":
                seen += 1
                if seen == 2:  # binary/__init__.py, after its directory is pinned
                    if replace_root:
                        self.root.rename(self.root.with_name("old-package"))
                        self._populate(self.root)
                    else:
                        old = self.root / "old-binary"
                        (self.root / "binary").rename(old)
                        if symlink_child:
                            (self.root / "binary").symlink_to(old, target_is_directory=True)
                        else:
                            (self.root / "binary").mkdir()
                            for relative in source.SOURCE_FILES:
                                if relative.startswith("binary/"):
                                    (self.root / relative).write_bytes(b"# replacement source\n")
            return result
        with patch.object(source, "_read_file", side_effect=read_and_swap):
            self.assert_unavailable(source.observe_package_source())

    def test_child_directory_replaced_after_open_is_unavailable(self):
        self._observe_with_directory_swap()

    def test_root_directory_replaced_after_open_is_unavailable(self):
        self._observe_with_directory_swap(replace_root=True)

    def test_child_directory_symlinked_after_open_is_unavailable(self):
        self._observe_with_directory_swap(symlink_child=True)


class SourceFallbackAndRenderingTests(unittest.TestCase):
    @unittest.skipUnless(source._SAFE_OPEN_SUPPORTED, "Requires POSIX bundle verification")
    def test_legacy_inventory_and_absent_metadata_remain_read_only_compatible(self):
        # Original synthetic saved artifacts: no estimator fit or old-study replay.
        from test_prediction_schema import manifest, write_bundle

        descriptors = [None]
        for omitted in ({"baselines.py", "kernel.py", "binary/saved_evaluation.py", "saved_scoring.py", "quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"kernel.py", "binary/saved_evaluation.py", "saved_scoring.py", "quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"binary/saved_evaluation.py", "saved_scoring.py", "quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"saved_scoring.py", "quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"},
                        {"group_time_json.py", "group_time_evaluation.py"}, {"group_time_evaluation.py"}, set()):
            names = [name for name in source.SOURCE_FILES
                     if name not in omitted]
            files = {name: hashlib.sha256(("invented source: " + name).encode()).hexdigest()
                     for name in names}
            body = {"schema_version": "1.0", "algorithm": "sha256",
                    "scope": "fixed_groupbench_python_sources_v1", "files": files}
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            descriptors.append({**body, "status": "available",
                                "sha256": hashlib.sha256(canonical).hexdigest()})
        self.assertEqual([len(item["files"]) for item in descriptors[1:]], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
        with tempfile.TemporaryDirectory() as temp, patch.object(source, "observe_package_source", side_effect=AssertionError("saved verification must not recapture installed source")):
            for version in ("1.0", "1.1", "1.2"):
                for index, descriptor in enumerate(descriptors):
                    with self.subTest(schema=version, inventory=index):
                        root = Path(temp) / (version + "-" + str(index))
                        root.mkdir()
                        value = manifest(version)
                        if descriptor is not None:
                            value["package_source"] = descriptor
                        write_bundle(root, value)
                        before = {p.name: p.read_bytes() for p in root.iterdir()}
                        self.assertEqual(verify_bundle(root).bundle_type, "prediction")
                        checked = validate_prediction_bundle_schema(root)
                        self.assertEqual(checked.schema_version, version)
                        self.assertEqual(checked.unchecked_metadata, "other_manifest_fields_not_validated")
                        self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_unsupported_and_relative_roots_are_unavailable_without_open(self):
        for changes in ({"_SAFE_OPEN_SUPPORTED": False}, {"_PACKAGE_ROOT": Path("relative-package")}):
            with patch.multiple(source, **changes), patch.object(source, "_open_run_directory") as opening:
                observed = source.observe_package_source()
                self.assertEqual(observed["status"], "unavailable")
                self.assertEqual(observed["reason"], "unsupported_source_access")
                opening.assert_not_called()

    def test_reports_show_recorded_digest_or_legacy_status_without_reading_source(self):
        for fixture, renderer in ((regression_fixture, render_html), (binary_fixture, render_binary_html)):
            result = fixture()
            with patch.object(source, "observe_package_source", side_effect=AssertionError("renderer must not recapture")):
                self.assertIn("not recorded", renderer(result))
                result.manifest["package_source"] = {"status": "available", "sha256": "b" * 64}
                before = copy.deepcopy(result.manifest)
                text = renderer(result)
                self.assertIn("b" * 64, text)
                self.assertIn("Self-reported source bytes", text)
                self.assertIn("checksums do not authenticate", text)
                self.assertEqual(before, result.manifest)
                result.manifest["package_source"] = {"status": "available", "sha256": '<script>private/path</script>'}
                text = renderer(result)
                self.assertIn("unavailable", text)
                self.assertNotIn("private/path", text)
                self.assertNotIn("<script>", text)
                result.manifest["package_source"] = {"status": "unavailable", "reason": "private/path"}
                self.assertNotIn("private/path", renderer(result))
        text = render_binary_html(binary_fixture())
        self.assertIn("未記錄", text)
        self.assertIn("僅為自行記錄的原始碼位元組", text)


class ResultSourceMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.DataFrame({"x": [float(i) for i in range(24)],
                                  "group": [f"synthetic-{i // 4}" for i in range(24)],
                                  "y": [float(i % 7) for i in range(24)]})
        cls.binary = cls.frame.assign(y=[i % 2 for i in range(24)])

    def test_four_routes_add_metadata_without_numerical_or_contract_changes(self):
        options = dict(target="y", group="group", inner_splits=2, baselines=("dummy",))
        routes = (
            ("groupbench.evaluation", lambda: benchmark(self.frame, outer_splits=3, **options)),
            ("groupbench.prediction", lambda: fit_predict(self.frame, self.frame[["x"]].head(2), **options)),
            ("groupbench.binary.workflow", lambda: benchmark_binary(self.binary, outer_splits=3, **options)),
            ("groupbench.binary.workflow", lambda: fit_predict_binary(self.binary, self.binary[["x"]].head(2), **options)),
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, (module, action) in enumerate(routes):
                with self.subTest(route=index):
                    first = action()
                    observed = first.manifest["package_source"]
                    self.assertIn(observed["status"], ("available", "unavailable"))
                    unavailable = {"status": "unavailable", "reason": "synthetic-test"}
                    with patch(module + ".observe_package_source", return_value=unavailable) as capture:
                        second = action()
                    capture.assert_called_once_with()
                    self.assertEqual(second.manifest["package_source"], unavailable)
                    pd.testing.assert_frame_equal(first.predictions, second.predictions)
                    if hasattr(first, "summary"):
                        self.assertEqual(first.summary, second.summary)
                    omit = {"package_source", "created_at_utc", "elapsed_seconds"}
                    self.assertEqual({k: v for k, v in first.manifest.items() if k not in omit},
                                     {k: v for k, v in second.manifest.items() if k not in omit})
                    output = Path(temp) / str(index)
                    first.write(output)
                    if source._SAFE_OPEN_SUPPORTED:
                        verify_bundle(output)
                        if index == 1:
                            checked = validate_prediction_bundle_schema(output)
                            self.assertEqual(checked.unchecked_metadata, "other_manifest_fields_not_validated")
                        # Reuse the same already-computed tiny software fixture;
                        # these additional saved-artifact checks perform no fit.
                        for legacy_mode in ("inventory_28", "no_source_metadata"):
                            saved = copy.deepcopy(first)
                            if legacy_mode == "no_source_metadata":
                                saved.manifest.pop("package_source", None)
                            else:
                                files = {name: hashlib.sha256(("invented legacy source: " + name).encode()).hexdigest()
                                         for name in source.SOURCE_FILES if name not in {"baselines.py", "kernel.py", "binary/saved_evaluation.py", "saved_scoring.py", "quantity_contract.py", "domain_selection.py", "group_time_plan.py", "group_time_json.py", "group_time_evaluation.py"}}
                                self.assertEqual(len(files), 28)
                                body = {"schema_version": "1.0", "algorithm": "sha256",
                                        "scope": "fixed_groupbench_python_sources_v1", "files": files}
                                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
                                saved.manifest["package_source"] = {**body, "status": "available",
                                                                    "sha256": hashlib.sha256(canonical).hexdigest()}
                            legacy_output = Path(temp) / f"{index}-{legacy_mode}"
                            saved.write(legacy_output)
                            before = {p.name: p.read_bytes() for p in legacy_output.iterdir()}
                            with patch.object(source, "observe_package_source", side_effect=AssertionError("no installed-source recapture")):
                                verify_bundle(legacy_output)
                                if index == 0:
                                    validate_evaluation_bundle(legacy_output)
                                elif index == 1:
                                    validate_prediction_bundle_schema(legacy_output)
                            self.assertEqual(before, {p.name: p.read_bytes() for p in legacy_output.iterdir()})


if __name__ == "__main__":
    unittest.main()
