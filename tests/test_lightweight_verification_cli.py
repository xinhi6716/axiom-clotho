"""Fresh-process byte verification contracts: no ML, fits or model payloads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from groupbench import verification

SOURCE = str(Path(__file__).resolve().parents[1] / "src")
# Exact help from the pre-change CLI at tree cda0981885701e71b2388936e5d0d192fa47c712.
HELP_SHA256 = "db2fb290c1731f6e632c609c01bd24eaafa8aff25ca70d0963ba82f968108325"
ROUTES = (
    ("console", "mlcalc", "groupbench.cli:main", []),
    ("console", "groupbench", "groupbench.cli:main", []),
    ("console", "clotho", "mlworkflow.cli:main", ["tabular"]),
    ("console", "mlcalc-workflow", "mlworkflow.cli:main", ["tabular"]),
    ("module", "groupbench", "groupbench", []),
    ("module", "groupbench.cli", "groupbench.cli", []),
    ("module", "mlworkflow", "mlworkflow", ["tabular"]),
    ("module", "mlworkflow.cli", "mlworkflow.cli", ["tabular"]),
)
INVENTORIES = {
    "prediction": ("manifest.json", "predictions.csv"),
    "evaluation": ("manifest.json", "predictions.csv", "report.html", "summary.json"),
    "temporal_evaluation": ("temporal_result.json",),
}
GUARD = '''import importlib.abc, os, sys
families = ('numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'threadpoolctl')
allowed = {'groupbench', 'groupbench.cli', 'groupbench.__main__', 'groupbench.verification'}
def forbidden(name):
    return (any(name == item or name.startswith(item + '.') for item in families)
            or (name.startswith('groupbench.') and name not in allowed))
assert not any(forbidden(name) for name in sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('Unexpected numerical/model/semantic import: ' + fullname)
sys.meta_path.insert(0, Block())
def read_only(event, args):
    if event == 'open':
        name, mode, flags = args
        if isinstance(mode, str):
            assert not any(char in mode for char in 'wax+'), (event, args)
        if isinstance(flags, int):
            assert flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) == 0, (event, args)
        if isinstance(name, (str, bytes)):
            assert os.path.basename(os.fsdecode(name)) not in {'unlisted', 'private-source.csv'}, (event, args)
    assert event not in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link',
                         'os.symlink', 'os.truncate', 'os.chmod', 'os.chown', 'os.utime'}, (event, args)
sys.addaudithook(read_only)
'''
ROUTE_RUNNER = GUARD + '''
import json, runpy
kind, label, target, arguments = json.loads(sys.argv[1])
sys.argv = [label, *arguments]
try:
    if kind == 'console':
        # Invoke the declared entry-point target, without installing wrappers.
        from importlib.metadata import EntryPoint
        status = EntryPoint(name=label, value=target, group='console_scripts').load()()
    else:
        runpy.run_module(target, run_name='__main__', alter_sys=True)
        status = 0
except SystemExit as exc:
    status = exc.code
assert not any(forbidden(name) for name in sys.modules)
raise SystemExit(status)
'''


def expected_stdout(kind, payloads):
    return (f"Verified {kind} bundle: {len(payloads)} files, {sum(map(len, payloads.values()))} payload bytes match checksums.json.\n"
            "Byte integrity only, not scientific/model correctness, authenticity, or proof of execution.\n"
            "Unlisted files are not checked. This is not an atomic directory snapshot; verify an idle directory.\n").encode()


class LightweightVerificationCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def child(self, code, *arguments):
        environment = dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1", COLUMNS="80",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        return subprocess.run([sys.executable, "-B", "-S", "-c", code, *arguments],
                              cwd=self.root, capture_output=True, env=environment, timeout=20)

    def invoke(self, route, args):
        kind, label, target, prefix = route
        return self.child(ROUTE_RUNNER, json.dumps([kind, label, target, prefix + ["verify", *args]]))

    def make_bundle(self, name, kind="prediction"):
        folder = self.root / name
        folder.mkdir()
        # Intentionally invalid JSON/CSV/HTML and non-UTF-8: opaque retained bytes.
        payloads = {filename: b"\x80\x04not a model\xff\x00\r\n" + filename.encode()
                    for filename in INVENTORIES[kind]}
        for filename, raw in payloads.items():
            (folder / filename).write_bytes(raw)
        records = {filename: hashlib.sha256(raw).hexdigest() for filename, raw in payloads.items()}
        (folder / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": records}))
        return folder, payloads

    def snapshot(self):
        state = {}
        for path in self.root.rglob("*"):
            info = path.lstat()
            value = os.readlink(path) if path.is_symlink() else (path.read_bytes() if path.is_file() else None)
            state[str(path.relative_to(self.root))] = (info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, value)
        return state

    def test_console_entrypoint_targets_match_project_declaration(self):
        source = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        for kind, label, target, _ in ROUTES:
            if kind == "console":
                self.assertIn(f'{label} = "{target}"', source)

    def test_help_is_exact_and_keeps_abbreviated_help_across_all_routes(self):
        before = self.snapshot()
        for route in ROUTES:
            for flag in ("--help", "--he", "-h"):
                with self.subTest(route=route[1], flag=flag):
                    result = self.invoke(route, [flag])
                    self.assertEqual((result.returncode, result.stderr), (0, b""))
                    self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), HELP_SHA256)
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(verification._SAFE_OPEN_SUPPORTED, "Requires POSIX no-follow descriptor access")
    def test_all_opaque_inventories_across_all_routes_are_read_only(self):
        (self.root / "private-source.csv").write_bytes(b"DO NOT OPEN")
        fixtures = {kind: self.make_bundle(kind, kind) for kind in INVENTORIES}
        for folder, _ in fixtures.values():
            (folder / "unlisted").symlink_to(self.root / "private-source.csv")
        before = self.snapshot()
        for route in ROUTES:
            for kind, (folder, payloads) in fixtures.items():
                with self.subTest(route=route[1], inventory=kind):
                    result = self.invoke(route, [str(folder)])
                    self.assertEqual((result.returncode, result.stderr), (0, b""))
                    self.assertEqual(result.stdout, expected_stdout(kind, payloads))
        self.assertEqual(self.snapshot(), before)

    def test_parse_failures_are_subcommand_focused_and_do_not_write(self):
        cases = (([], b"the following arguments are required: run_dir"),
                 (["unused", "--unknown"], b"unrecognized arguments: --unknown"),
                 (["unused", "extra"], b"unrecognized arguments: extra"))
        before = self.snapshot()
        for route in ROUTES:
            for args, error in cases:
                with self.subTest(route=route[1], args=args):
                    result = self.invoke(route, args)
                    self.assertEqual((result.returncode, result.stdout), (2, b""))
                    self.assertEqual(result.stderr, b"usage: groupbench verify [-h] run_dir\ngroupbench verify: error: " + error + b"\n")
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(verification._SAFE_OPEN_SUPPORTED, "Requires POSIX no-follow descriptor access")
    def test_io_hash_and_symlink_errors_have_no_partial_stdout_or_writes(self):
        good, _ = self.make_bundle("good")
        tampered, _ = self.make_bundle("tampered")
        (tampered / "predictions.csv").write_bytes(b"MISMATCH")
        payload_link, _ = self.make_bundle("payload_link")
        (payload_link / "manifest.json").unlink()
        (payload_link / "manifest.json").symlink_to(good / "manifest.json")
        checksum_link, _ = self.make_bundle("checksum_link")
        (checksum_link / "checksums.json").unlink()
        (checksum_link / "checksums.json").symlink_to(good / "checksums.json")
        alias = self.root / "alias"
        alias.symlink_to(good, target_is_directory=True)
        ancestor = self.root / "ancestor"
        ancestor.symlink_to(self.root, target_is_directory=True)
        cases = ((self.root / "missing", b"No such file or directory"),
                 (tampered, b"SHA-256 mismatch for predictions.csv; bundle byte integrity failed."),
                 (payload_link, b"Bundle file manifest.json must be a regular, non-symlink file."),
                 (checksum_link, b"Bundle file checksums.json must be a regular, non-symlink file."),
                 (alias, None), (ancestor / "good", None),
                 (self.root / ".." / self.root.name / "good", b"Use a direct run-directory path without '..' or a network-style root."))
        before = self.snapshot()
        for route in ROUTES:
            for path, expected in cases:
                with self.subTest(route=route[1], path=path):
                    result = self.invoke(route, [str(path)])
                    self.assertEqual((result.returncode, result.stdout), (2, b""))
                    self.assertTrue(result.stderr.startswith(b"groupbench: "), result.stderr)
                    self.assertNotIn(b"Traceback", result.stderr)
                    if expected is not None:
                        self.assertIn(expected, result.stderr)
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(verification._SAFE_OPEN_SUPPORTED, "Requires POSIX no-follow descriptor access")
    def test_double_dash_relative_path_keeps_original_argparse_behavior(self):
        _, payloads = self.make_bundle("--help")
        for route in ROUTES:
            with self.subTest(route=route[1]):
                result = self.invoke(route, ["--", "--help"])
                self.assertEqual((result.returncode, result.stderr), (0, b""))
                self.assertEqual(result.stdout, expected_stdout("prediction", payloads))

    def test_lazy_verifier_patch_explicit_process_argv_and_error_contracts(self):
        code = GUARD + '''
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from groupbench import cli
assert 'verify_bundle' not in vars(cli)
arguments = ['verify', 'name with spaces']
original = arguments.copy()
process = ['mlcalc', *arguments]
result = SimpleNamespace(bundle_type='patched', files=('opaque',), total_bytes=7)
with patch.object(cli, 'verify_bundle', return_value=result) as verify:
    with patch.object(cli, '_load_dependencies', wraps=cli._load_dependencies) as load:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert cli.main(arguments) == 0
            with patch.object(sys, 'argv', process):
                assert cli.main() == 0
                assert sys.argv == process
        assert arguments == original
        assert cli.verify_bundle is verify
        assert verify.call_count == 2
        assert all(call.args == (Path('name with spaces'),) for call in verify.call_args_list)
        assert [call.args for call in load.call_args_list] == [('verify_bundle',)] * 2
        assert not err.getvalue()
        assert out.getvalue().count('Verified patched bundle: 1 files, 7 payload bytes') == 2
for exception in (ValueError('value'), TypeError('type'), OSError('io')):
    with patch.object(cli, 'verify_bundle', side_effect=exception):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert cli.main(arguments) == 2
        assert out.getvalue() == ''
        assert err.getvalue() == 'groupbench: ' + str(exception) + '\\n'
with patch.object(cli, 'verify_bundle', side_effect=RuntimeError('programming defect')):
    try:
        cli.main(arguments)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Programming error hidden')
with patch.object(cli, '_load_dependencies', side_effect=AssertionError('Unneeded load')):
    for args, expected in ((['verify', '--help'], 0), (['verify'], 2), (['verify', 'x', '--unknown'], 2)):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                cli.main(args)
            except SystemExit as exc:
                assert exc.code == expected
            else:
                raise AssertionError('Expected argparse SystemExit')
assert not any(forbidden(name) for name in sys.modules)
'''
        result = self.child(code)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))


if __name__ == "__main__":
    unittest.main()
