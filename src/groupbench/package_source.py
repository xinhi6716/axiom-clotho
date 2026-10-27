"""Bounded observations of this package's source files, not executed-code proof.

Only fixed owned filenames are read, relative to this imported module. This is
self-reported metadata, not a build, origin, hardware or reproducibility attestation.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

from .verification import _SAFE_OPEN_SUPPORTED, _open_run_directory, _read_file, _signature


SOURCE_FILES = (
    "__init__.py", "__main__.py", "binary/__init__.py", "binary/cli.py",
    "binary/data.py", "binary/metrics.py", "binary/model.py", "binary/report.py",
    "binary/workflow.py", "budget_comparison.py", "categorical.py", "cli.py",
    "data.py", "evaluation.py", "feature_manifest.py", "input_support.py",
    "interval_evaluation.py", "metrics.py", "model.py", "package_source.py",
    "planning.py", "prediction.py", "prediction_schema.py", "report.py",
    "splitting.py", "verification.py",
)
MAX_SOURCE_FILE_BYTES = 256 * 1024
_PACKAGE_ROOT = Path(__file__).parent
_CONTRACT = {"schema_version": "1.0", "algorithm": "sha256",
             "scope": "fixed_groupbench_python_sources_v1"}


def observe_package_source():
    """Return a source-byte observation or unavailable, without blocking a run.

    No caller paths, Git, cwd discovery, directory scans or dependency reads.
    Source-free loaders and unsupported safe-open platforms are unavailable.
    Idle trusted storage is required; detected changes fail, but this is not an
    atomic snapshot or a check of bytecode already loaded into memory.
    """
    unavailable = {**_CONTRACT, "status": "unavailable"}
    if not _SAFE_OPEN_SUPPORTED or not _PACKAGE_ROOT.is_absolute():
        return {**unavailable, "reason": "unsupported_source_access"}
    try:
        with ExitStack() as stack:
            root_fd = stack.enter_context(_open_run_directory(_PACKAGE_ROOT))
            root_signature = _signature(os.fstat(root_fd))
            directories = {"": root_fd}
            records, signatures, directory_bindings = {}, [], []
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
            flags |= getattr(os, "O_CLOEXEC", 0)
            for relative in SOURCE_FILES:
                parts = relative.split("/")
                parent = ""
                for part in parts[:-1]:
                    child = f"{parent}/{part}" if parent else part
                    if child not in directories:
                        fd = os.open(part, flags, dir_fd=directories[parent])
                        stack.callback(os.close, fd)
                        directories[child] = fd
                        directory_bindings.append((directories[parent], part, _signature(os.fstat(fd))))
                    parent = child
                fd, name = directories[parent], parts[-1]
                digest, _, _, signature = _read_file(fd, name, MAX_SOURCE_FILE_BYTES)
                records[relative] = digest
                signatures.append((fd, name, signature))
            for fd, name, signature in signatures:
                if _signature(os.stat(name, dir_fd=fd, follow_symlinks=False)) != signature:
                    return {**unavailable, "reason": "source_unavailable_or_changed"}
            # Pinned descriptors alone can keep reading a directory that was
            # renamed/replaced. Recheck its entry in the pinned parent, too.
            for parent_fd, name, signature in directory_bindings:
                if _signature(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != signature:
                    return {**unavailable, "reason": "source_unavailable_or_changed"}
            # Reopen only the fixed package path, safely, to detect a replaced
            # root (or symlinked ancestor). No source is read through this fd.
            with _open_run_directory(_PACKAGE_ROOT) as current_root_fd:
                if _signature(os.fstat(current_root_fd)) != root_signature:
                    return {**unavailable, "reason": "source_unavailable_or_changed"}
        canonical = json.dumps({**_CONTRACT, "files": records}, sort_keys=True,
                               separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return {**_CONTRACT, "status": "available", "sha256": hashlib.sha256(canonical).hexdigest(),
                "files": records}
    except (OSError, ValueError):
        # Never serialize filesystem exceptions or partial records: they may
        # contain private installation paths. Modeling can still continue.
        return {**unavailable, "reason": "source_unavailable_or_changed"}


def package_source_display(manifest):
    """Return fixed status copy or a bounded digest, never arbitrary metadata."""
    if "package_source" not in manifest:
        return "not recorded"
    source = manifest["package_source"]
    if isinstance(source, dict) and source.get("status") == "available":
        digest = source.get("sha256")
        if isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            return digest
    return "unavailable"
