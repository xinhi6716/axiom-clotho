"""Bounded, read-only byte-integrity checks for the two existing bundle writers.

Payloads are never parsed, executed, rendered, or deserialized. A successful
check is not an authenticated origin, scientific validation, or atomic snapshot.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat


MAX_CHECKSUM_BYTES = 16 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
PREDICTION_FILES = frozenset({"manifest.json", "predictions.csv"})
EVALUATION_FILES = PREDICTION_FILES | {"report.html", "summary.json"}
_SAFE_OPEN_SUPPORTED = (
    os.name == "posix"
    and all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


@dataclass(frozen=True)
class VerificationResult:
    bundle_type: str
    files: tuple[str, ...]
    total_bytes: int


@contextmanager
def _open_run_directory(run_dir):
    """Pin the selected directory without following any symlink component."""
    if not _SAFE_OPEN_SUPPORTED:
        raise ValueError("Safe verification requires POSIX no-follow and directory-relative file access; this platform is unsupported.")
    path = Path(run_dir)
    if len(os.fsencode(path)) > 4096 or len(path.parts) > 256:
        raise ValueError("Run-directory path exceeds the verification size limit.")
    if ".." in path.parts or path.anchor not in ("", "/"):
        raise ValueError("Use a direct run-directory path without '..' or a network-style root.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path.anchor or ".", flags)
    try:
        parts = path.parts[1:] if path.anchor else path.parts
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _check_regular(info, name, limit):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Bundle file {name} must be a regular, non-symlink file.")
    if info.st_size > limit:
        raise ValueError(f"Bundle file {name} exceeds the {limit}-byte verification limit.")


def _read_file(directory_fd, name, limit, *, collect=False):
    """Read only a fixed expected basename, with both size and stream bounds."""
    # Checking before open avoids opening known devices/FIFOs. O_NONBLOCK and
    # the second check prevent a swapped FIFO from blocking or being read.
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _check_regular(before, name, limit)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(fd)
        _check_regular(opened, name, limit)
        if _signature(opened) != _signature(before):
            raise ValueError(f"Bundle file {name} changed before verification; retry with an idle directory.")
        digest, chunks, size = hashlib.sha256(), [], 0
        while True:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"Bundle file {name} exceeds the {limit}-byte verification limit.")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.fstat(fd)
        if size != opened.st_size or _signature(after) != _signature(opened):
            raise ValueError(f"Bundle file {name} changed during verification; retry with an idle directory.")
    finally:
        os.close(fd)
    return digest.hexdigest(), size, b"".join(chunks), _signature(after)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("checksums.json contains duplicate object keys.")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("checksums.json must contain strict JSON without non-finite constants.")


def _checksum_records(raw):
    try:
        checksums = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("checksums.json must contain a bounded, valid UTF-8 JSON object.") from exc
    if not isinstance(checksums, dict) or set(checksums) != {"algorithm", "files"}:
        raise ValueError("checksums.json must contain exactly algorithm and files.")
    if checksums["algorithm"] != "sha256":
        raise ValueError("checksums.json algorithm must be sha256.")
    records = checksums["files"]
    if not isinstance(records, dict) or set(records) not in (PREDICTION_FILES, EVALUATION_FILES):
        raise ValueError("checksums.json must list exactly the two prediction files or four evaluation files; other paths are forbidden.")
    for digest in records.values():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("checksums.json digests must be 64 lowercase hexadecimal SHA-256 characters.")
    return records


def _reject_stale_evaluation(directory_fd):
    for name in sorted(EVALUATION_FILES - PREDICTION_FILES):
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise ValueError("Prediction bundle contains unverified evaluation artifacts; use a separate run directory.")


def verify_bundle(run_dir) -> VerificationResult:
    """Verify exact writer file sets without loading models or reading sources.

    The checksum file is capped at 16 KiB, and each of at most four payloads at
    64 MiB. Unexpected files are neither opened nor verified. Prediction bundles
    must not contain stale report.html or summary.json entries. No writes occur.
    Directory/payload races detected by metadata checks fail, but this is not a
    lock or a guarantee of a filesystem-atomic snapshot under concurrent mutation.
    """
    with _open_run_directory(run_dir) as directory_fd:
        _, _, raw, signature = _read_file(directory_fd, "checksums.json", MAX_CHECKSUM_BYTES, collect=True)
        records = _checksum_records(raw)
        bundle_type = "prediction" if set(records) == PREDICTION_FILES else "evaluation"
        if bundle_type == "prediction":
            _reject_stale_evaluation(directory_fd)
        signatures = {"checksums.json": signature}
        total_bytes = 0
        for name in sorted(records):
            actual, size, _, signature = _read_file(directory_fd, name, MAX_PAYLOAD_BYTES)
            if actual != records[name]:
                raise ValueError(f"SHA-256 mismatch for {name}; bundle byte integrity failed.")
            signatures[name] = signature
            total_bytes += size
        # Also notice entries replaced after they were read. This still cannot
        # prevent changes after a check or provide whole-directory atomicity.
        for name, signature in signatures.items():
            if _signature(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) != signature:
                raise ValueError(f"Bundle file {name} changed during verification; retry with an idle directory.")
        if bundle_type == "prediction":
            _reject_stale_evaluation(directory_fd)
    return VerificationResult(bundle_type, tuple(sorted(records)), total_bytes)
