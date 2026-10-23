# Security and data boundaries

This alpha is a local Python/CLI tool, not a hardened service for untrusted multi-tenant execution. Run it with the permissions of an ordinary user and use trusted, bounded input files.

- It makes no network calls, downloads no model weights, and sends no telemetry.
- It does not load or save pickle/joblib executable model formats.
- HTML values are escaped and the report has a restrictive content security policy. No remote scripts or assets are used.
- Output bundles contain potentially sensitive group labels, target values, predictions, and feature names. Keep them private unless reviewed.
- CSV values are preserved exactly; spreadsheet applications may interpret formula-leading identifiers. Import untrusted CSV columns as text.
- Input-size checks and one-thread execution are not a sandbox or hard memory/time limit. Use OS quotas and isolated environments for untrusted files.
- CSV records are checked for consistent field counts before type inference. UTF-8 decoding, strict quoting, non-blank unique headers, and Python's CSV parser field-size limit apply; errors omit raw cell contents. These checks do not replace OS-level resource limits.
- CLI source hashes are computed incrementally from the same byte stream used for CSV validation, not from a later pathname lookup. This binds the digest to observed bytes but does not lock files or guarantee a filesystem-atomic snapshot during concurrent writes.
- Writes are staged, with atomic per-file no-clobber behavior unless overwrite is explicit. The bundle as a whole is not transactional.
- `mlcalc verify RUN_DIR` is read-only and accepts only the existing two-file prediction or four-file evaluation checksum sets. It rejects duplicate JSON keys, unexpected paths, malformed hashes, non-regular files, and symlinks in the selected path or payload entries. POSIX descriptor-relative no-follow access is required; unsupported platforms fail closed. Use direct paths without parent (`..`) traversal or network-style roots.
- Verification bounds checksum metadata to 16 KiB and each payload to 64 MiB, using streaming reads. Unlisted files are not verified; stale evaluation artifacts in prediction directories are rejected. Payloads remain opaque bytes, never executable deserialization or referenced source-file reads. Filesystem I/O can still stall on unsuitable storage; byte bounds are not a hard time limit.
- Verification checks observed bytes against unauthenticated checksums, not authorship, scientific/model correctness, proof of execution, or a whole-directory atomic snapshot. Do not modify a bundle during verification. An attacker who controls the payloads and checksum file can produce a matching bundle.

Do not publish exploit details alongside private data or credentials. Use the repository's private security reporting mechanism if one is available; otherwise provide only a minimal, non-sensitive description and request a private contact before sending sensitive details. No dedicated security response SLA is promised.
