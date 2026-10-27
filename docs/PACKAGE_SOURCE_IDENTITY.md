# Observed package-source metadata

**Current source only; not included in the frozen `v0.2.0a3` download.**
The package version can stay `0.2.0a3` while source changes. Retain the Git
revision for a checkout, but do not assume a Git command in your working directory
identifies the package Python actually imported.

Regression and binary evaluation/prediction manifests now add `package_source`.
The two evaluation reports display its recorded digest or availability status.
This is **self-reported metadata about observed source bytes**, not proof of the
code executed in memory, a build, hardware, reproducibility, authenticity or
safety. No commit, release or trusted origin is inferred from the digest.

## What is observed

The helper anchors reads to its own imported module's package directory, not the
current working directory. It reads a fixed allowlist of 26 owned `.py` files,
including itself and both report renderers, once when assembling each result
manifest after fitting. No directory scan, Git command, `.git`, dependency file,
user dataset, model file, bytecode or network access is involved. No absolute
installation path or source contents are recorded.

Available metadata contains:

- `schema_version: "1.0"`, `algorithm: "sha256"`;
- `scope: "fixed_groupbench_python_sources_v1"`;
- `status: "available"`;
- `files`: package-relative POSIX filenames mapped to lowercase SHA-256 hashes
  of their exact bytes;
- `sha256`: SHA-256 of the canonical UTF-8 bytes defined below.

For contract 1.0, the canonical object contains only `schema_version`,
`algorithm`, `scope` and `files` with the values above. Serialize with Python
`json.dumps(object, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`
and encode as UTF-8 without a trailing newline. `status` and the aggregate digest
are not themselves hashed. The allowlist is explicit in `package_source.py`;
maintainers must update it and review the contract when owned source files change.

Identical allowlisted bytes in another installation directory give the same
digest. Any changed allowlisted file, including a renderer or comment, changes
it. Unlisted files, documentation, dependency versions, timestamps and hardware
do not contribute. This is not a whole-repository or wheel fingerprint.

## Availability and limits

Source access reuses the byte verifier's POSIX descriptor-relative, no-follow
regular-file reader. Each file is capped at 256 KiB (26 files, at most 6.5 MiB
total); data is streamed and only digests are retained. Detected file changes,
symlinks in the source path, missing or oversized files and I/O errors return
`status: "unavailable"` with a fixed reason, no partial digest or local path.
Pinned child-directory bindings and the fixed package-root binding are rechecked
to detect directory replacements, including symlinks introduced after opening.
Unsupported safe-access platforms and source-free/zip imports can also be
unavailable. Modeling still returns its results. Bounds are not a hard I/O-time
limit; use trusted, idle local storage.

An old manifest without the field displays **not recorded**; it remains valid
under its existing contract. Missing metadata does not imply a particular
release. The HTML uses the saved record, not a fresh filesystem observation.
Reports do not authenticate or fully validate arbitrary supplied metadata.

Files can change after import or during execution. Even an available observation
does not establish correspondence with already loaded bytecode, monkey-patched
functions, dependency implementations or a filesystem-atomic snapshot. Restart
after changing source and retain the reviewed environment when comparing runs.

## Existing fields and verification are unchanged

- `groupbench_version` and CLI `--version` retain their package-version meaning.
- `source_file_sha256` / `prediction_source_file_sha256` still hash input CSV
  bytes, not Python source. Data fingerprints still describe normalized data.
- `run_key`, model selection, scores, outer schemas and output file sets do not
  change. A matching run key alone is not software identity.
- `mlcalc verify` checks saved payload bytes against unauthenticated checksums;
  it does not read or compare installed package source.
- `validate-prediction-bundle` permits this extra field but does **not** validate
  it. Older verifiers accepting the metadata is compatibility, not certification.

No new wheel, frozen-tag change, dependency lock, build attestation or scientific
reproducibility guarantee is implied. Plans and summary metrics do not acquire
this field; it is recorded in the result manifest.
