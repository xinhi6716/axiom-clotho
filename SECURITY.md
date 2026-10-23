# Security and data boundaries

This alpha is a local Python/CLI tool, not a hardened service for untrusted multi-tenant execution. Run it with the permissions of an ordinary user and use trusted, bounded input files.

- It makes no network calls, downloads no model weights, and sends no telemetry.
- It does not load or save pickle/joblib executable model formats.
- HTML values are escaped and the report has a restrictive content security policy. No remote scripts or assets are used.
- Output bundles contain potentially sensitive group labels, target values, predictions, and feature names. Keep them private unless reviewed.
- CSV values are preserved exactly; spreadsheet applications may interpret formula-leading identifiers. Import untrusted CSV columns as text.
- Input-size checks and one-thread execution are not a sandbox or hard memory/time limit. Use OS quotas and isolated environments for untrusted files.
- Writes are staged, with atomic per-file no-clobber behavior unless overwrite is explicit. The bundle as a whole is not transactional.

Do not publish exploit details alongside private data or credentials. Use the repository's private security reporting mechanism if one is available; otherwise provide only a minimal, non-sensitive description and request a private contact before sending sensitive details. No dedicated security response SLA is promised.
