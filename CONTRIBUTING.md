# Contributing

We want a small local ML calculator that people can actually trust and use. Correct protocols and understandable failure messages matter more than a long model list.

## Development

Use a clean Python environment and install this checkout. No GPU or hosted API is required.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
mlcalc demo --out runs/contribution-check
```

With dependencies already installed, use `PYTHONPATH=src` instead of installing. Do not add runtime network calls or install/download model weights during tests.

## Good contributions

- A minimal regression test and fix for a split, leakage, schema, or artifact bug.
- Clearer local workflows or accessibility improvements to the offline report.
- Well-scoped local adapters with explicit model/data licenses and resource budgets.
- Documentation that makes assumptions and limitations easier to understand.

For new model families, task types, data downloads, or persistence formats, discuss a design before implementation. Include why it helps a real user, its scientific estimand, resource cost, privacy boundary, and failure behavior.

## Required verification

- Run the complete test suite and add focused tests for the changed behavior.
- Preserve group-disjoint inner and outer splits. Prove that changing outer test targets cannot change their predictions or candidate selection.
- Keep all learned preprocessing inside each training partition.
- Keep `row_position`, predictions, and provenance aligned.
- Test new writes for pre-existing files, races, path aliases, interruption, and symlinks as applicable.
- Never claim synthetic results establish real-data performance, fairness, calibration, or state of the art.

## Data and licensing

Use synthetic test data by default. Do not include private measurements, credentials, personal information, manuscript/review content, or data with unclear redistribution rights. Public-data additions require a source, license, data card, and explicit scientific grouping rationale. Do not copy model weights or another project's code without a separate license review. Contributions must be compatible with this project's MIT license; dependencies retain their own terms.

AI-assisted contributions need the same human review, reproducible tests, and provenance as any other change. Do not submit generated results you did not actually run.

## Reporting a problem

Include Python/library versions, exact command or small API example, expected versus observed behavior, and a synthetic input that reproduces it. Omit private file paths and data. See SECURITY.md before reporting a security issue publicly.
