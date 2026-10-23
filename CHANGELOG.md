# Changelog

## Unreleased

- Reject malformed CSV records before fitting, preventing pandas from silently shifting extra fields into an index or padding absent fields.
- Validate non-blank headers, UTF-8, and quoting with line/count diagnostics that omit raw cell values. Preserve quoted commas/newlines, blank-line behavior, explicit empty fields, and textual group identifiers.
- Apply and document Python's CSV parser field-size guard for data cells as well as headers; keep the existing bounded row limit and add synthetic input regression tests.

## 0.1.0 — 2026-10-23

Initial alpha of ML Super Calculator, powered by the GroupBench Python engine.

- Nested group-disjoint regression evaluation with train-only preprocessing.
- Dummy, ridge, and Extra Trees selection, per-group metrics, and split audit.
- Local CSV/Python full-data refit and new-row prediction.
- Offline HTML/JSON reports, exact prediction CSVs, and integrity checksums.
- Fully synthetic materials-process example, documentation, and regression tests.

No public real-data benchmark, foundation-model training, hosted service, calibrated uncertainty, or production-readiness claim accompanies this release.
