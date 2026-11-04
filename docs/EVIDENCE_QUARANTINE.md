# Preview quantity-local evidence quarantine

Use this opt-in, standard-library-only helper when a source review has already
identified unresolved evidence conflicts that must remain outside the main
observation inventory. It checks **exact caller-declared metadata keys**, not
quantity values or source documents. It makes no model fits and does not change,
remove, approve, or join observations.

The useful distinction is local: an unresolved conflict about one endpoint in
one recipe group must not silently reappear in the inventory, but that conflict
does not automatically invalidate another endpoint from the same source/group.
An inventory-wide `ml_admitted=false` flag cannot justify mixing a quarantined
quantity back into the main observations. This helper checks the supplied rows
regardless of downstream ML admission.

## Run the invented example

From this source checkout:

```bash
PYTHONPATH=src python -S examples/evidence-quarantine/preview.py
```

The [example](../examples/evidence-quarantine/README.md) asserts four cases:
clean isolation, an injected exact quarantined key, a different endpoint from
the same source/group, and an unmapped quantity family. It contains only invented
metadata, no scientific measurements. It prints JSON and writes no files.

## Interface

Import these names explicitly from `groupbench.process_contract`:

```python
from groupbench.process_contract import (
    ObservationEvidenceKey, QuarantineScope, preview_evidence_quarantine,
)

observations = [
    ObservationEvidenceKey("invented-source", "recipe-2", "peak_area"),
]
unresolved_scopes = [
    QuarantineScope("invented-source", ("recipe-2",), ("peak_ratio",)),
]
report = preview_evidence_quarantine(
    observations, unresolved_scopes=unresolved_scopes,
)
assert report["status"] == "no_declared_quarantine_matches"
```

`ObservationEvidenceKey(source_id, recipe_group_id, endpoint_name)` represents
one row of the observation inventory being checked. It deliberately has no
quantity value, model-admission flag, inferred physical specimen identifier, or
file path. Duplicate keys remain separate input rows; they are not automatically
deduplicated and do not establish independent samples.

`QuarantineScope(source_id, recipe_group_ids, endpoint_names)` represents one
**known unresolved conflict selected by the caller**, not every source-review
issue. Both member collections are explicit tuples. Every named endpoint
applies to every named recipe group within this source. For a nonrectangular
mapping such as `(group-A, endpoint-X)` and `(group-B, endpoint-Y)`, supply two
separate scopes. Combining both groups and both endpoints in one scope would
also quarantine the two cross-pairs. The implementation scans only observed
keys, not a materialized Cartesian product.

Use `endpoint_names=None` when a quantity-family description has no reviewed,
explicit mapping to exact endpoint names. This is **not an empty scope** and
never yields a fully checked result. An empty tuple is invalid. There is no
substring matching, case folding, synonym inference, fuzzy matching, DOI
resolution, unit inference, or source-version/alias reconciliation.

## Project an existing inventory deliberately

Keep the evidence-review policy separate from this mechanical check:

1. Select the exact observation table to audit, not just its ML-selected subset.
   Project only source identity, recipe-group identity, and endpoint name into
   `ObservationEvidenceKey` objects. Retain row order to interpret report
   positions. A repeated row stays a repeated row, not an independent sample.
2. Select review issues explicitly designated for unresolved quarantine. An
   issue intentionally retained with a warning, such as an unresolved unit
   annotation under a different policy, is not automatically a quarantine.
   This helper neither selects review issues nor resolves their meaning.
3. Map each selected issue to its explicit source, recipe groups, and exact
   endpoint names. If a scope describes a family rather than exact endpoints,
   provide a reviewed explicit mapping or use `None`. Do not guess from the
   current inventory's strings or turn an absent string into proof of safety.
4. Preserve the selected issue order separately under appropriate permissions
   so that `scope_position` points back to the reviewed declaration. The report
   intentionally does not echo issue IDs, source IDs, group IDs, or endpoints.
5. Test the projection with a positive control that inserts a known quarantined
   key and a negative control that inserts a different endpoint in the same
   source/group. Neither control should add real measurements or become an
   observation in the study.

The unresolved-scope argument is required, even for an explicitly empty list.
An empty registry reports no declared matches; it does **not** establish that a
source has no unresolved conflicts or that all relevant issues were supplied.
Registry selection, mapping accuracy, completeness, and source truth remain
the caller's responsibility. Incorrect or aliased identities can miss a real
conflict; a broad or incorrect scope can overflag unrelated observations.

No CSV/HDF5 loader is provided. Do not deserialize unrelated object payloads,
read protected outcome partitions, or include quantity values merely to create
this metadata projection. This API is not called automatically by existing
training, prediction, ALD preview, or data-preparation routes.

## Interpret all report fields together

The report has `schema_version="1.0"`, `kind="evidence_quarantine_preview"`,
`action="audit_only_no_data_changes"`, and
`training_admission="not_performed"`. Status precedence is:

- `quarantine_violation_found`: at least one exact declared quarantine match
  is present. Unmapped scopes may also remain; inspect `n_unmapped_scopes`.
- `mapping_not_established`: no exact violation was found, but one or more
  scopes have unknown endpoint mappings, even if no rows from those groups
  currently appear in the supplied inventory.
- `no_declared_quarantine_matches`: the supplied exact scopes do not match
  the supplied rows. This is not a scientific pass, complete source review,
  permission to train, or approval of rows outside the declared scopes.

`n_observation_rows` counts input metadata rows. `n_rows_with_quarantine_matches`
counts distinct input row positions with exact matches. `issue_count` counts
every scope/row match plus one issue per unmapped scope. Overlapping or repeated
scope declarations can therefore contribute multiple issues for one row; the
affected-row count does not increase. No count is an independent-specimen N.

`n_rows_in_unmapped_scope_groups` counts distinct rows in the named source/group
combinations of unmapped scopes. These are potentially relevant rows, **not
confirmed quarantined quantities**. This set can overlap exact matched rows;
do not sum the two row counts or infer a safe remainder from them.

Each issue contains a fixed code and field label, plus zero-based
`scope_position` and `row_position`. An unmapped-scope issue has a null row
position. Exact matches use `quarantined_observation_present`; unmapped scopes
use `scope_endpoint_mapping_not_established`. Only the first 1,000 details are
returned, but aggregate counts cover every input row and scope. Check
`issues_truncated`; a missing detail is never evidence that a row was accepted.
Output is deterministic for the same ordered declarations.

Reports and fixed validation errors do not echo identities or values. Counts
and positions can still describe private research, so apply normal access
controls before sharing them. The helper does not preserve evidence documents
or write an audit trail for you.

## Bounds and non-goals

- Observations must be an exact built-in list or tuple of at most 20,000 exact
  `ObservationEvidenceKey` records. Empty inventories are allowed and do not
  establish completeness. Scopes must be an explicit list or tuple of at most
  1,000 exact `QuarantineScope` records.
- Each group/endpoint member tuple must be nonempty, contain no duplicates,
  and stay within the total limit of 20,000 references across all scopes.
  Only endpoint mapping may be `None`. Repeated scope declarations are allowed
  and counted separately; they do not create more physical observations.
- Identities are exact built-in strings, 1 to 256 characters, unpadded, without
  control characters or surrogate code points. Reserved `unknown`,
  `unreported`, and `not_reported` markers are invalid case-insensitively.
- All inputs, including frozen declarations, are rechecked each call. Invalid
  input raises a fixed-message `ValueError` before a report is constructed.
  Arbitrary iterators, mappings, subclasses, numeric identities, and implicit
  missing-value replacement are unsupported.

These are software declaration checks, not source conflict resolution,
measurement validation, physical plausibility, specimen pairing, causal
inference, data deduplication, split selection, or ML admission. In particular,
sharing a source and recipe group does not establish that film and device
measurements came from the same physical specimen.
