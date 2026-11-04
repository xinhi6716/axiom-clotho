# ALD declaration preview

This opt-in Python helper checks **caller-declared** atomic-layer-deposition
(ALD) metadata before anyone assembles a modeling table. It catches conflicting
temperature roles, mixed growth-rate definitions, some declared post-process
features, and inconsistent thickness/cycle arithmetic. It performs no fits,
network requests, file reads, repairs, row dropping, or training admission.

**`no_declared_conflicts` does not mean scientifically valid or model-ready.**
The helper cannot verify a paper, measurement, physical mechanism, provenance
claim, feature availability, causal interpretation, or sample independence.
It is not called automatically by any existing training or prediction API.
No existing input schema, split, model selection, prediction, or run key changes.

## Run the invented example

From this source checkout:

```bash
PYTHONPATH=src python -S examples/ald-contract-preview/preview.py
```

The [example](../examples/ald-contract-preview/README.md) creates declarations
in memory, asserts the intended conflicts, and prints JSON. It needs only the
Python standard library, makes zero model fits, and writes no files. All its
numbers and identities are invented, not extracted scientific observations.

Import explicitly from `groupbench.process_contract`:

- `ALDExperimentalContract`: expected temperature role/basis and target method,
  target column, selected feature declarations, and an explicit arithmetic
  tolerance, defaulting to exact agreement.
- `ALDObservation`: one declared observation, including source/condition
  metadata, temperature, target, source scope, batch and replicate identifiers.
- `ThicknessCycleDerivation`: a separately scoped thickness/cycle recipe for
  an average target, never evidence of a steady-growth slope.
- `ProcessFeatureDeclaration`: one selected input column's role and availability.
- `preview_ald_contract(records, contract=contract)`: returns a JSON-compatible
  report without mutating its inputs or producing a model-ready table.

The frozen dataclasses describe inputs, not proof of validation. Each preview
rechecks them. Invalid contract expectations or input container types raise a
fixed-message `ValueError`; bad observation declarations produce issue codes.

## Make the scientific choices explicitly

### Temperature and condition identity

Choose an exact temperature role: `substrate`, `stage`, `precursor_vessel`, or
`reactor`. These are not interchangeable. Choose `measured`, `setpoint`, or
`reported_unspecified` as the basis. An unspecified basis always produces
`unknown_temperature_basis`, even if the contract also requests that basis.
The helper does not guess which temperature a paper intended or replace one
temperature with another.

Every observation has a `condition_id`; its `temperature_condition_id` and
`target_condition_id` must match it exactly. Different valid identifiers produce
`condition_identity_mismatch`. This is an equality check on declarations, not
verification that two quantities were measured under the same conditions.

`temperature` is decimal text. The only accepted unit labels are `degC` and `K`.
Temperature values are not converted, compared, or checked for physical
plausibility. Even a well-formed negative Kelvin value requires a separate
physical review and can produce no declaration conflict.

### Distinguish target methods

The experimental contract accepts exactly two target methods:

- `steady_growth_slope`: a declared slope. No thickness/cycle recipe is allowed;
  attaching one produces `derivation_not_applicable_to_method`. The helper
  does not fit a slope or establish a steady-growth regime.
- `total_thickness_div_cycles`: a declared average. A
  `ThicknessCycleDerivation` is required. Its `source_id`, `source_version`, and
  `condition_id` must match the observation. Both locators must be explicit,
  but the recipe may refer to a different table/figure in that same source.

Targets use `angstrom/cycle` or `nm/cycle`; thickness uses `angstrom` or `nm`.
The exact factor is 10 angstrom per nm, consistent with the
[NIST length conversion table](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8).
The checker evaluates thickness divided by cycle count and compares it with
the declared target in angstrom/cycle using exact rational arithmetic. It
does not silently round. For example, the invented 12.5 nm / 250 cycles is
exactly 0.5 angstrom/cycle, an average rather than a steady-growth slope.

`derived_target_tolerance_angstrom_per_cycle` is bounded nonnegative decimal
text. Its default is `"0"`; equality at an explicitly supplied absolute
tolerance passes. A nonzero tolerance is a caller decision, not an inferred
measurement uncertainty or a recommended accuracy threshold. It applies only
to this arithmetic check. Global Decimal precision does not change the result.
There is no physical sign test: even negative thickness and a matching negative
average can agree arithmetically and still require a physical/source review.

### Evidence and experimental identity

An observation must declare `observation_kind="experimental"` and
`evidence_scope="full_text"` to avoid these issues. `simulation`, `theory`,
`abstract`, `snippet`, and `unknown` declarations remain explicitly flagged;
unsupported enum values are also flagged. **Declaring full text does not verify
access to a source, its contents, or the truth of a measurement.**

`source_id`, `source_version`, `source_locator`, and all condition identifiers
must be bounded, nonempty, unpadded text. The reserved strings `unknown`,
`unreported`, and `not_reported` (case-insensitive) are not evidence identities.
Use a reviewed source/version identifier and a precise page/table/row locator;
no DOI resolution, citation retrieval, document opening, or alias inference is
performed.

Keep unknown `batch_id` and `replicate_id` as `None`, not an invented zero or
invented group. Unknown declarations are flagged. The helper also flags every
row in an exact repeated `(source_id, condition_id, batch_id, replicate_id)`
key. Different source versions or locators do not create another observation;
replicate names are scoped to their batch. This limited collision check cannot
detect different identifiers for the same sample, reused data across papers,
or prove that different keys are independent. It does not choose split groups.

### Selected feature timing and role

Declare every proposed feature in the contract. Only `process_setting` or
`material_identity`, together with `before_process`, avoids feature issues.
Other supported roles are `postprocess_measurement`, `computed_target`,
`identifier`, and `unknown`. Availability is `before_process`, `after_process`,
or `unknown`. Selecting the exact target column always produces
`target_selected_as_feature`, regardless of its claimed role.

For example, a post-deposition thickness, an outcome-derived composition, or a
computed target must not be relabeled as pre-process information. The helper
only checks declarations. It cannot detect a dishonest label, target aliases,
derived leakage inside an apparently innocuous column, omitted features, or
whether a material identity was actually known before the process. It never
reads feature values or links declarations to an external table automatically.
The default empty feature tuple means no selected features were declared, not
that all columns in some external table were checked.

## Bounded numeric and metadata input

- Supply an exact built-in list or tuple of 1 to 20,000 `ALDObservation` objects,
  and a tuple of at most 100 unique `ProcessFeatureDeclaration` objects.
  Arbitrary iterators, mappings, and subclasses are not inferred or consumed.
- Numeric quantities are decimal **strings**, not floats, integers, booleans,
  or Decimal objects. This avoids pretending to recover digits lost before
  admission. The shared decimal parser permits at most 256 characters, 64
  coefficient digits, and absolute explicit/effective decimal exponents of 200.
  Whitespace, non-ASCII digits, missing values, NaN and infinity are rejected.
- `cycle_count` is an exact built-in integer from 1 to 1,000,000,000. Booleans,
  numeric strings, and floats are not cycle counts.
- Text metadata and feature names are 1 to 256 characters, unpadded, without
  control characters or surrogate code points. Enum labels and identities are
  case-sensitive except the reserved unknown identity markers above.
- These are software resource and declaration bounds, not an accepted ALD
  operating window, minimum sample size, physical sign restriction, or
  scientific inclusion criterion.

## Read the report without treating it as approval

The report has `schema_version="1.0"`, `kind="ald_declaration_preview"`,
`action="validation_only_no_data_changes"`, and
`training_admission="not_performed"`. Its status is either
`declaration_issues_found` or `no_declared_conflicts`.

`issue_count` and `issue_counts` count all detected conflicts. `n_rows_with_issues`
counts affected observation positions; `n_feature_declarations_with_issues`
separately counts affected feature declarations. A feature-only conflict can
therefore have zero affected rows while the overall status still shows issues.
Each detail contains a fixed code, fixed field label, and zero-based
`row_position` or `feature_position`, with the other position null. Some rows
can have several issues. Details are deterministic but capped at 1,000;
`issues_truncated` signals omitted details, while aggregate counts still cover
all supplied records. Do not infer an accepted subset from the detail list.

The report and validation errors do not echo quantities, identifiers, source
locators, or feature names. Counts and positions can still describe a study;
apply your usual access controls before sharing a report. The helper does not
retain provenance evidence or write audit files. Keep the source review and
declarations separately under appropriate permissions.

Review flagged declarations and source evidence yourself. Establish uncertainty,
batch/replicate structure, leakage controls, physical assumptions, and a held-out
evaluation design before deciding which observations belong in a study. A
conflict-free preview makes none of those decisions for you.
