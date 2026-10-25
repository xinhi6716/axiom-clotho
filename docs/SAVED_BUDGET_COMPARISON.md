# Saved two-budget comparison (descriptive only)

This optional local utility compares already saved scalar metric observations for
exactly two ordered methods and two ordered validation budgets. It does not fit,
retrain, select a model, contact a service, or read any research outputs. The
fixture is invented and establishes no scientific performance claim.

Unlike `examples/environment-heldout-comparison`, which computes device errors
for one fixed environment, this utility handles matched repeated observations
nested inside declared independent units and a selector-by-budget interaction.
It is not an automatic adapter for arbitrary benchmark reports: explicitly
prepare and check the input contract before use.

```sh
PYTHONPATH=src python -m groupbench.budget_comparison \
  examples/saved-budget-comparison/invented.json
```

Output goes only to stdout; errors go to stderr with exit code 2. No output file
is opened or overwritten. If saving via shell redirection, choose a new path
that is not the input: the shell itself can overwrite files before Python runs.
The Python API is `groupbench.budget_comparison.compare_saved_budgets(document)`;
`load_document(text)` rejects duplicate JSON keys and nonfinite constants.

The CLI accepts only a regular local input file. Explicit symlinks to regular
files remain supported; directories, named pipes (FIFOs), devices and other
non-regular inputs are rejected without reading their contents. The file type
is checked before opening and on the opened descriptor. Where supported,
nonblocking open also avoids waiting on a FIFO substituted between those checks.
Use trusted, idle local files: these checks do not provide a filesystem-atomic
snapshot, a hostile-filesystem sandbox, or a wall-clock deadline for storage
access. The existing UTF-8 decoding and 8-million-character limit are unchanged.

## Input contract

See the complete invented fixture. Version 1 requires exactly these fields:

- `schema_version`: integer 1 (not boolean).
- `methods`: two distinct strings, ordered A (reference), B (candidate selector).
- `budgets`: two distinct strings, ordered low then high. Order and labels are
  declarations; numeric cost/ordering is not independently verified.
- `metadata`: `metric_definition`, `value_unit`, `independent_unit_definition`,
  `normalization`, and `direction` (`minimize` or `maximize`). All are mandatory
  nonempty trimmed strings. Define metric denominator, aggregation, population,
  and normalization sufficiently precisely to make every supplied value comparable.
  Use `none; same scale in every cell` when no normalization was applied. This tool
  performs no normalization and cannot detect a false declaration.
- `rows`: every row contains exactly `unit_id`, `repeat_id`, `method`, `budget`,
  numeric `value`, `status` (must be `ok`), `metric_definition`, and `value_unit`.
  Row definitions/units must exactly match metadata. IDs are case-sensitive text;
  leading/trailing whitespace and control characters are rejected, not repaired.

Every `(unit_id, repeat_id)` must have all four method-by-budget combinations.
Duplicate keys, unknown methods/budgets, failures, invalid numbers, or missing
combinations reject the **whole comparison**. Do not preprocess away failures
or unsuccessful units to make input pass; diagnose and report the failed design.
A wholly omitted unit cannot be inferred from the supplied file: complete cohort
accounting is the user's responsibility. Different units may have different
numbers of repeats, but a unit's repeat IDs must match all four cells. A repeat
ID may occur in another unit; it is not an independent sample count.

The parser limits input to 8 million characters and 20,000 cells. Values allow
at most 100 significant decimal digits and absolute decimal exponent 400;
values and results must convert to nonzero finite float when nonzero. No silent
infinite or underflowed report is emitted. JSON decimal numbers are interpreted
exactly for rational aggregation/sign comparisons; reported values are rounded
to floats, so tiny displayed differences should not be reclassified using rounded
method means. There is no tolerance tie band: ties are exactly zero contrast.

## Estimand and accounting

For unit u, budget b and method m, first compute the arithmetic mean across its
matched repeat observations. Then the method-B gain is `A - B` for minimized
metrics and `B - A` for maximized metrics. Positive always favors B. The summary
averages gains **equally across units**, not across all repeat rows. Supply
already computed per-repeat scalar metrics; averaging RMSEs, for example, is
mean replicate RMSE, not pooled RMSE.

The selector-by-budget interaction for each unit is
`gain(high budget) - gain(low budget)`; the summary is its equal-unit mean.
Positive means the descriptive B advantage is larger at the second budget; it
is not a causal budget effect, and budget labels are not estimated doses.
Each budget reports strict wins, exact ties and harms across units. The complete
unit ledger preserves all means, contrasts and outcomes; the observation ledger
also preserves every adverse repeat even if aggregation makes a unit look good.

There are deliberately **no uncertainty estimates, significance tests, or
independence certification**. Unit independence, matched cohorts, identical
metric definitions, normalization, selection protocol, and data provenance are
user declarations, not verified facts. A checksum would not establish provenance
or independence either. Cross-unit averages are meaningful only on the declared
common scale. Do not use correlated folds or repeated seeds as independent
experimental units, and do not reinterpret descriptive output as a population
claim. Unknown fields, including requests for uncertainty, fail closed.
