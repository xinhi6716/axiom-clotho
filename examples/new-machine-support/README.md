# Count machines, not rows, for a new-machine cutoff

From the repository root, run the no-fit example with the current Rust workspace:

```sh
cargo run --manifest-path rust/Cargo.toml --locked -p clotho-core --example printer_roster_support
```

## Public source and synthetic replay

The October 2024 workbook in the [Calero Lumbreras, Hernández-del-Valle,
Schenk and Haranczyk Zenodo dataset, v1](https://doi.org/10.5281/zenodo.17366710)
is labeled [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Counting only settings fully crossed across printers gives 25 settings,
nine repeat-index pairs per setting, and three printer outcomes per pair:
675 weights, but only three declared machine IDs. Other settings are not
part of that common grid. The example copies only those aggregate counts.
All identifiers and residual scores in the program are invented; it uses
no workbook or measured weights.

At 95%, the group-max cutoff uses rank `ceil((3 + 1) × 0.95) = 4`.
Three declared machine scores cannot supply rank four, so the routine
returns `InsufficientFinite`. Incorrectly treating all 675 rows as machine
groups supplies a finite rank of 643, but answers the wrong question.
Actual machine independence and exchangeability are not established by a
spreadsheet key. This is not a measured coverage result.
