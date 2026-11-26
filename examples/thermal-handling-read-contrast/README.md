# Separating electrical reads from sample handling

```sh
python examples/thermal-handling-read-contrast/run.py
```

If samples leave controlled storage for intermediate measurements, the
extra-read arm experiences a different thermal and handling schedule even
when elapsed time is identical. The three arms in this example differ only
as follows; all share the start and end reference reads:

| Arm | Intermediate handling | Intermediate electrical reads |
| --- | --- | --- |
| A | No | No |
| B | Yes | No |
| C | Same handling as B | Yes |

Use the same contact and dwell procedures where possible, and record actual
sample temperature over time and electrical exposure separately. Assign and
compare independently handled trays or subsystems; if devices share a tray,
treat them as subsamples for handling. Equal tray weights here target the
average tray, not the average pooled cell.

In the exact toy, the signed outcome is change in log₂ conductance. One
assigned cell is missing and stays in the denominator, with a *hypothetical*
predeclared outcome range of [−4, 4]. The bounds are deterministic missing
outcome bounds, not confidence intervals:

| Contrast | Bound | Question |
| --- | --- | --- |
| B − A | [−3, −3] | Handling in the absence of additional electrical reads |
| C − B | [+1, +2] | Additional electrical reads with handling held fixed |
| C − A | [−2, −1] | Combined procedure relative to sparse measurement |

The combined contrast and the added-read contrast have opposite signs.
Let `H` denote handling and `R` denote additional reads. The arm without
handling but with extra reads (`H=0, R=1`) is absent; since `R=H×R` on
observed arms, the three arms do not estimate an interaction or the read
effect without handling. If constant-temperature in-situ reads are feasible,
a separate randomized sparse-versus-repeated comparison can target that
procedure.

The values and bounds are synthetic, not device limits, effect estimates,
or recommended operating parameters. An example of different storage and
measurement temperatures is [Frascaroli's thesis, §3.4.2][thesis].
The need to define physical groups and their interference boundaries follows
the [design distinction of Hudgens and Halloran][interference].

[thesis]: https://air.unimi.it/handle/2434/356780
[interference]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2600548/
