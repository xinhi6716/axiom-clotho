# Unresolved clock units can reverse a model comparison

```sh
python examples/ambiguous-clock-units/run.py
```

The [BIPM SI prefix table](https://www.bipm.org/en/measurement-units/si-prefixes)
defines milli as 10⁻³. That fixes the conversion, not which unit produced a
particular exported column. Bytes, headers, and processing code may disagree
on the physical interpretation. This fixture is wholly invented and audits
no individual dataset or implementation.

The clock origin is independently expressed in milliseconds at 10 ms.
Two saved candidate probabilities depend on features with raw ready stamps
2 and 4, on the same clock origin in either hypothetical unit. A fixed
comparator's predictions are independently ready. Compare the same
two endpoints, using a predefined neutral forecast of 1/2 when the
candidate feature cannot be used.

- If the stamps are certified milliseconds, both features are available.
  A's mean Brier is 0 and B's is 1/16.
- If the stamps are certified seconds, neither is available by 10 ms.
  A uses its fallback, giving 1/4 against the same 1/16.
- If provenance is unresolved, flag both features unavailable instead of
  selecting the unit which makes A win. The same fallback is evaluated,
  but this does not assert that the stamps are in seconds.

The ranking reversal compares prespecified policies on the same complete
roster, not changed labels, row dropping, fitting, or a claim about the
physical device. Before using the column establish its epoch, unit and
whether the field actually denotes availability on the required clock.
A plausible sampling interval, an axis label, or the smaller error does
not establish those facts.
