# Convert absolute temperature before taking its reciprocal

```sh
python examples/absolute-temperature-feature/run.py
```

The [BIPM SI Brochure, §2.3.1](https://www.bipm.org/en/publications/si-brochure)
defines Celsius temperature by subtracting the reference 273.15 K from
thermodynamic temperature. Temperature differences have the same
numerical magnitude in kelvin and degrees Celsius; the offset instead
applies to absolute values. The [NIST Arrhenius model](https://www.itl.nist.gov/div898/handbook/apr/section1/apr151.htm)
uses reciprocal thermodynamic temperature; the exact unit conversion
here follows the BIPM definition.

For invented values −25 °C and 25 °C, which are 248.15 K and 298.15 K,
the low-minus-high reciprocal feature contrast is 20000/29594369 K⁻¹
regardless of input units. Reciprocating the Celsius numbers directly
gives −2/25, reversing the sign. Standardizing that malformed feature
afterward does not recover the physical reciprocal. Zero degrees
Celsius is valid; zero kelvin has no finite reciprocal. A temperature
increment is not an absolute temperature.

This is a small input contract, not a unit-normalization library, a
fit, or a claim that an Arrhenius mechanism applies to any device.
It neither infers units from values nor authenticates a source label.
