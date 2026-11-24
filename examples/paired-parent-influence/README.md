# Check how much one declared parent changes a paired MAE comparison

With the current source checkout and dependencies, run from the repository root:

```sh
PYTHONPATH=src python examples/paired-parent-influence/run.py
```

This toy has six invented readings, three devices and two declared parents.
First average the paired losses within each device, then average devices within
each parent, then weight the two parents equally. The four repeat readings
from D1 do not give it four times the device weight. Parent P1 has B-minus-A
MAE of +2, P2 has −6, and the overall contrast is −2. Omitting one parent
leaves a contrast of either −6 or +2, so the apparent winner depends on which
parent remains. With only one parent, there is no omission range.

The omission range is a descriptive sensitivity check, not a confidence
interval or evidence that the two parents are physically independent. The
example declares expected row IDs and device/parent mappings separately from
the paired forecasts. Removing one repeated reading while retaining every
device and parent is rejected. No models are fitted; use pseudonymous parent
labels in any shared report because the result returns them verbatim.
