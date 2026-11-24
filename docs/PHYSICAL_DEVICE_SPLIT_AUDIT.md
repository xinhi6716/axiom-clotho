# Check declared physical-device overlap across a split

Distinct device labels do not necessarily mean distinct devices. An export can
rename a device, or two exports can use different labels for the same device.
Use `groupbench.splitting.audit_physical_device_split` before claiming a
predeclared split holds out devices. It compares only train and test and changes
no splitter, estimator, evaluation schema or defaults.

Supply the train and test row positions (as lists, tuples or one-dimensional
integer NumPy arrays), the aliases of **all** rows in the exact same positional
order as the split, and one binding for each distinct alias. Each alias is
scoped to its source. Many aliases, including aliases from different sources,
can map to the same physical device. Canonical physical IDs are scoped to an
identity namespace: to represent one device with aliases in multiple exports,
map them to the **same** namespace and ID.

IDs are literal unpadded strings of up to 256 characters or built-in signed
64-bit integers; integer `1` and string `"1"` differ. Source and device namespace
strings have the same text constraints. Supply bindings as a list or tuple so
duplicate mappings remain detectable. Partial coverage is allowed: unused rows
need bindings but do not affect overlap or device counts. Both partitions must
be nonempty, disjoint, and contain distinct in-bounds integers. Limit: 20,000
rows/bindings. A successful report gives counts, not the underlying IDs.

```python
from groupbench.splitting import (
    DeviceAlias, DeviceAliasBinding, PhysicalDevice, audit_physical_device_split,
)

a = DeviceAlias("original-export", "cell-04")
a_renamed = DeviceAlias("re-export", "cell-19")
b = DeviceAlias("re-export", "cell-04")
device_a = PhysicalDevice("bench-ledger", 7)
device_b = PhysicalDevice("bench-ledger", 9)
roster = (a, a_renamed, b)
aliases = (DeviceAliasBinding(a, device_a),
           DeviceAliasBinding(a_renamed, device_a),
           DeviceAliasBinding(b, device_b))
audit = audit_physical_device_split((0, 1), (2,), roster, bindings=aliases)
assert (audit["train_alias_count"], audit["train_physical_device_count"]) == (2, 1)

try:
    audit_physical_device_split((0,), (1,), roster, bindings=aliases)
except ValueError as error:
    assert "declared physical device(s) overlap" in str(error)
else:
    raise AssertionError("Same-device aliases crossed the split")
```

The audit checks supplied declarations, not whether the roster order, alias
mapping or identity namespace represents real hardware. It cannot detect a
missing alias relationship that was declared as two different devices. If
physical IDs are unknown, a disjoint address split is not evidence of device
transfer. This audit does not check wafer/die independence, chronological
availability, feature ancestry, model consumption, or outcomes. It does not
score or fit anything. Apply it to each actual fold for which you want to make
a disjoint-device claim.
