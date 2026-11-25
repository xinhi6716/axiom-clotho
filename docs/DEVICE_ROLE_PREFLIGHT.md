# Resolve resumed devices before assigning calibration roles

A device can resume in a later acquisition under a different local name.
Treating its two names as distinct calibration devices can change a group-max
rank or let the same physical device occur in training and calibration or
evaluation. This preflight resolves declared aliases, retains every read and
refuses such cross-role overlap. It is a separate Python API, not an extension
of `audit_physical_device_split`, a calibration algorithm, or part of an
automatic fit/score pipeline.

```python
from groupbench.splitting import (
    PhysicalDevice, RunDeviceAlias, RunDeviceBinding, SourceReadKey,
    preflight_device_roles,
)

former = RunDeviceAlias("bench", "run-1", "cell-4")
resumed = RunDeviceAlias("bench", "run-2", "cell-9")
train = RunDeviceAlias("bench", "run-1", "cell-5")
test = RunDeviceAlias("bench", "run-2", "cell-6")
cell = PhysicalDevice("device-ledger", 4)
bindings = (
    RunDeviceBinding(former, cell),
    RunDeviceBinding(resumed, cell),
    RunDeviceBinding(train, PhysicalDevice("device-ledger", 5)),
    RunDeviceBinding(test, PhysicalDevice("device-ledger", 6)),
)
first = SourceReadKey(former, "t1")
second = SourceReadKey(resumed, "t1")
training = (SourceReadKey(train, "t1"),)
evaluation = (SourceReadKey(test, "t1"),)
resolved = preflight_device_roles(
    train=training, calibration=(first, second), evaluation=evaluation,
    bindings=bindings,
)
assert tuple(r.key for r in resolved.calibration) == (first, second)
assert len({r.physical for r in resolved.calibration}) == 1

try:
    preflight_device_roles(
        train=training, calibration=(first,), evaluation=(*evaluation, second),
        bindings=bindings,
    )
except ValueError as err:
    assert "calibration and evaluation" in str(err)
else:
    raise AssertionError("One physical device was allowed across roles")
```

- Supply nonempty list/tuple rosters for train, calibration and evaluation
  with up to 20,000 reads across all three. Each original key is the compound
  `(source, run, raw_device, local_observation)`. Run is the declared episode
  namespace; local IDs may repeat across different episodes, but the full
  read key may not repeat. Distinct local reads on one device in one run are
  valid. The result retains every input key in its original role and order.
- Supply exactly one `RunDeviceBinding` per distinct alias
  `(source, run, raw_device)`, with no duplicates, missing or extra aliases.
  Cross-export aliases for the same device require the same canonical
  `PhysicalDevice` namespace and ID. The authority namespace is not inferred.
- Text is literal and case-sensitive, unpadded, 1–256 characters, without
  control or surrogate characters. Raw device, local observation and
  canonical device IDs can also be signed 64-bit built-in integers;
  `1` and `"1"` differ. `unknown`, `unreported`, and `not_reported`
  (any case) are invalid device IDs. There is no placeholder fallback.
- Use the **original read key** for aligned rows and the **canonical physical
  device** for downstream grouping. Do not deduplicate rows by physical
  device; a later read can have the largest error for that device.

This checks exact supplied metadata, without reading measurements, fitting
a model, computing a rank, or inspecting the provenance of the crosswalk.
It cannot detect two aliases wrongly declared as different entities. A pass
does not certify prospective reservation or physically independent wafers,
dies or protocols. Training and calibration chronology, held-out protocol
design, uncertainty coverage, and any subsequent group-max scores require
separate evidence.
