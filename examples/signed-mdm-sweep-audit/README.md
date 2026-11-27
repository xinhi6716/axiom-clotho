# Audit signed sweep identities before comparing body-bias measurements

The [IHP Open PDK repository](https://github.com/IHP-GmbH/IHP-Open-PDK)
identifies MOS/HBT measurements in MDM format. This optional local
example audits one restricted, DC drain-current/gate-voltage subset
with fixed drain, body and source voltages per block. It imports no
measurement files, downloads nothing and emits only aggregate metadata.

```sh
python examples/signed-mdm-sweep-audit/run.py path/to/local.mdm --sha256 EXPECTED_SHA256
```

Use a digest obtained from a separately trusted manifest or your frozen
source audit. The script checks the exact supplied bytes, rejects
duplicate numeric (Vd,Vb,Vs) keys including equivalent zero spellings,
and checks that all blocks have the same nonzero constant signed gate
step. It does not take absolute values to force NMOS and PMOS histories
onto one common sweep. The drain-current sign check is only for the
last gate point of each block, never for the entire trace.

Passes certify the named file structure only, not that every expected
sweep was attempted, devices are independent, labels share metrology,
or body bias caused any effect. Preserve source file identity along
with a block key; the same numeric tuple in two files need not identify
the same device or acquisition. Do not use these blocks interchangeably
as random row, physical-device, or geometry holdouts.
