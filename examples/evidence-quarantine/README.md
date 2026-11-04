# Check quantity-local quarantine without reading values

From the repository root:

```bash
PYTHONPATH=src python -S examples/evidence-quarantine/preview.py
```

This standard-library-only example creates invented source/group/endpoint
metadata, asserts four expected statuses, and prints JSON. It contains no
measurements, reads no research files, writes no files, and makes zero model
fits. No rows are deleted or admitted to training.

The cases show clean isolation, an injected exact quarantined key, an unrelated
endpoint in the same source/group, and a quantity family whose endpoint mapping
is unknown. An unknown mapping must stay `mapping_not_established`, not become
a clean result because its name is absent from the observations.

`no_declared_quarantine_matches` covers only the supplied exact scopes and
rows. It does not verify the completeness of a conflict registry, source truth,
scientific admissibility, physical specimen pairing, or independent samples.
Read the full [evidence-quarantine contract](../../docs/EVIDENCE_QUARANTINE.md)
before projecting your own metadata.
