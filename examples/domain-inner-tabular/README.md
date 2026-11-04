# Opt-in domain-inner tabular example

These six rows are invented software checks, not research data or evidence of
predictive performance. Run from the repository root with existing dependencies
(`PYTHONPATH=src python -m mlworkflow` can replace `clotho`). No installation or
network is needed. Plans only print JSON; run/predict fit and write local files.

```bash
clotho tabular plan run examples/domain-inner-tabular/train.csv --target target --group group --domain domain --inner-strategy leave_one_domain_out --outer-splits 3 --baselines dummy,ridge
clotho tabular plan predict examples/domain-inner-tabular/train.csv examples/domain-inner-tabular/query.csv --target target --group group --domain domain --inner-strategy leave_one_domain_out --baselines dummy,ridge
clotho tabular run examples/domain-inner-tabular/train.csv --target target --group group --domain domain --inner-strategy leave_one_domain_out --outer-splits 3 --baselines dummy,ridge --out runs/domain-inner-evaluation
clotho tabular predict examples/domain-inner-tabular/train.csv examples/domain-inner-tabular/query.csv --target target --group group --domain domain --inner-strategy leave_one_domain_out --baselines dummy,ridge --out runs/domain-inner-prediction
```

Choose fresh output directories. Query CSV contains only predictors, not domain,
group or target. Domain is removed from inferred features and rejected in explicit
feature allowlists (including `--features-file`). `--domain` and the nondefault
strategy must be supplied together. Omit `--inner-splits`: every explicitly supplied
count is rejected, even if it happens to match a domain count. Each selection holds
out one training domain at a time, so outer training partitions can have different
inner fold counts. The example has 38 evaluation fits and 14 full-training prediction
fits with dummy/ridge; plans and execution use the same schedules.

CSV domain labels are exact strings: `001`, `01`, and `NA` are distinct valid labels.
Labels must be nonempty, unpadded, no longer than 128 characters, and contain no
control characters. Every dependency group must belong to one domain, and every
training partition must retain at least two domains. The domain column must differ
from target/group. Invalid metadata is rejected before fitting.

The direct `GroupBenchRegressor.fit(..., domains=...)` API uses the same
exact-string rule. Python sequences that mix text with numbers, booleans or
bytes are rejected rather than converted into labels or merged with existing
text domains. Supply explicitly reviewed string labels. If the caller already
converted mixed values into a NumPy string array, their original types cannot
be recovered. Aligned pandas Series, string arrays and ordinary string lists
retain their exact labels and existing domain splits.

Default group selection still uses three inner folds and unchanged output schemas.
Opt-in evaluation uses schema 1.3; prediction uses 1.4. No-fit plans omit raw group
and domain labels, but saved fitted bundles include group/domain metadata. Review
artifacts before sharing. Labels and domain structure are user declarations, not
verified science. Outer evaluation still holds out groups, not necessarily domains;
this is not domain-transfer or causal validation, and weighting is still the chosen
row/group objective, not equal-domain weighting.

This is tabular regression only. Binary, demo, measurement-regression-v1 and FET
measurement-regression-v2 configuration schemas do not gain these options.
`mlcalc` and `groupbench` expose the same regression commands without `tabular`.
