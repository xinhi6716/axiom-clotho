# Product and methodology references

Checked 2026-10-23. These are external references, not bundled implementation or model assets.

## TabPFN: product ergonomics only

- Official repository: https://github.com/PriorLabs/TabPFN
- Official quickstart: https://docs.priorlabs.ai/quickstart
- Official code license: https://github.com/PriorLabs/TabPFN/blob/main/LICENSE
- Official license distinctions: https://github.com/PriorLabs/TabPFN/blob/main/README.md#license
- Official model license overview: https://docs.priorlabs.ai/models#tabpfn-model-license

The current source license is Apache 2.0. The README distinguishes code from separately licensed weights: recent weights use non-commercial terms; TabPFN-2 weights use a Prior Labs license with an additional attribution requirement. Always check the exact asset/version before using it. This is not legal advice or permission to redistribute another project's assets.

Design lessons: make the first useful example short, use a recognizable estimator interface, state local execution requirements, separate getting-started examples from deeper documentation, and distinguish code licensing from weight licensing. No algorithm, source code, weights, artwork, or copied documentation from TabPFN was incorporated. GroupBench is a different category: conventional local sklearn baselines plus group-aware evaluation and reporting.

## Implementation dependencies

- scikit-learn: https://scikit-learn.org/stable/
- pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/
- threadpoolctl: https://github.com/joblib/threadpoolctl

The project depends on these installed libraries; it does not vendor them. Their licenses and third-party notices remain applicable. Runtime computation requires no internet or model downloads.
