"""Local-first grouped tabular regression. No network or model downloads."""

from importlib import import_module as _import_module

__version__ = "0.3.0b7"

# Resolve original objects, not wrappers: identity, signatures and pickle lookup
# remain those of the defining modules. Star import intentionally loads all APIs.
__all__ = [
    'BenchmarkResult',
    'GroupBenchRegressor',
    'GroupKFold',
    'benchmark',
    'make_materials_fixture',
    'PredictionResult',
    'fit_predict',
    'plan_benchmark',
    'plan_prediction',
    'GroupBenchClassifier',
    'BinaryBenchmarkResult',
    'BinaryPredictionResult',
    'benchmark_binary',
    'fit_predict_binary',
    'plan_binary_benchmark',
    'plan_binary_prediction',
    'make_binary_fixture',
    'PredictionSchemaResult',
    'validate_prediction_bundle_schema',
    'EvaluationSchemaResult',
    'validate_evaluation_bundle',
    'GroupForwardEvaluationError',
    'GroupForwardEvaluationPreview',
    'GroupForwardEvaluationResult',
    'preview_group_forward_evaluation',
    'evaluate_group_forward_baselines',
    'GroupForwardBundleError',
    'GroupForwardBundleValidation',
    'write_group_forward_evaluation_bundle',
    'validate_group_forward_evaluation_bundle',
]

_EXPORT_MODULES = {
    'BenchmarkResult': '.evaluation',
    'GroupBenchRegressor': '.model',
    'GroupKFold': '.splitting',
    'benchmark': '.evaluation',
    'make_materials_fixture': '.data',
    'PredictionResult': '.prediction',
    'fit_predict': '.prediction',
    'plan_benchmark': '.planning',
    'plan_prediction': '.planning',
    'GroupBenchClassifier': '.binary',
    'BinaryBenchmarkResult': '.binary',
    'BinaryPredictionResult': '.binary',
    'benchmark_binary': '.binary',
    'fit_predict_binary': '.binary',
    'plan_binary_benchmark': '.binary',
    'plan_binary_prediction': '.binary',
    'make_binary_fixture': '.binary',
    'PredictionSchemaResult': '.prediction_schema',
    'validate_prediction_bundle_schema': '.prediction_schema',
    'EvaluationSchemaResult': '.evaluation_schema',
    'validate_evaluation_bundle': '.evaluation_schema',
    'GroupForwardEvaluationError': '.group_time_evaluation',
    'GroupForwardEvaluationPreview': '.group_time_evaluation',
    'GroupForwardEvaluationResult': '.group_time_evaluation',
    'preview_group_forward_evaluation': '.group_time_evaluation',
    'evaluate_group_forward_baselines': '.group_time_evaluation',
    'GroupForwardBundleError': '.group_time_saved',
    'GroupForwardBundleValidation': '.group_time_saved',
    'write_group_forward_evaluation_bundle': '.group_time_saved',
    'validate_group_forward_evaluation_bundle': '.group_time_saved',
}

# Preserve the bounded set of incidental submodule attributes exposed by the
# previous eager initializer. These are discoverable without importing them.
_SUBMODULES = frozenset({
    "baselines", "binary", "categorical", "data", "domain_selection",
    "evaluation", "evaluation_schema", "group_time_evaluation", "group_time_json",
    "group_time_plan", "group_time_saved", "input_support", "kernel", "metrics",
    "model", "package_source", "planning", "prediction", "prediction_schema",
    "report", "splitting", "verification",
})


def __getattr__(name):
    if name in _EXPORT_MODULES:
        value = getattr(_import_module(_EXPORT_MODULES[name], __name__), name)
    elif name in _SUBMODULES:
        value = _import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Do not cache a failed or partial resolution.
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
