"""Explicit local binary 0/1 workflows; no automatic regression task switching."""
from importlib import import_module as _import_module

__all__ = ["GroupBenchClassifier", "BinaryBenchmarkResult", "BinaryPredictionResult",
           "benchmark_binary", "fit_predict_binary", "plan_binary_benchmark",
           "plan_binary_prediction", "make_binary_fixture"]


_EXPORT_MODULES = dict.fromkeys(__all__, ".workflow")
_EXPORT_MODULES.update(make_binary_fixture=".data", GroupBenchClassifier=".model")


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
