"""Local-first grouped tabular regression. No network or model downloads."""

__version__ = "0.3.0b7"

from .data import make_materials_fixture
from .evaluation import BenchmarkResult, benchmark
from .model import GroupBenchRegressor
from .prediction import PredictionResult, fit_predict
from .planning import plan_benchmark, plan_prediction
from .splitting import GroupKFold

__all__ = ["BenchmarkResult", "GroupBenchRegressor", "GroupKFold", "benchmark", "make_materials_fixture", "PredictionResult", "fit_predict", "plan_benchmark", "plan_prediction"]

# Additive explicit binary APIs; regression entry points keep their contracts.
from .binary import (GroupBenchClassifier, BinaryBenchmarkResult, BinaryPredictionResult,
                     benchmark_binary, fit_predict_binary, plan_binary_benchmark,
                     plan_binary_prediction, make_binary_fixture)
__all__ += ["GroupBenchClassifier", "BinaryBenchmarkResult", "BinaryPredictionResult",
            "benchmark_binary", "fit_predict_binary", "plan_binary_benchmark",
            "plan_binary_prediction", "make_binary_fixture"]

# Explicit optional saved-prediction metadata consistency, separate from byte verification.
from .prediction_schema import PredictionSchemaResult, validate_prediction_bundle_schema
__all__ += ["PredictionSchemaResult", "validate_prediction_bundle_schema"]
