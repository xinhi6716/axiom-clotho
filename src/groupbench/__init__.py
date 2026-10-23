"""Local-first grouped tabular regression. No network or model downloads."""

__version__ = "0.1.0"

from .data import make_materials_fixture
from .evaluation import BenchmarkResult, benchmark
from .model import GroupBenchRegressor
from .prediction import PredictionResult, fit_predict
from .planning import plan_benchmark, plan_prediction
from .splitting import GroupKFold

__all__ = ["BenchmarkResult", "GroupBenchRegressor", "GroupKFold", "benchmark", "make_materials_fixture", "PredictionResult", "fit_predict", "plan_benchmark", "plan_prediction"]
