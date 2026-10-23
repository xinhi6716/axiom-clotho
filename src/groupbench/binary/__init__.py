"""Explicit local binary 0/1 workflows; no automatic regression task switching."""
from .data import make_binary_fixture
from .model import GroupBenchClassifier
from .workflow import (BinaryBenchmarkResult, BinaryPredictionResult, benchmark_binary,
                       fit_predict_binary, plan_binary_benchmark, plan_binary_prediction)

__all__ = ["GroupBenchClassifier", "BinaryBenchmarkResult", "BinaryPredictionResult",
           "benchmark_binary", "fit_predict_binary", "plan_binary_benchmark",
           "plan_binary_prediction", "make_binary_fixture"]
