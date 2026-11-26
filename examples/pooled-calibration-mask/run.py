"""Exact population calibration for two invented recipe strata."""
import json
from fractions import Fraction as F

EVENT_RATES = (F(1, 4), F(3, 4))


def evaluate(low_share):
    weights = (low_share, 1 - low_share)
    pooled_forecast = F(1, 2)
    errors = tuple(rate - pooled_forecast for rate in EVENT_RATES)
    expected_pooled_residual = sum(w * e for w, e in zip(weights, errors))

    def brier(forecasts):
        return sum(w * (p * (1 - q)**2 + (1 - p) * q**2)
                   for w, p, q in zip(weights, EVENT_RATES, forecasts))

    return {
        "pooled_calibration_residual": str(expected_pooled_residual),
        "within_stratum_residuals": list(map(str, errors)),
        "frozen_pooled_brier": str(brier((pooled_forecast,) * 2)),
        "frozen_stratum_brier": str(brier(EVENT_RATES)),
    }


def demonstrate():
    return {"balanced": evaluate(F(1, 2)), "changed_mix": evaluate(F(3, 4))}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
