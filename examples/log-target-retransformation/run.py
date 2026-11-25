"""Compare arithmetic-mean predictions from a frozen log-center toy."""

from fractions import Fraction as F

DEVELOPMENT = {"quiet": (F(1), F(1)), "variable": (F(1, 4), F(4))}
EVALUATION = {"quiet": (F(1), F(1)), "variable": (F(4), F(1, 4))}


def mean(values):
    return sum(values) / len(values)


def freeze_predictions(residual_factors):
    """Each row is 2**log_residual; fitted log2 centers are fixed at zero."""
    pooled = mean(tuple(value for rows in residual_factors.values() for value in rows))
    return {
        "direct inverse": {g: F(1) for g in residual_factors},
        "pooled smearing": {g: pooled for g in residual_factors},
        "stratum smearing": {g: mean(rows) for g, rows in residual_factors.items()},
    }


def squared_error(prediction, outcomes):
    return mean(tuple((prediction[g] - y)**2 for g, rows in outcomes.items() for y in rows))


if __name__ == "__main__":
    for name, prediction in freeze_predictions(DEVELOPMENT).items():
        print(f"{name}: quiet={prediction['quiet']}, variable={prediction['variable']}, "
              f"MSE={squared_error(prediction, EVALUATION)}")
