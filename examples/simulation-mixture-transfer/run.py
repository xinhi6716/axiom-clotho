"""An exact example where a mixed score rewards simulated-data transfer that fails."""
from fractions import Fraction
import json


def average(values):
    return sum(map(Fraction, values)) / len(values)


def mae(prediction, targets):
    return average([abs(prediction - target) for target in targets])


def demonstrate():
    train_measured, train_simulated = [0, 0], [8] * 6
    # Values alone do not make these the same observations. The lists
    # stand for independently reserved rows in the two named sources.
    test_measured, test_simulated = [0, 0], [8] * 6
    measured_only = average(train_measured)
    equal_row_hybrid = average(train_measured + train_simulated)
    output = {
        method: {
            "forecast": str(p),
            "mixed_mae": str(mae(p, test_measured + test_simulated)),
            "measured_mae": str(mae(p, test_measured)),
            "simulated_mae": str(mae(p, test_simulated)),
        }
        for method, p in (("measured_only", measured_only), ("equal_row_hybrid", equal_row_hybrid))
    }
    assert output["measured_only"]["mixed_mae"] == "6"
    assert output["equal_row_hybrid"]["mixed_mae"] == "3"
    assert output["equal_row_hybrid"]["measured_mae"] == "6"
    return output


if __name__ == "__main__":
    print(json.dumps(demonstrate(), sort_keys=True))
