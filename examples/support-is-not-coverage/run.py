"""Two invented outcome worlds with identical feature support and intervals."""

from decimal import Decimal

import pandas as pd

from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support
from groupbench.interval_evaluation import diagnose_saved_interval_whole_group_coverage


def main():
    training = validate_features(pd.DataFrame({"x": [0.0, 1.0, 2.0]}))
    query = pd.DataFrame({"x": [0.0, 1.0, 1.0, 2.0]})
    fixed_support = prediction_input_support(training, query)
    if fixed_support["rows_with_any_flag"] != 0:
        raise AssertionError("Unexpected support flag.")
    roster = ["r1", "r2", "r3", "r4"]

    def score(outcomes):
        return diagnose_saved_interval_whole_group_coverage({
            "schema_version": 1, "expected_row_ids": roster, "alpha": Decimal("0.2"),
            "metadata": {"target_definition": "invented response", "value_unit": "toy",
                         "group_definition": "toy device", "cohort_definition": "four fixed rows"},
            "rows": [{"row_id": row, "group": group, "observed": y,
                      "lower": 0, "upper": 2, "status": "ok"}
                     for row, group, y in zip(roster, ["D1", "D1", "D2", "D2"], outcomes)],
        })["whole_group"]["observed_coverage_exact"]

    unshifted = score([0, 1, 1, 2])
    shifted = score([3, 4, 4, 5])
    if (unshifted, shifted) != ("1", "0"):
        raise AssertionError("Unexpected observed whole-device coverage.")
    print("same feature inputs: 0 of 4 rows flagged in both worlds")
    print("observed whole-device coverage: 2/2 before shift, 0/2 after shift")


if __name__ == "__main__":
    main()
