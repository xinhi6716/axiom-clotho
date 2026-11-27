"""An acceptance flag can bias an otherwise complete randomized comparison."""
from fractions import Fraction
from itertools import combinations
import json


def gap(treated, control):
    return Fraction(sum(treated), len(treated)) - Fraction(sum(control), len(control))


def fixture():
    # Fixed binary outcomes under both possible assignments, not simulated noise.
    endpoint = (0, 0, 1, 1)
    units = set(range(len(endpoint)))
    full, accepted, empty_arm = [], [], 0
    for treated_tuple in combinations(sorted(units), 2):
        treated = set(treated_tuple)
        control = units - treated
        a = [endpoint[i] for i in treated]
        b = [endpoint[i] for i in control]
        full.append(gap(a, b))
        # All treated pass. Only robust controls pass. The endpoint equals the
        # hidden robustness in this toy, but remains fully observed in both arms.
        passed_b = [y for y in b if y == 1]
        if not passed_b:
            empty_arm += 1
            continue
        accepted.append(gap(a, passed_b))
    return {
        "balanced_assignments": len(full),
        "sharp_null_effect": "0",
        "full_roster_expected_gap": str(sum(full) / len(full)),
        "acceptance_comparable_assignments": len(accepted),
        "post_filter_empty_arm_assignments": empty_arm,
        "acceptance_only_expected_gap_given_both_arms_nonempty": str(
            sum(accepted) / len(accepted)
        ),
        "acceptance_only_negative_gaps": sum(d < 0 for d in accepted),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
