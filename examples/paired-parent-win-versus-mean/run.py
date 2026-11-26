"""Exact invented example; no empirical benchmark or model selection."""
import json
from fractions import Fraction
from itertools import product
from math import comb


def tail(n, wins, p):
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(wins, n + 1))


def main():
    gains = [1] * 7 + [-8]  # baseline loss minus candidate loss
    wins = sum(g > 0 for g in gains)
    sign_p = tail(len(gains), wins, Fraction(1, 2))
    exact_count = sum(sum(signs) >= wins for signs in product((0, 1), repeat=len(gains)))
    assert sign_p == Fraction(exact_count, 2**len(gains)) == Fraction(9, 256)
    p = Fraction(7, 8)
    zero_mean_loss = -p / (1 - p)
    true_mean = p + (1 - p) * zero_mean_loss
    alpha = Fraction(1, 20)
    rejection_count = next(k for k in range(len(gains) + 1)
                           if tail(len(gains), k, Fraction(1, 2)) <= alpha)
    wrong_null_rejection_prob = tail(len(gains), rejection_count, p)
    assert true_mean == 0 and rejection_count == 7
    print(json.dumps({
        "parent_wins": wins,
        "parents": len(gains),
        "parent_mean_gain": str(Fraction(sum(gains), len(gains))),
        "one_sided_fair_sign_p": str(sign_p),
        "zero_mean_example_gain_values": ["1", str(zero_mean_loss)],
        "zero_mean_example_gain_probabilities": [str(p), str(1 - p)],
        "fair_sign_test_rejection_probability_at_5pct_in_zero_mean_example":
            str(wrong_null_rejection_prob),
    }, indent=2))


if __name__ == "__main__":
    main()
