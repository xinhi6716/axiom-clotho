"""Exact educational replay of one draw from an unequal block lottery."""

from fractions import Fraction as F


def replay(rows):
    """One row per known block: (ID, target mass, draw probability, mean loss)."""
    keys = [key for key, _, _, _ in rows]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Expected one entry per target block.")
    if any(mass < 0 or prob < 0 or (mass > 0 and prob == 0) for _, mass, prob, _ in rows):
        raise ValueError("Every positive target mass needs known positive inclusion.")
    if sum(mass for _, mass, _, _ in rows) != 1 or sum(prob for _, _, prob, _ in rows) != 1:
        raise ValueError("Target and lottery masses each sum to one.")
    if any(loss is None for _, _, prob, loss in rows if prob > 0):
        raise ValueError("The exact toy replay requires known outcomes for possible draws.")
    possible = [(mass, prob, loss) for _, mass, prob, loss in rows if prob > 0]
    contributions = [(prob, mass * loss / prob) for mass, prob, loss in possible]
    mean = sum(prob * value for prob, value in contributions)
    return {
        "naive_expectation": sum(prob * loss for _, prob, loss in possible),
        "ht_expectation": mean,
        "ht_oracle_variance": sum(prob * (value - mean)**2 for prob, value in contributions),
        "clipped_at_two_expectation": sum(prob * mass * loss * min(F(2), 1/prob)
                                         for mass, prob, loss in possible),
    }


def main():
    for scope, weights in (("equal blocks", (F(1, 2), F(1, 2))),
                           ("equal devices", (F(1, 4), F(3, 4)))):
        a, b = weights
        result = replay((("A", a, F(3, 4), F(0)), ("B", b, F(1, 4), F(1))))
        print(scope + ": " + ", ".join(f"{k}={v}" for k,v in result.items()))


if __name__ == "__main__":
    main()
