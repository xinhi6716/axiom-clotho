"""Exact dependence of two contrasts sharing a control in each toy plate."""

from fractions import Fraction as F
from itertools import product


def moments():
    pairs = []
    for ca, aa, ba, cb, ab, bb in product((-1, 1), repeat=6):
        pairs.append((F(aa - ca + ab - cb, 2), F(ba - ca + bb - cb, 2)))
    n = len(pairs)
    m1, m2 = (sum(v[i] for v in pairs) / n for i in range(2))
    v1, v2 = (sum((v[i] - m)**2 for v in pairs) / n for i, m in enumerate((m1, m2)))
    covariance = sum((a - m1) * (b - m2) for a, b in pairs) / n
    true_variance = sum(((a + b - m1 - m2) / 2)**2 for a, b in pairs) / n
    return v1, v2, covariance, true_variance, (v1 + v2) / 4


if __name__ == "__main__":
    va, vb, covariance, true, naive = moments()
    print(f"contrast variances: {va}, {vb}; covariance: {covariance}")
    print(f"variance of mean contrast: {true}; assuming independence: {naive}")
