"""A paired loss ranking can change with the declared target history mix."""

from fractions import Fraction as F

# Matched candidate and baseline losses, two distinct toy rows per history.
ROWS = {
    "uncycled": ((1, 2), (0, 1)),
    "cycled": ((3, 1), (4, 2)),
}


def estimated_contrast(uncycled_share):
    p = uncycled_share
    if not 0 <= p <= 1:
        raise ValueError("Target share must be between zero and one.")

    def paired_mean(history):
        rows = ROWS[history]
        return F(sum(candidate - baseline for candidate, baseline in rows), len(rows))

    return p * paired_mean("uncycled") + (1 - p) * paired_mean("cycled")


if __name__ == "__main__":
    for p in (F(1, 2), F(4, 5), F(2, 3)):
        print(f"uncycled share={p}: candidate-minus-baseline={estimated_contrast(p)}")
