"""Exact fixed-state witness for separating geometric conductance terms."""

from fractions import Fraction as F
import json


def design(width, length):
    if any(type(x) not in (int, F) or x <= 0 for x in (width, length)):
        raise ValueError("Expected positive exact rational dimensions.")
    w, l = F(width), F(length)
    return F(1), w*l, 2*(w+l)


def determinant(rows):
    a, b, c = rows
    return (a[0]*(b[1]*c[2] - b[2]*c[1])
            - a[1]*(b[0]*c[2] - b[2]*c[0])
            + a[2]*(b[0]*c[1] - b[1]*c[0]))


def area_perimeter_fit(rows, y):
    d = determinant(rows)
    if not d:
        raise ValueError("The geometric contrast does not identify three terms.")
    return tuple(determinant([tuple(v if j == k else x for j, x in enumerate(row))
                              for row, v in zip(rows, y, strict=True)])/d
                 for k in range(3))


def predict(row, coef):
    return sum(x*b for x, b in zip(row, coef, strict=True))


def witness():
    train = [design(*shape) for shape in [(1, 1), (2, 2), (1, 4)]]
    held = design(3, 3)
    truth = (F(2), F(3), F(1))
    y = [predict(x, truth) for x in train]
    # Train only: exact least-squares slope/intercept, including duplicate area.
    a = [row[1] for row in train]
    n = len(y)
    slope = (n*sum(x*v for x, v in zip(a, y)) - sum(a)*sum(y)) / (
        n*sum(x*x for x in a) - sum(a)**2)
    baseline = (sum(y) - slope*sum(a))/n + slope*held[1]
    fitted = area_perimeter_fit(train, y)
    expected = predict(held, truth)
    return {
        "training_area_perimeter_conductance": [
            [str(x[1]), str(x[2]), str(v)] for x, v in zip(train, y)],
        "training_determinant": str(determinant(train)),
        "fitted_intercept_area_perimeter": list(map(str, fitted)),
        "held_truth": str(expected),
        "held_area_only": str(baseline),
        "held_area_only_abs_error": str(abs(baseline-expected)),
        "held_area_perimeter_abs_error": str(abs(predict(held, fitted)-expected)),
    }


if __name__ == "__main__":
    print(json.dumps(witness(), sort_keys=True))
