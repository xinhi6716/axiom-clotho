"""Exact out-of-fold comparison on invented, physically distinct devices."""

from collections import defaultdict
from fractions import Fraction


def compare(rows, hold_index):
    """Use the same held roster and train-only information for both rules."""
    folds = defaultdict(list)
    for row in rows:
        folds[row[hold_index]].append(row)
    errors = {}
    seen = 0
    for key, held in folds.items():
        train = [row for row in rows if row[hold_index] != key]
        train_devices = {(p, d) for p, d, _ in train}
        held_devices = {(p, d) for p, d, _ in held}
        if train_devices & held_devices:
            raise AssertionError("A physical device crossed the split.")
        train_protocols = defaultdict(list)
        for protocol, _, outcome in train:
            train_protocols[protocol].append(outcome)
        baseline = Fraction(sum(y for _, _, y in train), len(train))
        for p, d, y in held:
            if (p, d) in errors:
                raise AssertionError("A device was evaluated twice.")
            siblings = train_protocols.get(p)
            seen += bool(siblings)
            lookup = Fraction(sum(siblings), len(siblings)) if siblings else baseline
            errors[p, d] = (abs(y - lookup), abs(y - baseline))
    if len(errors) != len(rows):
        raise AssertionError("Evaluation omitted one or more devices.")
    return tuple(sum(err[i] for err in errors.values()) / len(rows) for i in (0, 1)) + (seen,)


def main():
    # Replicate slots are only fold labels, never a global physical cell ID.
    rows = [(p, d, y) for p, y in enumerate((-1, -1, 1, 1)) for d in range(5)]
    for index, name in ((1, "held devices at seen protocols"), (0, "held protocols")):
        lookup, mean, seen = compare(rows, index)
        print(f"{name}: seen={seen}/{len(rows)}, lookup MAE={lookup}, train-mean MAE={mean}")


if __name__ == "__main__":
    main()
