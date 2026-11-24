"""Audit invented paired errors against a separately declared parent roster."""

from groupbench.metrics import paired_parent_mae_influence


def main():
    # Frozen metadata: four readings from D1, one from D2, and one from D3.
    roster = [(f"r{i}", "D1", "P1") for i in range(1, 5)]
    roster += [("r5", "D2", "P1"), ("r6", "D3", "P2")]
    # Columns: row, device, parent, observation, prediction A, prediction B.
    rows = [
        ("r1", "D1", "P1", 0, 0, 1),
        ("r2", "D1", "P1", 0, 0, 1),
        ("r3", "D1", "P1", 0, 0, 1),
        ("r4", "D1", "P1", 0, 0, 1),
        ("r5", "D2", "P1", 0, 0, 3),
        ("r6", "D3", "P2", 0, 6, 0),
    ]
    audit = paired_parent_mae_influence(rows, expected_roster=roster)
    support = tuple(audit[key] for key in ("n_rows", "n_devices", "n_declared_parents"))
    effect = audit["mae_b_minus_a"]
    omission = audit["leave_one_parent_out_mae_b_minus_a"]
    if support != (6, 3, 2) or effect != -2 or omission != {"min": -6, "max": 2}:
        raise AssertionError("Unexpected support or parent-weighted influence.")
    print(f"rows={support[0]} devices={support[1]} declared_parents={support[2]}")
    print(f"equal-parent MAE B minus A: {effect:g}")
    print(f"omit-one-parent range: [{omission['min']:g}, {omission['max']:g}]")

    # One parent can still be described, but omission would leave no support.
    single = paired_parent_mae_influence(rows[:-1], expected_roster=roster[:-1])
    if single["leave_one_parent_out_mae_b_minus_a"] is not None:
        raise AssertionError("A one-parent input cannot have an omission range.")
    print("one parent: omission range unavailable")

    # Remove one repeated reading without removing its device or parent.
    try:
        paired_parent_mae_influence(rows[1:], expected_roster=roster)
    except ValueError as error:
        if "missing row" not in str(error):
            raise
        print(f"omission rejected: {error}")
    else:
        raise AssertionError("Missing scheduled reading was accepted.")


if __name__ == "__main__":
    main()
