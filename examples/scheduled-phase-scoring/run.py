"""Run an invented frozen phase-roster example without fitting or writing files."""

from groupbench.metrics import scheduled_phase_paired_mae


def main():
    cells = [("D1", "early"), ("D1", "late"), ("D2", "early"), ("D2", "late")]
    roster = [
        ("r1", "D1", "early"), ("r2", "D1", "early"), ("r3", "D1", "late"),
        ("r4", "D2", "early"), ("r5", "D2", "early"), ("r6", "D2", "early"),
        ("r7", "D2", "early"), ("r8", "D2", "late"),
    ]
    # Columns: row ID, device, phase, observation, forecast A, forecast B.
    paired = [
        ("r1", "D1", "early", 0, 0, 1), ("r2", "D1", "early", 0, 0, 1),
        ("r3", "D1", "late", 0, 0, 1), ("r4", "D2", "early", 0, 0, 1),
        ("r5", "D2", "early", 0, 0, 1), ("r6", "D2", "early", 0, 0, 1),
        ("r7", "D2", "early", 0, 0, 1), ("r8", "D2", "late", 0, 5, 1),
    ]

    def score(rows):
        return scheduled_phase_paired_mae(rows, expected_roster=roster,
            expected_cells=cells, phase_weights={"early": 1, "late": 1})

    result = score(paired)
    expected = {"micro": (0.625, "A"), "equal_device": (0.5, "A"),
                "fixed_phase_equal_device": (1.25, "B")}
    for name, pair in expected.items():
        metrics = result[name]
        if (metrics["mae_a"], metrics["better"]) != pair or metrics["mae_b"] != 1:
            raise AssertionError(f"Unexpected weighting result: {name}")
        print(f"{name}: A={metrics['mae_a']:g} B={metrics['mae_b']:g} winner={metrics['better']}")
    # Keep all four cells populated, but remove a scheduled row.
    try:
        score(paired[1:])
    except ValueError as error:
        if "missing row" not in str(error):
            raise
        print(f"omission rejected: {error}")
    else:
        raise AssertionError("A missing scheduled row was accepted")


if __name__ == "__main__":
    main()
