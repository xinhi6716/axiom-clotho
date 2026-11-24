use clotho_core::group_max_calibration::{calibrate_group_max, CalibrationRow, GroupMaxCutoff, RowKey};

fn main() {
    const SETTINGS: usize = 25;
    const REPEATS_PER_SETTING: usize = 3 * 3;
    const MACHINES: usize = 3;
    let names: Vec<_> = (0..SETTINGS * REPEATS_PER_SETTING * MACHINES)
        .map(|i| format!("synthetic-{i}"))
        .collect();
    let roster: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(i, row)| RowKey {
            group: ["toy-A", "toy-B", "toy-C"][i % MACHINES],
            row,
        })
        .collect();
    let rows: Vec<_> = roster
        .iter()
        .copied()
        .map(|key| CalibrationRow { key, observed: 1.0, prediction: 0.0 })
        .collect();

    let group_result = calibrate_group_max(&roster, &rows, 95, 100).unwrap();
    assert_eq!(group_result, GroupMaxCutoff::InsufficientFinite { groups: 3, rank: 4 });

    // Deliberately wrong grouping for the new-machine question, for comparison only.
    let row_roster: Vec<_> = roster.iter().map(|key| RowKey { group: key.row, row: key.row }).collect();
    let row_scores: Vec<_> = row_roster
        .iter()
        .copied()
        .map(|key| CalibrationRow { key, observed: 1.0, prediction: 0.0 })
        .collect();
    let row_result = calibrate_group_max(&row_roster, &row_scores, 95, 100).unwrap();
    assert!(matches!(row_result, GroupMaxCutoff::Finite { groups: 675, rank: 643, .. }));

    println!("675 synthetic scores, 3 declared machines: rank 4, insufficient finite support");
    println!("wrong unit (one group per row): 675 groups, finite rank 643");
}
