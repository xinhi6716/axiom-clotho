//! Synthetic paired-state readpoint example, not experimental device data.
//! cargo run --manifest-path rust/Cargo.toml -p clotho-core --example rram_paired_estimand
use clotho_core::rram::{extract_cohort, Acquisition, Branch, Conditions, Features, Sample, Settings, State};

fn branches() -> Vec<Branch> {
    let mut branches = Vec::new();
    for (i, (condition, hrs, lrs)) in [
        ("room", 10.0, 1.0),
        ("room", 20.0, 100.0),
        ("room", 30.0, 10.0),
        ("warm", 80.0, 10.0),
    ].into_iter().enumerate() {
        for (state, resistance, label) in [(State::Hrs, hrs, "high"), (State::Lrs, lrs, "low")] {
            branches.push(Branch {
                acquisition: Acquisition {
                    batch: "synthetic-batch".into(),
                    wafer: "synthetic-wafer".into(),
                    device: "one-device".into(),
                    cycle: format!("cycle-{}", i + 1),
                    replicate: format!("repeat-{}", i + 1),
                    branch: label.into(),
                    state,
                    condition: condition.into(),
                    protocol: "invented-protocol".into(),
                },
                conditions: Conditions {
                    read_voltage_v: 1.0,
                    temperature_k: if condition == "room" { 300.0 } else { 350.0 },
                    read_delay_s: 0.01,
                },
                samples: vec![Sample { source_row: 2 + branches.len(), voltage_v: 1.0,
                    current_a: 1.0 / resistance }],
            });
        }
    }
    branches
}

fn summarize(rows: &[Branch]) -> Vec<Features> {
    extract_cohort(rows, Settings {
        max_bracket_span_v: 0.1, assumptions_accepted: true, by_condition: true,
    }).expect("complete synthetic paired-state readpoints")
}

fn main() {
    println!("Synthetic only: ONE device, repeated cycles, conditions reported separately.");
    for f in summarize(&branches()) {
        println!("condition={}: cycles={}, median of paired HRS/LRS={:.6}, ratio of marginal medians={:.6}, inversions retained={}",
            f.condition, f.pairs.len(), f.resistance_ratio_median,
            f.hrs_resistance_median_ohm / f.lrs_resistance_median_ohm,
            f.ratio_below_one_cycle_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_estimand_and_condition_denominator_survive_branch_reordering() {
        let mut rows = branches();
        rows.reverse();
        let f = summarize(&rows);
        assert_eq!(f.len(), 2);
        assert_eq!((&f[0].condition[..], &f[1].condition[..]), ("room", "warm"));
        assert_eq!((f[0].pairs.len(), f[1].pairs.len()), (3, 1));
        assert_eq!((f[0].device.as_str(), f[1].device.as_str()), ("one-device", "one-device"));
        assert_eq!(f[0].resistance_ratio_median, 3.0);
        assert_eq!(f[0].hrs_resistance_median_ohm / f[0].lrs_resistance_median_ohm, 2.0);
        assert_eq!(f[0].ratio_below_one_cycle_count, 1);
        assert_eq!(f[0].pairs[1].hrs.selected_source_rows, vec![4]);
        assert_eq!(f[0].pairs[1].lrs.selected_source_rows, vec![5]);
        assert_eq!(f[1].resistance_ratio_median, 8.0);
    }
}
