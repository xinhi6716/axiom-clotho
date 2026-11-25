//! Run: cargo run --manifest-path rust/Cargo.toml -p clotho-core --example group_calibration_identity
//! Invented residuals only; never read a battery or other device export.
use clotho_core::group_max_calibration::{
    calibrate_group_max, CalibrationRow, GroupMaxCutoff, RowKey,
};

#[derive(Clone, Copy)]
enum Unit {
    ExportLabel,
    PhysicalDevice,
    Protocol,
}

fn cutoff(devices: usize, resumed_score: f64, unit: Unit) -> GroupMaxCutoff {
    let physical: Vec<_> = (1..=devices).map(|i| format!("device-{i}")).collect();
    let original_ids: Vec<_> = (1..=devices).map(|i| format!("run-a/device-{i}/t1")).collect();
    let group = |i: usize| match unit {
        Unit::ExportLabel | Unit::PhysicalDevice => physical[i].as_str(),
        Unit::Protocol => ["protocol-a", "protocol-b"][i % 2],
    };
    let mut roster: Vec<_> = original_ids
        .iter()
        .enumerate()
        .map(|(i, id)| RowKey { group: group(i), row: id })
        .collect();
    // A legitimate later read of the first device. It reuses local observation
    // ID t1, but the source-run namespace preserves both distinct reads.
    roster.push(RowKey {
        group: match unit {
            Unit::ExportLabel => "resumed-export-label",
            Unit::PhysicalDevice | Unit::Protocol => group(0),
        },
        row: "run-b/resumed-device/t1",
    });
    let rows: Vec<_> = roster.iter().copied().enumerate().map(|(i, key)| CalibrationRow {
        key,
        prediction: 0.0,
        observed: if i == devices { resumed_score } else { (i + 1) as f64 },
    }).collect();
    calibrate_group_max(&roster, &rows, 95, 100).expect("complete synthetic roster")
}

fn main() {
    println!("Synthetic residuals, nominal 95% threshold; not coverage evidence.");
    for (n, resumed) in [(18, 2.0), (38, 100.0)] {
        println!("physical_devices={n}, retained_reads={}:", n + 1);
        for (label, unit) in [
            ("export labels, alias untreated", Unit::ExportLabel),
            ("physical devices, alias canonicalized", Unit::PhysicalDevice),
            ("protocols, distinct devices still retained", Unit::Protocol),
        ] {
            println!("  {label}: {:?}", cutoff(n, resumed, unit));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use GroupMaxCutoff::{Finite, InsufficientFinite};

    #[test]
    fn alias_and_protocol_blocking_are_distinct() {
        assert_eq!(cutoff(18, 2.0, Unit::ExportLabel), Finite { groups: 19, rank: 19, radius: 18.0 });
        assert_eq!(cutoff(18, 2.0, Unit::PhysicalDevice), InsufficientFinite { groups: 18, rank: 19 });
        assert_eq!(cutoff(38, 100.0, Unit::ExportLabel), Finite { groups: 39, rank: 38, radius: 38.0 });
        assert_eq!(cutoff(38, 100.0, Unit::PhysicalDevice), Finite { groups: 38, rank: 38, radius: 100.0 });
        for (n, later) in [(18, 2.0), (38, 100.0)] {
            assert_eq!(cutoff(n, later, Unit::Protocol), InsufficientFinite { groups: 2, rank: 3 });
        }
    }
}
