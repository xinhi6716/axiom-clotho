//! An exact criterion readpoint does not determine log-current sweep shape.
//! cargo run --manifest-path rust/Cargo.toml -p clotho-core --example fet_readpoint_vs_shape
use clotho_core::fet::{extract_branch, Acquisition, Features, Sample, Settings};

fn sweep(device: &str, rate: i32, first_row: usize) -> Features {
    let acquisition = Acquisition {
        lot: "invented-lot".into(),
        wafer: "invented-wafer".into(),
        device: device.into(),
        condition: "invented-condition".into(),
        replicate: "one".into(),
        branch: "descending-gate".into(),
    };
    let samples: Vec<_> = (0..5)
        .map(|i| Sample {
            source_row: first_row + i,
            temperature_k: 298.0,
            drain_voltage_v: -0.1,
            body_voltage_v: 0.0,
            gate_voltage_v: -(i as f64),
            drain_current_a: -10f64.powi(-8 + rate * (i as i32 - 2)),
        })
        .collect();
    extract_branch(
        &acquisition,
        &samples,
        Settings {
            criterion_a: -10f64.powi(-8),
            positive_current: false,
            max_bracket_span_v: 1.0,
            window_min_v: -4.0,
            window_max_v: -1.0,
            assumptions_accepted: true,
        },
    )
    .expect("valid invented signed-current branch")
}

fn main() {
    println!("Invented sweeps only; no material or hardware comparison.");
    for f in [sweep("one", 1, 2), sweep("two", 2, 7)] {
        println!(
            "{}: criterion={:.1} V at rows {:?}; fixed-window log slope={:.1} dec/V on rows {:?}",
            f.acquisition.device,
            f.criterion_gate_voltage_v,
            f.criterion_source_rows,
            f.log_slope_decades_per_v,
            f.log_selected_source_rows,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_sweep_readpoint_ties_while_log_shape_differs() {
        let a = sweep("one", 1, 2);
        let b = sweep("two", 2, 7);
        assert_eq!((a.criterion_gate_voltage_v, b.criterion_gate_voltage_v), (-2.0, -2.0));
        assert_eq!((a.criterion_source_rows, b.criterion_source_rows), (vec![4], vec![9]));
        assert_eq!((a.log_selected_source_rows, b.log_selected_source_rows),
                   (vec![3, 4, 5, 6], vec![8, 9, 10, 11]));
        assert!((a.log_slope_decades_per_v + 1.0).abs() < 1e-12);
        assert!((b.log_slope_decades_per_v + 2.0).abs() < 1e-12);
    }
}
