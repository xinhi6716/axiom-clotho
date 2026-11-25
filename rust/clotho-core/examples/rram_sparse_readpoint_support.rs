//! A bracket-span check cannot identify the true response inside a sparse bracket.
//! cargo run --manifest-path rust/Cargo.toml -p clotho-core --example rram_sparse_readpoint_support
use clotho_core::rram::{extract_branch, Acquisition, Branch, Conditions, Readpoint, RramError,
                        Sample, Settings, State};

fn branch(midpoint: Option<f64>) -> Branch {
    let mut samples = vec![
        Sample { source_row: 2, voltage_v: 0.25, current_a: 0.25 },
        Sample { source_row: 4, voltage_v: 0.75, current_a: 0.75 },
    ];
    if let Some(current_a) = midpoint {
        samples.insert(1, Sample { source_row: 3, voltage_v: 0.5, current_a });
    }
    Branch {
        acquisition: Acquisition {
            batch: "invented-batch".into(), wafer: "invented-wafer".into(),
            device: "invented-device".into(), cycle: "one".into(), replicate: "one".into(),
            branch: "ascending".into(), state: State::Lrs, condition: "invented".into(),
            protocol: "invented".into(),
        },
        conditions: Conditions { read_voltage_v: 0.5, temperature_k: 300.0, read_delay_s: 0.0 },
        samples,
    }
}

fn evaluate(midpoint: Option<f64>, cap: f64) -> Result<Readpoint, RramError> {
    extract_branch(&branch(midpoint), Settings {
        max_bracket_span_v: cap, assumptions_accepted: true, by_condition: false,
    })
}

fn main() {
    println!("Invented sweeps only; no device-error bound.");
    let sparse = evaluate(None, 0.5).unwrap();
    let a = evaluate(Some(0.375), 0.5).unwrap();
    let b = evaluate(Some(0.625), 0.5).unwrap();
    println!("same endpoints, sparse linear readpoint={:.3} A", sparse.current_a);
    println!("two possible measured interior currents: {:.3} A and {:.3} A", a.current_a, b.current_a);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_sparse_endpoints_do_not_fix_interior() {
        let sparse = evaluate(None, 0.5).unwrap();
        assert_eq!((sparse.current_a, sparse.bracket_span_v), (0.5, 0.5));
        assert_eq!(sparse.selected_source_rows, vec![2, 4]);
        let rejected = evaluate(None, 0.25).unwrap_err();
        assert_eq!(rejected, RramError::BracketTooWide);
        for expected in [0.375, 0.625] {
            let observed = evaluate(Some(expected), 0.25).unwrap();
            assert_eq!(observed.current_a, expected);
            assert_eq!(observed.bracket_span_v, 0.0);
            assert_eq!(observed.selected_source_rows, vec![3]);
            assert_eq!((observed.current_a - sparse.current_a).abs(), 0.125);
        }
    }
}
