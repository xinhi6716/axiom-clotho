use clotho_core::fet::*;
fn acquisition() -> Acquisition {
    Acquisition {
        lot: "L".into(),
        wafer: "W".into(),
        device: "D".into(),
        condition: "C".into(),
        replicate: "R".into(),
        branch: "B".into(),
    }
}
fn samples() -> Vec<Sample> {
    (0..4)
        .map(|i| Sample {
            source_row: i + 2,
            temperature_k: 300.0,
            drain_voltage_v: 1.0,
            body_voltage_v: 0.0,
            gate_voltage_v: i as f64,
            drain_current_a: 10f64.powi(i as i32 - 9),
        })
        .collect()
}
fn settings() -> Settings {
    Settings {
        criterion_a: 1e-8,
        positive_current: true,
        max_bracket_span_v: 1.0,
        window_min_v: 0.0,
        window_max_v: 3.0,
        assumptions_accepted: true,
    }
}
#[test]
fn exact_and_log_slope() {
    let f = extract_branch(&acquisition(), &samples(), settings()).unwrap();
    assert_eq!(f.criterion_gate_voltage_v, 1.0);
    assert_eq!(f.criterion_source_rows, vec![3]);
    assert!((f.log_slope_decades_per_v - 1.0).abs() < 1e-14);
    assert!((f.log_intercept + 9.0).abs() < 1e-14);
    assert!(f.log_rmse_decades < 1e-14);
}
#[test]
fn linear_current_not_log_interpolation() {
    let mut p = settings();
    p.criterion_a = 5.5e-9;
    let f = extract_branch(&acquisition(), &samples(), p).unwrap();
    assert!((f.criterion_gate_voltage_v - 0.5).abs() < 1e-14);
    assert_eq!(f.criterion_source_rows, vec![2, 3]);
}
#[test]
fn reverse_branch_same_features() {
    let mut s = samples();
    s.reverse();
    let f = extract_branch(&acquisition(), &s, settings()).unwrap();
    assert_eq!(f.criterion_gate_voltage_v, 1.0);
    assert!((f.log_slope_decades_per_v - 1.0).abs() < 1e-14);
    assert_eq!(f.all_source_rows, vec![5, 4, 3, 2]);
}
#[test]
fn negative_polarity_preserved() {
    let mut s = samples();
    for q in &mut s {
        q.drain_current_a *= -1.0;
    }
    let mut p = settings();
    p.positive_current = false;
    p.criterion_a *= -1.0;
    let f = extract_branch(&acquisition(), &s, p).unwrap();
    assert_eq!(f.criterion_gate_voltage_v, 1.0);
    assert!((f.log_slope_decades_per_v - 1.0).abs() < 1e-14);
}
#[test]
fn no_assumed_units_or_consent() {
    let mut p = settings();
    p.assumptions_accepted = false;
    assert_eq!(
        extract_branch(&acquisition(), &samples(), p).unwrap_err(),
        FetError::InvalidSettings
    );
    p = settings();
    p.criterion_a = -1.0;
    assert_eq!(
        extract_branch(&acquisition(), &samples(), p).unwrap_err(),
        FetError::InvalidSettings
    );
}
#[test]
fn fixed_conditions_required() {
    for field in 0..3 {
        let mut s = samples();
        match field {
            0 => s[1].temperature_k = 301.0,
            1 => s[1].drain_voltage_v = 2.0,
            _ => s[1].body_voltage_v = 1.0,
        };
        assert_eq!(
            extract_branch(&acquisition(), &s, settings()).unwrap_err(),
            FetError::InvalidConditions
        );
    }
}
#[test]
fn no_sorting_or_duplicate_voltage() {
    let mut s = samples();
    s.swap(1, 2);
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::InvalidGateOrder
    );
    s = samples();
    s[1].gate_voltage_v = 0.0;
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::InvalidGateOrder
    );
}
#[test]
fn plateau_criterion_rejected() {
    let mut s = samples();
    s[2].drain_current_a = s[1].drain_current_a;
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::NonUniqueCriterion
    );
}
#[test]
fn cannot_extrapolate() {
    let mut p = settings();
    p.criterion_a = 1.0;
    assert_eq!(
        extract_branch(&acquisition(), &samples(), p).unwrap_err(),
        FetError::CriterionNotBracketed
    );
}
#[test]
fn bracket_bound() {
    let mut p = settings();
    p.criterion_a = 5.5e-9;
    p.max_bracket_span_v = 0.5;
    assert_eq!(
        extract_branch(&acquisition(), &samples(), p).unwrap_err(),
        FetError::BracketTooWide
    );
}
#[test]
fn window_is_inclusive_and_requires_three() {
    let mut p = settings();
    p.window_min_v = 1.0;
    let f = extract_branch(&acquisition(), &samples(), p).unwrap();
    assert_eq!(f.log_selected_source_rows, vec![3, 4, 5]);
    p.window_min_v = 1.1;
    assert_eq!(
        extract_branch(&acquisition(), &samples(), p).unwrap_err(),
        FetError::InsufficientWindow
    );
}
#[test]
fn zero_current_not_repaired() {
    let mut s = samples();
    s[0].drain_current_a = 0.0;
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::InvalidCurrent
    );
    let mut p = settings();
    p.window_min_v = 1.0;
    assert!(extract_branch(&acquisition(), &s, p).is_ok());
}
#[test]
fn no_nan_or_infinite_input() {
    for value in [f64::NAN, f64::INFINITY, 1e151, 1e-151] {
        let mut s = samples();
        s[0].gate_voltage_v = value;
        assert_eq!(
            extract_branch(&acquisition(), &s, settings()).unwrap_err(),
            FetError::InvalidSample
        );
    }
}
#[test]
fn duplicate_rows_rejected() {
    let mut s = samples();
    s[1].source_row = s[0].source_row;
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::InvalidSample
    );
}
#[test]
fn duplicate_device_even_different_branch() {
    let a = acquisition();
    let mut b = a.clone();
    b.branch = "other".into();
    assert_eq!(
        extract_cohort(&[(a, samples()), (b, samples())], settings()).unwrap_err(),
        FetError::DuplicateDevice
    );
}
#[test]
fn rejects_entire_cohort() {
    let a = acquisition();
    let mut b = a.clone();
    b.device = "D2".into();
    let mut s = samples();
    for q in &mut s {
        q.source_row += 4;
    }
    s[0].drain_voltage_v = 2.0;
    assert_eq!(
        extract_cohort(&[(a, samples()), (b, s)], settings()).unwrap_err(),
        FetError::InvalidConditions
    );
}
#[test]
fn negative_gate_slope_retained() {
    let mut s = samples();
    for q in &mut s {
        q.gate_voltage_v *= -1.0;
    }
    let mut p = settings();
    p.window_min_v = -3.0;
    p.window_max_v = 0.0;
    let f = extract_branch(&acquisition(), &s, p).unwrap();
    assert!((f.log_slope_decades_per_v + 1.0).abs() < 1e-14);
}
#[test]
fn privacy_errors_do_not_echo_identity() {
    let mut a = acquisition();
    a.device = " secret ".into();
    let e = extract_branch(&a, &samples(), settings())
        .unwrap_err()
        .to_string();
    assert!(!e.contains("secret"));
}
#[test]
fn nonmonotonic_current_rejected_even_with_unique_criterion() {
    let mut s = samples();
    s[3].drain_current_a = 5e-8;
    assert_eq!(
        extract_branch(&acquisition(), &s, settings()).unwrap_err(),
        FetError::NonMonotonicCurrent
    );
}
#[test]
fn distinct_currents_that_log_collapses_are_rejected() {
    let mut s = samples();
    s[0].drain_current_a = 1e-100;
    s[1].drain_current_a = f64::from_bits(s[0].drain_current_a.to_bits() + 1);
    assert_ne!(s[0].drain_current_a, s[1].drain_current_a);
    assert_eq!(s[0].drain_current_a.log10(), s[1].drain_current_a.log10());
    let mut p = settings();
    p.criterion_a = s[2].drain_current_a;
    assert_eq!(
        extract_branch(&acquisition(), &s, p).unwrap_err(),
        FetError::NumericalRange
    );
}
