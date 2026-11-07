use clotho_core::rram::*;
fn settings() -> Settings {
    Settings {
        max_bracket_span_v: 1.0,
        assumptions_accepted: true,
        by_condition: false,
    }
}
fn branch(state: State, cycle: &str, current: f64, row: usize) -> Branch {
    Branch {
        acquisition: Acquisition {
            batch: "B".into(),
            wafer: "W".into(),
            device: "D".into(),
            cycle: cycle.into(),
            replicate: cycle.into(),
            branch: format!("{state:?}"),
            state,
            condition: "C".into(),
            protocol: "P".into(),
        },
        conditions: Conditions {
            read_voltage_v: 1.0,
            temperature_k: 300.0,
            read_delay_s: 0.0,
        },
        samples: vec![Sample {
            source_row: row,
            voltage_v: 1.0,
            current_a: current,
        }],
    }
}
fn pair() -> Vec<Branch> {
    vec![
        branch(State::Hrs, "1", 0.01, 2),
        branch(State::Lrs, "1", 0.1, 3),
    ]
}
#[test]
fn signed_static_readpoints_and_reversed_interpolation() {
    let mut b = branch(State::Hrs, "1", 0.01, 2);
    b.samples = vec![
        Sample {
            source_row: 2,
            voltage_v: 1.5,
            current_a: 0.015,
        },
        Sample {
            source_row: 3,
            voltage_v: 0.5,
            current_a: 0.005,
        },
    ];
    let p = extract_branch(&b, settings()).unwrap();
    assert!((p.resistance_ohm - 100.0).abs() < 1e-12);
    assert_eq!(p.selected_source_rows, vec![2, 3]);
    assert_eq!(p.bracket_span_v, 1.0);
    b.conditions.read_voltage_v = -1.0;
    for s in &mut b.samples {
        s.voltage_v = -s.voltage_v;
        s.current_a = -s.current_a;
    }
    assert!((extract_branch(&b, settings()).unwrap().resistance_ohm - 100.0).abs() < 1e-12);
}
#[test]
fn medians_are_of_paired_ratios_not_ratio_of_medians() {
    let mut bs = pair();
    bs.extend([
        branch(State::Hrs, "2", 0.01, 4),
        branch(State::Lrs, "2", 0.001, 5),
    ]);
    let f = extract_cohort(&bs, settings()).unwrap().remove(0);
    assert_eq!(f.hrs_resistance_median_ohm, 100.0);
    assert_eq!(f.lrs_resistance_median_ohm, 505.0);
    assert_eq!(f.resistance_ratio_median, 5.05);
    assert_eq!(f.ratio_below_one_cycle_count, 1);
}
#[test]
fn rejects_incomplete_duplicate_states_and_reused_replicates() {
    let b = pair();
    assert_eq!(
        extract_cohort(&b[..1], settings()).unwrap_err(),
        RramError::MissingState
    );
    let mut bs = b.clone();
    let mut extra = branch(State::Hrs, "1", 0.01, 4);
    extra.acquisition.branch = "other".into();
    bs.push(extra);
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::DuplicateState
    );
    let mut bs = b;
    let mut extra = branch(State::Hrs, "2", 0.01, 4);
    extra.acquisition.replicate = "1".into();
    bs.push(extra);
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::ReplicateMismatch
    );
}
#[test]
fn fixed_global_identity_and_declared_conditions() {
    let mut bs = pair();
    bs[1].acquisition.wafer = "other".into();
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::IdentityMismatch
    );
    let mut bs = pair();
    bs[1].conditions.temperature_k = 301.0;
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::InvalidConditions
    );
    let mut bs = pair();
    bs[1].acquisition.replicate = "other".into();
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::ReplicateMismatch
    );
}
#[test]
fn conditions_remain_separate_and_require_opt_in() {
    let mut bs = pair();
    let mut second = vec![
        branch(State::Hrs, "2", 0.02, 4),
        branch(State::Lrs, "2", 0.1, 5),
    ];
    for b in &mut second {
        b.acquisition.condition = "C2".into();
        b.conditions.temperature_k = 310.0;
    }
    bs.extend(second);
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::IdentityMismatch
    );
    let mut p = settings();
    p.by_condition = true;
    let fs = extract_cohort(&bs, p).unwrap();
    assert_eq!(fs.len(), 2);
    assert_eq!(fs[0].resistance_ratio_median, 10.0);
    assert_eq!(fs[1].resistance_ratio_median, 5.0);
}
#[test]
fn rejects_nonmonotonic_duplicate_extrapolated_and_wide_branches() {
    let mut b = branch(State::Hrs, "1", 0.01, 2);
    b.samples.push(Sample {
        source_row: 3,
        voltage_v: 1.0,
        current_a: 0.02,
    });
    assert_eq!(
        extract_branch(&b, settings()).unwrap_err(),
        RramError::InvalidVoltageOrder
    );
    b.samples[1].voltage_v = 2.0;
    b.samples.push(Sample {
        source_row: 4,
        voltage_v: 0.5,
        current_a: 0.01,
    });
    assert_eq!(
        extract_branch(&b, settings()).unwrap_err(),
        RramError::InvalidVoltageOrder
    );
    b.samples.pop();
    b.conditions.read_voltage_v = 3.0;
    assert_eq!(
        extract_branch(&b, settings()).unwrap_err(),
        RramError::ReadVoltageNotBracketed
    );
    b.samples[0].voltage_v = 0.1;
    b.conditions.read_voltage_v = 1.0;
    assert_eq!(
        extract_branch(&b, settings()).unwrap_err(),
        RramError::BracketTooWide
    );
}
#[test]
fn rejects_invalid_current_numeric_range_source_rows_and_consent() {
    for i in [0.0, -0.01, f64::NAN, 1e-151] {
        let b = branch(State::Hrs, "1", i, 2);
        assert!(extract_branch(&b, settings()).is_err());
    }
    let mut bs = pair();
    bs[1].samples[0].source_row = 2;
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::InvalidSample
    );
    let mut p = settings();
    p.assumptions_accepted = false;
    assert_eq!(
        extract_cohort(&pair(), p).unwrap_err(),
        RramError::InvalidSettings
    );
    let mut bs = pair();
    bs[0].conditions.read_voltage_v = 1e150;
    bs[1].conditions.read_voltage_v = 1e150;
    bs[0].samples[0].voltage_v = 1e150;
    bs[1].samples[0].voltage_v = 1e150;
    bs[0].samples[0].current_a = 1e-150;
    bs[1].samples[0].current_a = 1e150;
    assert!(extract_cohort(&bs, settings()).is_ok());
    // Change common read voltage, preserving declarations; ratios remain finite across the supported input range.
    bs[0].conditions.read_voltage_v = 1.0;
    bs[1].conditions.read_voltage_v = 1.0;
    bs[0].samples[0].voltage_v = 1.0;
    bs[1].samples[0].voltage_v = 1.0;
    assert!(extract_cohort(&bs, settings()).is_ok());
}
#[test]
fn rejects_unsafe_identifiers_and_empty_cohort() {
    let mut bs = pair();
    bs[0].acquisition.device = "=private".into();
    assert_eq!(
        extract_cohort(&bs, settings()).unwrap_err(),
        RramError::InvalidIdentity
    );
    assert_eq!(
        extract_cohort(&[], settings()).unwrap_err(),
        RramError::ResourceLimit
    );
}

#[test]
fn derived_resistance_overflow_is_rejected_after_interpolation() {
    let mut b = branch(State::Hrs, "1", 0.01, 2);
    b.conditions.read_voltage_v = 0.5e150;
    let tiny = 1e-150_f64;
    b.samples = vec![
        Sample {
            source_row: 2,
            voltage_v: 0.0,
            current_a: f64::from_bits(tiny.to_bits() + 1),
        },
        Sample {
            source_row: 3,
            voltage_v: 1e150,
            current_a: -tiny,
        },
    ];
    let mut p = settings();
    p.max_bracket_span_v = 1e150;
    assert_eq!(
        extract_branch(&b, p).unwrap_err(),
        RramError::NumericalRange
    );
}
#[test]
fn branch_permutation_preserves_paired_statistics_and_rows() {
    let a = pair();
    let mut b = a.clone();
    b.reverse();
    let x = extract_cohort(&a, settings()).unwrap();
    let y = extract_cohort(&b, settings()).unwrap();
    assert_eq!(x[0].resistance_ratio_median, y[0].resistance_ratio_median);
    assert_eq!(
        x[0].pairs[0].hrs.selected_source_rows,
        y[0].pairs[0].hrs.selected_source_rows
    );
    assert_eq!(
        x[0].pairs[0].lrs.selected_source_rows,
        y[0].pairs[0].lrs.selected_source_rows
    );
}
#[test]
fn extreme_finite_paired_resistance_ratio_remains_finite() {
    let mut b = pair();
    b[0].conditions.read_voltage_v = 1e150;
    b[1].conditions.read_voltage_v = 1e150;
    b[0].samples[0].voltage_v = 1e150;
    b[1].samples[0].voltage_v = 1e150;
    b[0].samples[0].current_a = 1e-150;
    b[1].samples[0].current_a = 1e150;
    assert!(extract_cohort(&b, settings()).is_ok());
    // One signed read voltage shared by a pair bounds its ratio at 1e300;
    // independently derived resistances may be large, but must remain finite.
    let f = extract_cohort(&b, settings()).unwrap();
    assert!(f[0].resistance_ratio_median.is_finite());
}
