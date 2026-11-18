use clotho_core::evaluation::{regression_mse_skill, EvaluationError};

#[test]
fn hand_calculated_skill_retains_losses_and_ties() {
    let y = [1.0, 3.0];
    let baseline = [0.0, 1.0];
    assert!((regression_mse_skill(&y, &[1.0, 2.0], &baseline).unwrap().unwrap() - 0.8).abs() < 1e-14);
    assert_eq!(regression_mse_skill(&y, &baseline, &baseline), Ok(Some(0.0)));
    assert!((regression_mse_skill(&y, &[-1.0, -1.0], &baseline).unwrap().unwrap() + 3.0).abs() < 1e-14);
    assert_eq!(regression_mse_skill(&y, &y, &baseline), Ok(Some(1.0)));
}

#[test]
fn perfect_baseline_is_undefined_not_perfect_skill() {
    assert_eq!(regression_mse_skill(&[1.0], &[1.0], &[1.0]), Ok(None));
    assert_eq!(regression_mse_skill(&[1.0], &[2.0], &[1.0]), Ok(None));
    assert_eq!(regression_mse_skill(&[1.0], &[f64::NAN], &[1.0]), Err(EvaluationError::NonFinite));
}

#[test]
fn malformed_inputs_are_rejected() {
    assert_eq!(regression_mse_skill(&[], &[], &[]), Err(EvaluationError::Empty));
    assert_eq!(regression_mse_skill(&[0.0], &[], &[0.0]), Err(EvaluationError::Shape));
    assert_eq!(regression_mse_skill(&[0.0], &[0.0], &[]), Err(EvaluationError::Shape));
    assert_eq!(regression_mse_skill(&[0.0], &[0.0], &[f64::INFINITY]), Err(EvaluationError::NonFinite));
}

#[test]
fn scaled_errors_do_not_require_representable_raw_squares() {
    for scale in [1e200, 1e-200] {
        let score = regression_mse_skill(&[0.0, 0.0], &[scale, -scale], &[2.0*scale, -2.0*scale]).unwrap().unwrap();
        assert!((score - 0.75).abs() < 1e-14);
    }
    assert_eq!(regression_mse_skill(&[0.0], &[1e200], &[1e-200]), Err(EvaluationError::NumericOverflow));
}

#[test]
fn constant_targets_can_still_have_defined_baseline_skill() {
    assert_eq!(regression_mse_skill(&[7.0, 7.0], &[7.0, 7.0], &[6.0, 6.0]), Ok(Some(1.0)));
}


#[test]
fn subnormal_rmse_quantization_cannot_be_reported_as_a_skill_tie() {
    let tiny = f64::from_bits(1);
    // Exact MSE ratio is 1/2; rounded RMSE ratio would incorrectly give 1.
    assert_eq!(
        regression_mse_skill(&[0.0, 0.0], &[tiny, 0.0], &[tiny, tiny]),
        Err(EvaluationError::NumericOverflow),
    );
    // Reversing model/baseline should give -1, not a false tie either.
    assert_eq!(
        regression_mse_skill(&[0.0, 0.0], &[tiny, tiny], &[tiny, 0.0]),
        Err(EvaluationError::NumericOverflow),
    );
    for (model, baseline) in [(tiny, 1.0), (1.0, tiny), (tiny, tiny)] {
        assert_eq!(
            regression_mse_skill(&[0.0], &[model], &[baseline]),
            Err(EvaluationError::NumericOverflow),
        );
    }
    assert_eq!(regression_mse_skill(&[0.0], &[0.0], &[tiny]), Ok(Some(1.0)));
    assert_eq!(regression_mse_skill(&[0.0], &[tiny], &[0.0]), Ok(None));
    // Normal boundary and ordinary rescaling retain the old formula.
    assert_eq!(
        regression_mse_skill(&[0.0], &[f64::MIN_POSITIVE], &[f64::MIN_POSITIVE]),
        Ok(Some(0.0)),
    );
    let score = regression_mse_skill(&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0])
        .unwrap().unwrap();
    assert!((score - 0.5).abs() < 1e-15);
}
