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
