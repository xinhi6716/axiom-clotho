use clotho_core::regression::{MeanRegressor, RegressionError, RidgeRegressor};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:e}, expected {expected:e}, tolerance {tolerance:e}"
    );
}

#[test]
fn mean_baseline_uses_only_training_targets() {
    let model = MeanRegressor::fit(&[vec![0.0], vec![1.0], vec![2.0]], &[1.0, 3.0, 8.0]).unwrap();
    close(model.mean(), 4.0, 1e-14);
    assert_eq!(
        model.predict(&[vec![1000.0], vec![-10.0]]).unwrap(),
        vec![4.0, 4.0]
    );
}

#[test]
fn compensated_mean_preserves_cancellation_and_avoids_overflow() {
    let x = vec![vec![0.0]; 3];
    for y in [[1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [1.0, -1e16, 1e16]] {
        close(MeanRegressor::fit(&x, &y).unwrap().mean(), 1.0 / 3.0, 1e-15);
    }
    assert_eq!(
        MeanRegressor::fit(&x, &[f64::MAX; 3]).unwrap().mean(),
        f64::MAX
    );
    assert_eq!(MeanRegressor::fit(&x, &[0.0; 3]).unwrap().mean(), 0.0);
}

#[test]
fn unregularized_line_has_an_intercept_and_extrapolates() {
    let model =
        RidgeRegressor::fit(&[vec![1.0], vec![2.0], vec![3.0]], &[5.0, 7.0, 9.0], 0.0).unwrap();
    let predictions = model.predict(&[vec![0.0], vec![2.5], vec![10.0]]).unwrap();
    for (actual, expected) in predictions.into_iter().zip([3.0, 8.0, 23.0]) {
        close(actual, expected, 1e-12);
    }
}

#[test]
fn ridge_matches_hand_calculated_standardized_objective() {
    // Population-standardized x is [-sqrt(1.5), 0, sqrt(1.5)].
    // sum(z^2)=3, so alpha=3 halves the slope and leaves intercept=7.
    let x = [vec![1.0], vec![2.0], vec![3.0]];
    let model = RidgeRegressor::fit(&x, &[5.0, 7.0, 9.0], 3.0).unwrap();
    for (actual, expected) in model.predict(&x).unwrap().into_iter().zip([6.0, 7.0, 8.0]) {
        close(actual, expected, 1e-12);
    }
}

#[test]
fn multivariate_line_and_orthogonal_ridge_match_hand_math() {
    let x = [
        vec![-1.0, -1.0],
        vec![-1.0, 1.0],
        vec![1.0, -1.0],
        vec![1.0, 1.0],
    ];
    let y = [5.0, -1.0, 9.0, 3.0]; // 4 + 2*x0 - 3*x1
    let ols = RidgeRegressor::fit(&x, &y, 0.0).unwrap();
    for (actual, expected) in ols.predict(&x).unwrap().into_iter().zip(y) {
        close(actual, expected, 1e-12);
    }
    let ridge = RidgeRegressor::fit(&x, &y, 4.0).unwrap();
    for (actual, expected) in ridge
        .predict(&x)
        .unwrap()
        .into_iter()
        .zip([4.5, 1.5, 6.5, 3.5])
    {
        close(actual, expected, 1e-12);
    }
}

#[test]
fn redundant_columns_require_regularization() {
    let x = [vec![-1.0, -2.0], vec![0.0, 0.0], vec![1.0, 2.0]];
    assert_eq!(
        RidgeRegressor::fit(&x, &[1.0, 3.0, 5.0], 0.0).unwrap_err(),
        RegressionError::IllConditioned
    );
    // Identical standardized columns give a slope shrink factor 6/(6+6).
    let model = RidgeRegressor::fit(&x, &[1.0, 3.0, 5.0], 6.0).unwrap();
    for (actual, expected) in model.predict(&x).unwrap().into_iter().zip([2.0, 3.0, 4.0]) {
        close(actual, expected, 1e-12);
    }
}

#[test]
fn constant_features_and_one_row_require_regularization() {
    let x = [vec![7.0], vec![7.0], vec![7.0]];
    assert_eq!(
        RidgeRegressor::fit(&x, &[2.0, 4.0, 6.0], 0.0).unwrap_err(),
        RegressionError::IllConditioned
    );
    let model = RidgeRegressor::fit(&x, &[2.0, 4.0, 6.0], 1.0).unwrap();
    close(model.predict(&[vec![7.0]]).unwrap()[0], 4.0, 1e-12);
    let model = RidgeRegressor::fit(&[vec![7.0, -2.0]], &[9.0], 1.0).unwrap();
    close(model.predict(&[vec![100.0, -99.0]]).unwrap()[0], 9.0, 1e-12);
    assert_eq!(
        RidgeRegressor::fit(&[vec![1.0]], &[1.0], 0.0).unwrap_err(),
        RegressionError::IllConditioned
    );
}

#[test]
fn underdetermined_features_work_only_with_regularization() {
    let x = [vec![0.0, 1.0, 2.0], vec![2.0, 0.0, 3.0]];
    assert_eq!(
        RidgeRegressor::fit(&x, &[1.0, 5.0], 0.0).unwrap_err(),
        RegressionError::IllConditioned
    );
    let model = RidgeRegressor::fit(&x, &[1.0, 5.0], 6.0).unwrap();
    for (actual, expected) in model.predict(&x).unwrap().into_iter().zip([2.0, 4.0]) {
        close(actual, expected, 1e-12);
    }
}

#[test]
fn standardization_is_invariant_to_translation_and_feature_units() {
    let x = [
        vec![0.0, 2.0],
        vec![1.0, -1.0],
        vec![2.0, 4.0],
        vec![4.0, 1.0],
    ];
    let y = [3.0, 1.0, 9.0, 8.0];
    let query = [vec![1.5, 2.0], vec![5.0, -2.0]];
    let remap = |row: &Vec<f64>| vec![1e12 + row[0] * 2.0, -7.0 - row[1] * 1000.0];
    let translated: Vec<_> = x.iter().map(remap).collect();
    let translated_query: Vec<_> = query.iter().map(remap).collect();
    for alpha in [0.0, 0.5, 20.0] {
        let a = RidgeRegressor::fit(&x, &y, alpha)
            .unwrap()
            .predict(&query)
            .unwrap();
        let b = RidgeRegressor::fit(&translated, &y, alpha)
            .unwrap()
            .predict(&translated_query)
            .unwrap();
        for (actual, expected) in a.into_iter().zip(b) {
            close(actual, expected, 1e-11);
        }
    }
}

#[test]
fn prediction_scaling_is_training_only_and_batch_independent() {
    let x = [vec![0.0], vec![1.0], vec![2.0]];
    let model = RidgeRegressor::fit(&x, &[3.0, 5.0, 7.0], 3.0).unwrap();
    let alone = model.predict(&[vec![1.5]]).unwrap()[0];
    let batch = model.predict(&[vec![1.5], vec![1e8]]).unwrap();
    assert_eq!(alone, batch[0]);
    close(alone, 5.5, 1e-12);
    assert_eq!(model.predict(&[vec![1.5]]).unwrap()[0], alone);
}

#[test]
fn row_permutation_and_target_translation_preserve_predictions() {
    let x = [
        vec![0.0, 1.0],
        vec![2.0, -1.0],
        vec![-1.0, 4.0],
        vec![4.0, 3.0],
    ];
    let y = [1.0, 5.0, 0.0, 10.0];
    let query = [vec![1.0, 2.0]];
    let expected = RidgeRegressor::fit(&x, &y, 0.5)
        .unwrap()
        .predict(&query)
        .unwrap()[0];
    let reversed_x: Vec<_> = x.into_iter().rev().collect();
    let shifted_y: Vec<_> = y.into_iter().rev().map(|v| 100.0 + 2.0 * v).collect();
    let actual = RidgeRegressor::fit(&reversed_x, &shifted_y, 0.5)
        .unwrap()
        .predict(&query)
        .unwrap()[0];
    close(actual, 100.0 + 2.0 * expected, 1e-11);
}

#[test]
fn nearly_dependent_columns_are_rejected_without_sufficient_regularization() {
    let x = [
        vec![-1.0, -1.0],
        vec![0.0, 1e-14],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    assert_eq!(
        RidgeRegressor::fit(&x, &[1.0, 2.0, 3.0, 4.0], 0.0).unwrap_err(),
        RegressionError::IllConditioned
    );
    assert!(RidgeRegressor::fit(&x, &[1.0, 2.0, 3.0, 4.0], 0.1).is_ok());
}

#[test]
fn large_alpha_does_not_penalize_the_intercept() {
    let model =
        RidgeRegressor::fit(&[vec![0.0], vec![1.0], vec![2.0]], &[3.0, 5.0, 7.0], 1e100).unwrap();
    for prediction in model.predict(&[vec![-1e5], vec![1e5]]).unwrap() {
        close(prediction, 5.0, 1e-12);
    }
}

#[test]
fn finite_extremes_are_supported_or_explicitly_rejected() {
    let model = RidgeRegressor::fit(&[vec![-f64::MAX], vec![f64::MAX]], &[-1.0, 1.0], 2.0).unwrap();
    close(model.predict(&[vec![0.0]]).unwrap()[0], 0.0, 1e-12);
    let model =
        RidgeRegressor::fit(&[vec![0.0], vec![1.0]], &[f64::MAX / 2.0, f64::MAX], 0.0).unwrap();
    assert_eq!(
        model.predict(&[vec![100.0]]),
        Err(RegressionError::NumericFailure)
    );
    let model =
        RidgeRegressor::fit(&[vec![0.0], vec![f64::MIN_POSITIVE]], &[0.0, 1.0], 1.0).unwrap();
    assert_eq!(
        model.predict(&[vec![f64::MAX]]),
        Err(RegressionError::NumericFailure)
    );
    assert_eq!(
        RidgeRegressor::fit(
            &[vec![0.0], vec![1e-300], vec![f64::MAX]],
            &[1.0, 2.0, 3.0],
            1.0
        )
        .unwrap_err(),
        RegressionError::NumericFailure
    );
    assert_eq!(
        MeanRegressor::fit(&[vec![0.0], vec![1.0]], &[f64::MAX, f64::MIN_POSITIVE]).unwrap_err(),
        RegressionError::NumericFailure
    );
}

#[test]
fn rejects_empty_ragged_mismatched_nonfinite_and_over_capacity_inputs() {
    assert_eq!(
        RidgeRegressor::fit(&[], &[], 1.0).unwrap_err(),
        RegressionError::Empty
    );
    assert_eq!(
        MeanRegressor::fit(&[], &[]).unwrap_err(),
        RegressionError::Empty
    );
    for x in [vec![vec![]], vec![vec![0.0], vec![0.0, 1.0]]] {
        let y = vec![0.0; x.len()];
        assert_eq!(
            RidgeRegressor::fit(&x, &y, 1.0).unwrap_err(),
            RegressionError::Shape
        );
        assert_eq!(
            MeanRegressor::fit(&x, &y).unwrap_err(),
            RegressionError::Shape
        );
    }
    assert_eq!(
        RidgeRegressor::fit(&[vec![1.0]], &[], 1.0).unwrap_err(),
        RegressionError::Shape
    );
    assert_eq!(
        MeanRegressor::fit(&[vec![1.0]], &[]).unwrap_err(),
        RegressionError::Shape
    );
    for alpha in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            RidgeRegressor::fit(&[vec![1.0]], &[1.0], alpha).unwrap_err(),
            RegressionError::InvalidAlpha
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            RidgeRegressor::fit(&[vec![value]], &[1.0], 1.0).unwrap_err(),
            RegressionError::NonFinite
        );
        assert_eq!(
            RidgeRegressor::fit(&[vec![1.0]], &[value], 1.0).unwrap_err(),
            RegressionError::NonFinite
        );
        assert_eq!(
            MeanRegressor::fit(&[vec![value]], &[1.0]).unwrap_err(),
            RegressionError::NonFinite
        );
        assert_eq!(
            MeanRegressor::fit(&[vec![1.0]], &[value]).unwrap_err(),
            RegressionError::NonFinite
        );
    }
    for x in [vec![vec![0.0; 101]], vec![vec![0.0]; 20_001]] {
        let y = vec![0.0; x.len()];
        assert_eq!(
            RidgeRegressor::fit(&x, &y, 1.0).unwrap_err(),
            RegressionError::Capacity
        );
        assert_eq!(
            MeanRegressor::fit(&x, &y).unwrap_err(),
            RegressionError::Capacity
        );
    }
}

#[test]
fn prediction_validates_even_when_model_ignores_features() {
    let ridge = RidgeRegressor::fit(&[vec![1.0]], &[0.0], 1.0).unwrap();
    let mean = MeanRegressor::fit(&[vec![1.0]], &[0.0]).unwrap();
    for (x, error) in [
        (vec![], RegressionError::Empty),
        (vec![vec![]], RegressionError::Shape),
        (vec![vec![1.0, 2.0]], RegressionError::Shape),
        (vec![vec![f64::NAN]], RegressionError::NonFinite),
        (vec![vec![0.0]; 20_001], RegressionError::Capacity),
    ] {
        assert_eq!(ridge.predict(&x), Err(error));
        assert_eq!(mean.predict(&x), Err(error));
    }
}

#[test]
fn large_target_offsets_do_not_destroy_slopes_and_constants_are_exact() {
    let x = [vec![0.0], vec![1.0], vec![2.0]];
    let y = [1e15 + 2.0, 1e15 + 4.0, 1e15 + 6.0];
    let model = RidgeRegressor::fit(&x, &y, 0.0).unwrap();
    assert_eq!(model.predict(&[vec![3.0]]).unwrap()[0], 1e15 + 8.0);
    for target in [0.0, -3.0, f64::MAX, -f64::MAX] {
        let model = RidgeRegressor::fit(&x, &[target; 3], 1.0).unwrap();
        assert_eq!(model.predict(&[vec![100.0]]).unwrap()[0], target);
    }
}

#[test]
fn destructive_underflow_is_rejected_independent_of_row_order() {
    for x in [
        vec![vec![0.0], vec![1e-300], vec![f64::MAX]],
        vec![vec![f64::MAX], vec![0.0], vec![1e-300]],
    ] {
        assert_eq!(
            RidgeRegressor::fit(&x, &[1.0, 2.0, 3.0], 1.0).unwrap_err(),
            RegressionError::NumericFailure
        );
    }
    let least_subnormal = f64::from_bits(1);
    assert_eq!(
        MeanRegressor::fit(&[vec![0.0], vec![1.0]], &[least_subnormal, 0.0]).unwrap_err(),
        RegressionError::NumericFailure
    );
    assert_eq!(
        MeanRegressor::fit(&vec![vec![0.0]; 3], &[f64::MAX, -f64::MAX, 1e-15]).unwrap_err(),
        RegressionError::NumericFailure
    );
}
