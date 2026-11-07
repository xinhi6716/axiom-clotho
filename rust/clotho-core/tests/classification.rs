use clotho_core::classification::{ClassificationError, LogisticClassifier, MajorityClassifier};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:e}, expected {expected:e}, tolerance {tolerance:e}"
    );
}

#[test]
fn majority_uses_training_frequency_including_ties_and_single_class() {
    let x = [vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    for (y, expected) in [
        ([0.0, 0.0, 0.0, 1.0], 0.25),
        ([0.0, 1.0, 0.0, 1.0], 0.5),
        ([1.0; 4], 1.0),
        ([0.0; 4], 0.0),
    ] {
        let model = MajorityClassifier::fit(&x, &y).unwrap();
        assert_eq!(
            model.predict_proba(&[vec![-100.0], vec![100.0]]).unwrap(),
            vec![expected; 2]
        );
    }
}

#[test]
fn logistic_matches_hand_calculated_penalized_optimum() {
    // x is already population-standardized. At beta=ln(3), p(+1)=3/4.
    // Mean-loss derivative is -1/4, canceled by alpha*beta=1/4.
    let x = [vec![-1.0], vec![1.0]];
    let alpha = 0.25 / 3.0_f64.ln();
    let model = LogisticClassifier::fit(&x, &[0.0, 1.0], alpha).unwrap();
    let probabilities = model.predict_proba(&x).unwrap();
    close(probabilities[0], 0.25, 1e-9);
    close(probabilities[1], 0.75, 1e-9);
}

#[test]
fn mean_loss_objective_is_invariant_to_repeating_training_rows() {
    let x = [vec![-2.0], vec![-1.0], vec![1.0], vec![4.0]];
    let y = [0.0, 1.0, 0.0, 1.0];
    let repeated_x: Vec<_> = (0..7).flat_map(|_| x.iter().cloned()).collect();
    let repeated_y: Vec<_> = (0..7).flat_map(|_| y).collect();
    let expected = LogisticClassifier::fit(&x, &y, 0.2)
        .unwrap()
        .predict_proba(&x)
        .unwrap();
    let actual = LogisticClassifier::fit(&repeated_x, &repeated_y, 0.2)
        .unwrap()
        .predict_proba(&x)
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        close(actual, expected, 1e-9);
    }
}

#[test]
fn stronger_regularization_shrinks_discrimination_toward_the_prior() {
    let x = [vec![-1.0], vec![1.0]];
    let y = [0.0, 1.0];
    let mut previous = 1.0;
    for alpha in [1e-8, 1e-4, 0.1, 10.0, 1e8] {
        let p = LogisticClassifier::fit(&x, &y, alpha)
            .unwrap()
            .predict_proba(&x)
            .unwrap();
        assert!(p[1] > 0.5 && p[1] < previous);
        close(p[0] + p[1], 1.0, 1e-12);
        previous = p[1];
    }
}

#[test]
fn rare_positive_class_keeps_a_finite_unpenalized_intercept() {
    let x: Vec<_> = (0..1000).map(|i| vec![i as f64]).collect();
    let mut y = vec![0.0; x.len()];
    y[999] = 1.0;
    let p = LogisticClassifier::fit(&x, &y, 0.1)
        .unwrap()
        .predict_proba(&x)
        .unwrap();
    assert!(p.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    close(p.iter().sum::<f64>() / p.len() as f64, 0.001, 1e-9);
    assert!(p[999] > p[0]);
}

#[test]
fn unregularized_overlapping_labels_have_a_finite_optimum() {
    let x = [
        vec![-1.0],
        vec![-1.0],
        vec![-1.0],
        vec![-1.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
    ];
    let y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let p = LogisticClassifier::fit(&x, &y, 0.0)
        .unwrap()
        .predict_proba(&[vec![-1.0], vec![0.0], vec![1.0]])
        .unwrap();
    for (actual, expected) in p.into_iter().zip([0.25, 0.5, 0.75]) {
        close(actual, expected, 1e-9);
    }
}

#[test]
fn separable_unregularized_data_are_not_silently_declared_converged() {
    assert_eq!(
        LogisticClassifier::fit(&[vec![-1.0], vec![1.0]], &[0.0, 1.0], 0.0).unwrap_err(),
        ClassificationError::ConvergenceFailure
    );
}

#[test]
fn balanced_and_unbalanced_synthetic_sets_are_discriminated() {
    for threshold in [0, 32] {
        let x: Vec<_> = (-50..50).map(|i| vec![i as f64, (i % 7) as f64]).collect();
        let y: Vec<_> = (-50..50).map(|i| f64::from(i >= threshold)).collect();
        let model = LogisticClassifier::fit(&x, &y, 0.01).unwrap();
        let p = model.predict_proba(&x).unwrap();
        let correct = p
            .iter()
            .zip(&y)
            .filter(|(&p, &y)| f64::from(p >= 0.5) == y)
            .count();
        assert!(correct >= 95, "correct: {correct}");
        let mean_probability = p.iter().sum::<f64>() / p.len() as f64;
        let prevalence = y.iter().sum::<f64>() / y.len() as f64;
        close(mean_probability, prevalence, 1e-8); // Unpenalized intercept.
        let baseline = MajorityClassifier::fit(&x, &y)
            .unwrap()
            .predict_proba(&x)
            .unwrap();
        let loss = |p: &[f64]| {
            p.iter()
                .zip(&y)
                .map(|(p, y)| if *y == 1.0 { -p.ln() } else { -(1.0 - p).ln() })
                .sum::<f64>()
        };
        assert!(loss(&p) < loss(&baseline));
    }
}

#[test]
fn constant_features_return_prevalence_and_intercept_is_not_penalized() {
    let x = vec![vec![7.0, 0.0]; 5];
    let y = [0.0, 0.0, 0.0, 0.0, 1.0];
    for alpha in [0.0, 0.1, 1e100] {
        let model = LogisticClassifier::fit(&x, &y, alpha).unwrap();
        for p in model
            .predict_proba(&[vec![100.0, -100.0], vec![7.0, 0.0]])
            .unwrap()
        {
            close(p, 0.2, 1e-14);
        }
    }
}

#[test]
fn redundant_columns_are_supported_with_regularization() {
    let x = [vec![-1.0, -2.0], vec![0.0, 0.0], vec![1.0, 2.0]];
    let model = LogisticClassifier::fit(&x, &[0.0, 0.0, 1.0], 0.2).unwrap();
    let p = model.predict_proba(&x).unwrap();
    assert!(p[0] < p[1] && p[1] < p[2]);
    assert_eq!(
        LogisticClassifier::fit(&x, &[0.0, 0.0, 1.0], 0.0).unwrap_err(),
        ClassificationError::ConvergenceFailure
    );
}

#[test]
fn standardization_is_invariant_to_translation_and_feature_units() {
    let x = [
        vec![0.0, 2.0],
        vec![1.0, -1.0],
        vec![2.0, 4.0],
        vec![4.0, 1.0],
        vec![3.0, -2.0],
        vec![-1.0, 3.0],
    ];
    let y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let query = [vec![1.5, 2.0], vec![5.0, -2.0]];
    let remap = |row: &Vec<f64>| vec![1e12 + row[0] * 2.0, -7.0 - row[1] * 1000.0];
    let translated: Vec<_> = x.iter().map(remap).collect();
    let translated_query: Vec<_> = query.iter().map(remap).collect();
    for alpha in [0.01, 0.5, 20.0] {
        let a = LogisticClassifier::fit(&x, &y, alpha)
            .unwrap()
            .predict_proba(&query)
            .unwrap();
        let b = LogisticClassifier::fit(&translated, &y, alpha)
            .unwrap()
            .predict_proba(&translated_query)
            .unwrap();
        for (actual, expected) in a.into_iter().zip(b) {
            close(actual, expected, 1e-9);
        }
    }
}

#[test]
fn scaling_is_training_only_and_prediction_is_batch_independent() {
    let x = [vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
    let model = LogisticClassifier::fit(&x, &[0.0, 0.0, 1.0, 1.0], 0.1).unwrap();
    let alone = model.predict_proba(&[vec![0.7]]).unwrap()[0];
    let batch = model
        .predict_proba(&[vec![0.7], vec![1e8], vec![-1e8]])
        .unwrap();
    assert_eq!(alone, batch[0]);
    assert_eq!(batch[1], 1.0);
    assert_eq!(batch[2], 0.0);
    assert!(batch
        .iter()
        .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert_eq!(alone, model.predict_proba(&[vec![0.7]]).unwrap()[0]);
}

#[test]
fn row_permutation_and_label_complement_preserve_symmetry() {
    let x = [vec![-2.0], vec![-1.0], vec![1.0], vec![4.0]];
    let y = [0.0, 1.0, 0.0, 1.0];
    let query = [vec![0.5], vec![2.0]];
    let a = LogisticClassifier::fit(&x, &y, 0.1)
        .unwrap()
        .predict_proba(&query)
        .unwrap();
    let reversed_x: Vec<_> = x.into_iter().rev().collect();
    let reversed_y: Vec<_> = y.into_iter().rev().map(|v| 1.0 - v).collect();
    let b = LogisticClassifier::fit(&reversed_x, &reversed_y, 0.1)
        .unwrap()
        .predict_proba(&query)
        .unwrap();
    for (actual, complemented) in a.into_iter().zip(b) {
        close(actual, 1.0 - complemented, 1e-9);
    }
}

#[test]
fn extreme_finite_feature_ranges_are_scaled_or_explicitly_rejected() {
    for x in [
        [vec![-1e308], vec![1e308]],
        [vec![0.0], vec![f64::from_bits(1)]],
    ] {
        let p = LogisticClassifier::fit(&x, &[0.0, 1.0], 0.1)
            .unwrap()
            .predict_proba(&x)
            .unwrap();
        assert!(p[0] < 0.5 && p[1] > 0.5);
    }
    assert_eq!(
        LogisticClassifier::fit(&[vec![1e308], vec![1e-300]], &[0.0, 1.0], 0.1).unwrap_err(),
        ClassificationError::NumericFailure
    );
}

#[test]
fn invalid_training_inputs_return_fixed_errors_without_data_echo() {
    let cases = [
        (vec![], vec![], ClassificationError::Empty),
        (vec![vec![]], vec![0.0], ClassificationError::Shape),
        (
            vec![vec![1.0], vec![1.0, 2.0]],
            vec![0.0, 1.0],
            ClassificationError::Shape,
        ),
        (vec![vec![1.0]], vec![], ClassificationError::Shape),
        (
            vec![vec![f64::NAN]],
            vec![0.0],
            ClassificationError::NonFinite,
        ),
        (
            vec![vec![1.0]],
            vec![f64::INFINITY],
            ClassificationError::NonFinite,
        ),
        (
            vec![vec![1.0]],
            vec![0.5],
            ClassificationError::InvalidLabel,
        ),
        (
            vec![vec![1.0]],
            vec![-1.0],
            ClassificationError::InvalidLabel,
        ),
        (
            vec![vec![1.0]],
            vec![2.0],
            ClassificationError::InvalidLabel,
        ),
    ];
    for (x, y, error) in cases {
        assert_eq!(LogisticClassifier::fit(&x, &y, 0.1).unwrap_err(), error);
        assert_eq!(MajorityClassifier::fit(&x, &y).unwrap_err(), error);
    }
    for alpha in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            LogisticClassifier::fit(&[vec![0.0], vec![1.0]], &[0.0, 1.0], alpha).unwrap_err(),
            ClassificationError::InvalidAlpha
        );
    }
    for y in [[0.0, 0.0], [1.0, 1.0]] {
        assert_eq!(
            LogisticClassifier::fit(&[vec![0.0], vec![1.0]], &y, 0.1).unwrap_err(),
            ClassificationError::SingleClass
        );
    }
    assert_eq!(
        ClassificationError::InvalidLabel.to_string(),
        "Binary classification labels must be exactly 0 or 1."
    );
    assert_eq!(
        ClassificationError::SingleClass.to_string(),
        "Logistic training requires both binary classes."
    );
}

#[test]
fn predictions_validate_shapes_and_finite_features() {
    let x = [vec![0.0], vec![1.0]];
    let logistic = LogisticClassifier::fit(&x, &[0.0, 1.0], 0.1).unwrap();
    let majority = MajorityClassifier::fit(&x, &[0.0, 1.0]).unwrap();
    for (query, error) in [
        (vec![], ClassificationError::Empty),
        (vec![vec![]], ClassificationError::Shape),
        (vec![vec![0.0, 1.0]], ClassificationError::Shape),
        (
            vec![vec![f64::NEG_INFINITY]],
            ClassificationError::NonFinite,
        ),
    ] {
        assert_eq!(logistic.predict_proba(&query).unwrap_err(), error);
        assert_eq!(majority.predict_proba(&query).unwrap_err(), error);
    }
}

#[test]
fn row_and_feature_capacity_limits_apply_to_fit_and_predict() {
    let x = vec![vec![0.0]; 20_001];
    let y = vec![0.0; x.len()];
    assert_eq!(
        LogisticClassifier::fit(&x, &y, 0.1).unwrap_err(),
        ClassificationError::Capacity
    );
    assert_eq!(
        MajorityClassifier::fit(&x, &y).unwrap_err(),
        ClassificationError::Capacity
    );
    let wide = vec![vec![0.0; 101]; 2];
    assert_eq!(
        LogisticClassifier::fit(&wide, &[0.0, 1.0], 0.1).unwrap_err(),
        ClassificationError::Capacity
    );
    assert_eq!(
        MajorityClassifier::fit(&wide, &[0.0, 1.0]).unwrap_err(),
        ClassificationError::Capacity
    );
    let train = [vec![0.0], vec![1.0]];
    assert_eq!(
        LogisticClassifier::fit(&train, &[0.0, 1.0], 0.1)
            .unwrap()
            .predict_proba(&x)
            .unwrap_err(),
        ClassificationError::Capacity
    );
    assert_eq!(
        MajorityClassifier::fit(&train, &[0.0, 1.0])
            .unwrap()
            .predict_proba(&x)
            .unwrap_err(),
        ClassificationError::Capacity
    );
    let boundary = vec![vec![0.0; 100]; 20_000];
    let labels: Vec<_> = (0..20_000).map(|i| (i % 2) as f64).collect();
    let logistic = LogisticClassifier::fit(&boundary, &labels, 0.1).unwrap();
    assert_eq!(
        logistic.predict_proba(&[vec![0.0; 100]]).unwrap(),
        vec![0.5]
    );
    assert_eq!(
        MajorityClassifier::fit(&boundary, &labels)
            .unwrap()
            .predict_proba(&[vec![0.0; 100]])
            .unwrap(),
        vec![0.5]
    );
}
