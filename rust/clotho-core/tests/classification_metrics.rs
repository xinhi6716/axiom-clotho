use clotho_core::classification_metrics::{
    binary_metrics, BinaryMetrics, ClassificationMetricsError, LOG_LOSS_EPSILON,
};

fn close(actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:?} differs from {expected:?}"
    );
}

fn compare(actual: BinaryMetrics, expected: BinaryMetrics) {
    close(actual.accuracy, expected.accuracy);
    close(actual.log_loss, expected.log_loss);
    close(actual.brier, expected.brier);
    for (actual, expected) in [
        (actual.balanced_accuracy, expected.balanced_accuracy),
        (actual.precision, expected.precision),
        (actual.recall, expected.recall),
        (actual.f1, expected.f1),
        (actual.roc_auc, expected.roc_auc),
    ] {
        match (actual, expected) {
            (Some(actual), Some(expected)) => close(actual, expected),
            (None, None) => {}
            _ => panic!("Metrics have different definedness"),
        }
    }
    assert_eq!(actual.n_positive, expected.n_positive);
    assert_eq!(actual.n_negative, expected.n_negative);
    assert_eq!(actual.tp, expected.tp);
    assert_eq!(actual.fp, expected.fp);
    assert_eq!(actual.tn, expected.tn);
    assert_eq!(actual.fn_, expected.fn_);
}

fn pairwise_auc(labels: &[f64], probabilities: &[f64]) -> Option<f64> {
    let mut credit = 0.0;
    let mut pairs = 0;
    for (i, &label) in labels.iter().enumerate() {
        if label != 1.0 {
            continue;
        }
        for (j, &negative_label) in labels.iter().enumerate() {
            if negative_label != 0.0 {
                continue;
            }
            pairs += 1;
            if probabilities[i] > probabilities[j] {
                credit += 1.0;
            } else if probabilities[i] == probabilities[j] {
                credit += 0.5;
            }
        }
    }
    if pairs == 0 {
        None
    } else {
        Some(credit / pairs as f64)
    }
}

#[test]
fn analytic_probabilities_match_hand_calculation() {
    let metrics = binary_metrics(&[0.0, 0.0, 1.0, 1.0], &[0.1, 0.4, 0.35, 0.8], 0.5).unwrap();
    close(metrics.accuracy, 0.75);
    close(metrics.balanced_accuracy.unwrap(), 0.75);
    close(metrics.precision.unwrap(), 1.0);
    close(metrics.recall.unwrap(), 0.5);
    close(metrics.f1.unwrap(), 2.0 / 3.0);
    close(
        metrics.log_loss,
        -(0.9_f64.ln() + 0.6_f64.ln() + 0.35_f64.ln() + 0.8_f64.ln()) / 4.0,
    );
    close(metrics.brier, (0.01 + 0.16 + 0.4225 + 0.04) / 4.0);
    close(metrics.roc_auc.unwrap(), 0.75);
    assert_eq!((metrics.n_positive, metrics.n_negative), (2, 2));
    assert_eq!(
        (metrics.tp, metrics.fp, metrics.tn, metrics.fn_),
        (1, 0, 2, 1)
    );
}

#[test]
fn class_imbalance_distinguishes_balanced_accuracy() {
    let metrics = binary_metrics(&[0.0, 0.0, 0.0, 1.0], &[0.1, 0.1, 0.1, 0.1], 0.5).unwrap();
    close(metrics.accuracy, 0.75);
    assert_eq!(metrics.balanced_accuracy, Some(0.5));
    assert_eq!(metrics.precision, None);
    assert_eq!(metrics.recall, Some(0.0));
    assert_eq!(metrics.f1, Some(0.0));
    assert_eq!(metrics.roc_auc, Some(0.5));
}

#[test]
fn tied_auc_uses_average_ranks() {
    let metrics = binary_metrics(
        &[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        &[0.2, 0.2, 0.5, 0.5, 0.9, 0.9],
        0.5,
    )
    .unwrap();
    assert_eq!(metrics.roc_auc, Some(0.5));

    // Average positive ranks are 2 and 4; (2 + 4 - 3) / (2 * 2) = 0.75.
    let asymmetric = binary_metrics(&[0.0, 1.0, 0.0, 1.0], &[0.2, 0.2, 0.2, 0.9], 0.5).unwrap();
    assert_eq!(asymmetric.roc_auc, Some(0.75));
}

#[test]
fn perfect_and_inverted_rankings_have_endpoint_auc() {
    let labels = [0.0, 1.0, 0.0, 1.0];
    let perfect = binary_metrics(&labels, &[0.1, 0.8, 0.2, 0.9], 0.5).unwrap();
    assert_eq!(perfect.roc_auc, Some(1.0));
    assert_eq!(perfect.accuracy, 1.0);
    assert_eq!(perfect.f1, Some(1.0));
    let inverted = binary_metrics(&labels, &[0.9, 0.2, 0.8, 0.1], 0.5).unwrap();
    assert_eq!(inverted.roc_auc, Some(0.0));
    assert_eq!(inverted.accuracy, 0.0);
    assert_eq!(inverted.f1, Some(0.0));
}

#[test]
fn single_class_metrics_preserve_undefined_ratios() {
    let negative = binary_metrics(&[0.0, 0.0], &[0.1, 0.2], 0.5).unwrap();
    assert_eq!(negative.accuracy, 1.0);
    assert_eq!(negative.balanced_accuracy, None);
    assert_eq!(negative.precision, None);
    assert_eq!(negative.recall, None);
    assert_eq!(negative.f1, None);
    assert_eq!(negative.roc_auc, None);
    assert_eq!((negative.n_positive, negative.n_negative), (0, 2));

    let false_positive = binary_metrics(&[0.0, 0.0], &[0.1, 0.9], 0.5).unwrap();
    assert_eq!(false_positive.precision, Some(0.0));
    assert_eq!(false_positive.recall, None);
    assert_eq!(false_positive.f1, Some(0.0));
    assert_eq!(false_positive.balanced_accuracy, None);
    assert_eq!(false_positive.roc_auc, None);

    let positive = binary_metrics(&[1.0, 1.0], &[0.1, 0.9], 0.5).unwrap();
    assert_eq!(positive.balanced_accuracy, None);
    assert_eq!(positive.precision, Some(1.0));
    assert_eq!(positive.recall, Some(0.5));
    close(positive.f1.unwrap(), 2.0 / 3.0);
    assert_eq!(positive.roc_auc, None);
    assert_eq!((positive.n_positive, positive.n_negative), (2, 0));

    for label in [0.0, 1.0] {
        let single = binary_metrics(&[label], &[label], 0.5).unwrap();
        assert_eq!(single.accuracy, 1.0);
        assert_eq!(single.balanced_accuracy, None);
        assert_eq!(single.roc_auc, None);
    }
}

#[test]
fn all_equal_scores_receive_half_auc_credit() {
    for value in [0.0, 0.4, 1.0] {
        let metrics = binary_metrics(&[0.0, 1.0, 1.0, 1.0], &[value; 4], 0.5).unwrap();
        assert_eq!(metrics.roc_auc, Some(0.5));
    }
    // Both signed representations of zero are one tie group.
    let signed = binary_metrics(&[-0.0, 1.0, 0.0, 1.0], &[-0.0, 0.0, 0.0, -0.0], 0.5).unwrap();
    assert_eq!(signed.roc_auc, Some(0.5));
}

#[test]
fn inclusive_threshold_and_boundary_thresholds_are_explicit() {
    let labels = [0.0, 1.0, 0.0, 1.0];
    let probabilities = [0.0, 0.5, 0.5, 1.0];
    let middle = binary_metrics(&labels, &probabilities, 0.5).unwrap();
    assert_eq!((middle.tp, middle.fp, middle.tn, middle.fn_), (2, 1, 1, 0));
    let zero = binary_metrics(&labels, &probabilities, 0.0).unwrap();
    assert_eq!((zero.tp, zero.fp, zero.tn, zero.fn_), (2, 2, 0, 0));
    let one = binary_metrics(&labels, &probabilities, 1.0).unwrap();
    assert_eq!((one.tp, one.fp, one.tn, one.fn_), (1, 0, 2, 1));
    for metrics in [zero, one] {
        assert_eq!(metrics.roc_auc, middle.roc_auc);
        assert_eq!(metrics.log_loss, middle.log_loss);
        assert_eq!(metrics.brier, middle.brier);
    }
    compare(binary_metrics(&labels, &probabilities, -0.0).unwrap(), zero);
}

#[test]
fn endpoint_probabilities_have_finite_clipped_log_loss() {
    let labels = [0.0, 1.0];
    let perfect = binary_metrics(&labels, &[0.0, 1.0], 0.5).unwrap();
    assert!(perfect.log_loss > 0.0);
    assert!(perfect.log_loss < 2.0 * LOG_LOSS_EPSILON);
    assert_eq!(perfect.brier, 0.0);
    let inverted = binary_metrics(&labels, &[1.0, 0.0], 0.5).unwrap();
    close(
        inverted.log_loss,
        -((-(1.0 - LOG_LOSS_EPSILON)).ln_1p() + LOG_LOSS_EPSILON.ln()) / 2.0,
    );
    assert!(inverted.log_loss.is_finite());
    assert!(inverted.log_loss > 30.0);
    assert_eq!(inverted.brier, 1.0);

    // Clipping also covers near-endpoint values, but is not used for Brier/AUC.
    let probabilities = [LOG_LOSS_EPSILON * 0.25, LOG_LOSS_EPSILON * 0.5];
    let tiny = binary_metrics(&labels, &probabilities, LOG_LOSS_EPSILON * 0.375).unwrap();
    assert_eq!(tiny.accuracy, 1.0);
    assert_eq!(tiny.roc_auc, Some(1.0));
    let expected_loss = -((-LOG_LOSS_EPSILON).ln_1p() + LOG_LOSS_EPSILON.ln()) / 2.0;
    close(tiny.log_loss, expected_loss);
    assert_eq!(
        tiny.brier,
        (probabilities[0].powi(2) + (probabilities[1] - 1.0).powi(2)) / 2.0
    );
}

#[test]
fn row_permutations_preserve_metrics_and_inputs() {
    let labels = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let probabilities = vec![0.0, 0.1, 0.1, 0.7, 1.0, 0.7, 0.4];
    let original_labels = labels.clone();
    let original_probabilities = probabilities.clone();
    let expected = binary_metrics(&labels, &probabilities, 0.7).unwrap();
    for shift in 0..labels.len() {
        for reverse in [false, true] {
            let mut order: Vec<_> = (0..labels.len()).collect();
            order.rotate_left(shift);
            if reverse {
                order.reverse();
            }
            let permuted_labels: Vec<_> = order.iter().map(|&row| labels[row]).collect();
            let permuted_probabilities: Vec<_> =
                order.iter().map(|&row| probabilities[row]).collect();
            compare(
                binary_metrics(&permuted_labels, &permuted_probabilities, 0.7).unwrap(),
                expected,
            );
        }
    }
    assert_eq!(labels, original_labels);
    assert_eq!(probabilities, original_probabilities);
}

#[test]
fn ranked_auc_matches_independent_pairwise_reference() {
    let mut state = 12345_u64;
    for count in 2..90 {
        let mut labels = vec![0.0, 1.0];
        let mut probabilities = vec![0.0, 1.0];
        for _ in 2..count {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            labels.push(((state >> 32) % 2) as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            probabilities.push(((state >> 32) % 9) as f64 / 8.0);
        }
        let metrics = binary_metrics(&labels, &probabilities, 0.5).unwrap();
        close(
            metrics.roc_auc.unwrap(),
            pairwise_auc(&labels, &probabilities).unwrap(),
        );
        assert_eq!(metrics.n_positive + metrics.n_negative, count);
        assert_eq!(metrics.tp + metrics.fp + metrics.tn + metrics.fn_, count);

        let reversed: Vec<_> = probabilities.iter().map(|value| 1.0 - value).collect();
        close(
            binary_metrics(&labels, &reversed, 0.5)
                .unwrap()
                .roc_auc
                .unwrap(),
            1.0 - metrics.roc_auc.unwrap(),
        );
        let squared: Vec<_> = probabilities.iter().map(|value| value * value).collect();
        assert_eq!(
            binary_metrics(&labels, &squared, 0.5).unwrap().roc_auc,
            metrics.roc_auc
        );
    }
}

#[test]
fn invalid_inputs_are_rejected_without_echoing_values() {
    assert_eq!(
        binary_metrics(&[], &[], 0.5),
        Err(ClassificationMetricsError::Empty)
    );
    assert_eq!(
        binary_metrics(&[], &[0.2], 0.5),
        Err(ClassificationMetricsError::Shape)
    );
    assert_eq!(
        binary_metrics(&[1.0], &[], 0.5),
        Err(ClassificationMetricsError::Shape)
    );
    for invalid in [-0.1, 0.2, 2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            binary_metrics(&[0.0, invalid], &[0.2, 0.7], 0.5),
            Err(ClassificationMetricsError::InvalidLabel)
        );
    }
    for invalid in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            binary_metrics(&[0.0, 1.0], &[0.2, invalid], 0.5),
            Err(ClassificationMetricsError::InvalidProbability)
        );
        assert_eq!(
            binary_metrics(&[0.0, 1.0], &[0.2, 0.7], invalid),
            Err(ClassificationMetricsError::InvalidThreshold)
        );
    }
    let error = binary_metrics(&[123456.789], &[0.2], 0.5).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Binary labels must be finite and exactly zero or one."
    );
    assert!(!format!("{error:?}").contains("123456"));
}

#[test]
fn large_tied_input_keeps_auc_in_range() {
    let count = 20_000;
    let labels: Vec<_> = (0..count).map(|index| (index % 2) as f64).collect();
    let probabilities = vec![0.5; count];
    let metrics = binary_metrics(&labels, &probabilities, 0.5).unwrap();
    assert_eq!(metrics.roc_auc, Some(0.5));
    assert_eq!(metrics.n_positive, count / 2);
    assert_eq!(metrics.n_negative, count / 2);
    close(metrics.log_loss, 2.0_f64.ln());
    assert_eq!(metrics.brier, 0.25);
}
