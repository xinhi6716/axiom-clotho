use clotho_core::evaluation::{
    grouped_kfold, regression_metrics, EvaluationError, Fold, MAX_GROUPED_ROWS, MAX_GROUPED_SPLITS,
};
use std::collections::{BTreeMap, BTreeSet};

fn close(actual: f64, expected: f64) {
    let tolerance = expected.abs().max(1.0) * 1e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:?} differs from {expected:?}"
    );
}

fn labels(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

fn assert_partition(groups: &[String], folds: &[Fold]) {
    let mut test_counts = vec![0; groups.len()];
    let mut train_counts = vec![0; groups.len()];
    for fold in folds {
        assert!(!fold.train_indices.is_empty());
        assert!(!fold.test_indices.is_empty());
        let train: BTreeSet<_> = fold.train_indices.iter().copied().collect();
        let test: BTreeSet<_> = fold.test_indices.iter().copied().collect();
        assert_eq!(train.len(), fold.train_indices.len());
        assert_eq!(test.len(), fold.test_indices.len());
        assert!(train.is_disjoint(&test));
        assert_eq!(
            train.union(&test).copied().collect::<Vec<_>>(),
            (0..groups.len()).collect::<Vec<_>>()
        );
        assert!(fold.train_indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(fold.test_indices.windows(2).all(|pair| pair[0] < pair[1]));
        let train_groups: BTreeSet<_> = train.iter().map(|&row| &groups[row]).collect();
        let test_groups: BTreeSet<_> = test.iter().map(|&row| &groups[row]).collect();
        assert!(train_groups.is_disjoint(&test_groups));
        for &row in &fold.test_indices {
            test_counts[row] += 1;
        }
        for &row in &fold.train_indices {
            train_counts[row] += 1;
        }
    }
    assert!(test_counts.iter().all(|&count| count == 1));
    assert!(train_counts.iter().all(|&count| count == folds.len() - 1));
}

#[test]
fn regression_metrics_match_hand_calculation() {
    let metrics = regression_metrics(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0]).unwrap();
    close(metrics.mae, 0.5);
    close(metrics.rmse, 0.375_f64.sqrt());
    close(metrics.r2.unwrap(), 1.0 - 1.5 / 29.1875);
}

#[test]
fn perfect_predictions_and_negative_r2() {
    let perfect = regression_metrics(&[-2.0, 0.0, 2.0], &[-2.0, 0.0, 2.0]).unwrap();
    assert_eq!(perfect.mae, 0.0);
    assert_eq!(perfect.rmse, 0.0);
    assert_eq!(perfect.r2, Some(1.0));
    close(
        regression_metrics(&[0.0, 2.0], &[4.0, 4.0])
            .unwrap()
            .r2
            .unwrap(),
        -9.0,
    );
}

#[test]
fn constant_and_singleton_targets_have_no_r2() {
    for (truth, prediction) in [
        (vec![3.0, 3.0], vec![3.0, 3.0]),
        (vec![3.0, 3.0], vec![1.0, 5.0]),
        (vec![7.0], vec![6.0]),
        (vec![-0.0, 0.0], vec![1.0, 1.0]),
    ] {
        assert_eq!(regression_metrics(&truth, &prediction).unwrap().r2, None);
    }
}

#[test]
fn regression_rejects_empty_mismatched_and_nonfinite_inputs() {
    assert_eq!(regression_metrics(&[], &[]), Err(EvaluationError::Empty));
    assert_eq!(regression_metrics(&[], &[1.0]), Err(EvaluationError::Shape));
    assert_eq!(
        regression_metrics(&[1.0], &[1.0, 2.0]),
        Err(EvaluationError::Shape)
    );
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            regression_metrics(&[invalid], &[1.0]),
            Err(EvaluationError::NonFinite)
        );
        assert_eq!(
            regression_metrics(&[1.0], &[invalid]),
            Err(EvaluationError::NonFinite)
        );
    }
}

#[test]
fn large_errors_do_not_overflow_from_squaring() {
    let metrics = regression_metrics(&[-f64::MAX, f64::MAX], &[0.0, 0.0]).unwrap();
    assert_eq!(metrics.mae, f64::MAX);
    assert_eq!(metrics.rmse, f64::MAX);
    close(metrics.r2.unwrap(), 0.0);
    let constant = regression_metrics(&[f64::MAX, f64::MAX], &[0.0, 0.0]).unwrap();
    assert_eq!(constant.rmse, f64::MAX);
    assert_eq!(constant.r2, None);
}

#[test]
fn an_overflowing_raw_residual_can_have_finite_aggregate_metrics() {
    let metrics = regression_metrics(
        &[-f64::MAX, f64::MAX, -f64::MAX, f64::MAX],
        &[f64::MAX, f64::MAX, -f64::MAX, f64::MAX],
    )
    .unwrap();
    assert_eq!(metrics.mae, f64::MAX * 0.5);
    assert_eq!(metrics.rmse, f64::MAX);
    close(metrics.r2.unwrap(), 0.0);
}

#[test]
fn small_variation_survives_a_large_offset() {
    let base = 1e16;
    let metrics = regression_metrics(&[base, base + 2.0], &[base, base]).unwrap();
    close(metrics.mae, 1.0);
    close(metrics.rmse, 2.0_f64.sqrt());
    close(metrics.r2.unwrap(), -1.0);
}

#[test]
fn tiny_values_do_not_disappear_from_squaring() {
    let metrics = regression_metrics(&[-1e-200, 1e-200], &[0.0, 0.0]).unwrap();
    assert_eq!(metrics.mae, 1e-200);
    assert_eq!(metrics.rmse, 1e-200);
    close(metrics.r2.unwrap(), 0.0);
    let tiny = f64::from_bits(1);
    let metrics = regression_metrics(&[-tiny, tiny], &[0.0, 0.0]).unwrap();
    assert_eq!(metrics.mae, tiny);
    assert_eq!(metrics.rmse, tiny);
    close(metrics.r2.unwrap(), 0.0);
}

#[test]
fn unrepresentable_results_and_normalization_fail_explicitly() {
    assert_eq!(
        regression_metrics(&[f64::MAX], &[-f64::MAX]),
        Err(EvaluationError::NumericOverflow)
    );
    assert_eq!(
        regression_metrics(&[0.0, 1e-200], &[1e200, 1e200]),
        Err(EvaluationError::NumericOverflow)
    );
    assert_eq!(
        regression_metrics(&[0.0, 0.0], &[f64::MAX, f64::from_bits(1)]),
        Err(EvaluationError::NumericOverflow)
    );
    assert_eq!(
        regression_metrics(&[0.0, 0.0], &[f64::from_bits(1), 0.0]),
        Err(EvaluationError::NumericOverflow)
    );
}

#[test]
fn ordinary_values_match_an_independent_direct_reference() {
    let mut state = 12345_u64;
    for count in 2..40 {
        let mut truth = Vec::new();
        let mut predictions = Vec::new();
        for _ in 0..count {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((state >> 32) % 100_000) as f64 / 100.0 - 500.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let error = ((state >> 32) % 1_000) as f64 / 10.0 - 50.0;
            truth.push(value);
            predictions.push(value + error);
        }
        let mean = truth.iter().sum::<f64>() / count as f64;
        let absolute_error = truth
            .iter()
            .zip(&predictions)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        let squared_error = truth
            .iter()
            .zip(&predictions)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        let total_variation = truth
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        let metrics = regression_metrics(&truth, &predictions).unwrap();
        close(metrics.mae, absolute_error / count as f64);
        close(metrics.rmse, (squared_error / count as f64).sqrt());
        close(metrics.r2.unwrap(), 1.0 - squared_error / total_variation);
    }
}

#[test]
fn uneven_groups_are_greedily_balanced_without_leakage() {
    let groups = labels(&[
        "a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "d", "d", "e",
    ]);
    let folds = grouped_kfold(&groups, 3).unwrap();
    assert_partition(&groups, &folds);
    assert_eq!(
        folds
            .iter()
            .map(|fold| fold.test_indices.len())
            .collect::<Vec<_>>(),
        vec![5, 5, 5]
    );
    assert_eq!(folds[0].test_indices, vec![0, 1, 2, 3, 4]);
    assert_eq!(folds[1].test_indices, vec![5, 6, 7, 8, 14]);
    assert_eq!(folds[2].test_indices, vec![9, 10, 11, 12, 13]);
}

#[test]
fn equal_size_ties_use_labels_then_fold_indices() {
    let groups = labels(&["d", "b", "a", "c", "d", "a", "c", "b"]);
    let folds = grouped_kfold(&groups, 2).unwrap();
    assert_partition(&groups, &folds);
    assert_eq!(folds[0].test_indices, vec![2, 3, 5, 6]);
    assert_eq!(folds[1].test_indices, vec![0, 1, 4, 7]);
    for _ in 0..10 {
        assert_eq!(grouped_kfold(&groups, 2).unwrap(), folds);
    }
}

#[test]
fn group_assignment_is_stable_under_row_reordering() {
    fn assignments(groups: &[String], folds: &[Fold]) -> BTreeMap<String, usize> {
        folds
            .iter()
            .enumerate()
            .flat_map(|(fold, indices)| {
                indices
                    .test_indices
                    .iter()
                    .map(move |&row| (groups[row].clone(), fold))
            })
            .collect()
    }
    let groups = labels(&["z", "a", "z", "b", "a", "z", "c", "b", "d"]);
    let folds = grouped_kfold(&groups, 3).unwrap();
    let mut reversed = groups.clone();
    reversed.reverse();
    let reversed_folds = grouped_kfold(&reversed, 3).unwrap();
    assert_partition(&reversed, &reversed_folds);
    assert_eq!(
        assignments(&groups, &folds),
        assignments(&reversed, &reversed_folds)
    );
}

#[test]
fn one_fold_per_group_and_empty_labels_are_supported() {
    let groups = labels(&["", "A", "a", "", "A"]);
    let folds = grouped_kfold(&groups, 3).unwrap();
    assert_partition(&groups, &folds);
    for fold in folds {
        let group = &groups[fold.test_indices[0]];
        assert!(fold.test_indices.iter().all(|&row| &groups[row] == group));
    }
}

#[test]
fn dominant_group_remains_indivisible() {
    let mut groups = vec!["large".to_string(); 100];
    groups.extend(labels(&["b", "c", "d"]));
    let folds = grouped_kfold(&groups, 3).unwrap();
    assert_partition(&groups, &folds);
    assert_eq!(folds[0].test_indices.len(), 100);
    assert_eq!(folds[1].test_indices.len(), 2);
    assert_eq!(folds[2].test_indices.len(), 1);
}

#[test]
fn invalid_grouped_fold_requests_fail() {
    assert_eq!(grouped_kfold(&[], 2), Err(EvaluationError::Empty));
    let groups = labels(&["a", "a", "b"]);
    for splits in [0, 1] {
        assert_eq!(
            grouped_kfold(&groups, splits),
            Err(EvaluationError::InvalidSplits)
        );
    }
    for splits in [3, 4] {
        assert_eq!(
            grouped_kfold(&groups, splits),
            Err(EvaluationError::InsufficientGroups)
        );
    }
    assert_eq!(
        grouped_kfold(&labels(&["a"]), 2),
        Err(EvaluationError::InsufficientGroups)
    );
}

#[test]
fn grouped_folds_reject_excessive_materialized_output() {
    let too_many_rows = vec!["same".to_string(); MAX_GROUPED_ROWS + 1];
    assert_eq!(
        grouped_kfold(&too_many_rows, 2),
        Err(EvaluationError::Capacity)
    );
    let groups: Vec<_> = (0..=MAX_GROUPED_SPLITS)
        .map(|row| row.to_string())
        .collect();
    for splits in [MAX_GROUPED_SPLITS + 1, usize::MAX] {
        assert_eq!(
            grouped_kfold(&groups, splits),
            Err(EvaluationError::Capacity)
        );
    }
}

#[test]
fn grouped_folds_accept_inclusive_capacity_limits() {
    let groups: Vec<_> = (0..MAX_GROUPED_ROWS)
        .map(|row| (row % MAX_GROUPED_SPLITS).to_string())
        .collect();
    let folds = grouped_kfold(&groups, MAX_GROUPED_SPLITS).unwrap();
    assert_eq!(folds.len(), MAX_GROUPED_SPLITS);
    assert_eq!(
        folds
            .iter()
            .map(|fold| fold.train_indices.len() + fold.test_indices.len())
            .sum::<usize>(),
        MAX_GROUPED_ROWS * MAX_GROUPED_SPLITS
    );
    assert_partition(&groups, &folds);
}

#[test]
fn generated_group_shapes_preserve_all_partition_invariants() {
    for number_of_groups in 2..15 {
        let mut groups = Vec::new();
        for group in 0..number_of_groups {
            for _ in 0..(group * 7 + 3) % 11 + 1 {
                groups.push(format!("group-{group}"));
            }
        }
        for splits in 2..=number_of_groups {
            let folds = grouped_kfold(&groups, splits).unwrap();
            assert_partition(&groups, &folds);
            assert_eq!(grouped_kfold(&groups, splits).unwrap(), folds);
        }
    }
}
