use clotho_core::evaluation::{grouped_kfold, regression_metrics, EvaluationError};
use clotho_core::knn::{KnnError, KnnRegressor};
use clotho_core::regression::{MeanRegressor, RegressionError, RidgeRegressor};
use clotho_core::selection::{
    nested_grouped_regression, CandidateError, FailureStage, RegressionCandidate, SelectionError,
    MAX_SELECTION_CANDIDATES, MAX_SELECTION_FEATURES, MAX_SELECTION_ROWS, MAX_SELECTION_SPLITS,
};
use std::collections::BTreeSet;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * expected.abs().max(1.0),
        "actual {actual:?}, expected {expected:?}"
    );
}

fn fixture() -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut groups = Vec::new();
    for group in 0..9 {
        for row in 0..(1 + group % 4) {
            let a = (group * 3 + row) as f64;
            let b = ((group * 7 + row * 5) % 11) as f64;
            x.push(vec![a, b]);
            y.push(a * 0.7 - b * 0.2 + (group % 3) as f64);
            groups.push(format!("group-{group}"));
        }
    }
    (x, y, groups)
}

fn rows(x: &[Vec<f64>], y: &[f64], indices: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        indices.iter().map(|&i| x[i].clone()).collect(),
        indices.iter().map(|&i| y[i]).collect(),
    )
}

fn predict(
    candidate: RegressionCandidate,
    x: &[Vec<f64>],
    y: &[f64],
    test: &[Vec<f64>],
) -> Vec<f64> {
    match candidate {
        RegressionCandidate::Mean => MeanRegressor::fit(x, y).unwrap().predict(test).unwrap(),
        RegressionCandidate::Ridge { alpha } => RidgeRegressor::fit(x, y, alpha)
            .unwrap()
            .predict(test)
            .unwrap(),
        RegressionCandidate::Knn { k } => {
            KnnRegressor::fit(x, y, k).unwrap().predict(test).unwrap()
        }
    }
}

#[test]
fn nested_results_match_manual_training_only_fits_and_original_row_alignment() {
    let (x, y, groups) = fixture();
    let candidates = [
        RegressionCandidate::Mean,
        RegressionCandidate::Ridge { alpha: 0.2 },
        RegressionCandidate::Ridge { alpha: 2.0 },
        RegressionCandidate::Knn { k: 1 },
        RegressionCandidate::Knn { k: 2 },
    ];
    let report = nested_grouped_regression(&x, &y, &groups, 3, 3, &candidates, 48).unwrap();
    let outer_plan = grouped_kfold(&groups, 3).unwrap();
    let mut expected = vec![0.0; x.len()];
    let mut coverage = vec![0; x.len()];
    for (outer, result) in outer_plan.iter().zip(&report.outer_folds) {
        assert_eq!(outer.train_indices, result.train_indices);
        assert_eq!(outer.test_indices, result.test_indices);
        let outer_test_groups: BTreeSet<_> =
            outer.test_indices.iter().map(|&i| &groups[i]).collect();
        let training_groups: Vec<_> = outer
            .train_indices
            .iter()
            .map(|&i| groups[i].clone())
            .collect();
        let (train_x, train_y) = rows(&x, &y, &outer.train_indices);
        let inner_plan = grouped_kfold(&training_groups, 3).unwrap();
        let mut manual_scores = Vec::new();
        for (index, &candidate) in candidates.iter().enumerate() {
            let mut inner_predictions = vec![0.0; train_x.len()];
            for inner in &inner_plan {
                let inner_train_groups: BTreeSet<_> = inner
                    .train_indices
                    .iter()
                    .map(|&i| &training_groups[i])
                    .collect();
                let inner_test_groups: BTreeSet<_> = inner
                    .test_indices
                    .iter()
                    .map(|&i| &training_groups[i])
                    .collect();
                assert!(inner_train_groups.is_disjoint(&inner_test_groups));
                assert!(inner_train_groups.is_disjoint(&outer_test_groups));
                assert!(inner_test_groups.is_disjoint(&outer_test_groups));
                let (inner_x, inner_y) = rows(&train_x, &train_y, &inner.train_indices);
                let (test_x, _) = rows(&train_x, &train_y, &inner.test_indices);
                let predictions = predict(candidate, &inner_x, &inner_y, &test_x);
                for (&row, prediction) in inner.test_indices.iter().zip(predictions) {
                    inner_predictions[row] = prediction;
                }
            }
            let rmse = regression_metrics(&train_y, &inner_predictions)
                .unwrap()
                .rmse;
            manual_scores.push(rmse);
            let score = &result.inner_candidate_scores[index];
            assert_eq!(score.candidate_index, index);
            assert_eq!(score.fit_count, 3);
            assert_eq!(score.failure, None);
            close(score.pooled_rmse.unwrap(), rmse);
        }
        let selected = (0..candidates.len())
            .min_by(|&a, &b| {
                manual_scores[a]
                    .total_cmp(&manual_scores[b])
                    .then(a.cmp(&b))
            })
            .unwrap();
        assert_eq!(result.selected_candidate_index, selected);
        assert_eq!(result.fit_count, 16);
        let (test_x, _) = rows(&x, &y, &outer.test_indices);
        for (&row, prediction) in outer.test_indices.iter().zip(predict(
            candidates[selected],
            &train_x,
            &train_y,
            &test_x,
        )) {
            coverage[row] += 1;
            expected[row] = prediction;
        }
    }
    assert!(coverage.iter().all(|&count| count == 1));
    assert_eq!(report.out_of_fold_predictions, expected);
    assert_eq!(
        report.pooled_metrics,
        regression_metrics(&y, &expected).unwrap()
    );
    assert_eq!(report.fit_count, 48);
    assert_eq!(report.planned_fit_count, 48);
}

#[test]
fn changing_outer_test_targets_cannot_change_selection_or_their_predictions() {
    let (x, y, groups) = fixture();
    let candidates = [
        RegressionCandidate::Mean,
        RegressionCandidate::Ridge { alpha: 0.1 },
        RegressionCandidate::Knn { k: 1 },
    ];
    let original = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 21).unwrap();
    for (outer_index, original_fold) in original.outer_folds.iter().enumerate() {
        let mut changed_y = y.clone();
        for &row in &original_fold.test_indices {
            changed_y[row] = 10_000.0 + row as f64;
        }
        let changed =
            nested_grouped_regression(&x, &changed_y, &groups, 3, 2, &candidates, 21).unwrap();
        assert_eq!(original_fold, &changed.outer_folds[outer_index]);
        for &row in &original_fold.test_indices {
            assert_eq!(
                original.out_of_fold_predictions[row],
                changed.out_of_fold_predictions[row]
            );
        }
        assert_ne!(original.pooled_metrics, changed.pooled_metrics);
    }
}

#[test]
fn changing_outer_test_features_cannot_change_inner_scores_or_selection() {
    let (x, y, groups) = fixture();
    let candidates = [
        RegressionCandidate::Ridge { alpha: 1.0 },
        RegressionCandidate::Knn { k: 1 },
    ];
    let original = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15).unwrap();
    for (outer_index, original_fold) in original.outer_folds.iter().enumerate() {
        let mut changed_x = x.clone();
        for &row in &original_fold.test_indices {
            changed_x[row] = vec![100_000.0, -50_000.0];
        }
        let changed =
            nested_grouped_regression(&changed_x, &y, &groups, 3, 2, &candidates, 15).unwrap();
        assert_eq!(original_fold, &changed.outer_folds[outer_index]);
    }
}

#[test]
fn exact_ties_and_repeated_studies_follow_candidate_order() {
    let (x, _, groups) = fixture();
    let y = vec![7.0; x.len()];
    let candidates = [
        RegressionCandidate::Knn { k: 1 },
        RegressionCandidate::Mean,
        RegressionCandidate::Mean,
    ];
    let first = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 21).unwrap();
    let second = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 21).unwrap();
    assert_eq!(first, second);
    assert!(first
        .outer_folds
        .iter()
        .all(|fold| fold.selected_candidate_index == 0));
    assert!(first
        .outer_folds
        .iter()
        .flat_map(|fold| &fold.inner_candidate_scores)
        .all(|score| score.pooled_rmse == Some(0.0)));
    assert_eq!(first.out_of_fold_predictions, y);
    assert_eq!(first.pooled_metrics.r2, None);
}

#[test]
fn selection_uses_pooled_rmse_not_an_unweighted_average_of_fold_rmse() {
    let groups: Vec<_> = [7, 4, 3, 2, 1, 1]
        .iter()
        .enumerate()
        .flat_map(|(group, &count)| vec![group.to_string(); count])
        .collect();
    let x: Vec<_> = (0..groups.len()).map(|i| vec![i as f64]).collect();
    let y = vec![
        -3.0, 2.0, 5.0, 0.0, 0.0, 3.0, 2.0, 2.0, 5.0, -3.0, 4.0, -4.0, -1.0, -4.0, 2.0, -3.0, 5.0,
        -2.0,
    ];
    let candidates = [RegressionCandidate::Mean, RegressionCandidate::Knn { k: 1 }];
    let report = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15).unwrap();
    let outer = &report.outer_folds[2];
    let (train_x, train_y) = rows(&x, &y, &outer.train_indices);
    let training_groups: Vec<_> = outer
        .train_indices
        .iter()
        .map(|&i| groups[i].clone())
        .collect();
    let inner = grouped_kfold(&training_groups, 2).unwrap();
    let mut averages = Vec::new();
    for candidate in candidates {
        let average = inner
            .iter()
            .map(|fold| {
                let (inner_x, inner_y) = rows(&train_x, &train_y, &fold.train_indices);
                let (test_x, test_y) = rows(&train_x, &train_y, &fold.test_indices);
                regression_metrics(&test_y, &predict(candidate, &inner_x, &inner_y, &test_x))
                    .unwrap()
                    .rmse
            })
            .sum::<f64>()
            / 2.0;
        averages.push(average);
    }
    assert!(averages[0] < averages[1]);
    assert!(
        outer.inner_candidate_scores[1].pooled_rmse < outer.inner_candidate_scores[0].pooled_rmse
    );
    assert_eq!(outer.selected_candidate_index, 1);
}

#[test]
fn failed_candidate_has_no_partial_score_and_counts_its_attempted_fit() {
    let (_, y, groups) = fixture();
    let x = vec![vec![1.0]; y.len()];
    let candidates = [
        RegressionCandidate::Ridge { alpha: 0.0 },
        RegressionCandidate::Mean,
    ];
    let report = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15).unwrap();
    assert_eq!(report.planned_fit_count, 15);
    assert_eq!(report.fit_count, 12);
    for fold in report.outer_folds {
        assert_eq!(fold.fit_count, 4);
        assert_eq!(fold.selected_candidate_index, 1);
        let failed = &fold.inner_candidate_scores[0];
        assert_eq!(failed.pooled_rmse, None);
        assert_eq!(failed.fit_count, 1);
        let failure = failed.failure.as_ref().unwrap();
        assert_eq!(failure.inner_fold_index, Some(0));
        assert_eq!(failure.stage, FailureStage::Fit);
        assert_eq!(
            failure.error,
            CandidateError::Regression(RegressionError::IllConditioned)
        );
        assert_eq!(fold.inner_candidate_scores[1].fit_count, 2);
    }
}

#[test]
fn every_failed_candidate_fails_the_study_with_failure_details_and_fit_count() {
    let (_, y, groups) = fixture();
    let x = vec![vec![1.0]; y.len()];
    let candidates = [RegressionCandidate::Ridge { alpha: 0.0 }; 2];
    match nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15).unwrap_err() {
        SelectionError::NoValidCandidate {
            outer_fold_index,
            fit_count,
            inner_candidate_scores,
        } => {
            assert_eq!(outer_fold_index, 0);
            assert_eq!(fit_count, 2);
            assert_eq!(inner_candidate_scores.len(), 2);
            assert!(inner_candidate_scores
                .iter()
                .all(|score| score.pooled_rmse.is_none()
                    && score.failure.is_some()
                    && score.fit_count == 1));
        }
        other => panic!("Unexpected error: {other:?}"),
    }
}

#[test]
fn failure_after_a_successful_inner_fit_discards_the_partial_predictions() {
    let x = vec![
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
    ];
    let y = vec![1.0; x.len()];
    let groups: Vec<_> = (0..x.len()).map(|i| i.to_string()).collect();
    let candidates = [
        RegressionCandidate::Ridge { alpha: 0.0 },
        RegressionCandidate::Mean,
    ];
    let report = nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15).unwrap();
    let fold = &report.outer_folds[0];
    let failed = &fold.inner_candidate_scores[0];
    assert_eq!(fold.selected_candidate_index, 1);
    assert_eq!(fold.fit_count, 5);
    assert_eq!(failed.fit_count, 2);
    assert_eq!(failed.pooled_rmse, None);
    assert_eq!(failed.failure.as_ref().unwrap().inner_fold_index, Some(1));
    assert_eq!(failed.failure.as_ref().unwrap().stage, FailureStage::Fit);
    assert_eq!(report.out_of_fold_predictions, y);
    assert_eq!(
        report.fit_count,
        report.outer_folds.iter().map(|fold| fold.fit_count).sum()
    );
}

#[test]
fn pooled_scoring_failure_is_recorded_after_all_inner_fits() {
    let x = vec![vec![1.0]; 6];
    let y = vec![0.0, 1e300, 1e-300, 0.0, 0.0, 0.0];
    let groups: Vec<_> = (0..x.len()).map(|i| i.to_string()).collect();
    let candidates = [RegressionCandidate::Knn { k: 1 }];
    match nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 9).unwrap_err() {
        SelectionError::NoValidCandidate {
            outer_fold_index,
            fit_count,
            inner_candidate_scores,
        } => {
            assert_eq!(outer_fold_index, 0);
            assert_eq!(fit_count, 2);
            let failed = &inner_candidate_scores[0];
            assert_eq!(failed.fit_count, 2);
            assert_eq!(failed.pooled_rmse, None);
            let failure = failed.failure.as_ref().unwrap();
            assert_eq!(failure.inner_fold_index, None);
            assert_eq!(failure.stage, FailureStage::Score);
            assert_eq!(
                failure.error,
                CandidateError::Evaluation(EvaluationError::NumericOverflow)
            );
        }
        other => panic!("Unexpected error: {other:?}"),
    }
}

#[test]
fn finite_inner_rmse_is_not_rejected_when_unrequested_inner_r2_overflows() {
    let x = vec![
        vec![0.5],
        vec![0.0],
        vec![1.0],
        vec![3.0],
        vec![1e-200],
        vec![2.0],
    ];
    let y = vec![1e-200, 0.0, 0.0, 1e-200, 1e-200, 1e-200];
    let groups: Vec<_> = (0..x.len()).map(|i| i.to_string()).collect();
    let candidate = RegressionCandidate::Ridge { alpha: 1.0 };
    let report = nested_grouped_regression(&x, &y, &groups, 3, 2, &[candidate], 9).unwrap();
    let mut manual_outer_predictions = vec![0.0; x.len()];
    let mut saw_unrepresentable_inner_r2 = false;
    for (outer_index, outer) in grouped_kfold(&groups, 3).unwrap().iter().enumerate() {
        let (outer_x, outer_y) = rows(&x, &y, &outer.train_indices);
        let outer_groups: Vec<_> = outer
            .train_indices
            .iter()
            .map(|&i| groups[i].clone())
            .collect();
        let mut inner_predictions = vec![0.0; outer_x.len()];
        for inner in grouped_kfold(&outer_groups, 2).unwrap() {
            let (train_x, train_y) = rows(&outer_x, &outer_y, &inner.train_indices);
            let (test_x, _) = rows(&outer_x, &outer_y, &inner.test_indices);
            for (&row, prediction) in inner
                .test_indices
                .iter()
                .zip(predict(candidate, &train_x, &train_y, &test_x))
            {
                inner_predictions[row] = prediction;
            }
        }
        // Independent scaled RMSE reference: raw squared residuals would
        // underflow on these targets. Do not compute R² to select a candidate.
        let residuals: Vec<_> = outer_y
            .iter()
            .zip(&inner_predictions)
            .map(|(observed, predicted)| observed - predicted)
            .collect();
        let scale = residuals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let expected_rmse = scale
            * (residuals.iter().map(|v| (v / scale).powi(2)).sum::<f64>() / residuals.len() as f64)
                .sqrt();
        assert!(expected_rmse.is_finite() && expected_rmse > 0.0);
        let fold = &report.outer_folds[outer_index];
        let score = &fold.inner_candidate_scores[0];
        assert_eq!(fold.selected_candidate_index, 0);
        assert_eq!(score.failure, None);
        assert_eq!(score.fit_count, 2);
        assert!((score.pooled_rmse.unwrap() - expected_rmse).abs() <= expected_rmse * 1e-12);
        if regression_metrics(&outer_y, &inner_predictions) == Err(EvaluationError::NumericOverflow)
        {
            saw_unrepresentable_inner_r2 = true;
        }
        let (test_x, _) = rows(&x, &y, &outer.test_indices);
        for (&row, prediction) in outer
            .test_indices
            .iter()
            .zip(predict(candidate, &outer_x, &outer_y, &test_x))
        {
            manual_outer_predictions[row] = prediction;
        }
    }
    assert!(saw_unrepresentable_inner_r2);
    assert_eq!(report.fit_count, 9);
    assert_eq!(report.planned_fit_count, 9);
    assert_eq!(report.out_of_fold_predictions, manual_outer_predictions);
    assert_eq!(
        report.pooled_metrics,
        regression_metrics(&y, &manual_outer_predictions).unwrap()
    );
    assert!(report.pooled_metrics.r2.unwrap().is_finite());
}

#[test]
fn outer_prediction_failure_does_not_silently_switch_to_a_valid_runner_up() {
    let x = vec![
        vec![1e300],
        vec![1e-300],
        vec![1e-300],
        vec![1e-300],
        vec![1e-300],
        vec![1e-300],
    ];
    let y = vec![1.0; x.len()];
    let groups: Vec<_> = (0..x.len()).map(|i| i.to_string()).collect();
    let candidates = [RegressionCandidate::Knn { k: 1 }, RegressionCandidate::Mean];
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15),
        Err(SelectionError::OuterFailure {
            outer_fold_index: 0,
            candidate_index: 0,
            stage: FailureStage::Predict,
            fit_count: 5,
            error: CandidateError::Knn(KnnError::NumericOverflow)
        })
    );
}

#[test]
fn outer_refit_failure_is_fatal_and_counts_the_failed_fit() {
    // Inner training sets contain homogeneous target magnitudes. Their pooled
    // residuals and metrics are representable, but the outer refit's target
    // mean cannot preserve tiny/huge normalization and must fail honestly.
    let x = vec![vec![1.0]; 6];
    let y = vec![0.0, 1e300, 1e-300, 0.0, 1e300, 1e-300];
    let groups: Vec<_> = (0..x.len()).map(|i| i.to_string()).collect();
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups, 3, 2, &[RegressionCandidate::Mean], 9),
        Err(SelectionError::OuterFailure {
            outer_fold_index: 0,
            candidate_index: 0,
            stage: FailureStage::Fit,
            fit_count: 3,
            error: CandidateError::Regression(RegressionError::NumericFailure),
        })
    );
}

#[test]
fn empty_shapes_and_nonfinite_inputs_fail_preflight() {
    let (x, y, groups) = fixture();
    let c = [RegressionCandidate::Mean];
    assert_eq!(
        nested_grouped_regression(&[], &[], &[], 3, 2, &c, 9),
        Err(SelectionError::Empty)
    );
    for broken_x in [vec![vec![]; x.len()], {
        let mut broken = x.clone();
        broken[1].push(0.0);
        broken
    }] {
        assert_eq!(
            nested_grouped_regression(&broken_x, &y, &groups, 3, 2, &c, 9),
            Err(SelectionError::Shape)
        );
    }
    assert_eq!(
        nested_grouped_regression(&x, &y[1..], &groups, 3, 2, &c, 9),
        Err(SelectionError::Shape)
    );
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups[1..], 3, 2, &c, 9),
        Err(SelectionError::Shape)
    );
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad_x = x.clone();
        bad_x[0][0] = invalid;
        assert_eq!(
            nested_grouped_regression(&bad_x, &y, &groups, 3, 2, &c, 9),
            Err(SelectionError::NonFinite)
        );
        let mut bad_y = y.clone();
        bad_y[0] = invalid;
        assert_eq!(
            nested_grouped_regression(&x, &bad_y, &groups, 3, 2, &c, 9),
            Err(SelectionError::NonFinite)
        );
    }
}

#[test]
fn all_candidates_are_validated_before_any_fitting() {
    let (x, y, groups) = fixture();
    for alpha in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let candidates = [
            RegressionCandidate::Mean,
            RegressionCandidate::Ridge { alpha },
        ];
        assert_eq!(
            nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15),
            Err(SelectionError::InvalidCandidate {
                candidate_index: 1,
                error: CandidateError::Regression(RegressionError::InvalidAlpha)
            })
        );
    }
    let smallest = grouped_kfold(&groups, 3)
        .unwrap()
        .iter()
        .flat_map(|outer| {
            let gs: Vec<_> = outer
                .train_indices
                .iter()
                .map(|&i| groups[i].clone())
                .collect();
            grouped_kfold(&gs, 2)
                .unwrap()
                .into_iter()
                .map(|fold| fold.train_indices.len())
        })
        .min()
        .unwrap();
    for k in [0, smallest + 1, x.len() + 1, usize::MAX] {
        let candidates = [RegressionCandidate::Mean, RegressionCandidate::Knn { k }];
        assert_eq!(
            nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15),
            Err(SelectionError::InvalidCandidate {
                candidate_index: 1,
                error: CandidateError::Knn(KnnError::InvalidK)
            })
        );
    }
    assert!(nested_grouped_regression(
        &x,
        &y,
        &groups,
        3,
        2,
        &[RegressionCandidate::Knn { k: smallest }],
        9
    )
    .is_ok());
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups, 3, 2, &[], 0),
        Err(SelectionError::NoCandidates)
    );
}

#[test]
fn split_counts_and_every_outer_training_group_count_are_checked_upfront() {
    let (x, y, groups) = fixture();
    let c = [RegressionCandidate::Mean];
    for (outer, inner) in [(0, 2), (1, 2), (3, 0), (3, 1)] {
        assert_eq!(
            nested_grouped_regression(&x, &y, &groups, outer, inner, &c, 100),
            Err(SelectionError::InvalidSplits)
        );
    }
    assert_eq!(
        nested_grouped_regression(&x, &y, &vec![String::new(); x.len()], 3, 2, &c, 9),
        Err(SelectionError::InsufficientOuterGroups)
    );
    // Fold zero holds one large group and has three groups for inner splitting;
    // fold one holds the other three groups and has just one training group.
    // A first-fold-only preflight would fit models before discovering this.
    let groups: Vec<_> = ["large", "large", "large", "large", "a", "b", "c"]
        .iter()
        .map(|v| v.to_string())
        .collect();
    let x = vec![vec![1.0]; groups.len()];
    let y = vec![1.0; groups.len()];
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups, 2, 2, &c, 6),
        Err(SelectionError::InsufficientInnerGroups {
            outer_fold_index: 1
        })
    );
}

#[test]
fn budget_is_worst_case_even_when_a_candidate_would_fail() {
    let (_, y, groups) = fixture();
    let x = vec![vec![1.0]; y.len()];
    let candidates = [
        RegressionCandidate::Ridge { alpha: 0.0 },
        RegressionCandidate::Mean,
    ];
    for max_fits in [0, 12, 14] {
        assert_eq!(
            nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, max_fits),
            Err(SelectionError::BudgetExceeded {
                required_fits: 15,
                max_fits
            })
        );
    }
    assert_eq!(
        nested_grouped_regression(&x, &y, &groups, 3, 2, &candidates, 15)
            .unwrap()
            .fit_count,
        12
    );
}

#[test]
fn resource_limits_reject_oversized_requests_without_arithmetic_overflow() {
    let (x, y, groups) = fixture();
    for (outer, inner) in [
        (MAX_SELECTION_SPLITS + 1, 2),
        (3, MAX_SELECTION_SPLITS + 1),
        (usize::MAX, 2),
        (3, usize::MAX),
    ] {
        assert_eq!(
            nested_grouped_regression(
                &x,
                &y,
                &groups,
                outer,
                inner,
                &[RegressionCandidate::Mean],
                usize::MAX
            ),
            Err(SelectionError::Capacity)
        );
    }
    assert_eq!(
        nested_grouped_regression(
            &x,
            &y,
            &groups,
            3,
            2,
            &vec![RegressionCandidate::Mean; MAX_SELECTION_CANDIDATES + 1],
            usize::MAX
        ),
        Err(SelectionError::Capacity)
    );
    let wide = vec![vec![1.0; MAX_SELECTION_FEATURES + 1]; x.len()];
    assert_eq!(
        nested_grouped_regression(&wide, &y, &groups, 3, 2, &[RegressionCandidate::Mean], 9),
        Err(SelectionError::Capacity)
    );
    let count = MAX_SELECTION_ROWS + 1;
    assert_eq!(
        nested_grouped_regression(
            &vec![vec![1.0]; count],
            &vec![1.0; count],
            &vec![String::new(); count],
            3,
            2,
            &[RegressionCandidate::Mean],
            9
        ),
        Err(SelectionError::Capacity)
    );
}

#[test]
fn exact_row_and_feature_limits_are_supported() {
    for (count, width) in [(MAX_SELECTION_ROWS, 1), (6, MAX_SELECTION_FEATURES)] {
        let x = vec![vec![1.0; width]; count];
        let y = vec![2.0; count];
        let groups: Vec<_> = (0..count).map(|i| (i % 6).to_string()).collect();
        let report =
            nested_grouped_regression(&x, &y, &groups, 3, 2, &[RegressionCandidate::Mean], 9)
                .unwrap();
        assert_eq!(report.out_of_fold_predictions, y);
        assert_eq!(report.fit_count, 9);
    }
}

#[test]
fn exact_candidate_and_fold_limits_have_a_bounded_fit_count() {
    let count = 20;
    let x = vec![vec![1.0]; count];
    let y = vec![2.0; count];
    let groups: Vec<_> = (0..count).map(|i| i.to_string()).collect();
    let report = nested_grouped_regression(
        &x,
        &y,
        &groups,
        MAX_SELECTION_SPLITS,
        MAX_SELECTION_SPLITS,
        &[RegressionCandidate::Mean; MAX_SELECTION_CANDIDATES],
        1610,
    )
    .unwrap();
    assert_eq!(report.fit_count, 1610);
    assert_eq!(report.planned_fit_count, 1610);
    assert!(report
        .outer_folds
        .iter()
        .all(|fold| fold.selected_candidate_index == 0));
}

#[test]
fn zero_alpha_is_valid_when_folds_have_full_rank() {
    let x: Vec<_> = (0..12).map(|i| vec![i as f64]).collect();
    let y: Vec<_> = (0..12).map(|i| 2.0 * i as f64 - 1.0).collect();
    let groups: Vec<_> = (0..12).map(|i| format!("{i:02}")).collect();
    let report = nested_grouped_regression(
        &x,
        &y,
        &groups,
        3,
        2,
        &[RegressionCandidate::Ridge { alpha: 0.0 }],
        9,
    )
    .unwrap();
    assert_eq!(report.fit_count, 9);
    assert!(report.pooled_metrics.rmse < 1e-12);
}

#[test]
fn failure_text_does_not_echo_feature_target_or_group_values() {
    let error = SelectionError::OuterFailure {
        outer_fold_index: 2,
        candidate_index: 1,
        stage: FailureStage::Predict,
        fit_count: 12,
        error: CandidateError::Evaluation(EvaluationError::NumericOverflow),
    };
    let text = error.to_string();
    assert!(text.contains("outer fold 2"));
    assert!(text.contains("supported floating-point range"));
}
