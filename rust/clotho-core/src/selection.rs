//! Bounded, deterministic nested grouped model selection for numeric regression.
//!
//! Each inner fit sees only its inner-training rows, and the selected estimator
//! is freshly fitted on the outer-training rows. Estimator scaling is learned
//! inside those fits. The selection score is pooled inner out-of-fold RMSE,
//! not the mean of fold RMSEs. Exact ties use the first candidate supplied.
//!
//! Inputs must already be finite numeric measurements with consistent columns.
//! No missing-value imputation or category encoding is supplied. Do not pass
//! globally imputed, globally standardized, feature-selected, or otherwise
//! data-adaptively preprocessed inputs and describe this as fold-safe: this API
//! cannot undo upstream leakage. Group labels must express the study's actual
//! independence boundary. This protocol is not scientific validation or a claim
//! of compatibility with Python model-selection results.

use crate::evaluation::{
    grouped_kfold, regression_metrics, regression_rmse, EvaluationError, RegressionMetrics,
};
use crate::knn::{KnnError, KnnRegressor};
use crate::regression::{MeanRegressor, RegressionError, RidgeRegressor};
use std::fmt;

pub const MAX_SELECTION_ROWS: usize = 20_000;
pub const MAX_SELECTION_FEATURES: usize = 100;
pub const MAX_SELECTION_CANDIDATES: usize = 16;
pub const MAX_SELECTION_SPLITS: usize = 10;

/// Fixed numeric estimators; each fit learns its own scaling from training rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegressionCandidate {
    Mean,
    Ridge { alpha: f64 },
    Knn { k: usize },
}

/// Typed, fixed errors contain no feature, target, or group values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateError {
    Regression(RegressionError),
    Knn(KnnError),
    Evaluation(EvaluationError),
}

impl fmt::Display for CandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regression(error) => error.fmt(f),
            Self::Knn(error) => error.fmt(f),
            Self::Evaluation(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for CandidateError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureStage {
    Fit,
    Predict,
    Score,
}

/// A candidate stops at its first failure; it never receives a partial score.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateFailure {
    /// Zero-based inner fold, or `None` for pooled inner scoring.
    pub inner_fold_index: Option<usize>,
    pub stage: FailureStage,
    pub error: CandidateError,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InnerCandidateScore {
    pub candidate_index: usize,
    /// Present only after all inner folds and pooled scoring succeed.
    pub pooled_rmse: Option<f64>,
    pub failure: Option<CandidateFailure>,
    /// Attempted estimator fits, including the fit that failed, if any.
    pub fit_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OuterRegressionFold {
    /// Original input row positions, in ascending order.
    pub train_indices: Vec<usize>,
    /// Original input row positions, in ascending order.
    pub test_indices: Vec<usize>,
    pub selected_candidate_index: usize,
    /// One entry per supplied candidate, in the supplied order.
    pub inner_candidate_scores: Vec<InnerCandidateScore>,
    /// All attempted inner fits plus the selected candidate's outer refit.
    pub fit_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NestedRegressionReport {
    /// Exactly one held-out prediction per input row, in original row order.
    pub out_of_fold_predictions: Vec<f64>,
    pub outer_folds: Vec<OuterRegressionFold>,
    /// Actual attempted fits; may be below the plan when candidates fail early.
    pub fit_count: usize,
    /// Worst-case fit count checked against the caller's budget before any fit.
    pub planned_fit_count: usize,
    /// Unweighted metrics pooled over all original rows, not averaged by fold.
    pub pooled_metrics: RegressionMetrics,
}

/// Preflight errors perform zero fits. Execution errors retain attempted counts.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectionError {
    Empty,
    Shape,
    NonFinite,
    Capacity,
    InvalidSplits,
    NoCandidates,
    InvalidCandidate {
        candidate_index: usize,
        error: CandidateError,
    },
    InsufficientOuterGroups,
    InsufficientInnerGroups {
        outer_fold_index: usize,
    },
    BudgetExceeded {
        required_fits: usize,
        max_fits: usize,
    },
    NoValidCandidate {
        outer_fold_index: usize,
        fit_count: usize,
        inner_candidate_scores: Vec<InnerCandidateScore>,
    },
    OuterFailure {
        outer_fold_index: usize,
        candidate_index: usize,
        stage: FailureStage,
        fit_count: usize,
        error: CandidateError,
    },
    MetricsFailure {
        fit_count: usize,
        error: EvaluationError,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("Nested regression requires nonempty inputs."),
            Self::Shape => f.write_str("Feature widths must be nonzero and equal; row, target, and group counts must match."),
            Self::NonFinite => f.write_str("Features and targets must be finite; imputation is not supplied."),
            Self::Capacity => f.write_str("Nested regression supports at most 20,000 rows, 100 features, 16 candidates, and 10 folds at each level."),
            Self::InvalidSplits => f.write_str("Both outer and inner cross-validation require at least two folds."),
            Self::NoCandidates => f.write_str("At least one regression candidate is required."),
            Self::InvalidCandidate { candidate_index, error } => write!(f, "Candidate {candidate_index} is invalid before fitting: {error}"),
            Self::InsufficientOuterGroups => f.write_str("Too few distinct groups for the requested outer folds."),
            Self::InsufficientInnerGroups { outer_fold_index } => write!(f, "Outer fold {outer_fold_index} has too few training groups for the requested inner folds."),
            Self::BudgetExceeded { required_fits, max_fits } => write!(f, "The declared study requires a budget of {required_fits} fits; the supplied maximum is {max_fits}."),
            Self::NoValidCandidate { outer_fold_index, .. } => write!(f, "Every candidate failed inner validation in outer fold {outer_fold_index}."),
            Self::OuterFailure { outer_fold_index, candidate_index, stage, error, .. } => write!(f, "Selected candidate {candidate_index} failed during outer fold {outer_fold_index} {stage:?}: {error}"),
            Self::MetricsFailure { error, .. } => write!(f, "Pooled outer metrics failed: {error}"),
        }
    }
}
impl std::error::Error for SelectionError {}

/// Select on grouped inner out-of-fold RMSE and evaluate on untouched outer rows.
///
/// Preflight validates every input, candidate, split, and outer-training group's
/// ability to support the inner splits before fitting. `alpha` must be finite
/// and nonnegative; `k` must be positive and no larger than *every* inner-training
/// row count. The budget must cover `outer_splits * (candidates.len() *
/// inner_splits + 1)`, even when failures could reduce actual fit attempts.
///
/// An inner fit, prediction, or scoring failure invalidates that candidate for
/// that outer fold. Its first error and actual attempted fits are recorded;
/// remaining inner fits for it are skipped. If all candidates fail, the study
/// fails. An outer refit/prediction or final metrics error also fails the study,
/// without silently switching to a different candidate or returning partial
/// out-of-fold results. All reported fold/candidate indices are zero-based.
pub fn nested_grouped_regression(
    x: &[Vec<f64>],
    y: &[f64],
    groups: &[String],
    outer_splits: usize,
    inner_splits: usize,
    candidates: &[RegressionCandidate],
    max_fits: usize,
) -> Result<NestedRegressionReport, SelectionError> {
    validate_inputs(x, y, groups, outer_splits, inner_splits, candidates)?;
    // Validated bounds make this arithmetic small (at most 1,610 fits).
    let planned_fit_count = outer_splits * (candidates.len() * inner_splits + 1);
    if planned_fit_count > max_fits {
        return Err(SelectionError::BudgetExceeded {
            required_fits: planned_fit_count,
            max_fits,
        });
    }
    let outer_plan =
        grouped_kfold(groups, outer_splits).map_err(|_| SelectionError::InsufficientOuterGroups)?;
    let mut inner_plans = Vec::with_capacity(outer_splits);
    let mut smallest_training_count = usize::MAX;
    for (outer_fold_index, outer) in outer_plan.iter().enumerate() {
        let training_groups: Vec<_> = outer
            .train_indices
            .iter()
            .map(|&i| groups[i].clone())
            .collect();
        let folds = grouped_kfold(&training_groups, inner_splits)
            .map_err(|_| SelectionError::InsufficientInnerGroups { outer_fold_index })?;
        for fold in &folds {
            smallest_training_count = smallest_training_count.min(fold.train_indices.len());
        }
        inner_plans.push(folds);
    }
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if let RegressionCandidate::Knn { k } = candidate {
            if *k > smallest_training_count {
                return Err(SelectionError::InvalidCandidate {
                    candidate_index,
                    error: CandidateError::Knn(KnnError::InvalidK),
                });
            }
        }
    }

    let mut out_of_fold_predictions = vec![0.0; x.len()];
    let mut outer_folds = Vec::with_capacity(outer_splits);
    let mut fit_count = 0;
    for (outer_fold_index, (outer, inner_folds)) in
        outer_plan.into_iter().zip(inner_plans).enumerate()
    {
        let (outer_x, outer_y) = training_rows(x, y, &outer.train_indices);
        let start_fit_count = fit_count;
        let mut scores = Vec::with_capacity(candidates.len());
        let mut best: Option<(usize, f64)> = None;
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let mut score = InnerCandidateScore {
                candidate_index,
                pooled_rmse: None,
                failure: None,
                fit_count: 0,
            };
            let mut inner_predictions = vec![0.0; outer_x.len()];
            for (inner_fold_index, inner) in inner_folds.iter().enumerate() {
                let (train_x, train_y) = training_rows(&outer_x, &outer_y, &inner.train_indices);
                let test_x = feature_rows(&outer_x, &inner.test_indices);
                fit_count += 1;
                score.fit_count += 1;
                match fit_predict(*candidate, &train_x, &train_y, &test_x) {
                    Ok(predictions) => {
                        for (&row, prediction) in inner.test_indices.iter().zip(predictions) {
                            inner_predictions[row] = prediction;
                        }
                    }
                    Err((stage, error)) => {
                        score.failure = Some(CandidateFailure {
                            inner_fold_index: Some(inner_fold_index),
                            stage,
                            error,
                        });
                        break;
                    }
                }
            }
            if score.failure.is_none() {
                // Selection requires only RMSE. An unrelated R² overflow on
                // near-constant targets must not invalidate a finite RMSE.
                match regression_rmse(&outer_y, &inner_predictions) {
                    Ok(rmse) => {
                        score.pooled_rmse = Some(rmse);
                        if best.map_or(true, |(_, best_rmse)| rmse < best_rmse) {
                            best = Some((candidate_index, rmse));
                        }
                    }
                    Err(error) => {
                        score.failure = Some(CandidateFailure {
                            inner_fold_index: None,
                            stage: FailureStage::Score,
                            error: CandidateError::Evaluation(error),
                        });
                    }
                }
            }
            scores.push(score);
        }
        let selected_candidate_index = match best {
            Some((index, _)) => index,
            None => {
                return Err(SelectionError::NoValidCandidate {
                    outer_fold_index,
                    fit_count,
                    inner_candidate_scores: scores,
                })
            }
        };
        let test_x = feature_rows(x, &outer.test_indices);
        fit_count += 1;
        let predictions = fit_predict(
            candidates[selected_candidate_index],
            &outer_x,
            &outer_y,
            &test_x,
        )
        .map_err(|(stage, error)| SelectionError::OuterFailure {
            outer_fold_index,
            candidate_index: selected_candidate_index,
            stage,
            fit_count,
            error,
        })?;
        for (&row, prediction) in outer.test_indices.iter().zip(predictions) {
            out_of_fold_predictions[row] = prediction;
        }
        outer_folds.push(OuterRegressionFold {
            train_indices: outer.train_indices,
            test_indices: outer.test_indices,
            selected_candidate_index,
            inner_candidate_scores: scores,
            fit_count: fit_count - start_fit_count,
        });
    }
    let pooled_metrics = regression_metrics(y, &out_of_fold_predictions)
        .map_err(|error| SelectionError::MetricsFailure { fit_count, error })?;
    Ok(NestedRegressionReport {
        out_of_fold_predictions,
        outer_folds,
        fit_count,
        planned_fit_count,
        pooled_metrics,
    })
}

fn validate_inputs(
    x: &[Vec<f64>],
    y: &[f64],
    groups: &[String],
    outer: usize,
    inner: usize,
    candidates: &[RegressionCandidate],
) -> Result<(), SelectionError> {
    if x.is_empty() {
        return Err(SelectionError::Empty);
    }
    let width = x[0].len();
    if width == 0
        || y.len() != x.len()
        || groups.len() != x.len()
        || x.iter().any(|row| row.len() != width)
    {
        return Err(SelectionError::Shape);
    }
    if x.len() > MAX_SELECTION_ROWS
        || width > MAX_SELECTION_FEATURES
        || candidates.len() > MAX_SELECTION_CANDIDATES
        || outer > MAX_SELECTION_SPLITS
        || inner > MAX_SELECTION_SPLITS
    {
        return Err(SelectionError::Capacity);
    }
    if x.iter().flatten().chain(y).any(|value| !value.is_finite()) {
        return Err(SelectionError::NonFinite);
    }
    if outer < 2 || inner < 2 {
        return Err(SelectionError::InvalidSplits);
    }
    if candidates.is_empty() {
        return Err(SelectionError::NoCandidates);
    }
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let error = match candidate {
            RegressionCandidate::Ridge { alpha } if !alpha.is_finite() || *alpha < 0.0 => {
                Some(CandidateError::Regression(RegressionError::InvalidAlpha))
            }
            RegressionCandidate::Knn { k: 0 } => Some(CandidateError::Knn(KnnError::InvalidK)),
            _ => None,
        };
        if let Some(error) = error {
            return Err(SelectionError::InvalidCandidate {
                candidate_index,
                error,
            });
        }
    }
    Ok(())
}

fn feature_rows(x: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    indices.iter().map(|&i| x[i].clone()).collect()
}

fn training_rows(x: &[Vec<f64>], y: &[f64], indices: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        feature_rows(x, indices),
        indices.iter().map(|&i| y[i]).collect(),
    )
}

fn fit_predict(
    candidate: RegressionCandidate,
    x: &[Vec<f64>],
    y: &[f64],
    test: &[Vec<f64>],
) -> Result<Vec<f64>, (FailureStage, CandidateError)> {
    match candidate {
        RegressionCandidate::Mean => MeanRegressor::fit(x, y)
            .map_err(|e| (FailureStage::Fit, CandidateError::Regression(e)))?
            .predict(test)
            .map_err(|e| (FailureStage::Predict, CandidateError::Regression(e))),
        RegressionCandidate::Ridge { alpha } => RidgeRegressor::fit(x, y, alpha)
            .map_err(|e| (FailureStage::Fit, CandidateError::Regression(e)))?
            .predict(test)
            .map_err(|e| (FailureStage::Predict, CandidateError::Regression(e))),
        RegressionCandidate::Knn { k } => KnnRegressor::fit(x, y, k)
            .map_err(|e| (FailureStage::Fit, CandidateError::Knn(e)))?
            .predict(test)
            .map_err(|e| (FailureStage::Predict, CandidateError::Knn(e))),
    }
}
