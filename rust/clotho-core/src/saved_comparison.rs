//! Fit-free comparison of declared, ordered in-memory regression cohorts.
//!
//! Labels are caller declarations, not authenticated query bytes or physical
//! identity. This module does not load models, files, or fit any parameters.
use crate::evaluation::{mse_skill_from_checked_rmse, regression_rmse, EvaluationError};
use std::collections::BTreeSet;
use std::fmt;

pub const MAX_SAVED_ROWS: usize = 20_000;
pub const MAX_DECLARATION_BYTES: usize = 256;

/// Literal semantics shared by observations, model, and baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CohortContract<'a> {
    pub cohort: &'a str,
    pub target: &'a str,
    pub unit: &'a str,
    pub horizon: &'a str,
}

/// Values with a complete, ordered roster of opaque row keys.
#[derive(Clone, Copy, Debug)]
pub struct SavedRegressionSeries<'a> {
    pub contract: CohortContract<'a>,
    pub row_ids: &'a [&'a str],
    pub values: &'a [f64],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SavedComparisonError {
    Capacity,
    Declaration,
    ContractMismatch,
    DuplicateRow,
    RowOrderMismatch,
    Numeric(EvaluationError),
}

impl fmt::Display for SavedComparisonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity => f.write_str("Saved comparison exceeds the row limit."),
            Self::Declaration => f.write_str(
                "Declarations and row keys must be bounded, nonblank text without control characters.",
            ),
            Self::ContractMismatch => f.write_str(
                "Cohort, target, unit and horizon declarations must match exactly.",
            ),
            Self::DuplicateRow => f.write_str("Row keys must be unique within every series."),
            Self::RowOrderMismatch => f.write_str("Complete row rosters must match in the same order."),
            Self::Numeric(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for SavedComparisonError {}
impl From<EvaluationError> for SavedComparisonError {
    fn from(e: EvaluationError) -> Self {
        Self::Numeric(e)
    }
}

/// Descriptive equal-row scores only, not an independence or causal inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavedRegressionComparison<'a> {
    pub contract: CohortContract<'a>,
    pub n_rows: usize,
    pub model_rmse: f64,
    pub baseline_rmse: f64,
    /// Positive favors model; zero ties; negative favors baseline; a perfect
    /// baseline is undefined (`None`), even when both predictions are perfect.
    pub mse_skill: Option<f64>,
}

fn valid_text(s: &str) -> bool {
    !s.trim().is_empty() && s.len() <= MAX_DECLARATION_BYTES && !s.chars().any(char::is_control)
}

fn validate(
    series: SavedRegressionSeries<'_>,
    checked_roster: Option<&[&str]>,
) -> Result<bool, SavedComparisonError> {
    let n = series.values.len();
    if n > MAX_SAVED_ROWS || series.row_ids.len() > MAX_SAVED_ROWS {
        return Err(SavedComparisonError::Capacity);
    }
    if n != series.row_ids.len() {
        return Err(EvaluationError::Shape.into());
    }
    if n == 0 {
        return Err(EvaluationError::Empty.into());
    }
    let c = series.contract;
    if [c.cohort, c.target, c.unit, c.horizon]
        .iter()
        .any(|s| !valid_text(s))
    {
        return Err(SavedComparisonError::Declaration);
    }
    // Literal equality transfers the row-text and uniqueness checks from
    // the already validated observed roster. A mismatching roster is still
    // fully checked so its existing diagnostics take precedence.
    let same_roster = checked_roster == Some(series.row_ids);
    if !same_roster {
        let mut seen = BTreeSet::new();
        for id in series.row_ids {
            if !valid_text(id) {
                return Err(SavedComparisonError::Declaration);
            }
            if !seen.insert(*id) {
                return Err(SavedComparisonError::DuplicateRow);
            }
        }
    }
    if series.values.iter().any(|v| !v.is_finite()) {
        return Err(EvaluationError::NonFinite.into());
    }
    Ok(same_roster)
}

/// Compare existing predictions, rejecting the entire input on any mismatch.
///
/// All three contracts and ordered row rosters must agree literally; no join,
/// sorting, trimming, unit conversion or dropped rows occurs. The caller must
/// independently establish completeness, meaning, prediction-time availability
/// and provenance. Swapped values with unchanged false IDs cannot be detected.
/// This in-memory API does not implement Python saved-file byte verification.
pub fn compare_saved_regression<'a>(
    observed: SavedRegressionSeries<'a>,
    model: SavedRegressionSeries<'a>,
    baseline: SavedRegressionSeries<'a>,
) -> Result<SavedRegressionComparison<'a>, SavedComparisonError> {
    validate(observed, None)?;
    let model_roster_matches = validate(model, Some(observed.row_ids))?;
    let baseline_roster_matches = validate(baseline, Some(observed.row_ids))?;
    if observed.contract != model.contract || observed.contract != baseline.contract {
        return Err(SavedComparisonError::ContractMismatch);
    }
    if !model_roster_matches || !baseline_roster_matches {
        return Err(SavedComparisonError::RowOrderMismatch);
    }
    let model_rmse = regression_rmse(observed.values, model.values)?;
    let baseline_rmse = regression_rmse(observed.values, baseline.values)?;
    let mse_skill = mse_skill_from_checked_rmse(model_rmse, baseline_rmse)?;
    Ok(SavedRegressionComparison {
        contract: observed.contract,
        n_rows: observed.values.len(),
        model_rmse,
        baseline_rmse,
        mse_skill,
    })
}
