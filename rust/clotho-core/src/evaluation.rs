//! Dependency-free regression metrics and deterministic, group-disjoint folds.
//!
//! These utilities do not fit or transform data. Fit preprocessing and model
//! parameters using each fold's training rows only, and evaluate its held-out
//! rows. Group labels must describe the leakage boundary appropriate to the
//! study; splitting supplied labels cannot establish scientific independence.

use std::collections::BTreeMap;
use std::fmt;

/// Maximum input rows for materialized grouped folds.
pub const MAX_GROUPED_ROWS: usize = 20_000;

/// Maximum materialized folds. Together with [`MAX_GROUPED_ROWS`], this bounds
/// the returned training and test indices to 2,000,000 entries in total.
pub const MAX_GROUPED_SPLITS: usize = 100;

/// Invalid evaluation inputs or a result outside the supported numeric range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    Empty,
    Shape,
    NonFinite,
    InvalidSplits,
    InsufficientGroups,
    Capacity,
    NumericOverflow,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "Evaluation inputs must not be empty.",
            Self::Shape => "Observed and predicted target lengths must match.",
            Self::NonFinite => "Observed and predicted targets must be finite.",
            Self::InvalidSplits => "Grouped cross-validation requires at least two folds.",
            Self::InsufficientGroups => "Each fold requires at least one distinct group.",
            Self::Capacity => "Grouped folds support at most 20,000 rows and 100 folds.",
            Self::NumericOverflow => "Evaluation exceeds the supported floating-point range.",
        })
    }
}

impl std::error::Error for EvaluationError {}

/// Unweighted metrics over all supplied rows, in the target's original units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionMetrics {
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// `1 - sum((observed - predicted)^2) / sum((observed - mean)^2)`.
    ///
    /// `None` for an exactly constant observed target, including a single row,
    /// even when predictions are perfect. No artificial zero/one substitution
    /// is made. Negative values for worse-than-mean predictions are retained.
    pub r2: Option<f64>,
}

/// Calculate finite, unweighted MAE, RMSE, and R² without squaring raw values.
///
/// Inputs must be nonempty, equally sized, and finite. Scaled, compensated sums
/// avoid overflow for large finite errors. Targets are centered relative to an
/// observed value before scaling to preserve small variation on a large offset.
/// Unrepresentable outputs or loss of nonzero differences during normalization
/// return [`EvaluationError::NumericOverflow`] instead of misleading metrics.
pub fn regression_metrics(
    y_true: &[f64],
    y_pred: &[f64],
) -> Result<RegressionMetrics, EvaluationError> {
    if y_true.len() != y_pred.len() {
        return Err(EvaluationError::Shape);
    }
    if y_true.is_empty() {
        return Err(EvaluationError::Empty);
    }
    if y_true.iter().chain(y_pred).any(|value| !value.is_finite()) {
        return Err(EvaluationError::NonFinite);
    }

    let (errors, error_scale, error_factor) = differences(y_true, y_pred)?;
    let count = y_true.len() as f64;
    let abs_sum = compensated_sum(errors.iter().map(|value| value.abs()));
    let square_sum = compensated_sum(errors.iter().map(|value| value * value));
    let mae = restore_scale(abs_sum / count, error_scale, error_factor)?;
    let rmse = restore_scale((square_sum / count).sqrt(), error_scale, error_factor)?;

    let r2 = if y_true.iter().all(|value| *value == y_true[0]) {
        None
    } else if error_scale == 0.0 {
        Some(1.0)
    } else {
        let origin = vec![y_true[0]; y_true.len()];
        let (shifted, target_scale, target_factor) = differences(y_true, &origin)?;
        let mean = compensated_sum(shifted.iter().copied()) / count;
        let target_square_sum = compensated_sum(shifted.iter().map(|value| {
            let deviation = value - mean;
            deviation * deviation
        }));
        if target_scale == 0.0 || target_square_sum <= 0.0 {
            return Err(EvaluationError::NumericOverflow);
        }
        // The factors are one or one half. Taking the square root before
        // applying physical scales avoids overflow in either raw sum of squares.
        let error_to_variation = (error_scale / target_scale)
            * (target_factor / error_factor)
            * (square_sum / target_square_sum).sqrt();
        let score = 1.0 - error_to_variation * error_to_variation;
        if !score.is_finite() {
            return Err(EvaluationError::NumericOverflow);
        }
        Some(score)
    };

    Ok(RegressionMetrics { mae, rmse, r2 })
}

// Return normalized differences, their scale, and a factor applied before
// normalization. Halving only when necessary handles MAX - (-MAX), while direct
// subtraction preserves small differences between large, nearby target values.
fn differences(a: &[f64], b: &[f64]) -> Result<(Vec<f64>, f64, f64), EvaluationError> {
    let factor = if a.iter().zip(b).any(|(a, b)| !(a - b).is_finite()) {
        0.5
    } else {
        1.0
    };
    let mut values = Vec::with_capacity(a.len());
    let mut scale = 0.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        let direct = a - b;
        let difference = if direct.is_finite() {
            direct * factor
        } else {
            a * factor - b * factor
        };
        if !difference.is_finite() || (a != b && difference == 0.0) {
            return Err(EvaluationError::NumericOverflow);
        }
        scale = scale.max(difference.abs());
        values.push(difference);
    }
    if scale > 0.0 {
        for value in &mut values {
            let normalized = *value / scale;
            if *value != 0.0 && normalized == 0.0 {
                return Err(EvaluationError::NumericOverflow);
            }
            *value = normalized;
        }
    }
    Ok((values, scale, factor))
}

fn restore_scale(value: f64, scale: f64, factor: f64) -> Result<f64, EvaluationError> {
    let result = (value * scale) / factor;
    if !result.is_finite() || (value > 0.0 && scale > 0.0 && result == 0.0) {
        return Err(EvaluationError::NumericOverflow);
    }
    Ok(result)
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

/// Row indices for one group-disjoint cross-validation partition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fold {
    /// All rows outside this fold's held-out groups, in original row order.
    pub train_indices: Vec<usize>,
    /// All rows belonging to this fold's held-out groups, in original row order.
    pub test_indices: Vec<usize>,
}

/// Build deterministic, approximately row-balanced folds without group leakage.
///
/// Groups are assigned largest first to the currently least-loaded fold. Ties
/// use the lexical group label and then the lowest fold index. This greedy
/// heuristic does not promise equal sizes or an optimal partition; an indivisible
/// large group can dominate one fold. Labels are exact, case-sensitive strings;
/// an empty string is a valid label. Row reordering preserves group assignment.
///
/// Every row occurs in exactly one test fold and in all other training folds.
/// Requires nonempty input and `2 <= n_splits <= number_of_distinct_groups`.
/// At most [`MAX_GROUPED_ROWS`] rows and [`MAX_GROUPED_SPLITS`] folds are
/// accepted to bound the `rows * folds` index storage. This is a Rust grouping
/// protocol, not a reproduction of Python's random or encounter-order splits.
pub fn grouped_kfold(groups: &[String], n_splits: usize) -> Result<Vec<Fold>, EvaluationError> {
    if groups.is_empty() {
        return Err(EvaluationError::Empty);
    }
    if n_splits < 2 {
        return Err(EvaluationError::InvalidSplits);
    }
    if groups.len() > MAX_GROUPED_ROWS || n_splits > MAX_GROUPED_SPLITS {
        return Err(EvaluationError::Capacity);
    }
    let mut members: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, group) in groups.iter().enumerate() {
        members.entry(group.as_str()).or_default().push(index);
    }
    if n_splits > members.len() {
        return Err(EvaluationError::InsufficientGroups);
    }
    let mut ordered: Vec<_> = members.into_iter().collect();
    ordered.sort_unstable_by(|(label_a, rows_a), (label_b, rows_b)| {
        rows_b.len().cmp(&rows_a.len()).then(label_a.cmp(label_b))
    });
    let mut loads = vec![0_usize; n_splits];
    let mut assignment = vec![0_usize; groups.len()];
    for (_, rows) in ordered {
        // n_splits >= 2, and iteration order resolves load ties deterministically.
        let mut fold = 0;
        for candidate in 1..n_splits {
            if loads[candidate] < loads[fold] {
                fold = candidate;
            }
        }
        loads[fold] += rows.len();
        for row in rows {
            assignment[row] = fold;
        }
    }
    Ok(loads
        .into_iter()
        .enumerate()
        .map(|(fold, test_size)| {
            let mut train_indices = Vec::with_capacity(groups.len() - test_size);
            let mut test_indices = Vec::with_capacity(test_size);
            for (row, &assigned) in assignment.iter().enumerate() {
                if assigned == fold {
                    test_indices.push(row);
                } else {
                    train_indices.push(row);
                }
            }
            Fold {
                train_indices,
                test_indices,
            }
        })
        .collect())
}
