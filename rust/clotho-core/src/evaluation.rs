//! Dependency-free regression metrics and deterministic, group-disjoint folds.
//!
//! These utilities do not fit or transform data. Fit preprocessing and model
//! parameters using each fold's training rows only, and evaluate its held-out
//! rows. Group labels must describe the leakage boundary appropriate to the
//! study; splitting supplied labels cannot establish scientific independence.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
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

/// Compute only RMSE using the same robust error scaling as full metrics.
///
/// Model selection must not reject a finite RMSE merely because an unrelated
/// R² is outside binary64 range on a near-constant target. Full reporting can
/// still request [`regression_metrics`] separately.
pub fn regression_rmse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, EvaluationError> {
    if y_true.len() != y_pred.len() {
        return Err(EvaluationError::Shape);
    }
    if y_true.is_empty() {
        return Err(EvaluationError::Empty);
    }
    if y_true.iter().chain(y_pred).any(|v| !v.is_finite()) {
        return Err(EvaluationError::NonFinite);
    }
    validated_rmse(y_true, y_pred)
}

fn validated_rmse(y_true: &[f64], y_pred: &[f64]) -> Result<f64, EvaluationError> {
    // The usual path discovers the maximum finite difference in one pass,
    // then reduces normalized errors without a row-sized scratch vector.
    let (scale, factor) = difference_scale(|| y_true.iter().copied().zip(y_pred.iter().copied()))?;
    rmse_with_scale(y_true, y_pred, scale, factor)
}

fn rmse_with_scale(
    y_true: &[f64],
    y_pred: &[f64],
    scale: f64,
    factor: f64,
) -> Result<f64, EvaluationError> {
    let pairs = || y_true.iter().copied().zip(y_pred.iter().copied());
    if scale == 0.0 {
        return Ok(0.0);
    }
    let mut squares = CompensatedSum::default();
    if factor == 1.0 {
        // The scale pass proved each direct subtraction finite. Two distinct
        // finite binary64 values cannot subtract to zero with gradual
        // underflow: their exact difference is at least one subnormal unit.
        // Keep the separate division-underflow check.
        for (a, b) in pairs() {
            let normalized = normalize_error(a - b, scale)?;
            squares.add(normalized * normalized);
        }
    } else {
        for (a, b) in pairs() {
            let normalized = normalize_error(scaled_error(a, b, factor)?, scale)?;
            squares.add(normalized * normalized);
        }
    }
    restore_scale(
        (squares.total() / y_true.len() as f64).sqrt(),
        scale,
        factor,
    )
}

/// Evaluate two arms after their shared finite, nonempty roster is verified.
pub(crate) fn paired_validated_rmse(
    observed: &[f64],
    model: &[f64],
    baseline: &[f64],
) -> Result<(f64, f64), EvaluationError> {
    debug_assert!(!observed.is_empty());
    debug_assert_eq!(observed.len(), model.len());
    debug_assert_eq!(observed.len(), baseline.len());
    let triples = || observed.iter().zip(model).zip(baseline);
    let mut model_scale = 0.0_f64;
    let mut base_scale = 0.0_f64;
    for ((&actual, &predicted), &reference) in triples() {
        model_scale = model_scale.max((actual - predicted).abs());
        base_scale = base_scale.max((actual - reference).abs());
    }
    if !model_scale.is_finite() || !base_scale.is_finite() || model_scale == 0.0 || base_scale == 0.0 {
        let rmse = |arm: &[f64], scale: f64| {
            if scale.is_finite() {
                rmse_with_scale(observed, arm, scale, 1.0)
            } else {
                // Retain the existing half-scale rescue for raw overflow.
                validated_rmse(observed, arm)
            }
        };
        return Ok((rmse(model, model_scale)?, rmse(baseline, base_scale)?));
    }

    // Both arms retain their own error scales and their original row order.
    let mut model_squares = CompensatedSum::default();
    let mut base_squares = CompensatedSum::default();
    for ((&actual, &predicted), &reference) in triples() {
        let model_error = normalize_error(actual - predicted, model_scale)?;
        let base_error = normalize_error(actual - reference, base_scale)?;
        model_squares.add(model_error * model_error);
        base_squares.add(base_error * base_error);
    }
    let count = observed.len() as f64;
    Ok((
        restore_scale((model_squares.total() / count).sqrt(), model_scale, 1.0)?,
        restore_scale((base_squares.total() / count).sqrt(), base_scale, 1.0)?,
    ))
}

/// Relative squared-error improvement against explicit baseline predictions.
///
/// Computes `1 - MSE(model) / MSE(baseline)` on the same observations. Positive
/// scores favor the model, zero denotes equal error, and negative scores are
/// retained. This is not R² unless the baseline is the observed-target mean.
/// Supply prediction-time baselines (such as persistence) without fitting them
/// on held-out outcomes. This function cannot verify their provenance.
///
/// A perfect baseline makes the ratio undefined and returns `None`, including
/// when both forecasts are perfect. All three inputs must be equally sized,
/// nonempty and finite, even when the baseline is perfect. RMSE scaling avoids
/// overflowing raw squared errors; unrepresentable scores return an error.
/// When both errors are nonzero, subnormal RMSE values are rejected: rounding
/// their physical-unit values can materially corrupt the squared-error ratio.
/// Exact perfect-model and perfect-baseline cases retain their usual semantics.
pub fn regression_mse_skill(
    y_true: &[f64],
    y_pred: &[f64],
    baseline: &[f64],
) -> Result<Option<f64>, EvaluationError> {
    if y_true.len() != baseline.len() || y_true.len() != y_pred.len() {
        return Err(EvaluationError::Shape);
    }
    let model_rmse = regression_rmse(y_true, y_pred)?;
    let baseline_rmse = regression_rmse(y_true, baseline)?;
    mse_skill_from_checked_rmse(model_rmse, baseline_rmse)
}

// Both operands must come from successful regression_rmse reductions.
pub(crate) fn mse_skill_from_checked_rmse(
    model_rmse: f64,
    baseline_rmse: f64,
) -> Result<Option<f64>, EvaluationError> {
    debug_assert!(model_rmse.is_finite() && model_rmse >= 0.0);
    debug_assert!(baseline_rmse.is_finite() && baseline_rmse >= 0.0);
    if baseline_rmse == 0.0 {
        return Ok(None);
    }
    if model_rmse == 0.0 {
        return Ok(Some(1.0));
    }
    // Ratios of subnormal RMSEs can be materially wrong after quantization:
    // [min, 0] and [min, min] both round to RMSE=min although SSE differs 2x.
    // Keep the scalar-report formula unchanged in its supported normal range.
    if model_rmse.is_subnormal() || baseline_rmse.is_subnormal() {
        return Err(EvaluationError::NumericOverflow);
    }
    let ratio = model_rmse / baseline_rmse;
    let score = 1.0 - ratio * ratio;
    if !score.is_finite() {
        return Err(EvaluationError::NumericOverflow);
    }
    Ok(Some(score))
}

/// Calculate finite, unweighted MAE, RMSE, and R² without squaring raw values.
///
/// Inputs must be nonempty, equally sized, and finite. Scaled, compensated sums
/// avoid overflow for large finite errors and use constant additional storage.
/// Targets are centered relative to an
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

    let pairs = || y_true.iter().copied().zip(y_pred.iter().copied());
    let (error_scale, error_factor) = difference_scale(pairs)?;
    if error_scale == 0.0 {
        let varied = y_true.iter().any(|&value| value != y_true[0]);
        return Ok(RegressionMetrics { mae: 0.0, rmse: 0.0, r2: varied.then_some(1.0) });
    }
    let count = y_true.len() as f64;
    let (abs_sum, square_sum) = if error_factor == 1.0 {
        residual_moments(pairs().map(|(a, b)| normalize_error(a - b, error_scale)))?
    } else {
        residual_moments(pairs().map(|(a, b)| normalize_error(scaled_error(a, b, error_factor)?, error_scale)))?
    };
    let mae = restore_scale(abs_sum / count, error_scale, error_factor)?;
    let rmse = restore_scale((square_sum / count).sqrt(), error_scale, error_factor)?;

    let r2 = if y_true.iter().all(|value| *value == y_true[0]) {
        None
    } else {
        let origin = y_true[0];
        let target_pairs = || y_true.iter().copied().map(|value| (value, origin));
        let (target_scale, target_factor) = difference_scale(target_pairs)?;
        if target_scale == 0.0 {
            return Err(EvaluationError::NumericOverflow);
        }
        let target_square_sum = if target_factor == 1.0 {
            target_variation(|| y_true.iter().map(|&value| normalize_error(value - origin, target_scale)), count)?
        } else {
            target_variation(|| target_pairs().map(|(a, b)| normalize_error(scaled_error(a, b, target_factor)?, target_scale)), count)?
        };
        if target_square_sum <= 0.0 {
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

// When a scale pass proves all raw differences are finite, recomputing
// them is sufficient; distinct finite values cannot subtract to zero under
// gradual underflow. Half-scale recovery and normalization remain checked.
fn residual_moments(
    normalized: impl Iterator<Item = Result<f64, EvaluationError>>,
) -> Result<(f64, f64), EvaluationError> {
    let mut abs_sum = CompensatedSum::default();
    let mut square_sum = CompensatedSum::default();
    for value in normalized {
        let value = value?;
        abs_sum.add(value.abs());
        square_sum.add(value * value);
    }
    Ok((abs_sum.total(), square_sum.total()))
}

fn target_variation<I: Iterator<Item = Result<f64, EvaluationError>>>(
    normalized: impl Fn() -> I,
    count: f64,
) -> Result<f64, EvaluationError> {
    let mut sum = CompensatedSum::default();
    for value in normalized() {
        sum.add(value?);
    }
    let mean = sum.total() / count;
    let mut squares = CompensatedSum::default();
    for value in normalized() {
        let deviation = value? - mean;
        squares.add(deviation * deviation);
    }
    Ok(squares.total())
}

// Halve only when necessary to handle MAX - (-MAX). On the usual direct
// path, retaining the original subtraction preserves nearby large differences.
fn difference_scale<I: Iterator<Item = (f64, f64)>>(
    pairs: impl Fn() -> I,
) -> Result<(f64, f64), EvaluationError> {
    let direct = pairs().map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    if direct.is_finite() {
        return Ok((direct, 1.0));
    }
    let mut scale = 0.0_f64;
    for (a, b) in pairs() {
        scale = scale.max(scaled_error(a, b, 0.5)?.abs());
    }
    Ok((scale, 0.5))
}

fn scaled_error(a: f64, b: f64, factor: f64) -> Result<f64, EvaluationError> {
    let direct = a - b;
    let difference = if direct.is_finite() {
        direct * factor
    } else {
        a * factor - b * factor
    };
    if !difference.is_finite() || (a != b && difference == 0.0) {
        return Err(EvaluationError::NumericOverflow);
    }
    Ok(difference)
}

fn normalize_error(value: f64, scale: f64) -> Result<f64, EvaluationError> {
    let normalized = value / scale;
    if value != 0.0 && normalized == 0.0 {
        return Err(EvaluationError::NumericOverflow);
    }
    Ok(normalized)
}

fn restore_scale(value: f64, scale: f64, factor: f64) -> Result<f64, EvaluationError> {
    let result = (value * scale) / factor;
    if !result.is_finite() || (value > 0.0 && scale > 0.0 && result == 0.0) {
        return Err(EvaluationError::NumericOverflow);
    }
    Ok(result)
}

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
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
    grouped_kfold_labels(groups.iter().map(String::as_str), n_splits)
}

// The iterator's enumeration defines the local row index. Nested splits
// can traverse an outer training roster without cloning its group strings.
pub(crate) fn grouped_kfold_labels<'a>(
    mut groups: impl ExactSizeIterator<Item = &'a str> + Clone,
    n_splits: usize,
) -> Result<Vec<Fold>, EvaluationError> {
    let row_count = groups.len();
    if row_count == 0 {
        return Err(EvaluationError::Empty);
    }
    if n_splits < 2 {
        return Err(EvaluationError::InvalidSplits);
    }
    if row_count > MAX_GROUPED_ROWS || n_splits > MAX_GROUPED_SPLITS {
        return Err(EvaluationError::Capacity);
    }
    // Keep one flat row-to-group table rather than allocating a membership
    // vector for every group. Map iteration order never determines a fold.
    let mut group_ids = HashMap::with_capacity(row_count.min(16));
    // Both ordinals and eventual fold indices fit under the configured caps.
    const _: () = assert!(MAX_GROUPED_ROWS <= u16::MAX as usize);
    let mut assignment = Vec::with_capacity(row_count);
    // Recognize runs from a short prefix without a full second pass.
    // Uniformly interleaved labels still use ordinary row-wise hashing.
    let use_runs = if row_count >= 128 {
        let mut probe = groups.clone().take(64);
        let mut previous = probe.next();
        let mut repeats = 0;
        for label in probe {
            if previous == Some(label) {
                repeats += 1;
                if repeats == 4 {
                    break;
                }
            }
            previous = Some(label);
        }
        repeats == 4
    } else {
        false
    };
    if use_runs {
        let mut current_label = groups.next().ok_or(EvaluationError::Empty)?;
        let mut run_length = 1_u16;
        for label in groups {
            if label == current_label {
                run_length += 1;
            } else {
                add_group_run(&mut group_ids, &mut assignment, current_label, run_length);
                current_label = label;
                run_length = 1;
            }
        }
        add_group_run(&mut group_ids, &mut assignment, current_label, run_length);
    } else {
        for label in groups {
            let next_id = group_ids.len() as u16;
            let (id, count) = group_ids.entry(label).or_insert((next_id, 0_u16));
            *count += 1;
            assignment.push(*id);
        }
    }
    if n_splits > group_ids.len() {
        return Err(EvaluationError::InsufficientGroups);
    }
    let mut ordered: Vec<_> = group_ids
        .into_iter()
        .map(|(label, (id, count))| (label, count, id))
        .collect();
    ordered.sort_unstable_by(|(label_a, count_a, _), (label_b, count_b, _)| {
        count_b.cmp(count_a).then(label_a.cmp(label_b))
    });

    let mut loads = vec![0; n_splits];
    let mut group_folds = vec![0; ordered.len()];
    if n_splits > 8 && ordered[0].1 == ordered[ordered.len() - 1].1 {
        // All increments are equal, so the lowest-index lightest fold
        // cycles through 0..n_splits for each lexical group in order.
        let count = ordered[0].1 as usize;
        let mut fold = 0;
        for (_, _, id) in ordered {
            group_folds[id as usize] = fold as u16;
            loads[fold] += count;
            fold += 1;
            if fold == n_splits {
                fold = 0;
            }
        }
    } else if n_splits <= 8 {
        // A short direct scan avoids heap overhead on ordinary small folds.
        for (_, count, id) in ordered {
            let mut fold = 0;
            for candidate in 1..n_splits {
                if loads[candidate] < loads[fold] {
                    fold = candidate;
                }
            }
            loads[fold] += count as usize;
            group_folds[id as usize] = fold as u16;
        }
    } else {
        // The minimum pair picks the lightest fold and lowest index on ties.
        let mut available: BinaryHeap<_> = (0..n_splits).map(|fold| Reverse((0, fold))).collect();
        for (_, count, id) in ordered {
            let mut lightest = available.peek_mut().unwrap();
            let Reverse((load, fold)) = *lightest;
            *lightest = Reverse((load + count as usize, fold));
            group_folds[id as usize] = fold as u16;
        }
        for Reverse((load, fold)) in available {
            loads[fold] = load;
        }
    }
    let mut folds: Vec<_> = loads
        .into_iter()
        .map(|test_size| Fold {
            train_indices: Vec::with_capacity(row_count - test_size),
            test_indices: Vec::with_capacity(test_size),
        })
        .collect();
    // Each row contributes once to its test bucket in input order.
    for (row, id) in assignment.into_iter().enumerate() {
        let fold = group_folds[id as usize] as usize;
        folds[fold].test_indices.push(row);
    }
    // With exactly two folds, each training complement is the other test set.
    if let [left, right] = folds.as_mut_slice() {
        left.train_indices.extend_from_slice(&right.test_indices);
        right.train_indices.extend_from_slice(&left.test_indices);
        return Ok(folds);
    }
    // Sorted held indices split the training complement into contiguous runs.
    for fold in &mut folds {
        let mut start = 0;
        for &held in &fold.test_indices {
            fold.train_indices.extend(start..held);
            start = held + 1;
        }
        fold.train_indices.extend(start..row_count);
    }
    Ok(folds)
}

fn add_group_run<'a>(
    groups: &mut HashMap<&'a str, (u16, u16)>,
    assignment: &mut Vec<u16>,
    label: &'a str,
    run_length: u16,
) {
    let next_id = groups.len() as u16;
    let (id, count) = groups.entry(label).or_insert((next_id, 0));
    *count += run_length;
    if run_length == 1 {
        assignment.push(*id);
    } else {
        assignment.resize(assignment.len() + usize::from(run_length), *id);
    }
}

#[cfg(test)]
mod borrowed_fold_tests {
    use super::*;

    #[test]
    fn full_regression_moments_preserve_perfect_and_extreme_cases() {
        let perfect = regression_metrics(&[-0.0, 0.0], &[0.0, -0.0]).unwrap();
        assert_eq!([perfect.mae, perfect.rmse].map(f64::to_bits), [0, 0]);
        assert_eq!(perfect.r2, None);
        let perfect_varied = regression_metrics(&[f64::MAX, -f64::MAX], &[f64::MAX, -f64::MAX]).unwrap();
        assert_eq!(perfect_varied.r2, Some(1.0));
        let overflow_rescued = regression_metrics(
            &[f64::MAX, -f64::MAX, f64::MAX, -f64::MAX],
            &[-f64::MAX, -f64::MAX, f64::MAX, -f64::MAX],
        ).unwrap();
        assert_eq!(overflow_rescued.mae.to_bits(), (f64::MAX * 0.5).to_bits());
        assert_eq!(overflow_rescued.rmse.to_bits(), f64::MAX.to_bits());
        assert_eq!(overflow_rescued.r2, Some(0.0));
        assert_eq!(
            regression_metrics(&[0.0, 0.0], &[f64::MAX, f64::from_bits(1)]),
            Err(EvaluationError::NumericOverflow)
        );
    }

    #[test]
    fn indexed_labels_preserve_local_row_positions_and_group_identity() {
        let labels = ["", "ignored", "A", "a", "", "ignored", "a", "A"].map(str::to_owned);
        let roster = [7, 0, 3, 2, 4, 6];
        let split = grouped_kfold_labels(roster.iter().map(|&i| labels[i].as_str()), 3).unwrap();
        assert_eq!(split[0].test_indices, [1, 4]); // Empty label, lexical first.
        assert_eq!(split[1].test_indices, [0, 3]); // Uppercase group.
        assert_eq!(split[2].test_indices, [2, 5]); // Lowercase group.
        assert_eq!(split[0].train_indices, [0, 2, 3, 5]);
    }
}
