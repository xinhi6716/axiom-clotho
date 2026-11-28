//! Dependency-free binary classifiers for finite numeric tables.
//!
//! Logistic regression minimizes mean binary log loss plus
//! `alpha / 2 * sum(beta[j]^2)`. Features are centered and scaled by their
//! training population standard deviations; `beta` refers to those scaled
//! features. The intercept is unpenalized. Labels must be exactly zero or one.
//! These estimators do not perform model selection or scientific validation.

use std::fmt;

const MAX_ROWS: usize = 20_000;
const MAX_FEATURES: usize = 100;
const MAX_ITERATIONS: usize = 100;
const MAX_BACKTRACKS: usize = 50;
const GRADIENT_TOLERANCE: f64 = 1e-9;
const STEP_TOLERANCE: f64 = 1e-8;

struct ObjectiveDerivatives {
    value: f64,
    gradient: Vec<f64>,
    curvature: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationError {
    Empty,
    Shape,
    NonFinite,
    InvalidLabel,
    SingleClass,
    InvalidAlpha,
    Capacity,
    NumericFailure,
    ConvergenceFailure,
}

impl fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "Training and prediction tables must not be empty.",
            Self::Shape => "Feature widths and label lengths must match and be nonzero.",
            Self::NonFinite => "Features and labels must be finite.",
            Self::InvalidLabel => "Binary classification labels must be exactly 0 or 1.",
            Self::SingleClass => "Logistic training requires both binary classes.",
            Self::InvalidAlpha => "Logistic alpha must be finite and nonnegative.",
            Self::Capacity => "At most 20,000 rows and 100 features are supported.",
            Self::NumericFailure => "The data or result exceeds the supported floating-point range.",
            Self::ConvergenceFailure => "Logistic optimization did not converge; use stronger regularization or remove redundant features.",
        })
    }
}

impl std::error::Error for ClassificationError {}

/// Feature-independent baseline returning the training fraction of class one.
///
/// Thresholding at 0.5 predicts the training majority class (ties choose one).
/// The returned probability is the empirical class frequency, not a hard vote.
/// Single-class training sets are supported.
#[derive(Clone, Debug)]
pub struct MajorityClassifier {
    width: usize,
    probability: f64,
}

impl MajorityClassifier {
    pub fn fit(x: &[Vec<f64>], y: &[f64]) -> Result<Self, ClassificationError> {
        let width = validate_training(x, y)?;
        let positive_count = y.iter().filter(|&&label| label == 1.0).count();
        Ok(Self {
            width,
            probability: positive_count as f64 / y.len() as f64,
        })
    }

    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, ClassificationError> {
        validate_features(x, self.width)?;
        Ok(vec![self.probability; x.len()])
    }
}

/// Binary logistic regression with training-only numeric standardization.
///
/// Uses at most 100 damped Newton iterations, with at most 50 line-search
/// backtracks each. A fit succeeds only when both the gradient and Newton step
/// satisfy convergence tolerances. Singular curvature, failed line search, and
/// exhausted iterations return `ConvergenceFailure`; no fallback penalty or
/// best-effort unconverged model is substituted. Unregularized separable data
/// have no finite optimum and may fail to converge. Positive alpha is advised.
///
/// Constant training columns are ignored, including when predicting. Fit and
/// predict support 1..=20,000 rows and 1..=100 features. Prediction uses the same
/// feature order and units as training. No missing-value imputation is supplied.
#[derive(Clone, Debug)]
pub struct LogisticClassifier {
    width: usize,
    // Only nonconstant columns are included in the optimization.
    columns: Vec<usize>,
    scalers: Vec<FeatureScaler>,
    coefficients: Vec<f64>,
    intercept: f64,
}

impl LogisticClassifier {
    pub fn fit(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Result<Self, ClassificationError> {
        let width = validate_training(x, y)?;
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(ClassificationError::InvalidAlpha);
        }
        let positives = y.iter().filter(|&&label| label == 1.0).count();
        if positives == 0 || positives == y.len() {
            return Err(ClassificationError::SingleClass);
        }
        let columns = varying_columns(x, width);
        let initial_intercept = (positives as f64 / (y.len() - positives) as f64).ln();
        if columns.is_empty()
            && intercept_converges_without_a_design(y, initial_intercept).unwrap_or(false)
        {
            return Ok(Self {
                width,
                columns,
                scalers: Vec::new(),
                intercept: initial_intercept,
                coefficients: Vec::new(),
            });
        }
        // The design also stages checked centered deviations, avoiding
        // a per-column delta allocation and a second raw-difference pass.
        let design_width = columns.len() + 1;
        let mut design = vec![0.0; x.len() * design_width];
        let mut scalers = Vec::with_capacity(columns.len());
        for (index, &column) in columns.iter().enumerate() {
            scalers.push(FeatureScaler::fit_into(x, column, &mut design, design_width, index + 1)?);
        }
        for row in design.chunks_exact_mut(design_width) {
            row[0] = 1.0;
        }
        let mut parameters = vec![0.0; columns.len() + 1];
        parameters[0] = initial_intercept;
        let parameters = optimize(design.chunks_exact(design_width), y, alpha, parameters, MAX_ITERATIONS)?;
        Ok(Self {
            width,
            columns,
            scalers,
            intercept: parameters[0],
            coefficients: parameters[1..].to_vec(),
        })
    }

    /// Returns `P(y = 1)` in input-row order. Finite logits use a stable sigmoid.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, ClassificationError> {
        validate_features(x, self.width)?;
        if x.len() > 1 && self.columns.is_empty() {
            return repeated_intercept_probability(self.intercept, x.len());
        }
        x.iter()
            .map(|row| {
                let mut score = CompensatedSum::default();
                score.add(self.intercept)?;
                for ((&column, scaler), &coefficient) in self
                    .columns
                    .iter()
                    .zip(&self.scalers)
                    .zip(&self.coefficients)
                {
                    score.add(finite(scaler.transform(row[column])? * coefficient)?)?;
                }
                Ok(sigmoid(score.total()?))
            })
            .collect()
    }
}

// Keep this path separate from the usual nonconstant score loop. The
// compensated score and stable sigmoid are identical to that loop.
#[inline(never)]
fn repeated_intercept_probability(intercept: f64, count: usize) -> Result<Vec<f64>, ClassificationError> {
    let mut score = CompensatedSum::default();
    score.add(intercept)?;
    Ok(vec![sigmoid(score.total()?); count])
}

fn varying_columns(x: &[Vec<f64>], width: usize) -> Vec<usize> {
    let first = &x[0];
    // A row-major check avoids a full strided traversal per feature on
    // wide constant designs. Sample three rows first, so an ordinary
    // varying design doesn't repeat a long search.
    let mut remaining = &x[1..];
    if width >= 16 && x.len() >= 128
        && x[1] == *first && x[x.len() / 2] == *first && x[x.len() - 1] == *first
    {
        match x[2..x.len() - 1].iter().position(|row| row != first) {
            None => return Vec::new(),
            // Every previous row was already compared on all columns.
            Some(index) => remaining = &x[index + 2..],
        }
    }
    let mut columns = Vec::with_capacity(width);
    for column in 0..width {
        if !remaining.iter().all(|row| row[column] == first[column]) {
            columns.push(column);
        }
    }
    columns
}

// Test exactly the first optimizer iteration for a fixed intercept-only design.
// If it would not converge, the caller can still run the ordinary optimizer.
#[cold]
fn intercept_converges_without_a_design(
    y: &[f64],
    initial: f64,
) -> Result<bool, ClassificationError> {
    let score = dot(&[1.0], &[initial])?;
    let exponential = (-score.abs()).exp();
    let denominator = 1.0 + exponential;
    let weight = exponential / denominator.powi(2);
    let negative_label_residual = sigmoid_with_exp(score, exponential, denominator);
    let positive_label_residual = -sigmoid_with_exp(-score, exponential, denominator);
    let count = y.len() as f64;
    let mut objective = CompensatedSum::default();
    let mut gradient = 0.0;
    let mut curvature = 0.0;
    for &label in y {
        let residual = if label == 1.0 {
            positive_label_residual
        } else {
            negative_label_residual
        };
        objective.add(logistic_loss_term(score, label, exponential, count))?;
        gradient += residual * 1.0 / count;
        curvature += (weight * 1.0 / count) * 1.0;
    }
    // There are no slope coefficients, and the intercept is not penalized.
    objective.total()?;
    let step = solve_positive_definite(&mut [vec![curvature]], &[gradient])?;
    Ok(gradient.abs() <= GRADIENT_TOLERANCE
        && step[0].abs() <= STEP_TOLERANCE * (1.0 + initial.abs()))
}

fn optimize<'a>(
    x: impl ExactSizeIterator<Item = &'a [f64]> + Clone,
    y: &[f64],
    alpha: f64,
    mut parameters: Vec<f64>,
    max_iterations: usize,
) -> Result<Vec<f64>, ClassificationError> {
    // Keep a disjoint trial buffer. A rejected line-search point must not
    // update the current parameters; an accepted one swaps storage.
    let mut candidate = Vec::new();
    // Keep the packed rows allocated after Cholesky; the next derivative
    // pass zeroes their cells before accumulating into the same buffers.
    let mut workspace = Vec::new();
    for _ in 0..max_iterations {
        let ObjectiveDerivatives {
            value: objective,
            gradient,
            curvature,
        } = derivatives_into(x.clone(), y, alpha, &parameters, workspace)?;
        workspace = curvature;
        let step = solve_positive_definite(&mut workspace, &gradient)?;
        let small_gradient = gradient
            .iter()
            .all(|value| value.abs() <= GRADIENT_TOLERANCE);
        let small_step = step
            .iter()
            .zip(&parameters)
            .all(|(step, parameter)| step.abs() <= STEP_TOLERANCE * (1.0 + parameter.abs()));
        if small_gradient && small_step {
            return Ok(parameters);
        }
        let descent = dot(&gradient, &step)?;
        if descent <= 0.0 {
            return Err(ClassificationError::ConvergenceFailure);
        }
        let mut fraction = 1.0;
        let mut accepted = false;
        for _ in 0..MAX_BACKTRACKS {
            candidate.clear();
            candidate.extend(
                parameters.iter().zip(&step).map(|(value, step)| value - fraction * step),
            );
            // Large trial steps may overflow; backtrack rather than returning
            // a model or changing the objective's regularization strength.
            if let Ok(next_objective) = loss(x.clone(), y, alpha, &candidate) {
                if next_objective <= objective - 1e-4 * fraction * descent {
                    accepted = true;
                    break;
                }
            }
            fraction *= 0.5;
        }
        if !accepted {
            return Err(ClassificationError::ConvergenceFailure);
        }
        std::mem::swap(&mut parameters, &mut candidate);
    }
    Err(ClassificationError::ConvergenceFailure)
}

#[cfg(test)]
fn derivatives<'a>(
    x: impl ExactSizeIterator<Item = &'a [f64]>,
    y: &[f64],
    alpha: f64,
    parameters: &[f64],
) -> Result<ObjectiveDerivatives, ClassificationError> {
    derivatives_into(x, y, alpha, parameters, Vec::new())
}

fn derivatives_into<'a>(
    x: impl ExactSizeIterator<Item = &'a [f64]>,
    y: &[f64],
    alpha: f64,
    parameters: &[f64],
    mut curvature: Vec<Vec<f64>>,
) -> Result<ObjectiveDerivatives, ClassificationError> {
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(ClassificationError::NumericFailure);
    }
    let width = parameters.len();
    let mut objective = CompensatedSum::default();
    let mut gradient = vec![0.0; width];
    // Cholesky reads only the lower triangle. Store each active prefix.
    if curvature.is_empty() {
        curvature = (1..=width).map(|len| vec![0.0; len]).collect();
    } else {
        debug_assert_eq!(curvature.len(), width);
        for row in &mut curvature {
            row.fill(0.0);
        }
    }
    let count = x.len() as f64;
    for (row, &label) in x.zip(y) {
        let score = dot(row, parameters)?;
        let exponential = (-score.abs()).exp();
        let denominator = 1.0 + exponential;
        // Use -sigmoid(-score) rather than sigmoid(score) - 1 to retain
        // the positive-class residual in the tails. Either sigmoid sign
        // uses this same exponential, as do the loss and curvature.
        let residual = if label == 1.0 {
            -sigmoid_with_exp(-score, exponential, denominator)
        } else {
            sigmoid_with_exp(score, exponential, denominator)
        };
        objective.add(logistic_loss_term(score, label, exponential, count))?;
        let weight = exponential / denominator.powi(2);
        for ((gradient, lower_row), &value) in gradient.iter_mut().zip(&mut curvature).zip(row) {
            *gradient += residual * value / count;
            let weighted = weight * value / count;
            for (cell, &predictor) in lower_row.iter_mut().zip(row) {
                *cell += weighted * predictor;
            }
        }
    }
    for j in 0..width {
        if j > 0 {
            gradient[j] = finite(gradient[j] + finite(alpha * parameters[j])?)?;
            curvature[j][j] = finite(curvature[j][j] + alpha)?;
        }
    }
    Ok(ObjectiveDerivatives {
        value: penalized_objective(objective, parameters, alpha)?,
        gradient,
        curvature,
    })
}

fn loss<'a>(
    x: impl ExactSizeIterator<Item = &'a [f64]>,
    y: &[f64],
    alpha: f64,
    parameters: &[f64],
) -> Result<f64, ClassificationError> {
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(ClassificationError::NumericFailure);
    }
    let mut objective = CompensatedSum::default();
    let count = x.len() as f64;
    for (row, &label) in x.zip(y) {
        let score = dot(row, parameters)?;
        let exponential = (-score.abs()).exp();
        objective.add(logistic_loss_term(score, label, exponential, count))?;
    }
    penalized_objective(objective, parameters, alpha)
}

fn logistic_loss_term(score: f64, label: f64, exponential: f64, count: f64) -> f64 {
    let signed = if label == 1.0 { -score } else { score };
    // softplus(signed) avoids exp overflow and loss cancellation.
    // abs(signed) == abs(score), so the derivative pass can share its exp.
    (signed.max(0.0) + exponential.ln_1p()) / count
}

fn penalized_objective(
    mut objective: CompensatedSum,
    parameters: &[f64],
    alpha: f64,
) -> Result<f64, ClassificationError> {
    for &coefficient in &parameters[1..] {
        objective.add(finite(0.5 * finite(alpha * coefficient)? * coefficient)?)?;
    }
    objective.total()
}

/// Diagonally equilibrated Cholesky, without hidden diagonal jitter.
fn solve_positive_definite(
    matrix: &mut [Vec<f64>],
    right: &[f64],
) -> Result<Vec<f64>, ClassificationError> {
    let width = right.len();
    let scales: Vec<_> = (0..width).map(|j| matrix[j][j].sqrt()).collect();
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(ClassificationError::ConvergenceFailure);
    }
    for (row, &row_scale) in matrix.iter_mut().zip(&scales) {
        for (entry, &column_scale) in row.iter_mut().zip(&scales) {
            *entry = *entry / row_scale / column_scale;
        }
    }
    for j in 0..width {
        // Borrow the active row once. Earlier rows already contain the
        // factor and remain immutable while this row is solved left to right.
        let (previous, trailing) = matrix.split_at_mut(j);
        let row = &mut trailing[0];
        for (k, factored) in previous.iter().enumerate() {
            let mut value = row[k];
            for (a, b) in row[..k].iter().zip(&factored[..k]) {
                value -= a * b;
            }
            row[k] = value / factored[k];
        }
        let mut diagonal = row[j];
        for &value in &row[..j] {
            diagonal -= value * value;
        }
        // Refuse numerically singular curvature rather than silently
        // fitting an unrequested pseudoinverse/regularized objective.
        if diagonal <= 1e-14 || !diagonal.is_finite() {
            return Err(ClassificationError::ConvergenceFailure);
        }
        row[j] = diagonal.sqrt();
    }
    let mut solution = vec![0.0; width];
    for (j, (row, (&right, &scale))) in matrix.iter().zip(right.iter().zip(&scales)).enumerate() {
        let mut value = right / scale;
        for (entry, solved) in row[..j].iter().zip(&solution[..j]) {
            value -= entry * solved;
        }
        solution[j] = finite(value / row[j])?;
    }
    for j in (0..width).rev() {
        let mut value = solution[j];
        for (row, solved) in matrix[j + 1..].iter().zip(&solution[j + 1..]) {
            value -= row[j] * solved;
        }
        solution[j] = finite(value / matrix[j][j])?;
    }
    for (value, scale) in solution.iter_mut().zip(scales) {
        *value = finite(*value / scale)?;
    }
    Ok(solution)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn sigmoid_with_exp(value: f64, exponential: f64, denominator: f64) -> f64 {
    if value >= 0.0 {
        1.0 / denominator
    } else {
        exponential / denominator
    }
}

#[derive(Clone, Debug)]
struct FeatureScaler {
    origin: f64,
    magnitude: f64,
    mean_delta: f64,
    deviation: f64,
}

impl FeatureScaler {
    fn fit_into(
        x: &[Vec<f64>],
        column: usize,
        design: &mut [f64],
        width: usize,
        slot: usize,
    ) -> Result<Self, ClassificationError> {
        let origin = x[0][column];
        let magnitude = x
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0_f64, f64::max);
        let mut sum = CompensatedSum::default();
        for (row, staged) in x.iter().zip(design.chunks_exact_mut(width)) {
            let delta = scaled_difference(row[column], origin, magnitude)?;
            staged[slot] = delta;
            sum.add(delta)?;
        }
        let mean_delta = checked_div(sum.total()?, x.len() as f64)?;
        let norm = design
            .chunks_exact(width)
            .fold(0.0_f64, |norm, row| norm.hypot(row[slot] - mean_delta));
        let deviation = checked_div(norm, (x.len() as f64).sqrt())?;
        if deviation == 0.0 {
            return Err(ClassificationError::NumericFailure);
        }
        let scaler = Self { origin, magnitude, mean_delta, deviation };
        for row in design.chunks_exact_mut(width) {
            row[slot] = scaler.transform_delta(row[slot])?;
        }
        Ok(scaler)
    }

    fn transform(&self, value: f64) -> Result<f64, ClassificationError> {
        self.transform_delta(scaled_difference(value, self.origin, self.magnitude)?)
    }

    fn transform_delta(&self, delta: f64) -> Result<f64, ClassificationError> {
        checked_div(delta - self.mean_delta, self.deviation)
    }
}

fn scaled_difference(value: f64, origin: f64, scale: f64) -> Result<f64, ClassificationError> {
    // Detect destructive dynamic-range loss regardless of row ordering.
    checked_div(value, scale)?;
    let difference = value - origin;
    if difference.is_finite() {
        checked_div(difference, scale)
    } else {
        finite(checked_div(value, scale)? - checked_div(origin, scale)?)
    }
}

fn validate_training(x: &[Vec<f64>], y: &[f64]) -> Result<usize, ClassificationError> {
    let width = x.first().ok_or(ClassificationError::Empty)?.len();
    validate_features(x, width)?;
    if x.len() != y.len() {
        return Err(ClassificationError::Shape);
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(ClassificationError::NonFinite);
    }
    if y.iter().any(|&value| value != 0.0 && value != 1.0) {
        return Err(ClassificationError::InvalidLabel);
    }
    Ok(width)
}

fn validate_features(x: &[Vec<f64>], width: usize) -> Result<(), ClassificationError> {
    if x.is_empty() {
        return Err(ClassificationError::Empty);
    }
    if width == 0 || x.iter().any(|row| row.len() != width) {
        return Err(ClassificationError::Shape);
    }
    if x.len() > MAX_ROWS || width > MAX_FEATURES {
        return Err(ClassificationError::Capacity);
    }
    if x.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ClassificationError::NonFinite);
    }
    Ok(())
}

fn finite(value: f64) -> Result<f64, ClassificationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ClassificationError::NumericFailure)
    }
}

fn checked_div(numerator: f64, denominator: f64) -> Result<f64, ClassificationError> {
    let value = finite(numerator / denominator)?;
    if numerator != 0.0 && value == 0.0 {
        Err(ClassificationError::NumericFailure)
    } else {
        Ok(value)
    }
}

// Allocate no row- or feature-sized term buffers. Every caller keeps the
// original insertion order, Neumaier correction, and finite checks.
#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), ClassificationError> {
        let next = finite(self.sum + value)?;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        Ok(())
    }

    fn total(self) -> Result<f64, ClassificationError> {
        finite(self.sum + self.correction)
    }
}

fn dot(a: &[f64], b: &[f64]) -> Result<f64, ClassificationError> {
    let mut products = CompensatedSum::default();
    for (&left, &right) in a.iter().zip(b) {
        products.add(finite(left * right)?)?;
    }
    products.total()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_workspace_clears_factored_curvature_before_reuse() {
        let rows = [[1.0, 2.0], [1.0, -1.0], [1.0, 0.0], [1.0, 3.0]];
        let rows = || rows.iter().map(|row| row.as_slice());
        let y = [0.0, 1.0, 0.0, 1.0];
        let mut first = derivatives(rows(), &y, 0.75, &[0.0, 0.0]).unwrap();
        solve_positive_definite(&mut first.curvature, &first.gradient).unwrap();
        let allocated_row = first.curvature[1].as_ptr();
        let parameters = [1.0, -1.5];
        let fresh = derivatives(rows(), &y, 0.75, &parameters).unwrap();
        let reused = derivatives_into(rows(), &y, 0.75, &parameters, first.curvature).unwrap();
        assert_eq!(reused.curvature[1].as_ptr(), allocated_row);
        assert_eq!(fresh.value.to_bits(), reused.value.to_bits());
        assert!(fresh.gradient.iter().map(|v| v.to_bits()).eq(reused.gradient.iter().map(|v| v.to_bits())));
        assert!(fresh.curvature.iter().flatten().map(|v| v.to_bits())
            .eq(reused.curvature.iter().flatten().map(|v| v.to_bits())));
    }

    #[test]
    fn logistic_fit_never_drops_late_varying_columns_after_a_constant_prefix() {
        let mut rows = vec![vec![-0.0; 20]; 256];
        rows[1][0] = 0.0;
        rows[252][18] = -3.0;
        rows[254][2] = 4.0;
        assert_eq!(varying_columns(&rows, 20), [2, 18]);
        let labels: Vec<_> = (0..256).map(|i| f64::from(i % 3 == 0)).collect();
        let model = LogisticClassifier::fit(&rows, &labels, 1.0).unwrap();
        assert_eq!(model.columns, [2, 18]);
        rows[252][18] = 0.0;
        rows[254][2] = 0.0;
        assert!(varying_columns(&rows, 20).is_empty());
        rows[255][0] = f64::NAN;
        assert!(matches!(LogisticClassifier::fit(&rows, &labels, 1.0),
            Err(ClassificationError::NonFinite)));
    }

    #[test]
    fn flat_design_preserves_column_order_with_constants_interleaved() {
        let x = [
            [1.0, -0.0, 3.0, 4.0], [2.0, -0.0, 3.0, 1.0],
            [3.0, -0.0, 3.0, -2.0], [-4.0, 0.0, 3.0, -1.0],
            [-2.0, 0.0, 3.0, 7.0], [5.0, 0.0, 3.0, 8.0],
        ].map(Vec::from);
        let reduced: Vec<_> = x.iter().map(|row| vec![row[0], row[3]]).collect();
        let labels = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let with_constants = LogisticClassifier::fit(&x, &labels, 0.3).unwrap();
        let compact = LogisticClassifier::fit(&reduced, &labels, 0.3).unwrap();
        assert_eq!(with_constants.columns, [0, 3]);
        assert!(with_constants.predict_proba(&x).unwrap().into_iter().map(f64::to_bits)
            .eq(compact.predict_proba(&reduced).unwrap().into_iter().map(f64::to_bits)));
        let invalid = vec![vec![f64::MAX, 1.0], vec![f64::from_bits(1), 2.0]];
        assert!(matches!(LogisticClassifier::fit(&invalid, &[0.0, 1.0], 0.3),
            Err(ClassificationError::NumericFailure)));
    }

    #[test]
    fn only_intercept_predictions_match_single_row_bits_after_full_validation() {
        let fit = vec![vec![2.0, -0.0]; 7];
        for positives in [1, 3, 4, 6] {
            let y: Vec<_> = (0..7).map(|i| if i < positives { 1.0 } else { 0.0 }).collect();
            let model = LogisticClassifier::fit(&fit, &y, 1.0).unwrap();
            assert!(model.columns.is_empty());
            let mut query = vec![vec![1.0, 0.0], vec![-f64::MAX, f64::MAX], vec![-0.0, 2.0]];
            let p = model.predict_proba(&query).unwrap();
            for (row, observed) in query.iter().zip(p) {
                let single = model.predict_proba(std::slice::from_ref(row)).unwrap();
                assert_eq!(single[0].to_bits(), observed.to_bits());
            }
            query[2][1] = f64::NAN;
            assert_eq!(model.predict_proba(&query), Err(ClassificationError::NonFinite));
            query[0].pop();
            assert_eq!(model.predict_proba(&query), Err(ClassificationError::Shape));
        }
    }

    #[test]
    fn rejected_full_newton_step_does_not_change_the_current_iterate() {
        let x = vec![vec![1.0]; 4];
        let y = [0.0, 1.0, 0.0, 1.0];
        let initial = vec![4.0];
        let mut d = derivatives(x.iter().map(Vec::as_slice), &y, 0.0, &initial).unwrap();
        let step = solve_positive_definite(&mut d.curvature, &d.gradient).unwrap();
        assert!(loss(x.iter().map(Vec::as_slice), &y, 0.0, &[initial[0] - step[0]]).unwrap() > d.value);
        let solution = optimize(x.iter().map(Vec::as_slice), &y, 0.0, initial, MAX_ITERATIONS).unwrap();
        assert!(solution[0].abs() < 1e-9);
    }

    #[test]
    fn packed_lower_curvature_matches_zero_logit_outer_products() {
        let x = vec![vec![1.0, 2.0, -3.0], vec![1.0, -4.0, 5.0]];
        let d = derivatives(x.iter().map(Vec::as_slice), &[0.0, 1.0], 0.125, &[0.0; 3]).unwrap();
        assert_eq!(d.gradient, vec![0.0, 1.5, -2.0]);
        assert_eq!(
            d.curvature,
            vec![vec![0.25], vec![-0.25, 2.625], vec![0.25, -3.25, 4.375]]
        );
    }

    #[test]
    fn shared_exponential_residual_preserves_both_sigmoid_tails_bitwise() {
        for score in [
            -f64::MAX, -746.0, -745.0, -710.0, -40.0, -1.0,
            -f64::from_bits(1), -0.0, 0.0, f64::from_bits(1),
            1.0, 40.0, 710.0, 745.0, 746.0, f64::MAX,
        ] {
            let exponential = (-score.abs()).exp();
            let denominator = 1.0 + exponential;
            for signed in [score, -score] {
                assert_eq!(
                    sigmoid_with_exp(signed, exponential, denominator).to_bits(),
                    sigmoid(signed).to_bits()
                );
            }
        }
        let tail = (-40.0_f64).exp();
        assert!(-sigmoid_with_exp(-40.0, tail, 1.0 + tail) < 0.0);
    }


    #[test]
    fn streaming_dot_keeps_cancellation_and_rejects_nonfinite_products_or_sums() {
        let max = f64::MAX;
        assert_eq!(dot(&[max, 1.0, -max], &[1.0; 3]), Ok(1.0));
        assert_eq!(dot(&[-max, 1.0, max], &[1.0; 3]), Ok(1.0));
        assert_eq!(dot(&[max], &[2.0]), Err(ClassificationError::NumericFailure));
        assert_eq!(
            dot(&[max, max, -max], &[1.0; 3]),
            Err(ClassificationError::NumericFailure)
        );
    }

    #[test]
    fn exhausted_iteration_budget_never_returns_a_model() {
        let x = [vec![1.0, -1.0], vec![1.0, 1.0]];
        assert_eq!(
            optimize(x.iter().map(Vec::as_slice), &[0.0, 1.0], 0.1, vec![0.0; 2], 0),
            Err(ClassificationError::ConvergenceFailure)
        );
        assert_eq!(
            optimize(x.iter().map(Vec::as_slice), &[0.0, 1.0], 0.1, vec![0.0; 2], 1),
            Err(ClassificationError::ConvergenceFailure)
        );
    }

    #[test]
    fn combined_derivatives_match_the_trial_objective_bit_for_bit() {
        let design = [
            vec![1.0, -1.0, 0.0],
            vec![1.0, 0.0, -0.0],
            vec![1.0, 0.125, -0.75],
            vec![1.0, 1.0, 2.0],
            vec![1.0, -0.25, 0.375],
        ];
        let y = [1.0, 0.0, 1.0, 0.0, 0.0];
        for parameters in [[-0.0, 0.0, 0.0], [-0.1, 1.5, -0.4], [0.125, 1000.0, 0.1]] {
            for alpha in [0.0, 0.03125, 11.0] {
                let objective = loss(design.iter().map(Vec::as_slice), &y, alpha, &parameters).unwrap();
                let combined = derivatives(design.iter().map(Vec::as_slice), &y, alpha, &parameters).unwrap();
                assert_eq!(combined.value.to_bits(), objective.to_bits());
            }
        }
    }

    #[test]
    fn intercept_gate_matches_first_iteration_and_preserves_fallback() {
        for n in [2, 7, 113, 7_919, 20_000] {
            let y: Vec<_> = (0..n).map(|i| f64::from(i != 0)).collect();
            let design = vec![vec![1.0]; y.len()];
            let initial = f64::from(n - 1).ln();
            for trial in [initial, 0.5, -20.0] {
                let gated = intercept_converges_without_a_design(&y, trial).unwrap();
                let explicit = optimize(design.iter().map(Vec::as_slice), &y, 0.0, vec![trial], 1);
                assert_eq!(gated, explicit.is_ok());
            }
            let baseline = optimize(design.iter().map(Vec::as_slice), &y, 1e100, vec![initial], MAX_ITERATIONS).unwrap();
            let model =
                LogisticClassifier::fit(&vec![vec![7.0, -0.0]; y.len()], &y, 1e100).unwrap();
            assert_eq!(model.intercept.to_bits(), baseline[0].to_bits());
        }
    }

    #[test]
    fn logistic_loss_and_residuals_remain_stable_at_large_logits() {
        let x = [vec![1.0, -1.0], vec![1.0, 1.0]];
        let correct = loss(x.iter().map(Vec::as_slice), &[0.0, 1.0], 0.0, &[0.0, 1000.0]).unwrap();
        let wrong = loss(x.iter().map(Vec::as_slice), &[1.0, 0.0], 0.0, &[0.0, 1000.0]).unwrap();
        assert_eq!(correct, 0.0);
        assert_eq!(wrong, 1000.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        let result = derivatives(x.iter().map(Vec::as_slice), &[0.0, 1.0], 0.0, &[0.0, 40.0]).unwrap();
        assert!(result.gradient[1] < 0.0);
        assert!(result.curvature[1][1] > 0.0);
    }
    #[test]
    fn packed_cholesky_uses_only_factored_prefix_and_detects_singularity() {
        let mut matrix = vec![vec![4.0], vec![2.0, 5.0], vec![-2.0, 0.0, 10.25]];
        let actual = solve_positive_definite(&mut matrix, &[-8.0, -9.0, 29.75]).unwrap();
        for (actual, expected) in actual.into_iter().zip([0.5, -2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(
            solve_positive_definite(&mut [vec![1.0], vec![1.0, 1.0]], &[1.0, 1.0]),
            Err(ClassificationError::ConvergenceFailure)
        );
    }

}
