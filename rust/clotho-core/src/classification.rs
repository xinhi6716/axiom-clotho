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
        let mut columns = Vec::with_capacity(width);
        let mut scalers = Vec::with_capacity(width);
        for column in 0..width {
            if x.iter().all(|row| row[column] == x[0][column]) {
                continue;
            }
            columns.push(column);
            scalers.push(FeatureScaler::fit(x, column)?);
        }
        let mut design = Vec::with_capacity(x.len());
        for row in x {
            let mut transformed = Vec::with_capacity(columns.len() + 1);
            transformed.push(1.0);
            for (&column, scaler) in columns.iter().zip(&scalers) {
                transformed.push(scaler.transform(row[column])?);
            }
            design.push(transformed);
        }
        let mut parameters = vec![0.0; columns.len() + 1];
        parameters[0] = (positives as f64 / (y.len() - positives) as f64).ln();
        let parameters = optimize(&design, y, alpha, parameters, MAX_ITERATIONS)?;
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
        x.iter()
            .map(|row| {
                let mut terms = Vec::with_capacity(self.coefficients.len() + 1);
                terms.push(self.intercept);
                for ((&column, scaler), &coefficient) in self
                    .columns
                    .iter()
                    .zip(&self.scalers)
                    .zip(&self.coefficients)
                {
                    terms.push(finite(scaler.transform(row[column])? * coefficient)?);
                }
                Ok(sigmoid(compensated_sum(terms.into_iter())?))
            })
            .collect()
    }
}

fn optimize(
    x: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    mut parameters: Vec<f64>,
    max_iterations: usize,
) -> Result<Vec<f64>, ClassificationError> {
    for _ in 0..max_iterations {
        let ObjectiveDerivatives {
            value: objective,
            gradient,
            curvature,
        } = derivatives(x, y, alpha, &parameters)?;
        let step = solve_positive_definite(curvature, &gradient)?;
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
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACKS {
            let candidate: Vec<_> = parameters
                .iter()
                .zip(&step)
                .map(|(value, step)| value - fraction * step)
                .collect();
            // Large trial steps may overflow; backtrack rather than returning
            // a model or changing the objective's regularization strength.
            if let Ok(next_objective) = loss(x, y, alpha, &candidate) {
                if next_objective <= objective - 1e-4 * fraction * descent {
                    accepted = Some(candidate);
                    break;
                }
            }
            fraction *= 0.5;
        }
        parameters = accepted.ok_or(ClassificationError::ConvergenceFailure)?;
    }
    Err(ClassificationError::ConvergenceFailure)
}

fn derivatives(
    x: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    parameters: &[f64],
) -> Result<ObjectiveDerivatives, ClassificationError> {
    let width = parameters.len();
    let mut gradient = vec![0.0; width];
    let mut curvature = vec![vec![0.0; width]; width];
    let count = x.len() as f64;
    for (row, &label) in x.iter().zip(y) {
        let score = dot(row, parameters)?;
        // Avoid cancellation of sigmoid(score) - 1 for large positive scores.
        let residual = if label == 1.0 {
            -sigmoid(-score)
        } else {
            sigmoid(score)
        };
        let exponential = (-score.abs()).exp();
        let weight = exponential / (1.0 + exponential).powi(2);
        for j in 0..width {
            gradient[j] += residual * row[j] / count;
            let weighted = weight * row[j] / count;
            for k in 0..=j {
                curvature[j][k] += weighted * row[k];
            }
        }
    }
    for j in 0..width {
        if j > 0 {
            gradient[j] = finite(gradient[j] + finite(alpha * parameters[j])?)?;
            curvature[j][j] = finite(curvature[j][j] + alpha)?;
        }
    }
    // Cholesky reads only the lower triangle.
    Ok(ObjectiveDerivatives {
        value: loss(x, y, alpha, parameters)?,
        gradient,
        curvature,
    })
}

fn loss(
    x: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    parameters: &[f64],
) -> Result<f64, ClassificationError> {
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(ClassificationError::NumericFailure);
    }
    let mut terms = Vec::with_capacity(x.len() + parameters.len());
    for (row, &label) in x.iter().zip(y) {
        let score = dot(row, parameters)?;
        let signed = if label == 1.0 { -score } else { score };
        // softplus(signed) avoids exp overflow and loss cancellation.
        terms.push((signed.max(0.0) + (-signed.abs()).exp().ln_1p()) / x.len() as f64);
    }
    for &coefficient in &parameters[1..] {
        terms.push(finite(0.5 * finite(alpha * coefficient)? * coefficient)?);
    }
    compensated_sum(terms.into_iter())
}

/// Diagonally equilibrated Cholesky, without hidden diagonal jitter.
fn solve_positive_definite(
    mut matrix: Vec<Vec<f64>>,
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
    for j in 0..width {
        for k in 0..=j {
            matrix[j][k] = matrix[j][k] / scales[j] / scales[k];
        }
    }
    for j in 0..width {
        for k in 0..=j {
            let mut value = matrix[j][k];
            for (a, b) in matrix[j][..k].iter().zip(&matrix[k][..k]) {
                value -= a * b;
            }
            if j == k {
                // Refuse numerically singular curvature rather than silently
                // fitting an unrequested pseudoinverse/regularized objective.
                if value <= 1e-14 || !value.is_finite() {
                    return Err(ClassificationError::ConvergenceFailure);
                }
                matrix[j][j] = value.sqrt();
            } else {
                matrix[j][k] = value / matrix[k][k];
            }
        }
    }
    let mut solution = vec![0.0; width];
    for j in 0..width {
        let mut value = right[j] / scales[j];
        for k in 0..j {
            value -= matrix[j][k] * solution[k];
        }
        solution[j] = finite(value / matrix[j][j])?;
    }
    for j in (0..width).rev() {
        let mut value = solution[j];
        for k in j + 1..width {
            value -= matrix[k][j] * solution[k];
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

#[derive(Clone, Debug)]
struct FeatureScaler {
    origin: f64,
    magnitude: f64,
    mean_delta: f64,
    deviation: f64,
}

impl FeatureScaler {
    fn fit(x: &[Vec<f64>], column: usize) -> Result<Self, ClassificationError> {
        let origin = x[0][column];
        let magnitude = x
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0_f64, f64::max);
        let deltas = x
            .iter()
            .map(|row| scaled_difference(row[column], origin, magnitude))
            .collect::<Result<Vec<_>, _>>()?;
        let mean_delta = checked_div(compensated_sum(deltas.iter().copied())?, x.len() as f64)?;
        let norm = deltas
            .iter()
            .fold(0.0_f64, |norm, delta| norm.hypot(delta - mean_delta));
        let deviation = checked_div(norm, (x.len() as f64).sqrt())?;
        if deviation == 0.0 {
            return Err(ClassificationError::NumericFailure);
        }
        Ok(Self {
            origin,
            magnitude,
            mean_delta,
            deviation,
        })
    }

    fn transform(&self, value: f64) -> Result<f64, ClassificationError> {
        checked_div(
            scaled_difference(value, self.origin, self.magnitude)? - self.mean_delta,
            self.deviation,
        )
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

fn compensated_sum(values: impl Iterator<Item = f64>) -> Result<f64, ClassificationError> {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = finite(sum + value)?;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    finite(sum + correction)
}

fn dot(a: &[f64], b: &[f64]) -> Result<f64, ClassificationError> {
    let products = a
        .iter()
        .zip(b)
        .map(|(a, b)| finite(a * b))
        .collect::<Result<Vec<_>, _>>()?;
    compensated_sum(products.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_iteration_budget_never_returns_a_model() {
        let x = [vec![1.0, -1.0], vec![1.0, 1.0]];
        assert_eq!(
            optimize(&x, &[0.0, 1.0], 0.1, vec![0.0; 2], 0),
            Err(ClassificationError::ConvergenceFailure)
        );
        assert_eq!(
            optimize(&x, &[0.0, 1.0], 0.1, vec![0.0; 2], 1),
            Err(ClassificationError::ConvergenceFailure)
        );
    }

    #[test]
    fn logistic_loss_and_residuals_remain_stable_at_large_logits() {
        let x = [vec![1.0, -1.0], vec![1.0, 1.0]];
        let correct = loss(&x, &[0.0, 1.0], 0.0, &[0.0, 1000.0]).unwrap();
        let wrong = loss(&x, &[1.0, 0.0], 0.0, &[0.0, 1000.0]).unwrap();
        assert_eq!(correct, 0.0);
        assert_eq!(wrong, 1000.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        let result = derivatives(&x, &[0.0, 1.0], 0.0, &[0.0, 40.0]).unwrap();
        assert!(result.gradient[1] < 0.0);
        assert!(result.curvature[1][1] > 0.0);
    }
}
