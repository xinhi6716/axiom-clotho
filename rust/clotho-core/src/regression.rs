//! Dependency-free mean and ridge baselines for finite numeric tables.
//!
//! Ridge minimizes `sum((y - intercept - Z * beta)^2) + alpha * sum(beta^2)`,
//! where `Z` uses training-only population-standard-deviation scaling. The
//! intercept is never penalized. Consequently alpha applies to standardized
//! coefficients, not coefficients in the caller's original feature units.
//! These estimators do not perform model selection or scientific validation.

use std::fmt;

const MAX_ROWS: usize = 20_000;
const MAX_FEATURES: usize = 100;
// Reject solves that may lose twelve or more decimal digits. This is the
// reciprocal infinity-norm condition of the equilibrated triangular factor,
// not an estimate of the condition of unscaled, original-unit features.
const MIN_RECIPROCAL_CONDITION: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegressionError {
    Empty,
    Shape,
    NonFinite,
    InvalidAlpha,
    Capacity,
    IllConditioned,
    NumericFailure,
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "Training and prediction tables must not be empty.",
            Self::Shape => "Feature widths and target lengths must match and be nonzero.",
            Self::NonFinite => "Features and targets must be finite.",
            Self::InvalidAlpha => "Ridge alpha must be finite and nonnegative.",
            Self::Capacity => "At most 20,000 rows and 100 features are supported.",
            Self::IllConditioned => "The regression system is rank deficient or ill-conditioned; use stronger regularization or remove redundant features.",
            Self::NumericFailure => "The data or result exceeds the supported floating-point range.",
        })
    }
}
impl std::error::Error for RegressionError {}

/// Predicts the training-target mean and validates prediction feature shape.
#[derive(Clone, Debug)]
pub struct MeanRegressor {
    width: usize,
    mean: f64,
}

impl MeanRegressor {
    pub fn fit(x: &[Vec<f64>], y: &[f64]) -> Result<Self, RegressionError> {
        let width = validate_training(x, y)?;
        let scale = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let mean = if scale == 0.0 {
            0.0
        } else {
            let normalized = y
                .iter()
                .map(|v| checked_div(*v, scale))
                .collect::<Result<Vec<_>, _>>()?;
            let average = checked_div(compensated_sum(normalized.into_iter())?, y.len() as f64)?
                .clamp(-1.0, 1.0);
            checked_mul(average, scale)?
        };
        Ok(Self { width, mean })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, RegressionError> {
        validate_features(x, self.width)?;
        Ok(vec![self.mean; x.len()])
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }
}

/// Linear ridge regression with an unpenalized intercept.
///
/// Uses column-pivoted Householder QR on an augmented least-squares system,
/// rather than forming the normal equations. Column equilibration and an
/// explicit condition check precede back-substitution. Rank-deficient or
/// ill-conditioned unregularized fits return an error; they do not silently
/// receive a pseudoinverse or a different regularization parameter.
///
/// Fit and predict support 1..=20,000 rows and 1..=100 features. Constant
/// features are supported with positive alpha. With alpha zero they make the
/// system rank deficient. Features must have identical order and units in fit
/// and predict. No missing-value imputation or uncertainty estimate is supplied.
#[derive(Clone, Debug)]
pub struct RidgeRegressor {
    scalers: Vec<FeatureScaler>,
    // Coefficients and intercept are in normalized target units, avoiding
    // overflow when targets have large magnitudes.
    coefficients: Vec<f64>,
    intercept: f64,
    targets: TargetScaler,
}

impl RidgeRegressor {
    pub fn fit(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Result<Self, RegressionError> {
        let width = validate_training(x, y)?;
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(RegressionError::InvalidAlpha);
        }
        let penalty = alpha.sqrt();
        let row_count = x.len() + if alpha > 0.0 { width } else { 0 };
        let column_count = width + 1;
        if row_count < column_count {
            return Err(RegressionError::IllConditioned);
        }
        // Column-major storage makes the QR updates contiguous. Only the
        // feature columns have regularization rows; the intercept has none.
        let mut columns = vec![vec![0.0; row_count]; column_count];
        columns[0][..x.len()].fill(1.0);
        let mut scalers = Vec::with_capacity(width);
        for j in 0..width {
            let scaler = FeatureScaler::fit(x, j)?;
            for (i, row) in x.iter().enumerate() {
                columns[j + 1][i] = scaler.transform(row[j])?;
            }
            if alpha > 0.0 {
                columns[j + 1][x.len() + j] = penalty;
            }
            scalers.push(scaler);
        }
        let targets = TargetScaler::fit(x, y)?;
        let mut target = vec![0.0; row_count];
        for (b, value) in target.iter_mut().zip(y) {
            *b = targets.transform(*value)?;
        }
        let solution = solve_qr(columns, target)?;
        Ok(Self {
            scalers,
            intercept: solution[0],
            coefficients: solution[1..].to_vec(),
            targets,
        })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, RegressionError> {
        validate_features(x, self.scalers.len())?;
        x.iter()
            .map(|row| {
                let mut terms = Vec::with_capacity(row.len() + 1);
                terms.push(self.intercept);
                for ((value, scaler), coefficient) in
                    row.iter().zip(&self.scalers).zip(&self.coefficients)
                {
                    terms.push(checked_mul(scaler.transform(*value)?, *coefficient)?);
                }
                self.targets.inverse(compensated_sum(terms.into_iter())?)
            })
            .collect()
    }
}

/// Centers targets before solving, preserving slopes when labels share a large
/// offset. The two scales also accommodate finite labels whose difference would
/// exceed f64::MAX. The mean is computed separately with compensated summation.
#[derive(Clone, Debug)]
struct TargetScaler {
    origin: f64,
    magnitude: f64,
    deviation: f64,
}

impl TargetScaler {
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Result<Self, RegressionError> {
        let origin = MeanRegressor::fit(x, y)?.mean();
        let magnitude = y.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        let magnitude = if magnitude == 0.0 { 1.0 } else { magnitude };
        let mut deviation = 0.0_f64;
        for value in y {
            deviation = deviation.max(scaled_difference(*value, origin, magnitude)?.abs());
        }
        let deviation = if deviation == 0.0 { 1.0 } else { deviation };
        Ok(Self {
            origin,
            magnitude,
            deviation,
        })
    }

    fn transform(&self, value: f64) -> Result<f64, RegressionError> {
        checked_div(
            scaled_difference(value, self.origin, self.magnitude)?,
            self.deviation,
        )
    }

    fn inverse(&self, value: f64) -> Result<f64, RegressionError> {
        let scale = self.magnitude * self.deviation;
        if scale.is_finite() && scale != 0.0 {
            let difference = value * scale;
            if difference.is_finite() {
                return finite(self.origin + checked_mul(value, scale)?);
            }
        }
        // The intermediate original-unit difference can overflow even when
        // adding the target origin would yield a finite result.
        let normalized = compensated_sum(
            [
                checked_div(self.origin, self.magnitude)?,
                checked_mul(value, self.deviation)?,
            ]
            .into_iter(),
        )?;
        checked_mul(normalized, self.magnitude)
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
    fn fit(x: &[Vec<f64>], column: usize) -> Result<Self, RegressionError> {
        let origin = x[0][column];
        let magnitude = x
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0_f64, f64::max);
        let magnitude = if magnitude == 0.0 { 1.0 } else { magnitude };
        // Subtract before dividing whenever possible. This preserves feature
        // differences when a large common offset is added to a column.
        let deltas = x
            .iter()
            .map(|row| scaled_difference(row[column], origin, magnitude))
            .collect::<Result<Vec<_>, _>>()?;
        let mean_delta = checked_div(compensated_sum(deltas.iter().copied())?, x.len() as f64)?;
        let norm = deltas
            .iter()
            .fold(0.0_f64, |norm, delta| norm.hypot(delta - mean_delta));
        let deviation = if norm == 0.0 {
            1.0
        } else {
            checked_div(norm, (x.len() as f64).sqrt())?
        };
        Ok(Self {
            origin,
            magnitude,
            mean_delta,
            deviation,
        })
    }

    fn transform(&self, value: f64) -> Result<f64, RegressionError> {
        checked_div(
            scaled_difference(value, self.origin, self.magnitude)? - self.mean_delta,
            self.deviation,
        )
    }
}

fn scaled_difference(value: f64, origin: f64, scale: f64) -> Result<f64, RegressionError> {
    // Reject destructive underflow independently of which training row supplied
    // the origin. Otherwise reordering extreme data could hide lost values.
    checked_div(value, scale)?;
    let difference = value - origin;
    if difference.is_finite() {
        checked_div(difference, scale)
    } else {
        // Opposite, very large finite inputs can overflow a subtraction even
        // though their normalized difference is representable.
        finite(checked_div(value, scale)? - checked_div(origin, scale)?)
    }
}

fn validate_training(x: &[Vec<f64>], y: &[f64]) -> Result<usize, RegressionError> {
    let width = x.first().ok_or(RegressionError::Empty)?.len();
    validate_features(x, width)?;
    if x.len() != y.len() {
        return Err(RegressionError::Shape);
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(RegressionError::NonFinite);
    }
    Ok(width)
}

fn validate_features(x: &[Vec<f64>], width: usize) -> Result<(), RegressionError> {
    if x.is_empty() {
        return Err(RegressionError::Empty);
    }
    if width == 0 || x.iter().any(|row| row.len() != width) {
        return Err(RegressionError::Shape);
    }
    if x.len() > MAX_ROWS || width > MAX_FEATURES {
        return Err(RegressionError::Capacity);
    }
    if x.iter().flatten().any(|value| !value.is_finite()) {
        return Err(RegressionError::NonFinite);
    }
    Ok(())
}

fn finite(value: f64) -> Result<f64, RegressionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RegressionError::NumericFailure)
    }
}

fn checked_div(a: f64, b: f64) -> Result<f64, RegressionError> {
    let value = finite(a / b)?;
    if a != 0.0 && value == 0.0 {
        Err(RegressionError::NumericFailure)
    } else {
        Ok(value)
    }
}

fn checked_mul(a: f64, b: f64) -> Result<f64, RegressionError> {
    let value = finite(a * b)?;
    if a != 0.0 && b != 0.0 && value == 0.0 {
        Err(RegressionError::NumericFailure)
    } else {
        Ok(value)
    }
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> Result<f64, RegressionError> {
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

fn dot(a: &[f64], b: &[f64]) -> Result<f64, RegressionError> {
    let products = a
        .iter()
        .zip(b)
        .map(|(a, b)| checked_mul(*a, *b))
        .collect::<Result<Vec<_>, _>>()?;
    compensated_sum(products.into_iter())
}

fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |n, value| n.hypot(*value))
}

/// Returns coefficients in original column order and scale.
fn solve_qr(mut columns: Vec<Vec<f64>>, mut target: Vec<f64>) -> Result<Vec<f64>, RegressionError> {
    let count = columns.len();
    let mut permutation: Vec<usize> = (0..count).collect();
    let mut scales = Vec::with_capacity(count);
    for column in &mut columns {
        let scale = norm(column);
        if scale == 0.0 || !scale.is_finite() {
            return Err(RegressionError::IllConditioned);
        }
        for value in column {
            *value = checked_div(*value, scale)?;
        }
        scales.push(scale);
    }
    for k in 0..count {
        let pivot = (k..count)
            .max_by(|a, b| norm(&columns[*a][k..]).total_cmp(&norm(&columns[*b][k..])))
            .ok_or(RegressionError::IllConditioned)?;
        columns.swap(k, pivot);
        permutation.swap(k, pivot);
        scales.swap(k, pivot);
        let length = norm(&columns[k][k..]);
        if length <= 64.0 * f64::EPSILON {
            return Err(RegressionError::IllConditioned);
        }
        let diagonal = -length.copysign(columns[k][k]);
        let mut reflector = columns[k][k..].to_vec();
        reflector[0] -= diagonal;
        let reflector_norm = norm(&reflector);
        for value in &mut reflector {
            *value = checked_div(*value, reflector_norm)?;
        }
        for column in &mut columns[k + 1..] {
            let projection = 2.0 * dot(&reflector, &column[k..])?;
            for (value, direction) in column[k..].iter_mut().zip(&reflector) {
                *value = finite(*value - checked_mul(projection, *direction)?)?;
            }
        }
        let projection = 2.0 * dot(&reflector, &target[k..])?;
        for (value, direction) in target[k..].iter_mut().zip(&reflector) {
            *value = finite(*value - checked_mul(projection, *direction)?)?;
        }
        columns[k][k] = diagonal;
        columns[k][k + 1..].fill(0.0);
    }
    check_condition(&columns)?;
    let mut solution = vec![0.0; count];
    for i in (0..count).rev() {
        let products = (i + 1..count)
            .map(|j| checked_mul(columns[j][i], solution[j]))
            .collect::<Result<Vec<_>, _>>()?;
        let contribution = compensated_sum(products.into_iter())?;
        solution[i] = checked_div(target[i] - contribution, columns[i][i])?;
    }
    let mut unpermuted = vec![0.0; count];
    for i in 0..count {
        unpermuted[permutation[i]] = checked_div(solution[i], scales[i])?;
    }
    Ok(unpermuted)
}

/// Computes the infinity-norm condition of R, using triangular solves for its
/// inverse. The dimension is capped at 101, so this O(p^3) check is bounded.
fn check_condition(columns: &[Vec<f64>]) -> Result<(), RegressionError> {
    let count = columns.len();
    let matrix_norm = (0..count)
        .map(|i| (i..count).map(|j| columns[j][i].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let mut inverse_row_sums = vec![0.0_f64; count];
    for j in 0..count {
        let mut inverse_column = vec![0.0; count];
        for i in (0..=j).rev() {
            let contribution =
                compensated_sum((i + 1..=j).map(|k| columns[k][i] * inverse_column[k]))?;
            let rhs = if i == j { 1.0 } else { 0.0 };
            inverse_column[i] = (rhs - contribution) / columns[i][i];
            if !inverse_column[i].is_finite() {
                return Err(RegressionError::IllConditioned);
            }
            inverse_row_sums[i] += inverse_column[i].abs();
        }
    }
    let inverse_norm = inverse_row_sums.into_iter().fold(0.0_f64, f64::max);
    let reciprocal_condition = (1.0 / matrix_norm) / inverse_norm;
    if !reciprocal_condition.is_finite() || reciprocal_condition <= MIN_RECIPROCAL_CONDITION {
        Err(RegressionError::IllConditioned)
    } else {
        Ok(())
    }
}
