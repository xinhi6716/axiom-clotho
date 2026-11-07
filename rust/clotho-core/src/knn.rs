//! Dependency-free, numeric-only k-nearest-neighbor regression baseline.
//! Scaling is fitted on training features only. This is not a scientific
//! validation or a replacement for grouped evaluation/model selection.
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnError {
    Empty,
    Shape,
    NonFinite,
    InvalidK,
    Capacity,
    NumericOverflow,
}
impl fmt::Display for KnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "Training and prediction tables must not be empty.",
            Self::Shape => "Feature widths and target lengths must match.",
            Self::NonFinite => "Features and targets must be finite.",
            Self::InvalidK => "k must be between one and the training row count.",
            Self::Capacity => "At most 20,000 rows and 100 features are supported.",
            Self::NumericOverflow => "Values exceed the numeric range supported by fitted scaling.",
        })
    }
}
impl std::error::Error for KnnError {}

/// Uniform-neighbor regression with training-only maximum-absolute scaling.
///
/// Feature order and units must match between fit and predict. No missing-value
/// imputation, category encoding, extrapolation guarantee, or uncertainty estimate
/// is provided. Distance ties use training-row order deterministically.
#[derive(Clone, Debug)]
pub struct KnnRegressor {
    training: Vec<Vec<f64>>,
    targets: Vec<f64>,
    scales: Vec<f64>,
    k: usize,
}
impl KnnRegressor {
    pub fn fit(x: &[Vec<f64>], y: &[f64], k: usize) -> Result<Self, KnnError> {
        let width = x.first().ok_or(KnnError::Empty)?.len();
        validate(x, width)?;
        if y.len() != x.len() {
            return Err(KnnError::Shape);
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(KnnError::NonFinite);
        }
        if k == 0 || k > x.len() {
            return Err(KnnError::InvalidK);
        }
        let mut scales = vec![0.0_f64; width];
        for row in x {
            for (s, v) in scales.iter_mut().zip(row) {
                *s = s.max(v.abs());
            }
        }
        for s in &mut scales {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        let training: Vec<Vec<f64>> = x
            .iter()
            .map(|r| r.iter().zip(&scales).map(|(v, s)| v / s).collect())
            .collect();
        if x.iter()
            .flatten()
            .zip(training.iter().flatten())
            .any(|(original, scaled)| *original != 0.0 && *scaled == 0.0)
        {
            return Err(KnnError::NumericOverflow);
        }
        Ok(Self {
            training,
            targets: y.to_vec(),
            scales,
            k,
        })
    }
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, KnnError> {
        validate(x, self.scales.len())?;
        x.iter()
            .map(|row| {
                let q: Vec<f64> = row.iter().zip(&self.scales).map(|(v, s)| v / s).collect();
                if q.iter()
                    .zip(row)
                    .any(|(v, original)| !v.is_finite() || (*original != 0.0 && *v == 0.0))
                {
                    return Err(KnnError::NumericOverflow);
                }
                let mut distances = Vec::with_capacity(self.training.len());
                for (i, train) in self.training.iter().enumerate() {
                    let d = q
                        .iter()
                        .zip(train)
                        .fold(0.0_f64, |acc, (a, b)| acc.hypot(a - b));
                    if !d.is_finite() {
                        return Err(KnnError::NumericOverflow);
                    }
                    distances.push((d, i));
                }
                distances.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                // Normalize targets before summation to avoid overflowing a finite mean.
                let scale = distances[..self.k]
                    .iter()
                    .map(|(_, i)| self.targets[*i].abs())
                    .fold(0.0_f64, f64::max);
                if scale == 0.0 {
                    return Ok(0.0);
                }
                // Neumaier compensation retains small terms during cancellation.
                // The normalized sum is bounded by k, so it cannot overflow.
                let mut sum = 0.0_f64;
                let mut correction = 0.0_f64;
                for (_, i) in &distances[..self.k] {
                    let value = self.targets[*i] / scale;
                    if self.targets[*i] != 0.0 && value == 0.0 {
                        return Err(KnnError::NumericOverflow);
                    }
                    let next = sum + value;
                    correction += if sum.abs() >= value.abs() {
                        (sum - next) + value
                    } else {
                        (value - next) + sum
                    };
                    sum = next;
                }
                let mean = ((sum + correction) / self.k as f64).clamp(-1.0, 1.0) * scale;
                Ok(mean)
            })
            .collect()
    }
}
fn validate(x: &[Vec<f64>], width: usize) -> Result<(), KnnError> {
    if x.is_empty() {
        return Err(KnnError::Empty);
    }
    if width == 0 || x.iter().any(|r| r.len() != width) {
        return Err(KnnError::Shape);
    }
    if x.len() > 20_000 || width > 100 {
        return Err(KnnError::Capacity);
    }
    if x.iter().flatten().any(|v| !v.is_finite()) {
        return Err(KnnError::NonFinite);
    }
    Ok(())
}
