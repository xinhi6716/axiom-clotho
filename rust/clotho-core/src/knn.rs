//! Dependency-free, numeric-only k-nearest-neighbor regression baseline.
//! Scaling is fitted on training features only. This is not a scientific
//! validation or a replacement for grouped evaluation/model selection.
use std::fmt;

#[path = "knn_exact_mean.rs"]
mod exact_mean;

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
    // Keep raw coordinates: separately rounded scaled values can erase a
    // real difference and invent a nearest-neighbor tie.
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
        // Keep validating original-to-scaled underflow, but the model
        // stores original coordinates, not a second normalized table.
        if x.iter().any(|row| {
            row.iter()
                .zip(&scales)
                .any(|(&value, &scale)| value != 0.0 && value / scale == 0.0)
        }) {
            return Err(KnnError::NumericOverflow);
        }
        Ok(Self {
            training: x.to_vec(),
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
                    let mut d = 0.0_f64;
                    for ((a, b), scale) in row.iter().zip(train).zip(&self.scales) {
                        let direct = a - b;
                        let delta = if direct.is_finite() {
                            direct / scale
                        } else {
                            // Opposite extreme values may overflow subtraction
                            // while their scaled difference remains representable.
                            a / scale - b / scale
                        };
                        if !delta.is_finite() || (a != b && delta == 0.0) {
                            return Err(KnnError::NumericOverflow);
                        }
                        d = d.hypot(delta);
                    }
                    if !d.is_finite() {
                        return Err(KnnError::NumericOverflow);
                    }
                    distances.push((d, i));
                }
                let neighbors = nearest_subset(&mut distances, self.k);
                exact_mean::checked_mean(neighbors.iter().map(|(_, i)| self.targets[*i]), self.k)
            })
            .collect()
    }
}
// Only membership matters: the exact mean and its normalization check
// are independent of target order. Distances arrive in original row order.
fn nearest_subset(distances: &mut [(f64, usize)], k: usize) -> &[(f64, usize)] {
    if k == distances.len() {
        return distances;
    }
    // Fit guarantees 1 <= k; the shorter request leaves at least two rows.
    if distances[0].0.total_cmp(&distances[1].0).is_le() {
        // Equal distances already follow the training index tie-break.
        if distances.windows(2).all(|w| w[0].0.total_cmp(&w[1].0).is_le()) {
            return &distances[..k];
        }
    } else if distances.windows(2).all(|w| w[0].0.total_cmp(&w[1].0).is_gt()) {
        // Strict decrease only: a reversed tied run would change membership.
        return &distances[distances.len() - k..];
    }
    distances.select_nth_unstable_by(k - 1, |a, b| {
        a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
    });
    &distances[..k]
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
