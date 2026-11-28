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
        if self.k == self.training.len() {
            // The whole roster always contributes, but distance guards
            // remain part of prediction and still run for every query.
            let mut rows = x.iter();
            let first = rows.next().ok_or(KnnError::Empty)?;
            self.validate_query_scaling(first)?;
            self.visit_distances(first, |_, _| {})?;
            let mean = exact_mean::checked_mean(self.targets.iter().copied(), self.k)?;
            for row in rows {
                self.validate_query_scaling(row)?;
                self.visit_distances(row, |_, _| {})?;
            }
            return Ok(vec![mean; x.len()]);
        }
        if self.k == 1 {
            return x
                .iter()
                .map(|row| {
                    self.validate_query_scaling(row)?;
                    let mut best_distance = f64::INFINITY;
                    let mut best_index = 0;
                    // Still validate every distance, even after a perfect
                    // match. Strict improvement preserves the first-row tie.
                    self.visit_distances(row, |index, distance| {
                        if distance < best_distance {
                            best_distance = distance;
                            best_index = index;
                        }
                    })?;
                    exact_mean::checked_mean(std::iter::once(self.targets[best_index]), 1)
                })
                .collect();
        }
        if self.k == 2 && self.training.len() >= 8 {
            return self.predict_two(x);
        }
        // Every query needs fresh distances, but their capacity depends only
        // on training size. Allocate lazily after the first scaling check.
        let mut distance_workspace = None;
        x.iter()
            .map(|row| {
                self.validate_query_scaling(row)?;
                let distances = distance_workspace
                    .get_or_insert_with(|| Vec::with_capacity(self.training.len()));
                distances.clear();
                // Finite Euclidean distances are nonnegative and use +0.
                // Binary64 bits therefore have exactly the same order.
                self.visit_distances(row, |i, d| distances.push((d.to_bits(), i)))?;
                let neighbors = nearest_subset(distances, self.k);
                exact_mean::checked_mean(neighbors.iter().map(|(_, i)| self.targets[*i]), self.k)
            })
            .collect()
    }

    #[inline(never)]
    fn predict_two(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, KnnError> {
        x.iter()
            .map(|row| {
                self.validate_query_scaling(row)?;
                let mut first = (f64::INFINITY, 0);
                let mut second = first;
                // All visits still run, including distances unable to win.
                // Strict comparisons keep older rows ahead on exact ties.
                self.visit_distances(row, |index, distance| {
                    if distance < first.0 {
                        second = first;
                        first = (distance, index);
                    } else if distance < second.0 {
                        second = (distance, index);
                    }
                })?;
                exact_mean::checked_mean([self.targets[first.1], self.targets[second.1]].into_iter(), 2)
            })
            .collect()
    }

    fn validate_query_scaling(&self, row: &[f64]) -> Result<(), KnnError> {
        for (&value, &scale) in row.iter().zip(&self.scales) {
            let scaled = value / scale;
            if !scaled.is_finite() || (value != 0.0 && scaled == 0.0) {
                return Err(KnnError::NumericOverflow);
            }
        }
        Ok(())
    }

    fn visit_distances(
        &self,
        row: &[f64],
        mut visit: impl FnMut(usize, f64),
    ) -> Result<(), KnnError> {
        for (i, train) in self.training.iter().enumerate() {
            // For a finite first component, hypot(0, delta) is exactly
            // abs(delta). Start there; remaining components retain hypot's
            // scaling and evaluation order.
            let mut d = distance_component(row[0], train[0], self.scales[0])?.abs();
            for ((&a, &b), &scale) in row[1..].iter().zip(&train[1..]).zip(&self.scales[1..]) {
                d = d.hypot(distance_component(a, b, scale)?);
            }
            if !d.is_finite() {
                return Err(KnnError::NumericOverflow);
            }
            visit(i, d);
        }
        Ok(())
    }
}
fn distance_component(a: f64, b: f64, scale: f64) -> Result<f64, KnnError> {
    let direct = a - b;
    let delta = if direct.is_finite() {
        direct / scale
    } else {
        // Opposite extreme values may overflow subtraction even though
        // the scaled difference remains representable.
        a / scale - b / scale
    };
    if !delta.is_finite() || (a != b && delta == 0.0) {
        return Err(KnnError::NumericOverflow);
    }
    Ok(delta)
}

// Only membership matters: the exact mean and its normalization check
// are independent of target order. Distances arrive in original row order.
fn nearest_subset(distances: &mut [(u64, usize)], k: usize) -> &[(u64, usize)] {
    if k == distances.len() {
        return distances;
    }
    // Fit guarantees 1 <= k; the shorter request leaves at least two rows.
    if distances[0].0 <= distances[1].0 {
        // Equal distances already follow the training index tie-break.
        if distances.windows(2).all(|w| w[0].0 <= w[1].0) {
            return &distances[..k];
        }
    } else if distances.windows(2).all(|w| w[0].0 > w[1].0) {
        // Strict decrease only: a reversed tied run would change membership.
        return &distances[distances.len() - k..];
    }
    distances.select_nth_unstable(k - 1);
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

#[cfg(test)]
mod stable_two_tests {
    use super::*;

    #[test]
    fn streaming_pair_keeps_training_order_on_ties_and_checks_late_distances() {
        let training = [[-1.0], [1.0], [1.0], [-1.0], [-1.0], [1.0], [1.0], [-1.0]].map(Vec::from);
        let model = KnnRegressor::fit(&training, &[2.0, 4.0, 100.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2).unwrap();
        assert_eq!(model.predict(&[vec![0.0]]).unwrap(), [3.0]);

        let query = 1e-23_f64;
        let late = f64::from_bits(query.to_bits() + 1);
        let training = [[query], [query], [1e300], [query], [query], [query], [query], [late]].map(Vec::from);
        let model = KnnRegressor::fit(&training, &[1.0; 8], 2).unwrap();
        assert_eq!(model.predict(&[vec![query]]), Err(KnnError::NumericOverflow));
    }
}
