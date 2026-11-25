//! Dependency-free, unweighted metrics for binary labels and positive-class
//! probabilities. These summaries do not establish calibration, independence,
//! or scientific validity, and do not select a decision threshold.

use std::{cmp::Ordering, fmt};

/// Probabilities used for log loss are clipped to
/// `[LOG_LOSS_EPSILON, 1 - LOG_LOSS_EPSILON]`.
///
/// This fixed value, rather than a platform-dependent tolerance, makes losses
/// finite for probabilities exactly zero or one. Clipping affects only log
/// loss, not the decision threshold, Brier score, or ROC AUC.
pub const LOG_LOSS_EPSILON: f64 = 1e-15;

/// Invalid inputs to [`binary_metrics`]. Errors contain no individual values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationMetricsError {
    /// No observations were supplied.
    Empty,
    /// Label and probability lengths differ.
    Shape,
    /// A label is not exactly zero or one, or is not finite.
    InvalidLabel,
    /// A probability is not finite or is outside the closed unit interval.
    InvalidProbability,
    /// The threshold is not finite or is outside the closed unit interval.
    InvalidThreshold,
    /// At least one probability error is nonzero, but its mean square rounded to zero.
    BrierUnderflow,
}

impl fmt::Display for ClassificationMetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "Classification inputs must not be empty.",
            Self::Shape => "Label and probability lengths must match.",
            Self::InvalidLabel => "Binary labels must be finite and exactly zero or one.",
            Self::InvalidProbability => "Probabilities must be finite and between zero and one.",
            Self::InvalidThreshold => {
                "The decision threshold must be finite and between zero and one."
            }
            Self::BrierUnderflow => {
                "Brier score underflowed to zero despite a nonzero probability error."
            }
        })
    }
}

impl std::error::Error for ClassificationMetricsError {}

/// Unweighted binary classification metrics, with one as the positive class.
///
/// Undefined ratios are represented by `None`, not an artificial zero or one.
/// The confusion counts sum to `n_positive + n_negative`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryMetrics {
    /// Fraction of labels predicted correctly, `(tp + tn) / n`.
    pub accuracy: f64,
    /// Mean of sensitivity and specificity. `None` when either actual class
    /// is absent, including single-row inputs.
    pub balanced_accuracy: Option<f64>,
    /// `tp / (tp + fp)`. `None` when no row is predicted positive.
    pub precision: Option<f64>,
    /// `tp / (tp + fn_)`. `None` when no actual positive label is present.
    pub recall: Option<f64>,
    /// `2 * tp / (2 * tp + fp + fn_)`. `None` only when there are neither
    /// actual nor predicted positives. Zero when positive predictions or
    /// labels exist but none are correctly identified.
    pub f1: Option<f64>,
    /// Mean negative log likelihood, using natural logarithms and probabilities
    /// clipped with [`LOG_LOSS_EPSILON`]. Even perfect endpoint predictions
    /// therefore have a small positive log loss.
    pub log_loss: f64,
    /// Mean squared error of the original, unclipped probabilities.
    /// A returned zero requires all probabilities to equal their labels.
    pub brier: f64,
    /// Area under the receiver operating characteristic curve. Tied scores
    /// receive average ranks (equivalently, half credit for a tied pair).
    /// `None` when either actual class is absent. Uses original probabilities,
    /// independent of the decision threshold.
    pub roc_auc: Option<f64>,
    /// Number of actual positive labels.
    pub n_positive: usize,
    /// Number of actual negative labels.
    pub n_negative: usize,
    /// Number of actual positives predicted positive.
    pub tp: usize,
    /// Number of actual negatives predicted positive.
    pub fp: usize,
    /// Number of actual negatives predicted negative.
    pub tn: usize,
    /// Number of actual positives predicted negative. The trailing underscore
    /// avoids Rust's reserved `fn` keyword.
    pub fn_: usize,
}

/// Evaluate binary labels against probabilities of the positive class.
///
/// Inputs must be nonempty and equally sized. Labels must be exactly `0.0` or
/// `1.0`; signed zero is accepted as zero. Probabilities and the threshold must
/// be finite and in `[0, 1]`. A row is predicted positive when its original
/// probability is **greater than or equal to** `threshold`. Thus threshold
/// zero predicts every row positive, while threshold one predicts positive
/// only for probabilities exactly one.
///
/// The function uses `O(n log n)` time and `O(n)` additional storage for
/// average-rank ROC AUC. Metric sums use compensated addition. No inputs are
/// changed, and errors do not include raw row values or indices. A Brier score
/// that rounds to zero despite a nonzero error returns
/// [`ClassificationMetricsError::BrierUnderflow`]. Preliminary Brier results
/// at most twice the smallest normal are checked by summing exact squares.
/// The exact mean is rounded once when it falls in the minimum-subnormal
/// spacing range; other scores use compensated addition.
pub fn binary_metrics(
    y_true: &[f64],
    probability: &[f64],
    threshold: f64,
) -> Result<BinaryMetrics, ClassificationMetricsError> {
    if y_true.len() != probability.len() {
        return Err(ClassificationMetricsError::Shape);
    }
    if y_true.is_empty() {
        return Err(ClassificationMetricsError::Empty);
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(ClassificationMetricsError::InvalidThreshold);
    }
    if y_true.iter().any(|&label| label != 0.0 && label != 1.0) {
        return Err(ClassificationMetricsError::InvalidLabel);
    }
    if probability
        .iter()
        .any(|&value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(ClassificationMetricsError::InvalidProbability);
    }

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut tn = 0_usize;
    let mut fn_ = 0_usize;
    let mut loss_sum = CompensatedSum::default();
    let mut brier_sum = CompensatedSum::default();

    for (&label, &value) in y_true.iter().zip(probability) {
        match (label == 1.0, value >= threshold) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
        let clipped = value.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON);
        loss_sum.add(if label == 1.0 {
            -clipped.ln()
        } else {
            -(-clipped).ln_1p()
        });
        let error = value - label;
        brier_sum.add(error * error);
    }

    let n_positive = tp + fn_;
    let n_negative = tn + fp;
    let count = y_true.len() as f64;
    let mut brier = brier_sum.total() / count;
    // Do not use the smallest normal itself as the preliminary cutoff:
    // rounded squares can place an exact boundary result just above it.
    if brier <= 2.0 * f64::MIN_POSITIVE && y_true.iter().zip(probability).any(|(y, p)| y != p) {
        brier = tiny_brier_exact(y_true, probability).unwrap_or(brier);
        if brier == 0.0 {
            return Err(ClassificationMetricsError::BrierUnderflow);
        }
    }
    let recall = ratio(tp as f64, n_positive as f64);
    let specificity = ratio(tn as f64, n_negative as f64);
    let balanced_accuracy = recall
        .zip(specificity)
        .map(|(sensitivity, specificity)| (sensitivity + specificity) * 0.5);
    // Convert before doubling or multiplying counts, avoiding usize overflow.
    let twice_tp = 2.0 * tp as f64;

    Ok(BinaryMetrics {
        accuracy: (tp + tn) as f64 / count,
        balanced_accuracy,
        precision: ratio(tp as f64, (tp + fp) as f64),
        recall,
        f1: ratio(twice_tp, twice_tp + fp as f64 + fn_ as f64),
        log_loss: loss_sum.total() / count,
        brier,
        roc_auc: ranked_auc(y_true, probability, n_positive, n_negative),
        n_positive,
        n_negative,
        tp,
        fp,
        tn,
        fn_,
    })
}

// Integer units are 2^-2148, the square of the minimum subnormal. Even a
// sum of (usize::MAX) unit probability errors needs at most 2212 bits on a
// 64-bit target, so adding exact 106-bit squares cannot exhaust 35 limbs.
fn tiny_brier_exact(labels: &[f64], probabilities: &[f64]) -> Option<f64> {
    let mut sum = [0_u64; 35];
    for (&label, &p) in labels.iter().zip(probabilities) {
        // If a positive label has a nonzero error, its magnitude is at least
        // 2^-53 even before dividing by n, far outside this tiny-score path.
        if label == 1.0 && p != 1.0 {
            return None;
        }
        if label == 1.0 || p == 0.0 {
            continue;
        }
        let bits = p.to_bits();
        let exponent = (bits >> 52) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let mantissa = u128::from(if exponent == 0 { fraction } else { fraction | (1_u64 << 52) });
        let square = mantissa * mantissa;
        let shift = 2 * exponent.saturating_sub(1);
        add_shifted(&mut sum, square as u64, shift);
        add_shifted(&mut sum, (square >> 64) as u64, shift + 64);
    }

    // Divide by n * 2^1074 to convert to minimum-subnormal quanta. Split
    // above bit 1074 (16*64+50), leaving the low suffix for midpoint checks.
    if sum[18] >> 50 != 0 || sum[19..].iter().any(|&word| word != 0) {
        return None;
    }
    let high = u128::from(sum[16] >> 50)
        | (u128::from(sum[17]) << 14)
        | (u128::from(sum[18]) << 78);
    let n = labels.len() as u128;
    let quotient = high / n;
    let twice_remainder = 2 * (high % n);
    let suffix_top = sum[16] & ((1_u64 << 50) - 1);
    let lower_nonzero = || sum[..16].iter().any(|&word| word != 0);
    let midpoint = match twice_remainder.cmp(&n) {
        Ordering::Equal if suffix_top != 0 || lower_nonzero() => Ordering::Greater,
        Ordering::Less if twice_remainder + 1 == n => match suffix_top.cmp(&(1_u64 << 49)) {
            Ordering::Equal if lower_nonzero() => Ordering::Greater,
            other => other,
        },
        other => other,
    };
    let rounds_up =
        midpoint == Ordering::Greater || (midpoint == Ordering::Equal && quotient % 2 != 0);
    let rounded = quotient + u128::from(rounds_up);

    // The minimum-subnormal bit spacing persists through the first normal
    // binade. This defensive check keeps the specialized conversion scoped
    // to that spacing, including a carry into the next binade's first value.
    (rounded <= 1_u128 << 53).then(|| f64::from_bits(rounded as u64))
}

fn add_shifted(sum: &mut [u64; 35], word: u64, shift: usize) {
    let index = shift / 64;
    let intra = shift % 64;
    add_word(sum, word << intra, index);
    if intra != 0 {
        add_word(sum, word >> (64 - intra), index + 1);
    }
}

fn add_word(sum: &mut [u64; 35], mut word: u64, mut index: usize) {
    while word != 0 {
        let (value, carry) = sum[index].overflowing_add(word);
        sum[index] = value;
        word = u64::from(carry);
        index += 1;
    }
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn ranked_auc(
    y_true: &[f64],
    probability: &[f64],
    n_positive: usize,
    n_negative: usize,
) -> Option<f64> {
    if n_positive == 0 || n_negative == 0 {
        return None;
    }
    // A validated probability can have its sign bit set only for -0.0.
    // Shifting drops that sign (so both zeros tie), preserves all 63 score
    // bits in their numeric order, and leaves a low bit for the binary label.
    let mut ranked: Vec<u64> = probability
        .iter()
        .zip(y_true)
        .map(|(&score, &label)| (score.to_bits() << 1) | u64::from(label == 1.0))
        .collect();
    // Compare only score bits to retain the sort's fast path for an
    // already ordered sequence that includes ties with differing labels.
    ranked.sort_unstable_by(|left, right| (left >> 1).cmp(&(right >> 1)));

    let mut start = 0;
    let mut negatives_before = 0_usize;
    let mut concordance = CompensatedSum::default();
    while start < ranked.len() {
        let mut end = start;
        let mut positives = 0_usize;
        let score_key = ranked[start] >> 1;
        while end < ranked.len() && ranked[end] >> 1 == score_key {
            positives += (ranked[end] & 1) as usize;
            end += 1;
        }
        let negatives = end - start - positives;
        // Equivalent to assigning each tied observation its average rank and
        // subtracting the positive-class rank baseline. Counting concordant
        // pairs directly avoids subtracting nearly equal large rank sums.
        concordance.add(positives as f64 * (negatives_before as f64 + 0.5 * negatives as f64));
        negatives_before += negatives;
        start = end;
    }
    let pairs = n_positive as f64 * n_negative as f64;
    Some((concordance.total() / pairs).clamp(0.0, 1.0))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brier_tiny_mean_rounds_once_instead_of_rounding_squares_first() {
        let scale = 2.0_f64.powi(-537);
        let actual = binary_metrics(&[0.0, 0.0], &[1.125 * scale, 0.0], 0.5).unwrap();
        assert_eq!(actual.brier.to_bits(), 1); // Exact mean: 81/128 min-subnormal.

        assert_eq!(
            binary_metrics(&[0.0, 0.0, 1.0], &[0.75 * scale, 0.75 * scale, 1.0], 0.5),
            Err(ClassificationMetricsError::BrierUnderflow) // Exact mean: 3/8 min-subnormal.
        );
    }

    #[test]
    fn brier_tiny_mean_resolves_midpoints_with_even_parity() {
        let scale = 2.0_f64.powi(-537);
        let below = f64::from_bits(scale.to_bits() - 1);
        let above = f64::from_bits(scale.to_bits() + 1);
        for p in [below, scale] {
            assert_eq!(
                binary_metrics(&[0.0, 0.0], &[p, 0.0], 0.5),
                Err(ClassificationMetricsError::BrierUnderflow)
            );
        }
        assert_eq!(
            binary_metrics(&[0.0, 0.0], &[above, 0.0], 0.5)
                .unwrap()
                .brier
                .to_bits(),
            1
        );
        // An odd denominator puts the midpoint inside the low integer suffix.
        let half = 0.5 * scale;
        let prev_half = f64::from_bits(half.to_bits() - 1);
        let next_half = f64::from_bits(half.to_bits() + 1);
        for p in [prev_half, half] {
            assert_eq!(
                binary_metrics(&[0.0; 3], &[p, scale, half], 0.5),
                Err(ClassificationMetricsError::BrierUnderflow)
            );
        }
        let odd = binary_metrics(&[0.0; 3], &[next_half, scale, half], 0.5).unwrap();
        assert_eq!(odd.brier.to_bits(), 1);

        // Other ties round toward the even bit pattern, in either direction.
        for (probabilities, expected) in [
            ([1.5 * scale, half, half, half], 1),
            ([2.0 * scale, scale, scale, 0.0], 2), // 6 / 4 = 1.5
            ([3.0 * scale, scale, 0.0, 0.0], 2), // 10 / 4 = 2.5
        ] {
            let actual = binary_metrics(&[0.0; 4], &probabilities, 0.5).unwrap();
            assert_eq!(actual.brier.to_bits(), expected);
        }
    }

    #[test]
    fn brier_tiny_mean_preserves_the_subnormal_normal_boundary() {
        let root_min = 2.0_f64.powi(-511);
        for (p, expected) in [
            (
                f64::from_bits(root_min.to_bits() - 1),
                f64::MIN_POSITIVE.to_bits() - 1,
            ),
            (root_min, f64::MIN_POSITIVE.to_bits()),
            (
                f64::from_bits(root_min.to_bits() + 1),
                f64::MIN_POSITIVE.to_bits() + 2,
            ),
        ] {
            assert_eq!(binary_metrics(&[0.0], &[p], 0.5).unwrap().brier.to_bits(), expected);
        }
    }

    #[test]
    fn brier_boundary_rechecks_a_preliminary_normal_successor() {
        let probabilities = [
            f64::from_bits(0x200b_b67a_e858_4caa),
            f64::from_bits(0x1e63_ad69_f7f3_f385),
            f64::from_bits(0x1e63_ad69_f7f3_f385),
        ];
        // Independently rounding the squares yields minimum normal + 1 ulp,
        // but their exact mean rounds to minimum normal.
        let actual = binary_metrics(&[0.0; 3], &probabilities, 0.5).unwrap();
        assert_eq!(actual.brier.to_bits(), f64::MIN_POSITIVE.to_bits());
    }

    #[test]
    fn brier_zero_distinguishes_perfect_predictions_from_underflow() {
        for (labels, probabilities) in [
            (&[0.0][..], &[1e-200][..]),
            (&[0.0, 1.0, 0.0][..], &[2.3e-162, 1.0, 0.0][..]),
        ] {
            assert_eq!(
                binary_metrics(labels, probabilities, 0.5),
                Err(ClassificationMetricsError::BrierUnderflow)
            );
        }

        let perfect = binary_metrics(&[0.0, 1.0, 0.0], &[-0.0, 1.0, 0.0], 0.5).unwrap();
        assert_eq!(perfect.brier.to_bits(), 0);
        assert!(perfect.log_loss > 0.0); // Only log loss clips the endpoints.

        let tiny = binary_metrics(&[0.0], &[2.3e-162], 0.5).unwrap();
        assert_eq!(tiny.brier.to_bits(), 1);

        let mixed = binary_metrics(&[0.0, 1.0], &[1e-200, 0.5], 0.5).unwrap();
        assert_eq!(mixed.brier, 0.125);
    }
}
