//! Dependency-free, unweighted metrics for binary labels and positive-class
//! probabilities. These summaries do not establish calibration, independence,
//! or scientific validity, and do not select a decision threshold.

use std::fmt;

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
/// changed, and errors do not include raw row values or indices.
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
        brier: brier_sum.total() / count,
        roc_auc: ranked_auc(y_true, probability, n_positive, n_negative),
        n_positive,
        n_negative,
        tp,
        fp,
        tn,
        fn_,
    })
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
    let mut ranked: Vec<_> = probability
        .iter()
        .copied()
        .zip(y_true.iter().map(|&label| label == 1.0))
        .collect();
    // All probabilities have been validated as finite. Signed zero compares
    // equal when collecting ties, even though total_cmp orders its two forms.
    ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    let mut start = 0;
    let mut negatives_before = 0_usize;
    let mut concordance = CompensatedSum::default();
    while start < ranked.len() {
        let mut end = start;
        let mut positives = 0_usize;
        while end < ranked.len() && ranked[end].0 == ranked[start].0 {
            positives += usize::from(ranked[end].1);
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
