//! Bounded, borrowed little-endian binary64 columns for numeric diagnostics.
//!
//! NaNs encode missing values in this API only. Infinities are rejected. Values
//! are decoded one at a time with safe byte copies, without decoded column buffers.

use crate::input_support::{
    summarize_row_flags, FeatureCounts, FeatureSupport, InputSupport, TrainingSupport,
    MAX_FEATURES, MAX_ROWS, MISSING, OUTSIDE, UNSUPPORTED,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct NumericBytesColumn<'a> {
    pub name: &'a str,
    pub values: &'a [u8],
}

/// Fixed errors disclose neither values nor feature names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericBytesError {
    FeatureCount,
    RowCount,
    InvalidByteLength,
    RaggedColumns,
    DuplicateNames,
    SchemaMismatch,
    InfiniteNumeric,
}

impl fmt::Display for NumericBytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FeatureCount => "Expected 1 to 100 source features.",
            Self::RowCount => "Expected 1 to 20,000 rows in each table.",
            Self::InvalidByteLength => {
                "Columns must contain complete little-endian binary64 values."
            }
            Self::RaggedColumns => "Columns must have the same row count.",
            Self::DuplicateNames => "Feature names must be unique.",
            Self::SchemaMismatch => "Prediction feature names must exactly match training.",
            Self::InfiniteNumeric => "Numeric values must be finite or NaN for missing.",
        })
    }
}
impl std::error::Error for NumericBytesError {}

fn validate_table(columns: &[NumericBytesColumn<'_>]) -> Result<usize, NumericBytesError> {
    if !(1..=MAX_FEATURES).contains(&columns.len()) {
        return Err(NumericBytesError::FeatureCount);
    }
    let bytes = columns[0].values.len();
    let mut names = HashSet::new();
    for column in columns {
        if column.values.len() % 8 != 0 {
            return Err(NumericBytesError::InvalidByteLength);
        }
        if !(1..=MAX_ROWS).contains(&(column.values.len() / 8)) {
            return Err(NumericBytesError::RowCount);
        }
        if column.values.len() != bytes {
            return Err(NumericBytesError::RaggedColumns);
        }
        if !names.insert(column.name) {
            return Err(NumericBytesError::DuplicateNames);
        }
    }
    Ok(bytes / 8)
}

fn decode(chunk: &[u8]) -> f64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(chunk);
    f64::from_le_bytes(bytes)
}

fn summarize_training(
    bytes: &[u8],
    rows: usize,
) -> Result<(usize, Option<(f64, f64)>), NumericBytesError> {
    let mut missing = 0;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for chunk in bytes.chunks_exact(8) {
        let value = decode(chunk);
        if value.is_finite() {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        } else if value.is_nan() {
            missing += 1;
        } else {
            return Err(NumericBytesError::InfiniteNumeric);
        }
    }
    Ok((missing, (missing != rows).then_some((minimum, maximum))))
}

#[inline]
fn compare_bounded<const BRANCHLESS: bool>(
    bytes: &[u8],
    minimum: f64,
    maximum: f64,
    flags: &mut [u8],
) -> Result<(usize, usize, usize), NumericBytesError> {
    let mut missing = 0;
    let mut below = 0;
    let mut above = 0;
    for (chunk, flag) in bytes.chunks_exact(8).zip(flags.iter_mut()) {
        let value = decode(chunk);
        if value.is_finite() {
            if BRANCHLESS {
                let low = value < minimum;
                let high = value > maximum;
                below += usize::from(low);
                above += usize::from(high);
                *flag |= u8::from(low | high) * OUTSIDE;
            } else if value < minimum {
                below += 1;
                *flag |= OUTSIDE;
            } else if value > maximum {
                above += 1;
                *flag |= OUTSIDE;
            }
        } else if value.is_nan() {
            missing += 1;
            *flag |= MISSING;
        } else {
            return Err(NumericBytesError::InfiniteNumeric);
        }
    }
    Ok((missing, below, above))
}

fn compare_column(
    bytes: &[u8],
    bounds: Option<(f64, f64)>,
    flags: &mut [u8],
) -> Result<(usize, usize, FeatureCounts), NumericBytesError> {
    let mut missing = 0;
    let mut unsupported = 0;
    let mut below = 0;
    let mut above = 0;
    // Training support is column-level state. Dispatch once per column rather
    // than checking it on every finite query value.
    if let Some((minimum, maximum)) = bounds {
        // Most deployment inputs fall inside the training bounds. Retain the
        // no-write loop for that case and only use branchless counts when a
        // cheap, evenly spaced probe finds frequently out-of-range values.
        // It chooses the kernel only; both loops still validate every value.
        let rows = bytes.len() / 8;
        let mixed = rows >= 1024
            && (0..32)
                .filter(|i| {
                    let start = i * (rows / 32) * 8;
                    let value = decode(&bytes[start..start + 8]);
                    value.is_finite() && ((value < minimum) | (value > maximum))
                })
                .take(5)
                .count() == 5;
        (missing, below, above) = if mixed {
            compare_bounded::<true>(bytes, minimum, maximum, flags)?
        } else {
            compare_bounded::<false>(bytes, minimum, maximum, flags)?
        };
    } else {
        for (chunk, flag) in bytes.chunks_exact(8).zip(flags.iter_mut()) {
            let value = decode(chunk);
            if value.is_finite() {
                unsupported += 1;
                *flag |= UNSUPPORTED;
            } else if value.is_nan() {
                missing += 1;
                *flag |= MISSING;
            } else {
                return Err(NumericBytesError::InfiniteNumeric);
            }
        }
    }
    Ok((
        missing,
        unsupported,
        FeatureCounts::Numeric {
            below_training_min_count: bounds.map(|_| below),
            above_training_max_count: bounds.map(|_| above),
        },
    ))
}

/// Compare numeric source-feature support without modifying inputs.
///
/// Both tables must contain 1..=100 uniquely named columns and 1..=20,000
/// rows. Names must match exactly; incoming columns may be reordered. All
/// lengths and schemas are validated before counting. NaN payloads and signs
/// are all missing; infinities are errors. No categorical conversion occurs.
/// Returned counts are marginal diagnostics, not calibrated uncertainty or an
/// admission gate. The original Option-based API keeps its own NaN rejection.
pub fn prediction_input_support_bytes(
    training: &[NumericBytesColumn<'_>],
    incoming: &[NumericBytesColumn<'_>],
) -> Result<InputSupport, NumericBytesError> {
    let n_training_rows = validate_table(training)?;
    let n_prediction_rows = validate_table(incoming)?;
    let by_name: HashMap<_, _> = incoming.iter().map(|c| (c.name, c.values)).collect();
    if training.len() != incoming.len() || training.iter().any(|c| !by_name.contains_key(c.name)) {
        return Err(NumericBytesError::SchemaMismatch);
    }
    let mut flags = vec![0u8; n_prediction_rows];
    let mut features = Vec::with_capacity(training.len());
    for source in training {
        let (training_missing_count, bounds) = summarize_training(source.values, n_training_rows)?;
        let (prediction_missing_count, unsupported, counts) =
            compare_column(by_name[source.name], bounds, &mut flags)?;
        features.push(FeatureSupport {
            name: source.name.to_owned(),
            training_missing_count,
            training_support: if bounds.is_none() {
                TrainingSupport::UnavailableAllMissing
            } else {
                TrainingSupport::ObservedNonMissingValues
            },
            prediction_missing_count,
            prediction_non_missing_without_training_values_count: unsupported,
            counts,
        });
    }
    let (rows_with_any_flag, row_flag_counts) = summarize_row_flags(&flags);
    Ok(InputSupport {
        schema_version: "1.0",
        reference: "full_training_features_before_imputation",
        action: "diagnostic_only_no_input_changes",
        n_training_rows,
        n_prediction_rows,
        rows_with_any_flag,
        row_flag_counts,
        features,
    })
}

/// Owned numeric training summary for repeated, explicit diagnostic queries.
///
/// Construction scans validated training bytes once, retaining only names,
/// missing counts and extrema, never input buffers. No targets or fitted model
/// state are consumed. There is no serialization or automatic global cache.
/// Fields are private and no Debug implementation exposes the stored extrema.
pub struct NumericSupportReference {
    n_training_rows: usize,
    features: Vec<NumericReferenceFeature>,
}

struct NumericReferenceFeature {
    name: String,
    missing: usize,
    bounds: Option<(f64, f64)>,
}

impl NumericSupportReference {
    /// Snapshot the training support. Infinities fail before a reference exists.
    pub fn from_training(training: &[NumericBytesColumn<'_>]) -> Result<Self, NumericBytesError> {
        let n_training_rows = validate_table(training)?;
        let mut features = Vec::with_capacity(training.len());
        for source in training {
            let (missing, bounds) = summarize_training(source.values, n_training_rows)?;
            features.push(NumericReferenceFeature {
                name: source.name.to_owned(),
                missing,
                bounds,
            });
        }
        Ok(Self {
            n_training_rows,
            features,
        })
    }

    /// Compare one bounded batch against the original immutable reference.
    ///
    /// Query rows never update or expand it. Names must match exactly but may
    /// arrive in a different order. Each returned report counts only this batch;
    /// a previous successful or failed call cannot affect the result.
    pub fn compare(
        &self,
        incoming: &[NumericBytesColumn<'_>],
    ) -> Result<InputSupport, NumericBytesError> {
        let n_prediction_rows = validate_table(incoming)?;
        let by_name: HashMap<_, _> = incoming.iter().map(|c| (c.name, c.values)).collect();
        if self.features.len() != incoming.len()
            || self
                .features
                .iter()
                .any(|c| !by_name.contains_key(c.name.as_str()))
        {
            return Err(NumericBytesError::SchemaMismatch);
        }
        let mut flags = vec![0u8; n_prediction_rows];
        let mut features = Vec::with_capacity(self.features.len());
        for source in &self.features {
            let (prediction_missing_count, unsupported, counts) =
                compare_column(by_name[source.name.as_str()], source.bounds, &mut flags)?;
            features.push(FeatureSupport {
                name: source.name.clone(),
                training_missing_count: source.missing,
                training_support: if source.bounds.is_none() {
                    TrainingSupport::UnavailableAllMissing
                } else {
                    TrainingSupport::ObservedNonMissingValues
                },
                prediction_missing_count,
                prediction_non_missing_without_training_values_count: unsupported,
                counts,
            });
        }
        let (rows_with_any_flag, row_flag_counts) = summarize_row_flags(&flags);
        Ok(InputSupport {
            schema_version: "1.0",
            reference: "full_training_features_before_imputation",
            action: "diagnostic_only_no_input_changes",
            n_training_rows: self.n_training_rows,
            n_prediction_rows,
            rows_with_any_flag,
            row_flag_counts,
            features,
        })
    }
}
