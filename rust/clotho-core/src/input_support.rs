//! Source-feature support counts against training data only.
//!
//! Callers must preserve explicit missing values and normalize categorical values
//! to strings before this boundary. Numeric values are already binary64; this
//! API cannot recover precision lost upstream or certify physical units.

use std::collections::{HashMap, HashSet};
use std::fmt;

pub const MAX_ROWS: usize = 20_000;
pub const MAX_FEATURES: usize = 100;

#[derive(Clone, Debug, PartialEq)]
pub enum FeatureValues {
    Numeric(Vec<Option<f64>>),
    Categorical(Vec<Option<String>>),
}

impl FeatureValues {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Categorical(values) => values.len(),
        }
    }

    fn all_missing(&self) -> bool {
        match self {
            Self::Numeric(values) => values.iter().all(Option::is_none),
            Self::Categorical(values) => values.iter().all(Option::is_none),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureColumn {
    pub name: String,
    pub values: FeatureValues,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingSupport {
    ObservedNonMissingValues,
    UnavailableAllMissing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeatureCounts {
    Numeric {
        below_training_min_count: Option<usize>,
        above_training_max_count: Option<usize>,
    },
    Categorical {
        unseen_category_count: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureSupport {
    pub name: String,
    pub training_missing_count: usize,
    pub training_support: TrainingSupport,
    pub prediction_missing_count: usize,
    pub prediction_non_missing_without_training_values_count: usize,
    pub counts: FeatureCounts,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RowFlagCounts {
    pub missing_value: usize,
    pub numeric_outside_training_range: usize,
    pub unseen_category: usize,
    pub non_missing_value_in_all_missing_training_feature: usize,
}

// Keep row flags and their aggregation shared with the borrowed byte API.
pub(crate) const MISSING: u8 = 1;
pub(crate) const OUTSIDE: u8 = 2;
pub(crate) const UNSEEN: u8 = 4;
pub(crate) const UNSUPPORTED: u8 = 8;

pub(crate) fn summarize_row_flags(flags: &[u8]) -> (usize, RowFlagCounts) {
    let mut any = 0;
    let mut counts = RowFlagCounts::default();
    for &flag in flags {
        any += usize::from(flag != 0);
        counts.missing_value += usize::from(flag & MISSING != 0);
        counts.numeric_outside_training_range += usize::from(flag & OUTSIDE != 0);
        counts.unseen_category += usize::from(flag & UNSEEN != 0);
        counts.non_missing_value_in_all_missing_training_feature +=
            usize::from(flag & UNSUPPORTED != 0);
    }
    (any, counts)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputSupport {
    pub schema_version: &'static str,
    pub reference: &'static str,
    pub action: &'static str,
    pub n_training_rows: usize,
    pub n_prediction_rows: usize,
    pub rows_with_any_flag: usize,
    pub row_flag_counts: RowFlagCounts,
    pub features: Vec<FeatureSupport>,
}

/// Errors contain no feature values, category inventories or row identifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportError {
    FeatureCount,
    RowCount,
    RaggedColumns,
    DuplicateNames,
    SchemaMismatch,
    TypeMismatch,
    NonFiniteNumeric,
    NulCategory,
}

impl fmt::Display for SupportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FeatureCount => "Expected 1 to 100 source features.",
            Self::RowCount => "Expected 1 to 20,000 rows in each table.",
            Self::RaggedColumns => "Columns must have the same row count.",
            Self::DuplicateNames => "Feature names must be unique.",
            Self::SchemaMismatch => "Prediction feature names must exactly match training.",
            Self::TypeMismatch => "Prediction feature types differ from training.",
            Self::NonFiniteNumeric => "Numeric values must be finite; use None for missing.",
            Self::NulCategory => "Categorical feature values cannot contain NUL.",
        })
    }
}

impl std::error::Error for SupportError {}

fn validate_table(columns: &[FeatureColumn]) -> Result<usize, SupportError> {
    if !(1..=MAX_FEATURES).contains(&columns.len()) {
        return Err(SupportError::FeatureCount);
    }
    let rows = columns[0].values.len();
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(SupportError::RowCount);
    }
    let mut names = HashSet::new();
    for column in columns {
        if !names.insert(column.name.as_str()) {
            return Err(SupportError::DuplicateNames);
        }
        if column.values.len() != rows {
            return Err(SupportError::RaggedColumns);
        }
        match &column.values {
            FeatureValues::Numeric(values) => {
                if values.iter().flatten().any(|value| !value.is_finite()) {
                    return Err(SupportError::NonFiniteNumeric);
                }
            }
            FeatureValues::Categorical(values) => {
                if values.iter().flatten().any(|value| value.contains('\0')) {
                    return Err(SupportError::NulCategory);
                }
            }
        }
    }
    Ok(rows)
}

enum CategorySupport<'a> {
    Single(&'a str),
    Set(HashSet<&'a str>),
}

fn summarize_categories(values: &[Option<String>]) -> (CategorySupport<'_>, usize) {
    let mut observed = values.iter().flatten().map(String::as_str);
    let Some(first) = observed.next() else {
        return (CategorySupport::Set(HashSet::new()), 0);
    };
    let mut non_missing = 1;
    while let Some(value) = observed.next() {
        non_missing += 1;
        if value != first {
            // Only materialize a vocabulary once a second category exists.
            // Earlier observations were identical to the first category.
            let mut categories = HashSet::new();
            categories.insert(first);
            categories.insert(value);
            for value in observed {
                non_missing += 1;
                categories.insert(value);
            }
            return (CategorySupport::Set(categories), non_missing);
        }
    }
    (CategorySupport::Single(first), non_missing)
}

/// Compare source-feature support without changing inputs or making predictions.
///
/// Names must match exactly; incoming columns may be reordered. An entirely
/// missing incoming column is accepted with either variant, matching Python's
/// all-missing type accommodation. No other type conversion is inferred.
/// Returned counts may overlap. They are marginal diagnostics, not an OOD test,
/// calibrated uncertainty, an admission gate, or evidence of future accuracy.
pub fn prediction_input_support(
    training: &[FeatureColumn],
    incoming: &[FeatureColumn],
) -> Result<InputSupport, SupportError> {
    let n_training_rows = validate_table(training)?;
    let n_prediction_rows = validate_table(incoming)?;
    let query_by_name: HashMap<_, _> = incoming
        .iter()
        .map(|column| (column.name.as_str(), &column.values))
        .collect();
    if training.len() != incoming.len()
        || training
            .iter()
            .any(|column| !query_by_name.contains_key(column.name.as_str()))
    {
        return Err(SupportError::SchemaMismatch);
    }

    // Bit flags keep aggregate row unions bounded by the prediction row limit.
    let mut flags = vec![0u8; n_prediction_rows];
    let mut features = Vec::with_capacity(training.len());
    for source in training {
        let query = query_by_name[source.name.as_str()];
        if std::mem::discriminant(&source.values) != std::mem::discriminant(query)
            && !query.all_missing()
        {
            return Err(SupportError::TypeMismatch);
        }

        let mut prediction_missing_count = 0;
        let mut unsupported_count = 0;
        let source_missing;
        let counts = match &source.values {
            FeatureValues::Numeric(values) => {
                let mut observed = 0;
                let mut minimum = f64::INFINITY;
                let mut maximum = f64::NEG_INFINITY;
                for &value in values.iter().flatten() {
                    observed += 1;
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                }
                source_missing = n_training_rows - observed;
                let mut below = 0;
                let mut above = 0;
                if let FeatureValues::Numeric(query_values) = query {
                    for (value, flag) in query_values.iter().zip(&mut flags) {
                        match value {
                            None => {
                                prediction_missing_count += 1;
                                *flag |= MISSING;
                            }
                            Some(_) if observed == 0 => {
                                unsupported_count += 1;
                                *flag |= UNSUPPORTED;
                            }
                            Some(value) => {
                                // Direct comparison avoids overflowing range subtraction.
                                if *value < minimum {
                                    below += 1;
                                    *flag |= OUTSIDE;
                                } else if *value > maximum {
                                    above += 1;
                                    *flag |= OUTSIDE;
                                }
                            }
                        }
                    }
                } else {
                    // The accepted alternate variant is entirely missing.
                    prediction_missing_count = n_prediction_rows;
                    flags.iter_mut().for_each(|flag| *flag |= MISSING);
                }
                FeatureCounts::Numeric {
                    below_training_min_count: (observed != 0).then_some(below),
                    above_training_max_count: (observed != 0).then_some(above),
                }
            }
            FeatureValues::Categorical(values) => {
                let (support, non_missing) = summarize_categories(values);
                source_missing = n_training_rows - non_missing;
                let mut unseen = 0;
                if let FeatureValues::Categorical(query_values) = query {
                    match support {
                        CategorySupport::Single(category) => {
                            for (value, flag) in query_values.iter().zip(&mut flags) {
                                match value {
                                    None => {
                                        prediction_missing_count += 1;
                                        *flag |= MISSING;
                                    }
                                    Some(value) if value != category => {
                                        unseen += 1;
                                        *flag |= UNSEEN;
                                    }
                                    Some(_) => {}
                                }
                            }
                        }
                        CategorySupport::Set(observed) => {
                            for (value, flag) in query_values.iter().zip(&mut flags) {
                                match value {
                                    None => {
                                        prediction_missing_count += 1;
                                        *flag |= MISSING;
                                    }
                                    Some(_) if observed.is_empty() => {
                                        // An unseen category can also lack training support.
                                        unsupported_count += 1;
                                        unseen += 1;
                                        *flag |= UNSUPPORTED | UNSEEN;
                                    }
                                    Some(value) if !observed.contains(value.as_str()) => {
                                        unseen += 1;
                                        *flag |= UNSEEN;
                                    }
                                    Some(_) => {}
                                }
                            }
                        }
                    }
                } else {
                    prediction_missing_count = n_prediction_rows;
                    flags.iter_mut().for_each(|flag| *flag |= MISSING);
                }
                FeatureCounts::Categorical {
                    unseen_category_count: unseen,
                }
            }
        };
        features.push(FeatureSupport {
            name: source.name.clone(),
            training_missing_count: source_missing,
            training_support: if source_missing == n_training_rows {
                TrainingSupport::UnavailableAllMissing
            } else {
                TrainingSupport::ObservedNonMissingValues
            },
            prediction_missing_count,
            prediction_non_missing_without_training_values_count: unsupported_count,
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

#[cfg(test)]
mod row_flag_tests {
    use super::*;

    #[test]
    fn overlapping_flags_count_once_in_union_and_once_per_kind() {
        let flags: Vec<_> = (0..16).collect();
        assert_eq!(
            summarize_row_flags(&flags),
            (
                15,
                RowFlagCounts {
                    missing_value: 8,
                    numeric_outside_training_range: 8,
                    unseen_category: 8,
                    non_missing_value_in_all_missing_training_feature: 8,
                }
            )
        );
    }
}
