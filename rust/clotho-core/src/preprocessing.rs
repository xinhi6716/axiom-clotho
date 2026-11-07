//! Training-only preprocessing for mixed numeric and categorical tables.
//!
//! Numeric features retain their units and use the training median for missing
//! values (zero when training is entirely missing). Estimators own any scaling.
//! Categorical features use a sorted training vocabulary and a separate missing
//! indicator; query-only strings produce all-zero indicators. No query values
//! can change a fitted vocabulary or numeric fill value.
//!
//! Each categorical feature supports at most 64 output indicators, including
//! its always-present missing indicator, hence at most 63 observed strings.
//! Exceeding that cap is an error. Unlike Python encoders' `max_categories`,
//! this implementation does not pool infrequent categories or silently drop
//! categories. The dense result has at most 6,400 columns; a downstream
//! estimator can impose a smaller feature limit.

use crate::input_support::{FeatureColumn, FeatureValues, MAX_FEATURES, MAX_ROWS};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

/// Maximum indicators per categorical feature, including missing.
pub const MAX_CATEGORIES: usize = 64;
/// Maximum number of columns in a transformed dense table.
pub const MAX_OUTPUT_FEATURES: usize = MAX_FEATURES * MAX_CATEGORIES;

/// Errors deliberately omit feature values, category inventories and row data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreprocessingError {
    FeatureCount,
    RowCount,
    RaggedColumns,
    DuplicateNames,
    SchemaMismatch,
    TypeMismatch,
    NonFiniteNumeric,
    NulCategory,
    CategoryLimit,
    OutputFeatureLimit,
}

impl fmt::Display for PreprocessingError {
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
            Self::CategoryLimit => "A categorical feature supports at most 63 observed strings and one reserved missing indicator (64 total).",
            Self::OutputFeatureLimit => "Preprocessing supports at most 6,400 output features.",
        })
    }
}

impl std::error::Error for PreprocessingError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureKind {
    Numeric,
    Categorical,
}

/// Fitted aggregate metadata, without source records or category inventories.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureMetadata {
    pub name: String,
    pub kind: FeatureKind,
    pub training_missing_count: usize,
    pub output_start: usize,
    pub output_width: usize,
    /// Present for numeric features, including the zero all-missing fallback.
    pub numeric_imputation_value: Option<f64>,
    /// Present for categorical features; excludes the reserved missing slot.
    pub observed_category_count: Option<usize>,
}

#[derive(Clone)]
enum FittedValues {
    Numeric { fill: f64 },
    Categorical { vocabulary: Vec<String> },
}

#[derive(Clone)]
struct FittedColumn {
    name: String,
    values: FittedValues,
    training_missing_count: usize,
    output_start: usize,
}

/// Immutable, reusable preprocessing learned exclusively from a training table.
///
/// Fit this object independently inside every model-selection training fold.
/// Fitting it on the complete dataset before splitting would leak information.
/// Only medians and categorical vocabularies are retained, not source rows.
/// Output feature order follows the training source-column order. Within each
/// categorical feature, missing comes first, then strings in Rust's lexical
/// order (case-sensitive, without Unicode normalization).
#[derive(Clone)]
pub struct Preprocessor {
    columns: Vec<FittedColumn>,
    feature_names: Vec<String>,
    training_row_count: usize,
}

// Do not accidentally dump the fitted category inventory via Debug logging.
impl fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Preprocessor")
            .field("training_row_count", &self.training_row_count)
            .field("features", &self.metadata())
            .finish()
    }
}

impl Preprocessor {
    /// Fit medians and vocabularies from 1..=20,000 training rows and
    /// 1..=100 source columns. Missing values must be explicit `None` values.
    pub fn fit(training: &[FeatureColumn]) -> Result<Self, PreprocessingError> {
        let training_row_count = validate_table(training)?;
        let mut columns = Vec::with_capacity(training.len());
        let mut feature_names = Vec::new();
        for (source_index, column) in training.iter().enumerate() {
            let output_start = feature_names.len();
            let (values, training_missing_count) = match &column.values {
                FeatureValues::Numeric(values) => {
                    let mut observed: Vec<_> = values.iter().flatten().copied().collect();
                    let training_missing_count = values.len() - observed.len();
                    let fill = median(&mut observed);
                    feature_names.push(format!("feature_{}__numeric", source_index));
                    (FittedValues::Numeric { fill }, training_missing_count)
                }
                FeatureValues::Categorical(values) => {
                    let mut observed = BTreeSet::new();
                    let mut training_missing_count = 0;
                    for value in values {
                        match value {
                            Some(value) => {
                                observed.insert(value.as_str());
                                // Include the reserved missing indicator.
                                if observed.len() >= MAX_CATEGORIES {
                                    return Err(PreprocessingError::CategoryLimit);
                                }
                            }
                            None => training_missing_count += 1,
                        }
                    }
                    let vocabulary: Vec<_> = observed.into_iter().map(str::to_owned).collect();
                    feature_names.push(format!("feature_{}__missing", source_index));
                    for index in 0..vocabulary.len() {
                        feature_names.push(format!("feature_{}__category_{}", source_index, index));
                    }
                    (
                        FittedValues::Categorical { vocabulary },
                        training_missing_count,
                    )
                }
            };
            if feature_names.len() > MAX_OUTPUT_FEATURES {
                return Err(PreprocessingError::OutputFeatureLimit);
            }
            columns.push(FittedColumn {
                name: column.name.clone(),
                values,
                training_missing_count,
                output_start,
            });
        }
        Ok(Self {
            columns,
            feature_names,
            training_row_count,
        })
    }

    /// Transform rows without updating any fitted state.
    ///
    /// Incoming columns may be reordered but must have the exact fitted names
    /// and types. An entirely missing incoming column may use either enum
    /// variant, because its values are unambiguous; no other coercion occurs.
    /// Empty tables, nonfinite numeric values and NUL category strings fail.
    pub fn transform(
        &self,
        incoming: &[FeatureColumn],
    ) -> Result<Vec<Vec<f64>>, PreprocessingError> {
        let rows = validate_table(incoming)?;
        let query_by_name: HashMap<_, _> = incoming
            .iter()
            .map(|column| (column.name.as_str(), &column.values))
            .collect();
        if incoming.len() != self.columns.len()
            || self
                .columns
                .iter()
                .any(|column| !query_by_name.contains_key(column.name.as_str()))
        {
            return Err(PreprocessingError::SchemaMismatch);
        }
        // Validate every type before allocating a potentially large dense table.
        for column in &self.columns {
            let query = query_by_name[column.name.as_str()];
            let same_type = matches!(
                (&column.values, query),
                (FittedValues::Numeric { .. }, FeatureValues::Numeric(_))
                    | (
                        FittedValues::Categorical { .. },
                        FeatureValues::Categorical(_)
                    )
            );
            if !same_type && !all_missing(query) {
                return Err(PreprocessingError::TypeMismatch);
            }
        }
        let mut output = vec![vec![0.0; self.feature_names.len()]; rows];
        for column in &self.columns {
            let query = query_by_name[column.name.as_str()];
            let start = column.output_start;
            match (&column.values, query) {
                (FittedValues::Numeric { fill }, FeatureValues::Numeric(values)) => {
                    for (row, value) in output.iter_mut().zip(values) {
                        row[start] = value.unwrap_or(*fill);
                    }
                }
                (FittedValues::Numeric { fill }, FeatureValues::Categorical(_)) => {
                    // The mismatched variant was verified to be entirely missing.
                    for row in &mut output {
                        row[start] = *fill;
                    }
                }
                (FittedValues::Categorical { vocabulary }, FeatureValues::Categorical(values)) => {
                    for (row, value) in output.iter_mut().zip(values) {
                        match value {
                            None => row[start] = 1.0,
                            Some(value) => {
                                if let Ok(index) = vocabulary.binary_search(value) {
                                    row[start + index + 1] = 1.0;
                                }
                                // Query-only strings intentionally remain all zero.
                            }
                        }
                    }
                }
                (FittedValues::Categorical { .. }, FeatureValues::Numeric(_)) => {
                    // The mismatched variant was verified to be entirely missing.
                    for row in &mut output {
                        row[start] = 1.0;
                    }
                }
            }
        }
        Ok(output)
    }

    /// Collision-free generated column names, aligned with transformed rows.
    /// Names use training source-column and vocabulary positions rather than raw
    /// labels or category strings. Use metadata to recover source-column names.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn training_row_count(&self) -> usize {
        self.training_row_count
    }

    pub fn output_feature_count(&self) -> usize {
        self.feature_names.len()
    }

    /// Return aggregate fitted metadata without retained category strings.
    pub fn metadata(&self) -> Vec<FeatureMetadata> {
        self.columns
            .iter()
            .map(|column| {
                let (kind, width, fill, category_count) = match &column.values {
                    FittedValues::Numeric { fill } => (FeatureKind::Numeric, 1, Some(*fill), None),
                    FittedValues::Categorical { vocabulary } => (
                        FeatureKind::Categorical,
                        vocabulary.len() + 1,
                        None,
                        Some(vocabulary.len()),
                    ),
                };
                FeatureMetadata {
                    name: column.name.clone(),
                    kind,
                    training_missing_count: column.training_missing_count,
                    output_start: column.output_start,
                    output_width: width,
                    numeric_imputation_value: fill,
                    observed_category_count: category_count,
                }
            })
            .collect()
    }
}

fn row_count(values: &FeatureValues) -> usize {
    match values {
        FeatureValues::Numeric(values) => values.len(),
        FeatureValues::Categorical(values) => values.len(),
    }
}

fn all_missing(values: &FeatureValues) -> bool {
    match values {
        FeatureValues::Numeric(values) => values.iter().all(Option::is_none),
        FeatureValues::Categorical(values) => values.iter().all(Option::is_none),
    }
}

fn validate_table(columns: &[FeatureColumn]) -> Result<usize, PreprocessingError> {
    if !(1..=MAX_FEATURES).contains(&columns.len()) {
        return Err(PreprocessingError::FeatureCount);
    }
    let rows = row_count(&columns[0].values);
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(PreprocessingError::RowCount);
    }
    let mut names = HashSet::with_capacity(columns.len());
    for column in columns {
        if !names.insert(column.name.as_str()) {
            return Err(PreprocessingError::DuplicateNames);
        }
        if row_count(&column.values) != rows {
            return Err(PreprocessingError::RaggedColumns);
        }
        match &column.values {
            FeatureValues::Numeric(values) => {
                if values.iter().flatten().any(|value| !value.is_finite()) {
                    return Err(PreprocessingError::NonFiniteNumeric);
                }
            }
            FeatureValues::Categorical(values) => {
                if values.iter().flatten().any(|value| value.contains('\0')) {
                    return Err(PreprocessingError::NulCategory);
                }
            }
        }
    }
    Ok(rows)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let upper = values.len() / 2;
    if values.len() % 2 == 1 {
        values[upper]
    } else {
        let (lower, upper) = (values[upper - 1], values[upper]);
        let sum = lower + upper;
        // Use the sum when finite to preserve subnormals. Only divide operands
        // first when their same-sign sum overflows; neither can underflow then.
        if sum.is_finite() {
            sum * 0.5
        } else {
            lower * 0.5 + upper * 0.5
        }
    }
}
