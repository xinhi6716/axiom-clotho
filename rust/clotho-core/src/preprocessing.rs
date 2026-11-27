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
        // Selection mutates its input; reuse a scratch copy but never
        // carry observations from one training column into another.
        let mut observed_numeric = Vec::new();
        for (source_index, column) in training.iter().enumerate() {
            let output_start = feature_names.len();
            let (values, training_missing_count) = match &column.values {
                FeatureValues::Numeric(values) => {
                    let (fill, training_missing_count) = numeric_fill(values, &mut observed_numeric);
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
        if incoming.len() != self.columns.len() {
            return Err(PreprocessingError::SchemaMismatch);
        }
        // Native fit/evaluation usually constructs query columns in fitted order.
        // Borrow that order directly; preserve name alignment on reordered input.
        let ordered = self.columns.iter().zip(incoming).all(|(fit, query)| fit.name == query.name);
        let query_by_name = if ordered {
            None
        } else {
            let by_name: HashMap<_, _> = incoming
                .iter()
                .map(|column| (column.name.as_str(), &column.values))
                .collect();
            if self.columns.iter().any(|column| !by_name.contains_key(column.name.as_str())) {
                return Err(PreprocessingError::SchemaMismatch);
            }
            Some(by_name)
        };
        // Validate every type before allocating a potentially large dense table.
        let mut matched = Vec::with_capacity(self.columns.len());
        for (column, original) in self.columns.iter().zip(incoming) {
            let query = match &query_by_name {
                None => &original.values,
                Some(by_name) => by_name[column.name.as_str()],
            };
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
            // Hash once per larger compatible batch, retaining each
            // training vocabulary's existing lexical indicator index.
            let lookup = match (&column.values, query) {
                (FittedValues::Categorical { vocabulary }, FeatureValues::Categorical(_))
                    if rows >= 128 && vocabulary.len() >= 16 =>
                {
                    Some(
                        vocabulary.iter().enumerate().map(|(i, s)| (s.as_str(), i))
                            .collect::<HashMap<_, _>>(),
                    )
                }
                _ => None,
            };
            matched.push((column, query, lookup));
        }

        // Keep just-initialized rows near the processor while writing the
        // column-oriented input. Schema and name binding run only once.
        const TILE_ROWS: usize = 128;
        let mut output = Vec::with_capacity(rows);
        for start_row in (0..rows).step_by(TILE_ROWS) {
            let end_row = (start_row + TILE_ROWS).min(rows);
            output.resize_with(end_row, || vec![0.0; self.feature_names.len()]);
            let tile = &mut output[start_row..end_row];
            for (column, query, lookup) in &matched {
                let start = column.output_start;
                match (&column.values, query) {
                    (FittedValues::Numeric { fill }, FeatureValues::Numeric(values)) => {
                        for (row, value) in tile.iter_mut().zip(&values[start_row..end_row]) {
                            row[start] = value.unwrap_or(*fill);
                        }
                    }
                    (FittedValues::Numeric { fill }, FeatureValues::Categorical(_)) => {
                        // The mismatched variant was verified to be entirely missing.
                        for row in &mut *tile {
                            row[start] = *fill;
                        }
                    }
                    (FittedValues::Categorical { vocabulary }, FeatureValues::Categorical(values)) => {
                        let values = &values[start_row..end_row];
                        if let Some(lookup) = lookup {
                            fill_categorical_tile(tile, values, start, |s| lookup.get(s).copied());
                        } else {
                            for (row, value) in tile.iter_mut().zip(values) {
                                match value {
                                    None => row[start] = 1.0,
                                    Some(value) => {
                                        if let Ok(index) = vocabulary.binary_search(value) {
                                            row[start + index + 1] = 1.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    (FittedValues::Categorical { .. }, FeatureValues::Numeric(_)) => {
                        // The mismatched variant was verified to be entirely missing.
                        for row in &mut *tile {
                            row[start] = 1.0;
                        }
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

fn fill_categorical_tile(
    tile: &mut [Vec<f64>],
    values: &[Option<String>],
    start: usize,
    indicator: impl Fn(&str) -> Option<usize>,
) {
    for (row, value) in tile.iter_mut().zip(values) {
        match value {
            None => row[start] = 1.0,
            Some(s) => {
                if let Some(index) = indicator(s) {
                    row[start + index + 1] = 1.0;
                }
                // Query-only strings intentionally remain all zero.
            }
        }
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

// Finite values were validated already. Check the borrowed column first so
// a monotonic column, including a constant column, needs no scratch copy.
fn numeric_fill(values: &[Option<f64>], scratch: &mut Vec<f64>) -> (f64, usize) {
    let mut iter = values.iter().flatten().copied();
    let Some(mut previous) = iter.next() else {
        return (0.0, values.len());
    };
    let mut observations = 1;
    let mut order = std::cmp::Ordering::Equal;
    for current in iter {
        let direction = previous.total_cmp(&current);
        if direction != std::cmp::Ordering::Equal {
            if order == direction.reverse() {
                // A genuine inversion requires selection on mutable storage.
                scratch.clear();
                scratch.extend(values.iter().flatten().copied());
                return (unordered_median(scratch), values.len() - scratch.len());
            }
            order = direction;
        }
        previous = current;
        observations += 1;
    }

    let descending = order.is_gt();
    let odd = observations % 2 != 0;
    let low_index = (observations - 1) / 2;
    let missing = values.len() - observations;
    let (first, second) = if missing == 0 {
        (values[low_index].unwrap(), values.get(low_index + 1).copied().flatten())
    } else {
        let mut middle_values = values.iter().flatten().skip(low_index);
        (*middle_values.next().unwrap(), middle_values.next().copied())
    };
    let fill = if odd {
        first
    } else if descending {
        midpoint(second.unwrap(), first)
    } else {
        midpoint(first, second.unwrap())
    };
    (fill, missing)
}

fn unordered_median(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let odd = values.len() % 2 == 1;
    let (lower_half, upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if odd {
        *upper
    } else {
        // The lower half need not be sorted: its maximum is the lower middle.
        let lower = lower_half.iter().copied().max_by(f64::total_cmp).unwrap();
        midpoint(lower, *upper)
    }
}

fn midpoint(lower: f64, upper: f64) -> f64 {
    let sum = lower + upper;
    // Use the sum when finite to preserve subnormals. Only divide operands
    // first when their same-sign sum overflows; neither can underflow then.
    if sum.is_finite() {
        sum * 0.5
    } else {
        lower * 0.5 + upper * 0.5
    }
}

#[test]
fn ordered_and_reordered_transform_preserve_bits_and_validation() {
    let ordered = vec![
        FeatureColumn {
            name: "axis".into(),
            values: FeatureValues::Numeric(vec![Some(-0.0), None, Some(2.0)]),
        },
        FeatureColumn {
            name: "condition".into(),
            values: FeatureValues::Categorical(vec![Some("a".into()), None, Some("b".into())]),
        },
    ];
    let pre = Preprocessor::fit(&ordered).unwrap();
    let mut reordered = ordered.clone();
    reordered.reverse();
    let bits = |rows: Vec<Vec<f64>>| -> Vec<u64> {
        rows.into_iter().flatten().map(f64::to_bits).collect()
    };
    let a = bits(pre.transform(&ordered).unwrap());
    assert_eq!(a[0], (-0.0_f64).to_bits());
    assert_eq!(bits(pre.transform(&reordered).unwrap()), a);
    reordered[1].values = FeatureValues::Categorical(vec![None, None, Some("x".into())]);
    assert_eq!(pre.transform(&reordered), Err(PreprocessingError::TypeMismatch));
    reordered[0].name = "unknown".into();
    assert_eq!(pre.transform(&reordered), Err(PreprocessingError::SchemaMismatch));
    reordered[1].values = FeatureValues::Numeric(vec![Some(f64::NAN); 3]);
    assert_eq!(pre.transform(&reordered), Err(PreprocessingError::NonFiniteNumeric));
}

#[cfg(test)]
mod median_tests {
    use super::*;

    #[test]
    fn borrowed_median_skips_scratch_and_preserves_full_sort_bits() {
        let tiny = f64::from_bits(1);
        let candidates = [
            None,
            Some(-f64::MAX),
            Some(-tiny),
            Some(-0.0),
            Some(0.0),
            Some(tiny),
            Some(f64::MAX),
        ];
        let mut scratch = vec![72.0, 91.0];
        for value in [
            vec![None; 8],
            vec![Some(-0.0); 3],
            vec![None, Some(-tiny), Some(-0.0), None, Some(0.0), Some(tiny)],
            vec![None, Some(f64::MAX), Some(f64::MAX), None],
        ] {
            let (a, missing) = numeric_fill(&value, &mut scratch);
            let mut reversed = value;
            reversed.reverse();
            let (b, reverse_missing) = numeric_fill(&reversed, &mut scratch);
            assert_eq!(a.to_bits(), b.to_bits());
            assert_eq!(missing, reverse_missing);
            assert_eq!(scratch, [72.0, 91.0]);
        }

        for mut code in 0..7_usize.pow(5) {
            let values: Vec<_> = (0..5).map(|_| {
                let v = candidates[code % candidates.len()];
                code /= candidates.len();
                v
            }).collect();
            let mut ordered: Vec<_> = values.iter().flatten().copied().collect();
            ordered.sort_unstable_by(f64::total_cmp);
            let fill = match ordered.len() {
                0 => 0.0,
                n if n % 2 == 1 => ordered[n / 2],
                n => midpoint(ordered[n / 2 - 1], ordered[n / 2]),
            };
            let (actual, missing) = numeric_fill(&values, &mut scratch);
            assert_eq!(actual.to_bits(), fill.to_bits());
            assert_eq!(missing, values.len() - ordered.len());
        }
    }
}
