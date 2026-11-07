use clotho_core::input_support::{FeatureColumn, FeatureValues, MAX_FEATURES, MAX_ROWS};
use clotho_core::preprocessing::{
    FeatureKind, PreprocessingError, Preprocessor, MAX_CATEGORIES, MAX_OUTPUT_FEATURES,
};
use std::collections::HashSet;

fn numeric(name: &str, values: &[Option<f64>]) -> FeatureColumn {
    FeatureColumn {
        name: name.to_owned(),
        values: FeatureValues::Numeric(values.to_vec()),
    }
}

fn categorical(name: &str, values: &[Option<&str>]) -> FeatureColumn {
    FeatureColumn {
        name: name.to_owned(),
        values: FeatureValues::Categorical(
            values
                .iter()
                .map(|value| value.map(str::to_owned))
                .collect(),
        ),
    }
}

fn numbered_categories(name: &str, count: usize) -> FeatureColumn {
    FeatureColumn {
        name: name.to_owned(),
        values: FeatureValues::Categorical(
            (0..count)
                .map(|value| Some(format!("category_{value:02}")))
                .collect(),
        ),
    }
}

#[test]
fn mixed_fit_imputes_training_median_without_scaling_and_orders_categories() {
    let training = vec![
        numeric("quantity", &[Some(9.0), None, Some(1.0), Some(5.0)]),
        categorical("kind", &[Some("z"), None, Some("a"), Some("z")]),
    ];
    let fitted = Preprocessor::fit(&training).unwrap();
    assert_eq!(
        fitted.transform(&training).unwrap(),
        vec![
            vec![9.0, 0.0, 0.0, 1.0],
            vec![5.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![5.0, 0.0, 0.0, 1.0],
        ]
    );
    assert_eq!(
        fitted.feature_names(),
        &[
            "feature_0__numeric",
            "feature_1__missing",
            "feature_1__category_0",
            "feature_1__category_1",
        ]
    );
    assert_eq!(fitted.training_row_count(), 4);
    assert_eq!(fitted.output_feature_count(), 4);
}

#[test]
fn transform_cannot_leak_query_values_into_fitted_statistics() {
    let training = vec![
        numeric("number", &[Some(1.0), Some(3.0), None]),
        categorical("kind", &[Some("b"), Some("a"), None]),
    ];
    let fitted = Preprocessor::fit(&training).unwrap();
    let metadata = fitted.metadata();
    let names = fitted.feature_names().to_vec();
    let probe = vec![
        numeric("number", &[None]),
        categorical("kind", &[Some("unseen")]),
    ];
    let before = fitted.transform(&probe).unwrap();
    assert_eq!(before, vec![vec![2.0, 0.0, 0.0, 0.0]]);
    let query = vec![
        numeric("number", &[Some(1e300), Some(-1e300), None]),
        categorical("kind", &[Some("unseen"), Some("another new value"), None]),
    ];
    assert_eq!(
        fitted.transform(&query).unwrap(),
        vec![
            vec![1e300, 0.0, 0.0, 0.0],
            vec![-1e300, 0.0, 0.0, 0.0],
            vec![2.0, 1.0, 0.0, 0.0],
        ]
    );
    assert_eq!(fitted.transform(&probe).unwrap(), before);
    assert_eq!(fitted.metadata(), metadata);
    assert_eq!(fitted.feature_names(), names);
}

#[test]
fn missing_is_distinct_even_when_absent_in_training() {
    let fitted = Preprocessor::fit(&[categorical("kind", &[Some("known")])]).unwrap();
    assert_eq!(
        fitted
            .transform(&[categorical("kind", &[None, Some("new"), Some("known")])])
            .unwrap(),
        vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 1.0]]
    );
}

#[test]
fn missing_and_empty_or_sentinel_like_strings_are_distinct() {
    let values = [
        None,
        Some(""),
        Some("__missing"),
        Some("None"),
        Some("null"),
    ];
    let table = [categorical("kind", &values)];
    let fitted = Preprocessor::fit(&table).unwrap();
    let result = fitted.transform(&table).unwrap();
    assert_eq!(result.len(), 5);
    assert!(result.iter().all(|row| row.iter().sum::<f64>() == 1.0));
    for left in 0..result.len() {
        for right in left + 1..result.len() {
            assert_ne!(result[left], result[right]);
        }
    }
}

#[test]
fn all_missing_training_features_keep_a_stable_output_schema() {
    let training = [
        numeric("number", &[None, None]),
        categorical("kind", &[None, None]),
    ];
    let fitted = Preprocessor::fit(&training).unwrap();
    assert_eq!(
        fitted.transform(&training).unwrap(),
        vec![vec![0.0, 1.0]; 2]
    );
    assert_eq!(
        fitted
            .transform(&[
                numeric("number", &[None, Some(7.0)]),
                categorical("kind", &[None, Some("new")]),
            ])
            .unwrap(),
        vec![vec![0.0, 1.0], vec![7.0, 0.0]]
    );
    assert_eq!(fitted.metadata()[0].numeric_imputation_value, Some(0.0));
    assert_eq!(fitted.metadata()[1].observed_category_count, Some(0));
}

#[test]
fn even_median_handles_overflow_opposite_signs_and_subnormals() {
    let tiny = f64::from_bits(1);
    let cases = [
        (vec![1.0, 5.0, 3.0, 9.0], 4.0),
        (vec![f64::MAX, f64::MAX], f64::MAX),
        (vec![-f64::MAX, -f64::MAX], -f64::MAX),
        (vec![-f64::MAX, f64::MAX], 0.0),
        (vec![tiny, tiny], tiny),
        (vec![tiny, tiny * 2.0], tiny * 2.0),
        (vec![-tiny * 2.0, -tiny], -tiny * 2.0),
        (
            vec![f64::MIN_POSITIVE, f64::MIN_POSITIVE],
            f64::MIN_POSITIVE,
        ),
    ];
    for (values, expected) in cases {
        let values: Vec<_> = values.into_iter().map(Some).collect();
        let fitted = Preprocessor::fit(&[numeric("number", &values)]).unwrap();
        assert_eq!(
            fitted.transform(&[numeric("number", &[None])]).unwrap()[0][0],
            expected
        );
    }
}

#[test]
fn finite_observed_values_are_preserved_including_signed_zero() {
    let table = [numeric(
        "number",
        &[Some(-0.0), Some(f64::MAX), Some(-f64::MAX)],
    )];
    let fitted = Preprocessor::fit(&table).unwrap();
    let result = fitted.transform(&table).unwrap();
    assert_eq!(result[0][0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(result[1][0], f64::MAX);
    assert_eq!(result[2][0], -f64::MAX);
}

#[test]
fn vocabulary_is_deterministic_under_training_row_reordering() {
    let first = [categorical(
        "kind",
        &[Some("β"), Some("a"), None, Some("A"), Some("a")],
    )];
    let second = [categorical(
        "kind",
        &[None, Some("A"), Some("a"), Some("a"), Some("β")],
    )];
    let left = Preprocessor::fit(&first).unwrap();
    let right = Preprocessor::fit(&second).unwrap();
    let query = [categorical(
        "kind",
        &[Some("A"), Some("a"), Some("β"), None],
    )];
    assert_eq!(left.metadata(), right.metadata());
    assert_eq!(left.feature_names(), right.feature_names());
    assert_eq!(
        left.transform(&query).unwrap(),
        right.transform(&query).unwrap()
    );
    assert_eq!(
        left.transform(&query).unwrap(),
        vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ]
    );
}

#[test]
fn incoming_columns_can_be_reordered_by_exact_name() {
    let training = [
        numeric("amount", &[Some(10.0)]),
        categorical("kind", &[Some("a")]),
        numeric("other", &[Some(20.0)]),
    ];
    let fitted = Preprocessor::fit(&training).unwrap();
    let query = [
        numeric("other", &[Some(80.0)]),
        categorical("kind", &[Some("a")]),
        numeric("amount", &[Some(40.0)]),
    ];
    assert_eq!(
        fitted.transform(&query).unwrap(),
        vec![vec![40.0, 0.0, 1.0, 80.0]]
    );
}

#[test]
fn completely_missing_query_columns_accept_either_variant() {
    let fitted = Preprocessor::fit(&[
        numeric("number", &[Some(4.0)]),
        categorical("kind", &[Some("a")]),
    ])
    .unwrap();
    let query = [
        categorical("number", &[None, None]),
        numeric("kind", &[None, None]),
    ];
    assert_eq!(
        fitted.transform(&query).unwrap(),
        vec![vec![4.0, 1.0, 0.0]; 2]
    );
}

#[test]
fn type_changes_with_any_observed_query_value_are_rejected() {
    for (training, query) in [
        (
            numeric("x", &[Some(4.0)]),
            categorical("x", &[None, Some("4")]),
        ),
        (
            categorical("x", &[Some("4")]),
            numeric("x", &[None, Some(4.0)]),
        ),
        (numeric("x", &[None]), categorical("x", &[Some("4")])),
        (categorical("x", &[None]), numeric("x", &[Some(4.0)])),
    ] {
        let fitted = Preprocessor::fit(&[training]).unwrap();
        assert_eq!(
            fitted.transform(&[query]),
            Err(PreprocessingError::TypeMismatch)
        );
    }
}

#[test]
fn category_cap_includes_the_reserved_missing_slot_without_pooling() {
    let table = [numbered_categories("kind", MAX_CATEGORIES - 1)];
    let fitted = Preprocessor::fit(&table).unwrap();
    assert_eq!(fitted.output_feature_count(), MAX_CATEGORIES);
    assert_eq!(
        fitted.metadata()[0].observed_category_count,
        Some(MAX_CATEGORIES - 1)
    );
    let over_limit = [numbered_categories("kind", MAX_CATEGORIES)];
    assert_eq!(
        Preprocessor::fit(&over_limit).unwrap_err(),
        PreprocessingError::CategoryLimit
    );
    // Transform accepts any number of novel strings, without fitting them.
    let output = fitted.transform(&over_limit).unwrap();
    assert_eq!(output.last().unwrap(), &vec![0.0; MAX_CATEGORIES]);
}

#[test]
fn category_frequency_does_not_affect_the_cap_or_vocabulary_order() {
    let table = [categorical(
        "kind",
        &[Some("z"), Some("a"), Some("z"), Some("z")],
    )];
    let fitted = Preprocessor::fit(&table).unwrap();
    assert_eq!(fitted.output_feature_count(), 3);
    assert_eq!(
        fitted
            .transform(&[categorical("kind", &[Some("a")])])
            .unwrap(),
        vec![vec![0.0, 1.0, 0.0]]
    );
}

#[test]
fn output_expansion_reaches_but_does_not_exceed_the_documented_bound() {
    let table: Vec<_> = (0..MAX_FEATURES)
        .map(|column| numbered_categories(&format!("feature_{column}"), MAX_CATEGORIES - 1))
        .collect();
    let fitted = Preprocessor::fit(&table).unwrap();
    assert_eq!(fitted.output_feature_count(), MAX_OUTPUT_FEATURES);
    let query: Vec<_> = (0..MAX_FEATURES)
        .map(|column| categorical(&format!("feature_{column}"), &[None]))
        .collect();
    let result = fitted.transform(&query).unwrap();
    assert_eq!(result[0].len(), MAX_OUTPUT_FEATURES);
    assert_eq!(result[0].iter().sum::<f64>(), MAX_FEATURES as f64);
}

#[test]
fn indexed_output_names_do_not_collide_with_adversarial_source_names() {
    let table = [
        numeric("feature_1__missing", &[None]),
        categorical("", &[Some("feature_0__numeric")]),
        categorical("__category_1", &[None]),
        numeric("__numeric", &[Some(3.0)]),
        categorical("source\0name", &[Some("odd\ncategory")]),
    ];
    let fitted = Preprocessor::fit(&table).unwrap();
    let names: HashSet<_> = fitted.feature_names().iter().collect();
    assert_eq!(names.len(), fitted.output_feature_count());
    assert!(names
        .iter()
        .all(|name| !name.contains('\0') && !name.contains('\n')));
}

#[test]
fn metadata_contains_aggregate_schema_and_not_category_inventory() {
    let fitted = Preprocessor::fit(&[
        numeric("number", &[None, Some(8.0)]),
        categorical("kind", &[Some("private category"), None]),
    ])
    .unwrap();
    let metadata = fitted.metadata();
    assert_eq!(metadata[0].name, "number");
    assert_eq!(metadata[0].kind, FeatureKind::Numeric);
    assert_eq!(metadata[0].training_missing_count, 1);
    assert_eq!(metadata[0].output_start, 0);
    assert_eq!(metadata[0].output_width, 1);
    assert_eq!(metadata[0].numeric_imputation_value, Some(8.0));
    assert_eq!(metadata[0].observed_category_count, None);
    assert_eq!(metadata[1].kind, FeatureKind::Categorical);
    assert_eq!(metadata[1].training_missing_count, 1);
    assert_eq!(metadata[1].output_start, 1);
    assert_eq!(metadata[1].output_width, 2);
    assert_eq!(metadata[1].numeric_imputation_value, None);
    assert_eq!(metadata[1].observed_category_count, Some(1));
    assert!(!format!("{fitted:?}").contains("private category"));
    assert!(!format!("{metadata:?}").contains("private category"));
    assert!(fitted
        .feature_names()
        .iter()
        .all(|name| !name.contains("private category")));
}

#[test]
fn fit_and_transform_reject_empty_and_oversized_tables() {
    let fitted = Preprocessor::fit(&[numeric("number", &[Some(1.0)])]).unwrap();
    for (table, expected) in [
        (vec![], PreprocessingError::FeatureCount),
        (vec![numeric("number", &[])], PreprocessingError::RowCount),
        (
            vec![numeric("number", &vec![None; MAX_ROWS + 1])],
            PreprocessingError::RowCount,
        ),
        (
            (0..=MAX_FEATURES)
                .map(|index| numeric(&format!("n{index}"), &[None]))
                .collect(),
            PreprocessingError::FeatureCount,
        ),
    ] {
        assert_eq!(Preprocessor::fit(&table).unwrap_err(), expected);
        assert_eq!(fitted.transform(&table), Err(expected));
    }
    let max_rows = [numeric("number", &vec![None; MAX_ROWS])];
    let largest = Preprocessor::fit(&max_rows).unwrap();
    assert_eq!(largest.transform(&max_rows).unwrap().len(), MAX_ROWS);
}

#[test]
fn fit_and_transform_reject_ragged_or_duplicate_columns() {
    let fitted =
        Preprocessor::fit(&[numeric("a", &[Some(1.0)]), numeric("b", &[Some(2.0)])]).unwrap();
    for (table, expected) in [
        (
            vec![numeric("a", &[None]), numeric("b", &[None, None])],
            PreprocessingError::RaggedColumns,
        ),
        (
            vec![numeric("a", &[None]), numeric("a", &[None])],
            PreprocessingError::DuplicateNames,
        ),
    ] {
        assert_eq!(Preprocessor::fit(&table).unwrap_err(), expected);
        assert_eq!(fitted.transform(&table), Err(expected));
    }
}

#[test]
fn schema_requires_exact_names_and_no_extra_or_missing_columns() {
    let fitted = Preprocessor::fit(&[numeric("a", &[Some(1.0)])]).unwrap();
    for table in [
        vec![numeric("A", &[None])],
        vec![numeric("a ", &[None])],
        vec![numeric("b", &[None])],
        vec![numeric("a", &[None]), numeric("b", &[None])],
    ] {
        assert_eq!(
            fitted.transform(&table),
            Err(PreprocessingError::SchemaMismatch)
        );
    }
    let two_features = Preprocessor::fit(&[numeric("a", &[None]), numeric("b", &[None])]).unwrap();
    assert_eq!(
        two_features.transform(&[numeric("a", &[None])]),
        Err(PreprocessingError::SchemaMismatch)
    );
}

#[test]
fn fit_and_transform_reject_nonfinite_numeric_values() {
    let fitted = Preprocessor::fit(&[numeric("x", &[Some(1.0)])]).unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let table = [numeric("x", &[None, Some(value)])];
        assert_eq!(
            Preprocessor::fit(&table).unwrap_err(),
            PreprocessingError::NonFiniteNumeric
        );
        assert_eq!(
            fitted.transform(&table),
            Err(PreprocessingError::NonFiniteNumeric)
        );
    }
}

#[test]
fn fit_and_transform_reject_nul_category_strings_without_echoing_them() {
    let fitted = Preprocessor::fit(&[categorical("x", &[Some("normal")])]).unwrap();
    let table = [categorical("x", &[Some("private\0value")])];
    for error in [
        Preprocessor::fit(&table).unwrap_err(),
        fitted.transform(&table).unwrap_err(),
    ] {
        assert_eq!(error, PreprocessingError::NulCategory);
        assert!(!error.to_string().contains("private"));
    }
}

#[test]
fn fitting_and_transforming_do_not_mutate_caller_tables() {
    let table = [
        numeric("number", &[Some(9.0), None, Some(1.0)]),
        categorical("kind", &[Some("z"), None, Some("a")]),
    ];
    let original = table.clone();
    let fitted = Preprocessor::fit(&table).unwrap();
    fitted.transform(&table).unwrap();
    assert_eq!(table, original);
    assert_eq!(fitted.clone().transform(&table), fitted.transform(&table));
}
