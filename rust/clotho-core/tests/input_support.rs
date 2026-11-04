use clotho_core::input_support::*;

fn numeric(name: &str, values: &[Option<f64>]) -> FeatureColumn {
    FeatureColumn {
        name: name.into(),
        values: FeatureValues::Numeric(values.to_vec()),
    }
}

fn categorical(name: &str, values: &[Option<&str>]) -> FeatureColumn {
    FeatureColumn {
        name: name.into(),
        values: FeatureValues::Categorical(
            values.iter().map(|value| value.map(String::from)).collect(),
        ),
    }
}

#[test]
fn numeric_boundaries_are_inclusive_and_constant_ranges_work() {
    let train = vec![
        numeric("x", &[Some(1.), Some(3.), None]),
        numeric("constant", &[Some(2.); 3]),
    ];
    let query = vec![
        numeric("constant", &[Some(2.), Some(1.), Some(3.), Some(2.), None]),
        numeric("x", &[Some(1.), Some(3.), Some(0.), Some(4.), None]),
    ];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.rows_with_any_flag, 4);
    assert_eq!(
        result.row_flag_counts,
        RowFlagCounts {
            missing_value: 1,
            numeric_outside_training_range: 3,
            ..Default::default()
        }
    );
    assert_eq!(result.features[0].name, "x");
    assert_eq!(result.features[0].training_missing_count, 1);
    for feature in result.features {
        assert_eq!(
            feature.counts,
            FeatureCounts::Numeric {
                below_training_min_count: Some(1),
                above_training_max_count: Some(1)
            }
        );
    }
}

#[test]
fn all_missing_training_does_not_invent_numeric_bounds() {
    let train = vec![numeric("x", &[None; 2]), categorical("kind", &[None; 2])];
    let query = vec![
        numeric("x", &[Some(1.), None, Some(-1.)]),
        categorical("kind", &[Some("new"), None, Some("__GROUPBENCH_MISSING__")]),
    ];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.rows_with_any_flag, 3);
    assert_eq!(
        result.row_flag_counts,
        RowFlagCounts {
            missing_value: 1,
            unseen_category: 2,
            non_missing_value_in_all_missing_training_feature: 2,
            numeric_outside_training_range: 0,
        }
    );
    for feature in &result.features {
        assert_eq!(
            feature.training_support,
            TrainingSupport::UnavailableAllMissing
        );
        assert_eq!(
            feature.prediction_non_missing_without_training_values_count,
            2
        );
    }
    assert_eq!(
        result.features[0].counts,
        FeatureCounts::Numeric {
            below_training_min_count: None,
            above_training_max_count: None,
        }
    );
}

#[test]
fn entirely_missing_prediction_columns_accommodate_either_type() {
    let train = vec![numeric("x", &[Some(1.)]), categorical("kind", &[Some("a")])];
    let query = vec![categorical("x", &[None; 2]), numeric("kind", &[None; 2])];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.rows_with_any_flag, 2);
    assert_eq!(result.row_flag_counts.missing_value, 2);
    assert_eq!(result.row_flag_counts.unseen_category, 0);
    assert_eq!(
        result.features[1].counts,
        FeatureCounts::Categorical {
            unseen_category_count: 0
        }
    );
}

#[test]
fn literal_missing_markers_are_ordinary_categories() {
    let train = vec![categorical("kind", &[Some("A"), None])];
    let query = vec![categorical(
        "kind",
        &[Some("__GROUPBENCH_MISSING__"), None, Some("A"), Some("")],
    )];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(
        result.features[0].counts,
        FeatureCounts::Categorical {
            unseen_category_count: 2
        }
    );
    assert_eq!(result.row_flag_counts.missing_value, 1);
}

#[test]
fn comparisons_do_not_subtract_extreme_endpoints() {
    let train = vec![
        numeric("x", &[Some(-f64::MAX), Some(f64::MAX)]),
        numeric("constant", &[Some(f64::MAX); 2]),
    ];
    let query = vec![
        numeric("x", &[Some(-f64::MAX), Some(0.), Some(f64::MAX)]),
        numeric("constant", &[Some(-f64::MAX), Some(0.), Some(f64::MAX)]),
    ];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.row_flag_counts.numeric_outside_training_range, 2);
    assert_eq!(
        result.features[0].counts,
        FeatureCounts::Numeric {
            below_training_min_count: Some(0),
            above_training_max_count: Some(0),
        }
    );
}

#[test]
fn query_never_expands_reference_and_inputs_are_unchanged() {
    let train = vec![
        numeric("x", &[Some(0.), Some(1.)]),
        categorical("kind", &[Some("A"), Some("B")]),
    ];
    let query = vec![
        numeric("x", &[Some(2.), Some(3.), Some(1e100)]),
        categorical("kind", &[Some("new"), Some("new"), Some("other")]),
    ];
    let before = (train.clone(), query.clone());
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.row_flag_counts.numeric_outside_training_range, 3);
    assert_eq!(result.row_flag_counts.unseen_category, 3);
    assert_eq!((train, query), before);
}

#[test]
fn aggregate_counts_are_row_unions_not_sums() {
    let train = vec![numeric("x", &[Some(0.)]), numeric("y", &[Some(0.)])];
    let query = vec![
        numeric("x", &[Some(1.), None]),
        numeric("y", &[Some(2.), None]),
    ];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.rows_with_any_flag, 2);
    assert_eq!(result.row_flag_counts.numeric_outside_training_range, 1);
    assert_eq!(result.row_flag_counts.missing_value, 1);
}

#[test]
fn signed_zero_and_smallest_subnormal_are_not_rounded_or_rejected() {
    let train = vec![numeric("x", &[Some(-0.), Some(f64::from_bits(1))])];
    let query = vec![numeric(
        "x",
        &[
            Some(0.),
            Some(-0.),
            Some(f64::from_bits(1)),
            Some(-f64::from_bits(1)),
        ],
    )];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(
        result.features[0].counts,
        FeatureCounts::Numeric {
            below_training_min_count: Some(1),
            above_training_max_count: Some(0),
        }
    );
}

#[test]
fn exact_unicode_and_whitespace_category_identity_is_preserved() {
    let train = vec![categorical("製程", &[Some("材料"), Some(""), Some("A ")])];
    let query = vec![categorical(
        "製程",
        &[Some("材料"), Some(""), Some("A"), Some("a ")],
    )];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.row_flag_counts.unseen_category, 2);
}

#[test]
fn full_category_inventory_is_supported_but_not_returned() {
    let train = vec![FeatureColumn {
        name: "kind".into(),
        values: FeatureValues::Categorical(
            (0..MAX_ROWS)
                .map(|i| Some(format!("private-category-{i}")))
                .collect(),
        ),
    }];
    let query = vec![categorical(
        "kind",
        &[
            Some("private-category-0"),
            Some("private-category-19999"),
            Some("new-secret"),
        ],
    )];
    let result = prediction_input_support(&train, &query).unwrap();
    assert_eq!(result.row_flag_counts.unseen_category, 1);
    let report = format!("{result:?}");
    assert!(!report.contains("private-category"));
    assert!(!report.contains("new-secret"));
}

#[test]
fn reject_zero_or_excessive_rows() {
    for count in [0, MAX_ROWS + 1] {
        let bad = vec![numeric("x", &vec![None; count])];
        let good = vec![numeric("x", &[Some(1.)])];
        assert_eq!(
            prediction_input_support(&bad, &good),
            Err(SupportError::RowCount)
        );
        assert_eq!(
            prediction_input_support(&good, &bad),
            Err(SupportError::RowCount)
        );
    }
}

#[test]
fn reject_zero_or_excessive_features() {
    for count in [0, MAX_FEATURES + 1] {
        let bad: Vec<_> = (0..count)
            .map(|i| numeric(&format!("x{i}"), &[Some(1.)]))
            .collect();
        assert_eq!(
            prediction_input_support(&bad, &bad),
            Err(SupportError::FeatureCount)
        );
    }
}

#[test]
fn accept_exact_row_and_feature_limits() {
    let columns: Vec<_> = (0..MAX_FEATURES)
        .map(|i| numeric(&format!("x{i}"), &vec![None; MAX_ROWS]))
        .collect();
    let result = prediction_input_support(&columns, &columns).unwrap();
    assert_eq!(result.features.len(), MAX_FEATURES);
    assert_eq!(result.row_flag_counts.missing_value, MAX_ROWS);
}

#[test]
fn reject_ragged_columns_on_both_sides() {
    let good = vec![numeric("x", &[Some(1.)]), numeric("y", &[Some(1.)])];
    let bad = vec![numeric("x", &[Some(1.)]), numeric("y", &[Some(1.); 2])];
    assert_eq!(
        prediction_input_support(&bad, &good),
        Err(SupportError::RaggedColumns)
    );
    assert_eq!(
        prediction_input_support(&good, &bad),
        Err(SupportError::RaggedColumns)
    );
}

#[test]
fn reject_duplicate_names_on_both_sides() {
    let good = vec![numeric("x", &[Some(1.)]), numeric("y", &[Some(1.)])];
    let bad = vec![numeric("x", &[Some(1.)]), numeric("x", &[Some(1.)])];
    assert_eq!(
        prediction_input_support(&bad, &good),
        Err(SupportError::DuplicateNames)
    );
    assert_eq!(
        prediction_input_support(&good, &bad),
        Err(SupportError::DuplicateNames)
    );
}

#[test]
fn reject_missing_extra_or_changed_names() {
    let train = vec![numeric("x", &[Some(1.)]), numeric("y", &[Some(1.)])];
    for query in [
        vec![numeric("x", &[Some(1.)])],
        vec![numeric("x", &[Some(1.)]), numeric("z", &[Some(1.)])],
        vec![
            numeric("x", &[Some(1.)]),
            numeric("y", &[Some(1.)]),
            numeric("z", &[Some(1.)]),
        ],
    ] {
        assert_eq!(
            prediction_input_support(&train, &query),
            Err(SupportError::SchemaMismatch)
        );
    }
}

#[test]
fn reject_non_missing_type_mismatch_including_all_missing_training() {
    for train in [vec![numeric("x", &[Some(1.)])], vec![numeric("x", &[None])]] {
        let query = vec![categorical("x", &[Some("1")])];
        assert_eq!(
            prediction_input_support(&train, &query),
            Err(SupportError::TypeMismatch)
        );
        assert_eq!(
            prediction_input_support(&query, &train),
            if train[0].values == FeatureValues::Numeric(vec![None]) {
                prediction_input_support(&query, &[categorical("x", &[None])])
            } else {
                Err(SupportError::TypeMismatch)
            }
        );
    }
}

#[test]
fn reject_nan_and_infinity_rather_than_infer_missing() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let bad = vec![numeric("x", &[Some(value)])];
        let good = vec![numeric("x", &[Some(1.)])];
        assert_eq!(
            prediction_input_support(&bad, &good),
            Err(SupportError::NonFiniteNumeric)
        );
        assert_eq!(
            prediction_input_support(&good, &bad),
            Err(SupportError::NonFiniteNumeric)
        );
    }
}

#[test]
fn reject_nul_categories_without_echoing_values() {
    let bad = vec![categorical("x", &[Some("private\0value")])];
    let good = vec![categorical("x", &[Some("safe")])];
    assert_eq!(
        prediction_input_support(&bad, &good),
        Err(SupportError::NulCategory)
    );
    assert_eq!(
        prediction_input_support(&good, &bad),
        Err(SupportError::NulCategory)
    );
    assert!(!SupportError::NulCategory.to_string().contains("private"));
}
