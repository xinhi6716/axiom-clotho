use clotho_core::input_support::{
    prediction_input_support, FeatureColumn, FeatureValues, SupportError,
};
use clotho_core::numeric_bytes::{
    prediction_input_support_bytes, NumericBytesColumn, NumericBytesError,
};

fn bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn col<'a>(name: &'a str, values: &'a [u8]) -> NumericBytesColumn<'a> {
    NumericBytesColumn { name, values }
}
fn compare(training: &[(&str, Vec<f64>)], incoming: &[(&str, Vec<f64>)]) {
    let old = |table: &[(&str, Vec<f64>)]| {
        table
            .iter()
            .map(|(name, values)| FeatureColumn {
                name: name.to_string(),
                values: FeatureValues::Numeric(
                    values
                        .iter()
                        .map(|v| if v.is_nan() { None } else { Some(*v) })
                        .collect(),
                ),
            })
            .collect::<Vec<_>>()
    };
    let train_bytes: Vec<_> = training.iter().map(|(_, v)| bytes(v)).collect();
    let query_bytes: Vec<_> = incoming.iter().map(|(_, v)| bytes(v)).collect();
    let t: Vec<_> = training
        .iter()
        .zip(&train_bytes)
        .map(|((n, _), v)| col(n, v))
        .collect();
    let q: Vec<_> = incoming
        .iter()
        .zip(&query_bytes)
        .map(|((n, _), v)| col(n, v))
        .collect();
    assert_eq!(
        prediction_input_support_bytes(&t, &q).unwrap(),
        prediction_input_support(&old(training), &old(incoming)).unwrap()
    );
}

#[test]
fn counts_unions_reorder_and_unusual_names() {
    compare(
        &[
            ("製程\0", vec![1., 3., f64::NAN]),
            ("", vec![f64::NAN; 3]),
            ("constant", vec![2.; 3]),
        ],
        &[
            ("constant", vec![1., 2., 3., f64::NAN, 2.]),
            ("", vec![1., f64::NAN, 2., 3., f64::NAN]),
            ("製程\0", vec![0., 1., 4., f64::NAN, 3.]),
        ],
    );
}
#[test]
fn all_nan_encodings_are_missing() {
    let nan = [
        f64::from_bits(0x7ff0000000000001),
        f64::from_bits(0xfff8000000000001),
        f64::NAN,
    ];
    compare(
        &[("x", nan.to_vec())],
        &[("x", vec![1., nan[0], nan[1], nan[2]])],
    );
    compare(&[("x", vec![1.])], &[("x", nan.to_vec())]);
}
#[test]
fn binary64_boundaries() {
    compare(
        &[
            ("x", vec![-0., f64::from_bits(1)]),
            ("y", vec![-f64::MAX, f64::MAX]),
        ],
        &[
            (
                "x",
                vec![
                    -f64::from_bits(1),
                    -0.,
                    0.,
                    f64::from_bits(1),
                    f64::MIN_POSITIVE,
                ],
            ),
            ("y", vec![-f64::MAX, 0., f64::MAX, f64::NAN, -0.]),
        ],
    );
}
#[test]
fn deterministic_finite_random_bits() {
    let mut state = 20261104_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f64::from_bits(state & 0xffefffffffffffff)
    };
    for _ in 0..32 {
        let t: Vec<_> = (0..100).map(|_| next()).collect();
        let q: Vec<_> = (0..137).map(|_| next()).collect();
        compare(&[("x", t)], &[("x", q)]);
    }
}
#[test]
fn exact_little_endian_bytes_and_unaligned_slice() {
    let mut storage = vec![99];
    storage.extend([0, 0, 0, 0, 0, 0, 0xf0, 0x3f]); // 1.0 LE
    let query = bytes(&[0., 1., 2.]);
    let report =
        prediction_input_support_bytes(&[col("x", &storage[1..])], &[col("x", &query)]).unwrap();
    assert_eq!(report.rows_with_any_flag, 2);
}
#[test]
fn rejects_infinity_in_either_table() {
    let good = bytes(&[1.]);
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let bad = bytes(&[v]);
        for (t, q) in [(&good, &bad), (&bad, &good)] {
            assert_eq!(
                prediction_input_support_bytes(&[col("x", t)], &[col("x", q)]),
                Err(NumericBytesError::InfiniteNumeric)
            );
        }
    }
}
#[test]
fn rejects_shape_schema_and_duplicates_on_both_sides() {
    let good = bytes(&[1.]);
    let long = bytes(&[1., 2.]);
    let oversized = vec![0; 20001 * 8];
    let many_names: Vec<_> = (0..101).map(|i| i.to_string()).collect();
    let cases = vec![
        (vec![], NumericBytesError::FeatureCount),
        (vec![col("x", &[])], NumericBytesError::RowCount),
        (vec![col("x", &oversized)], NumericBytesError::RowCount),
        (
            vec![col("x", &[0; 9])],
            NumericBytesError::InvalidByteLength,
        ),
        (
            vec![col("x", &good), col("y", &[0; 9])],
            NumericBytesError::InvalidByteLength,
        ),
        (
            vec![col("x", &good), col("y", &long)],
            NumericBytesError::RaggedColumns,
        ),
        (
            vec![col("x", &good), col("x", &good)],
            NumericBytesError::DuplicateNames,
        ),
        (vec![col("y", &good)], NumericBytesError::SchemaMismatch),
        (
            vec![col("x", &good), col("y", &good)],
            NumericBytesError::SchemaMismatch,
        ),
        (
            many_names.iter().map(|n| col(n, &good)).collect(),
            NumericBytesError::FeatureCount,
        ),
    ];
    for (invalid, error) in cases {
        assert_eq!(
            prediction_input_support_bytes(&invalid, &[col("x", &good)]),
            Err(error)
        );
        assert_eq!(
            prediction_input_support_bytes(&[col("x", &good)], &invalid),
            Err(error)
        );
    }
}
#[test]
fn validates_all_shapes_before_counting() {
    let infinite = bytes(&[f64::INFINITY]);
    assert_eq!(
        prediction_input_support_bytes(&[col("x", &infinite)], &[col("x", &[0; 9])]),
        Err(NumericBytesError::InvalidByteLength)
    );
}
#[test]
fn inclusive_row_and_feature_limits() {
    let names: Vec<_> = (0..100).map(|i| format!("x{i}")).collect();
    let values = vec![0; 20000 * 8];
    let t: Vec<_> = names.iter().map(|n| col(n, &values)).collect();
    let q: Vec<_> = t.iter().copied().rev().collect();
    let report = prediction_input_support_bytes(&t, &q).unwrap();
    assert_eq!(report.n_training_rows, 20000);
    assert_eq!(report.n_prediction_rows, 20000);
    assert_eq!(report.features.len(), 100);
    assert_eq!(report.rows_with_any_flag, 0);
}
#[test]
fn old_option_api_still_rejects_nan() {
    let t = [FeatureColumn {
        name: "x".into(),
        values: FeatureValues::Numeric(vec![Some(f64::NAN)]),
    }];
    assert_eq!(
        prediction_input_support(&t, &t),
        Err(SupportError::NonFiniteNumeric)
    );
}
#[test]
fn errors_are_fixed_and_private() {
    for error in [
        NumericBytesError::FeatureCount,
        NumericBytesError::RowCount,
        NumericBytesError::InvalidByteLength,
        NumericBytesError::RaggedColumns,
        NumericBytesError::DuplicateNames,
        NumericBytesError::SchemaMismatch,
        NumericBytesError::InfiniteNumeric,
    ] {
        assert!(!error.to_string().contains("secret"));
        assert!(!error.to_string().is_empty());
    }
    let secret = bytes(&[f64::INFINITY]);
    let e = prediction_input_support_bytes(&[col("secret", &secret)], &[col("secret", &secret)])
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "Numeric values must be finite or NaN for missing."
    );
}
