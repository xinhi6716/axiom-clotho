use clotho_core::numeric_bytes::{
    prediction_input_support_bytes, NumericBytesColumn, NumericBytesError, NumericSupportReference,
};
use std::sync::Arc;

fn bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn col<'a>(name: &'a str, values: &'a [u8]) -> NumericBytesColumn<'a> {
    NumericBytesColumn { name, values }
}

#[test]
fn summary_owns_names_and_not_training_buffers() {
    let reference = {
        let mut values = bytes(&[0., 1., f64::NAN]);
        let mut name = String::from("temperature");
        let result = NumericSupportReference::from_training(&[col(&name, &values)]).unwrap();
        values.fill(0);
        name.clear();
        result
    };
    let q = bytes(&[-1., 2., f64::NAN]);
    let original = bytes(&[0., 1., f64::NAN]);
    assert_eq!(
        reference.compare(&[col("temperature", &q)]).unwrap(),
        prediction_input_support_bytes(&[col("temperature", &original)], &[col("temperature", &q)])
            .unwrap()
    );
}

#[test]
fn query_order_and_failure_cannot_change_reference() {
    let t = bytes(&[0., 1.]);
    let reference = NumericSupportReference::from_training(&[col("x", &t)]).unwrap();
    let q = bytes(&[2., f64::NAN]);
    let expected = reference.compare(&[col("x", &q)]).unwrap();
    for _ in 0..4 {
        let outside = bytes(&[-100., 100.]);
        reference.compare(&[col("x", &outside)]).unwrap();
        let invalid = bytes(&[f64::INFINITY]);
        assert_eq!(
            reference.compare(&[col("x", &invalid)]),
            Err(NumericBytesError::InfiniteNumeric)
        );
        assert_eq!(reference.compare(&[col("x", &q)]).unwrap(), expected);
    }
}

#[test]
fn complete_report_reordering_missing_and_unicode() {
    let a = bytes(&[f64::NAN; 3]);
    let b = bytes(&[-0., f64::from_bits(1), f64::NAN]);
    let qa = bytes(&[0., f64::NAN, 2., -1.]);
    let qb = bytes(&[-f64::from_bits(1), 0., f64::from_bits(1), f64::NAN]);
    let t = [col("製程\0", &a), col("", &b)];
    let q = [col("", &qb), col("製程\0", &qa)];
    assert_eq!(
        NumericSupportReference::from_training(&t)
            .unwrap()
            .compare(&q)
            .unwrap(),
        prediction_input_support_bytes(&t, &q).unwrap()
    );
}

#[test]
fn arbitrary_finite_bits_and_nan_payloads() {
    let mut state = 731_u64;
    let mut generate = |n| -> Vec<f64> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                f64::from_bits(state & 0xffef_ffff_ffff_ffff)
            })
            .collect()
    };
    for _ in 0..64 {
        let mut t = generate(31);
        t.extend([
            f64::from_bits(0x7ff0000000000001),
            f64::from_bits(0xfff8000000000001),
        ]);
        let tb = bytes(&t);
        let reference = NumericSupportReference::from_training(&[col("x", &tb)]).unwrap();
        for _ in 0..3 {
            let qb = bytes(&generate(17));
            assert_eq!(
                reference.compare(&[col("x", &qb)]).unwrap(),
                prediction_input_support_bytes(&[col("x", &tb)], &[col("x", &qb)]).unwrap()
            );
        }
    }
}

#[test]
fn construction_rejects_invalid_shapes_schema_and_infinity() {
    let one = bytes(&[1.]);
    let two = bytes(&[1., 2.]);
    let inf = bytes(&[f64::NEG_INFINITY]);
    let cases = [
        (vec![], NumericBytesError::FeatureCount),
        (vec![col("x", &one); 101], NumericBytesError::FeatureCount),
        (vec![col("x", &[])], NumericBytesError::RowCount),
        (
            vec![col("x", &[1, 2])],
            NumericBytesError::InvalidByteLength,
        ),
        (
            vec![col("x", &one), col("x", &one)],
            NumericBytesError::DuplicateNames,
        ),
        (
            vec![col("x", &one), col("y", &two)],
            NumericBytesError::RaggedColumns,
        ),
        (vec![col("x", &inf)], NumericBytesError::InfiniteNumeric),
    ];
    for (columns, error) in cases {
        assert!(
            matches!(NumericSupportReference::from_training(&columns), Err(actual) if actual == error)
        );
    }
    let too_large = vec![0; 20_001 * 8];
    assert!(matches!(
        NumericSupportReference::from_training(&[col("x", &too_large)]),
        Err(NumericBytesError::RowCount)
    ));
}

#[test]
fn queries_reject_invalid_shapes_schema_and_infinity() {
    let one = bytes(&[1.]);
    let two = bytes(&[1., 2.]);
    let inf = bytes(&[f64::INFINITY]);
    let reference = NumericSupportReference::from_training(&[col("x", &one)]).unwrap();
    let cases = [
        (vec![], NumericBytesError::FeatureCount),
        (vec![col("x", &one); 101], NumericBytesError::FeatureCount),
        (vec![col("x", &[])], NumericBytesError::RowCount),
        (
            vec![col("x", &[0; 9])],
            NumericBytesError::InvalidByteLength,
        ),
        (
            vec![col("x", &one), col("x", &one)],
            NumericBytesError::DuplicateNames,
        ),
        (
            vec![col("x", &one), col("y", &two)],
            NumericBytesError::RaggedColumns,
        ),
        (vec![col("y", &one)], NumericBytesError::SchemaMismatch),
        (
            vec![col("x", &one), col("y", &one)],
            NumericBytesError::SchemaMismatch,
        ),
        (vec![col("x", &inf)], NumericBytesError::InfiniteNumeric),
    ];
    for (columns, error) in cases {
        assert_eq!(reference.compare(&columns), Err(error));
    }
    let too_large = vec![0; 20_001 * 8];
    assert_eq!(
        reference.compare(&[col("x", &too_large)]),
        Err(NumericBytesError::RowCount)
    );
}

#[test]
fn shared_reference_is_send_sync_and_repeatable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NumericSupportReference>();
    let t = bytes(&[-f64::MAX, f64::MAX, f64::NAN]);
    let reference = Arc::new(NumericSupportReference::from_training(&[col("x", &t)]).unwrap());
    let q = bytes(&[0., f64::NAN]);
    let expected = reference.compare(&[col("x", &q)]).unwrap();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let r = reference.clone();
            let q = q.clone();
            std::thread::spawn(move || r.compare(&[col("x", &q)]).unwrap())
        })
        .collect();
    for handle in threads {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn inclusive_feature_and_row_limits() {
    let t = bytes(&vec![0.; 20_000]);
    let q = bytes(&vec![1.; 20_000]);
    let names: Vec<_> = (0..100).map(|i| format!("x{i}")).collect();
    let training: Vec<_> = names.iter().map(|n| col(n, &t)).collect();
    let incoming: Vec<_> = names.iter().rev().map(|n| col(n, &q)).collect();
    assert_eq!(
        NumericSupportReference::from_training(&training)
            .unwrap()
            .compare(&incoming)
            .unwrap(),
        prediction_input_support_bytes(&training, &incoming).unwrap()
    );
}

#[test]
fn specialized_query_loops_preserve_flags_and_reject_late_infinities() {
    let valid = f64::from_bits(1);
    let signed_nan = f64::from_bits(0xfff0_0000_0000_0001);
    let train_a = bytes(&[-0.0, valid]);
    let train_b = bytes(&[signed_nan, f64::NAN]);
    let query_a = bytes(&[-0.0, 0.0, valid, -valid, 2.0, signed_nan]);
    let query_b = bytes(&[7.0, signed_nan, 8.0, signed_nan, 9.0, signed_nan]);
    let training = [col("bounded", &train_a), col("missing", &train_b)];
    let incoming = [col("missing", &query_b), col("bounded", &query_a)];
    let reference = NumericSupportReference::from_training(&training).unwrap();
    let report = reference.compare(&incoming).unwrap();
    assert_eq!(
        prediction_input_support_bytes(&training, &incoming),
        Ok(report.clone())
    );
    assert_eq!(report.rows_with_any_flag, 6);
    assert_eq!(report.row_flag_counts.numeric_outside_training_range, 2);
    assert_eq!(report.row_flag_counts.missing_value, 3);
    assert_eq!(
        report.row_flag_counts.non_missing_value_in_all_missing_training_feature,
        3
    );

    for column in 0..incoming.len() {
        let mut invalid_bytes = incoming[column].values.to_vec();
        invalid_bytes[5 * 8..].copy_from_slice(&f64::INFINITY.to_le_bytes());
        let mut invalid = incoming;
        invalid[column].values = &invalid_bytes;
        assert_eq!(
            reference.compare(&invalid),
            Err(NumericBytesError::InfiniteNumeric)
        );
        assert_eq!(
            prediction_input_support_bytes(&training, &invalid),
            Err(NumericBytesError::InfiniteNumeric)
        );
    }
    assert_eq!(reference.compare(&incoming), Ok(report));
}

#[test]
fn large_clean_and_mixed_columns_keep_exact_counts() {
    let training_bytes = bytes(&[-1.0, 1.0]);
    let training = [col("x", &training_bytes)];
    let reference = NumericSupportReference::from_training(&training).unwrap();
    for (mixed, outside, missing) in [(false, 0, 0), (true, 512, 256)] {
        let mut query_bytes = bytes(
            &(0..1024)
                .map(|row| match (mixed, row % 4) {
                    (true, 0) => -2.0,
                    (true, 1) => 2.0,
                    (true, 2) => f64::NAN,
                    _ => 1.0,
                })
                .collect::<Vec<_>>(),
        );
        let query = [col("x", &query_bytes)];
        let report = reference.compare(&query).unwrap();
        assert_eq!(report.rows_with_any_flag, outside + missing);
        assert_eq!(report.row_flag_counts.numeric_outside_training_range, outside);
        assert_eq!(report.row_flag_counts.missing_value, missing);
        assert_eq!(prediction_input_support_bytes(&training, &query), Ok(report));

        query_bytes[1023 * 8..].copy_from_slice(&f64::NEG_INFINITY.to_le_bytes());
        let invalid = [col("x", &query_bytes)];
        assert_eq!(reference.compare(&invalid), Err(NumericBytesError::InfiniteNumeric));
        assert_eq!(
            prediction_input_support_bytes(&training, &invalid),
            Err(NumericBytesError::InfiniteNumeric)
        );
    }
}
