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
