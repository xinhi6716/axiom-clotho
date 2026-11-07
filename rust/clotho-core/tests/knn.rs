use clotho_core::knn::{KnnError, KnnRegressor};
#[test]
fn predicts_known_neighbor_and_uniform_mean() {
    let x = vec![vec![0.0], vec![2.0], vec![4.0]];
    assert_eq!(
        KnnRegressor::fit(&x, &[1.0, 5.0, 9.0], 1)
            .unwrap()
            .predict(&[vec![1.8]])
            .unwrap(),
        vec![5.0]
    );
    assert!(
        (KnnRegressor::fit(&x, &[1.0, 5.0, 9.0], 2)
            .unwrap()
            .predict(&[vec![1.0]])
            .unwrap()[0]
            - 3.0)
            .abs()
            < 1e-14
    );
}
#[test]
fn uses_training_scales_and_is_batch_independent() {
    let model = KnnRegressor::fit(&[vec![1.0, 1000.0], vec![2.0, 0.0]], &[10.0, 20.0], 1).unwrap();
    let alone = model.predict(&[vec![1.9, 900.0]]).unwrap();
    assert_eq!(alone, vec![10.0]);
    assert_eq!(
        model
            .predict(&[vec![1.9, 900.0], vec![10000.0, 0.0]])
            .unwrap()[0],
        alone[0]
    );
}
#[test]
fn ties_use_row_order_and_constant_features_work() {
    let m = KnnRegressor::fit(&[vec![0.0, 0.0], vec![0.0, 0.0]], &[2.0, 4.0], 1).unwrap();
    assert_eq!(m.predict(&[vec![0.0, 0.0]]).unwrap(), vec![2.0]);
}
#[test]
fn extreme_targets_and_distances_do_not_silently_overflow() {
    let m =
        KnnRegressor::fit(&[vec![-f64::MAX], vec![f64::MAX]], &[f64::MAX, f64::MAX], 2).unwrap();
    assert_eq!(m.predict(&[vec![0.0]]).unwrap(), vec![f64::MAX]);
    let tiny = KnnRegressor::fit(&[vec![f64::MIN_POSITIVE]], &[1.0], 1).unwrap();
    assert_eq!(
        tiny.predict(&[vec![f64::MAX]]),
        Err(KnnError::NumericOverflow)
    );
}
#[test]
fn rejects_invalid_inputs() {
    assert_eq!(KnnRegressor::fit(&[], &[], 1).unwrap_err(), KnnError::Empty);
    assert_eq!(
        KnnRegressor::fit(&[vec![]], &[1.0], 1).unwrap_err(),
        KnnError::Shape
    );
    assert_eq!(
        KnnRegressor::fit(&[vec![1.0]], &[], 1).unwrap_err(),
        KnnError::Shape
    );
    for k in [0, 2] {
        assert_eq!(
            KnnRegressor::fit(&[vec![1.0]], &[1.0], k).unwrap_err(),
            KnnError::InvalidK
        );
    }
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            KnnRegressor::fit(&[vec![v]], &[1.0], 1).unwrap_err(),
            KnnError::NonFinite
        );
        assert_eq!(
            KnnRegressor::fit(&[vec![1.0]], &[v], 1).unwrap_err(),
            KnnError::NonFinite
        );
    }
    let m = KnnRegressor::fit(&[vec![1.0]], &[1.0], 1).unwrap();
    assert_eq!(m.predict(&[]), Err(KnnError::Empty));
    assert_eq!(m.predict(&[vec![1.0, 2.0]]), Err(KnnError::Shape));
    assert_eq!(m.predict(&[vec![f64::NAN]]), Err(KnnError::NonFinite));
}

#[test]
fn compensated_mean_preserves_cancelling_small_terms() {
    let x = vec![vec![0.0]; 3];
    for targets in [[1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [1.0, -1e16, 1e16]] {
        let result = KnnRegressor::fit(&x, &targets, 3)
            .unwrap()
            .predict(&[vec![0.0]])
            .unwrap()[0];
        assert!((result - 1.0 / 3.0).abs() < 1e-15);
    }
}

#[test]
fn rejects_ragged_and_over_capacity_tables() {
    assert_eq!(
        KnnRegressor::fit(&[vec![1.0], vec![1.0, 2.0]], &[1.0, 2.0], 1).unwrap_err(),
        KnnError::Shape
    );
    assert_eq!(
        KnnRegressor::fit(&[vec![0.0; 101]], &[1.0], 1).unwrap_err(),
        KnnError::Capacity
    );
    assert_eq!(
        KnnRegressor::fit(&vec![vec![0.0]; 20_001], &vec![1.0; 20_001], 1).unwrap_err(),
        KnnError::Capacity
    );
    let m = KnnRegressor::fit(&[vec![0.0]], &[1.0], 1).unwrap();
    assert_eq!(m.predict(&vec![vec![0.0]; 20_001]), Err(KnnError::Capacity));
}

#[test]
fn rejects_destructive_scaling_underflow() {
    assert_eq!(
        KnnRegressor::fit(
            &[vec![0.0], vec![1e-300], vec![f64::MAX]],
            &[10.0, 20.0, 30.0],
            1
        )
        .unwrap_err(),
        KnnError::NumericOverflow
    );
    let m = KnnRegressor::fit(&[vec![f64::MAX]], &[1.0], 1).unwrap();
    assert_eq!(m.predict(&[vec![1e-300]]), Err(KnnError::NumericOverflow));
    let m = KnnRegressor::fit(
        &vec![vec![0.0]; 3],
        &[f64::MAX, -f64::MAX, f64::MIN_POSITIVE],
        3,
    )
    .unwrap();
    assert_eq!(m.predict(&[vec![0.0]]), Err(KnnError::NumericOverflow));
}
#[test]
fn zero_and_negative_targets_are_valid() {
    for targets in [[0.0, 0.0], [-2.0, -4.0], [-4.0, 2.0]] {
        let m = KnnRegressor::fit(&[vec![0.0], vec![1.0]], &targets, 2).unwrap();
        assert!(
            (m.predict(&[vec![0.5]]).unwrap()[0] - (targets[0] + targets[1]) / 2.0).abs() < 1e-14
        );
    }
}
