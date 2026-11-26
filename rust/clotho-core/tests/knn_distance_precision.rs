use clotho_core::knn::{KnnError, KnnRegressor};

#[test]
fn exact_neighbor_is_not_lost_to_a_rounded_scaling_tie() {
    let adjacent = f64::from_bits(7.0_f64.to_bits() + 1);
    assert_eq!(7.0 / 12.0, adjacent / 12.0);
    for (x, y) in [
        (vec![vec![7.0], vec![adjacent], vec![12.0]], vec![0.0, 1.0, 2.0]),
        (vec![vec![adjacent], vec![7.0], vec![12.0]], vec![1.0, 0.0, 2.0]),
    ] {
        let model = KnnRegressor::fit(&x, &y, 1).unwrap();
        assert_eq!(model.predict(&[vec![adjacent]]).unwrap(), vec![1.0]);
        assert_eq!(model.predict(&[vec![7.0]]).unwrap(), vec![0.0]);
    }
}

#[test]
fn large_origins_overflow_fallback_and_genuine_ties_remain_supported() {
    let adjacent = f64::from_bits(7.0_f64.to_bits() + 1);
    let scale = 2.0_f64.powi(900);
    let x = vec![vec![7.0 * scale], vec![adjacent * scale], vec![12.0 * scale]];
    let model = KnnRegressor::fit(&x, &[0.0, 1.0, 2.0], 1).unwrap();
    assert_eq!(model.predict(&[x[1].clone()]).unwrap(), vec![1.0]);
    let extremes = KnnRegressor::fit(
        &[vec![-f64::MAX], vec![f64::MAX]], &[0.0, 1.0], 1,
    ).unwrap();
    assert_eq!(extremes.predict(&[vec![f64::MAX]]).unwrap(), vec![1.0]);
    assert_eq!(extremes.predict(&[vec![0.0]]).unwrap(), vec![0.0]);
    let duplicates = KnnRegressor::fit(&[vec![7.0], vec![7.0]], &[3.0, 4.0], 1).unwrap();
    assert_eq!(duplicates.predict(&[vec![7.0]]).unwrap(), vec![3.0]);
}

#[test]
fn nonzero_distance_lost_in_normalization_is_rejected_not_tied() {
    let tiny = f64::from_bits(1);
    let model = KnnRegressor::fit(
        &[vec![tiny * 2.0], vec![tiny * 3.0], vec![2.0]], &[0.0, 1.0, 2.0], 1,
    ).unwrap();
    assert_eq!(
        model.predict(&[vec![tiny * 3.0]]),
        Err(KnnError::NumericOverflow),
    );
}

#[test]
fn leading_absolute_component_preserves_selection_at_binary64_boundaries() {
    let m = f64::from_bits(1);
    let x = [vec![-0.0], vec![m], vec![2.0 * m], vec![f64::MIN_POSITIVE]];
    let fitted = KnnRegressor::fit(&x, &[-1.0, 2.0, 4.0, 8.0], 1).unwrap();
    assert_eq!(
        fitted.predict(&[vec![0.0], vec![m], vec![3.0 * m], vec![f64::MIN_POSITIVE]]).unwrap(),
        [-1.0, 2.0, 4.0, 8.0]
    );
    let x = [vec![-f64::MAX, 1.0], vec![f64::MAX, 2.0]];
    let fitted = KnnRegressor::fit(&x, &[1.0, 2.0], 1).unwrap();
    assert_eq!(fitted.predict(&[vec![f64::MAX, 2.0]]).unwrap(), [2.0]);
}
