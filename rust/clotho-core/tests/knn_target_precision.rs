use clotho_core::knn::KnnRegressor;

#[test]
fn cancellation_residual_is_not_quantized_by_max_target_scaling() {
    let small = 1e-15;
    for y in [
        [1e308, -1e308, small], [1e308, small, -1e308],
        [-1e308, 1e308, small], [-1e308, small, 1e308],
        [small, 1e308, -1e308], [small, -1e308, 1e308],
    ] {
        let model = KnnRegressor::fit(&[vec![0.0], vec![0.0], vec![0.0]], &y, 3).unwrap();
        assert_eq!(model.predict(&[vec![0.0]]).unwrap()[0].to_bits(), (small / 3.0).to_bits());
    }
}

#[test]
fn overflowing_raw_sums_use_scaled_fallback_and_selected_neighbors_only() {
    for target in [f64::MAX, -f64::MAX] {
        let model = KnnRegressor::fit(&[vec![0.0], vec![0.0]], &[target, target], 2).unwrap();
        assert_eq!(model.predict(&[vec![0.0]]).unwrap(), vec![target]);
    }
    let model = KnnRegressor::fit(&[vec![0.0], vec![1.0], vec![100.0]], &[2.0, 4.0, f64::MAX], 2).unwrap();
    assert_eq!(model.predict(&[vec![0.0]]).unwrap(), vec![3.0]);
}
