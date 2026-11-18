use clotho_core::regression::{MeanRegressor, RegressionError};

#[test]
fn restores_validated_summary_and_preserves_prediction_checks() {
    assert_eq!(MeanRegressor::from_mean(0, 1.0).unwrap_err(), RegressionError::Shape);
    assert_eq!(MeanRegressor::from_mean(101, 1.0).unwrap_err(), RegressionError::Capacity);
    for mean in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(MeanRegressor::from_mean(1, mean).unwrap_err(), RegressionError::NonFinite);
    }
    for mean in [0.0, -0.0, f64::from_bits(1), f64::MAX, -f64::MAX] {
        let model = MeanRegressor::from_mean(1, mean).unwrap();
        assert_eq!(model.predict(&[vec![1.0]]).unwrap()[0].to_bits(), mean.to_bits());
        assert!(model.predict(&[]).is_err());
        assert!(model.predict(&[vec![]]).is_err());
        assert!(model.predict(&[vec![f64::NAN]]).is_err());
        assert!(model.predict(&vec![vec![1.0]; 20_001]).is_err());
    }
}
