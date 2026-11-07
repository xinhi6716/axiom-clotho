//! Synthetic API demonstration only, not measured semiconductor data.
use clotho_core::knn::KnnRegressor;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Columns: anneal temperature (C), film thickness (nm).
    // Targets: illustrative dimensionless response, NOT experimental measurements.
    let training = vec![vec![300.0, 10.0], vec![400.0, 10.0], vec![500.0, 20.0]];
    let model = KnnRegressor::fit(&training, &[1.0, 2.0, 4.0], 2)?;
    let predictions = model.predict(&[vec![350.0, 10.0]])?;
    println!(
        "synthetic uniform 2-neighbor prediction: {}",
        predictions[0]
    );
    assert!((predictions[0] - 1.5).abs() < 1e-12);
    Ok(())
}
