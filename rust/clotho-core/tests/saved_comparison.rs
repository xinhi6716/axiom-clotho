use clotho_core::evaluation::EvaluationError;
use clotho_core::saved_comparison::*;

fn contract() -> CohortContract<'static> {
    CohortContract {
        cohort: "chip-batch-1",
        target: "next resistance",
        unit: "ohm",
        horizon: "1 write",
    }
}

fn series<'a>(ids: &'a [&'a str], values: &'a [f64]) -> SavedRegressionSeries<'a> {
    SavedRegressionSeries {
        contract: contract(),
        row_ids: ids,
        values,
    }
}

#[test]
fn aligned_scores_match_hand_calculation_and_retain_contract() {
    let ids = ["dev1:event1", "dev2:event1"];
    let result = compare_saved_regression(
        series(&ids, &[1., 3.]),
        series(&ids, &[1., 2.]),
        series(&ids, &[0., 1.]),
    )
    .unwrap();
    assert!((result.mse_skill.unwrap() - 0.8).abs() < 1e-14);
    assert_eq!(result.contract, contract());
    assert_eq!(result.n_rows, 2);
}

#[test]
fn swapped_roster_is_rejected_for_every_series() {
    let ids = ["a", "b"];
    let swapped = ["b", "a"];
    let s = series(&ids, &[1., 3.]);
    let p = series(&swapped, &[3., 1.]);
    for inputs in [[p, s, s], [s, p, s], [s, s, p]] {
        assert_eq!(
            compare_saved_regression(inputs[0], inputs[1], inputs[2]),
            Err(SavedComparisonError::RowOrderMismatch)
        );
    }
}

#[test]
fn duplicate_keys_fail_even_when_every_series_matches() {
    let s = series(&["a", "a"], &[1., 2.]);
    assert_eq!(
        compare_saved_regression(s, s, s),
        Err(SavedComparisonError::DuplicateRow)
    );
}

#[test]
fn all_semantic_declarations_must_match() {
    let s = series(&["a"], &[1.]);
    for field in 0..4 {
        let mut p = s;
        match field {
            0 => p.contract.cohort = "other",
            1 => p.contract.target = "conductance",
            2 => p.contract.unit = "kohm",
            _ => p.contract.horizon = "2 writes",
        }
        assert_eq!(
            compare_saved_regression(s, p, s),
            Err(SavedComparisonError::ContractMismatch)
        );
        assert_eq!(
            compare_saved_regression(s, s, p),
            Err(SavedComparisonError::ContractMismatch)
        );
    }
}

#[test]
fn malformed_and_nonfinite_inputs_never_get_partial_scores() {
    let s = series(&["a"], &[1.]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            compare_saved_regression(s, series(&["a"], &[bad]), s),
            Err(SavedComparisonError::Numeric(EvaluationError::NonFinite))
        );
    }
    assert_eq!(
        compare_saved_regression(s, series(&[], &[1.]), s),
        Err(SavedComparisonError::Numeric(EvaluationError::Shape))
    );
    assert_eq!(
        compare_saved_regression(s, series(&[], &[]), s),
        Err(SavedComparisonError::Numeric(EvaluationError::Empty))
    );
    assert_eq!(
        compare_saved_regression(s, series(&["b"], &[1.]), s),
        Err(SavedComparisonError::RowOrderMismatch)
    );
}

#[test]
fn invalid_declarations_and_bounded_capacity() {
    let s = series(&["a"], &[1.]);
    for id in ["", " ", "x\0y", "x\ny"] {
        assert_eq!(
            compare_saved_regression(s, series(&[id], &[1.]), s),
            Err(SavedComparisonError::Declaration)
        );
    }
    let long = "x".repeat(MAX_DECLARATION_BYTES + 1);
    let mut p = s;
    p.contract.horizon = &long;
    assert_eq!(
        compare_saved_regression(s, p, s),
        Err(SavedComparisonError::Declaration)
    );
    let ids = vec!["a"; MAX_SAVED_ROWS + 1];
    let values = vec![1.; ids.len()];
    assert_eq!(
        compare_saved_regression(s, series(&ids, &values), s),
        Err(SavedComparisonError::Capacity)
    );
}

#[test]
fn perfect_baseline_is_undefined_ties_and_losses_are_preserved() {
    let ids = ["a", "b"];
    let y = series(&ids, &[1., 3.]);
    let b = series(&ids, &[0., 1.]);
    assert_eq!(compare_saved_regression(y, y, y).unwrap().mse_skill, None);
    assert_eq!(compare_saved_regression(y, b, b).unwrap().mse_skill, Some(0.));
    assert!(
        compare_saved_regression(y, series(&ids, &[-1., -1.]), b)
            .unwrap()
            .mse_skill
            .unwrap()
            < 0.
    );
}

#[test]
fn same_keys_do_not_authenticate_values() {
    let ids = ["a", "b"];
    let y = series(&ids, &[1., 3.]);
    let b = series(&ids, &[0., 1.]);
    // Identical declared keys are not provenance or identity proof. Swapping
    // values while falsely retaining those keys is undetectable by this API.
    let wrong_values = series(&ids, &[3., 1.]);
    assert!(compare_saved_regression(y, wrong_values, b).is_ok());
}
