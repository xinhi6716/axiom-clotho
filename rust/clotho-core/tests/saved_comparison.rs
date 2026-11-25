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

#[test]
fn scalar_reuse_preserves_perfect_model_and_subnormal_boundaries() {
    let ids = ["a"];
    let min = f64::from_bits(1);
    let observed = series(&ids, &[0.0]);
    let tiny_values = [min];
    let tiny = series(&ids, &tiny_values);
    let ordinary = series(&ids, &[1.0]);

    let perfect = compare_saved_regression(observed, observed, tiny).unwrap();
    assert_eq!(perfect.model_rmse.to_bits(), 0);
    assert_eq!(perfect.baseline_rmse.to_bits(), 1);
    assert_eq!(perfect.mse_skill, Some(1.0));

    let undefined = compare_saved_regression(observed, tiny, observed).unwrap();
    assert_eq!(undefined.model_rmse.to_bits(), 1);
    assert_eq!(undefined.baseline_rmse.to_bits(), 0);
    assert_eq!(undefined.mse_skill, None);

    for (model, baseline) in [
        (tiny, ordinary),
        (ordinary, tiny),
        (series(&ids, &[f64::MAX]), series(&ids, &[f64::MIN_POSITIVE])),
    ] {
        assert_eq!(
            compare_saved_regression(observed, model, baseline),
            Err(SavedComparisonError::Numeric(EvaluationError::NumericOverflow))
        );
    }
}

#[test]
fn independent_roster_storage_can_match_but_values_still_validate() {
    let ids = ["lot|α:1", "lot|α:2"];
    let model_storage = ids.map(str::to_owned);
    let baseline_storage = ids.map(str::to_owned);
    let model_ids = [model_storage[0].as_str(), model_storage[1].as_str()];
    let baseline_ids = [baseline_storage[0].as_str(), baseline_storage[1].as_str()];
    let observations = series(&ids, &[1.0, 2.0]);
    let model = series(&model_ids, &[2.0, 1.0]);
    let baseline = series(&baseline_ids, &[3.0, 4.0]);
    assert_eq!(
        compare_saved_regression(observations, model, baseline)
            .unwrap()
            .mse_skill,
        Some(0.75)
    );
    let malformed = series(&model_ids, &[2.0, f64::NAN]);
    assert_eq!(
        compare_saved_regression(observations, malformed, baseline),
        Err(SavedComparisonError::Numeric(EvaluationError::NonFinite))
    );
}

#[test]
fn mismatching_rosters_preserve_individual_diagnostics_before_comparison() {
    let observed = series(&["a", "b"], &[1.0, 2.0]);
    let reordered = series(&["b", "a"], &[1.0, 2.0]);
    let duplicate = series(&["a", "a"], &[1.0, 2.0]);
    let invalid_text = series(&["a", "\n"], &[1.0, 2.0]);
    let mut wrong_contract = reordered;
    wrong_contract.contract.unit = "different";

    // Existing series-local validity precedes cross-series checks even if
    // an earlier valid series already disagrees with the observed roster.
    for (model, baseline, expected) in [
        (reordered, duplicate, SavedComparisonError::DuplicateRow),
        (duplicate, wrong_contract, SavedComparisonError::DuplicateRow),
        (wrong_contract, invalid_text, SavedComparisonError::Declaration),
        (wrong_contract, observed, SavedComparisonError::ContractMismatch),
        (reordered, observed, SavedComparisonError::RowOrderMismatch),
    ] {
        assert_eq!(
            compare_saved_regression(observed, model, baseline),
            Err(expected)
        );
    }
}
