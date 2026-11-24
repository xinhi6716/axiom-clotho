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
fn overflowing_raw_sums_are_finite_and_use_only_selected_neighbors() {
    for target in [f64::MAX, -f64::MAX] {
        let model = KnnRegressor::fit(&[vec![0.0], vec![0.0]], &[target, target], 2).unwrap();
        assert_eq!(model.predict(&[vec![0.0]]).unwrap(), vec![target]);
    }
    let model = KnnRegressor::fit(&[vec![0.0], vec![1.0], vec![100.0]], &[2.0, 4.0, f64::MAX], 2).unwrap();
    assert_eq!(model.predict(&[vec![0.0]]).unwrap(), vec![3.0]);
}

fn assert_mean_bits(y: &[f64], expected: f64) {
    let model = KnnRegressor::fit(&vec![vec![0.0]; y.len()], y, y.len()).unwrap();
    let prediction = model.predict(&[vec![0.0]]).unwrap()[0];
    assert_eq!(prediction.to_bits(), expected.to_bits(), "inputs {y:?}");
}

#[test]
fn finite_cancellation_never_invents_a_residual_or_flips_its_sign() {
    for scale in [1.0, 2_f64.powi(110)] {
        let mut terms = [2.5, 1.3, -1e-30, -1.3, -2.5, 1e-30].map(|v| v * scale);
        // Enumerate all permutations without an additional dependency.
        fn permutations(terms: &mut [f64], used: usize) {
            if used == terms.len() {
                assert_mean_bits(terms, 0.0);
                return;
            }
            for i in used..terms.len() {
                terms.swap(used, i);
                permutations(terms, used + 1);
                terms.swap(used, i);
            }
        }
        permutations(&mut terms, 0);
    }
    assert_mean_bits(
        &[2.5, 1.3, -1e-30, 1e-35, -1.3, -2.5, 1e-30],
        1e-35 / 7.0,
    );
}

#[test]
fn extreme_sums_are_exact_across_orders_without_relaxing_range_checks() {
    let max = f64::MAX;
    for small in [1e-15, -1e-15, 1.0] {
        for terms in [
            [max, max, -max, -max, small],
            [max, -max, max, -max, small],
            [small, -max, -max, max, max],
        ] {
            assert_mean_bits(&terms, small / 5.0);
        }
    }
    for target in [max, -max] {
        assert_mean_bits(&vec![target; 20_000], target);
    }
    let model = KnnRegressor::fit(&[vec![0.0], vec![0.0]], &[max, 1e-16], 2).unwrap();
    assert_eq!(
        model.predict(&[vec![0.0]]),
        Err(clotho_core::knn::KnnError::NumericOverflow)
    );
}

#[test]
fn final_rounding_is_nearest_even_in_normal_and_subnormal_ranges() {
    let tiny = f64::from_bits(1);
    let min_normal = f64::MIN_POSITIVE;
    let max_subnormal = f64::from_bits(min_normal.to_bits() - 1);
    let one_next = f64::from_bits(1.0_f64.to_bits() + 1);
    let one_next_even = f64::from_bits(1.0_f64.to_bits() + 2);
    let two_prev = f64::from_bits(2.0_f64.to_bits() - 1);
    let binade2 = min_normal * 2.0;
    let binade2_next = f64::from_bits(binade2.to_bits() + 1);
    for (targets, want) in [
        ([tiny, 0.0], 0.0),
        ([-tiny, 0.0], -0.0),
        ([-0.0, -0.0], 0.0),
        ([tiny, 2.0 * tiny], 2.0 * tiny),
        ([2.0 * tiny, 3.0 * tiny], 2.0 * tiny),
        ([max_subnormal, min_normal], min_normal),
        ([1.0, one_next], 1.0),
        ([one_next, one_next_even], one_next_even),
        ([two_prev, 2.0], 2.0),
    ] {
        assert_mean_bits(&targets, want);
    }
    // Integer quotient lands on the midpoint; the exact division remainder
    // places the mean strictly above it, even though the retained bit is even.
    assert_mean_bits(&[binade2, binade2_next, binade2_next], binade2_next);
}
