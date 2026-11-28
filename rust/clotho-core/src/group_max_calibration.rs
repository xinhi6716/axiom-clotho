//! Fit-free split calibration of one maximum residual per declared group.
//!
//! Any group-level guarantee requires exchangeable calibration and new-group
//! scores under the same roster construction rule and a predictor fixed
//! independently of calibration. This function checks neither assumption.
//! It does not consume query labels, construct bounds, or read any files.

use std::collections::{BTreeMap, HashMap, HashSet};

pub const MAX_CALIBRATION_ROWS: usize = 20_000;
pub const MAX_IDENTIFIER_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowKey<'a> {
    pub group: &'a str,
    pub row: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct CalibrationRow<'a> {
    pub key: RowKey<'a>,
    pub observed: f64,
    pub prediction: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroupMaxCutoff {
    Finite {
        groups: usize,
        rank: usize,
        radius: f64,
    },
    InsufficientFinite {
        groups: usize,
        rank: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupMaxError {
    Capacity,
    Coverage,
    EmptyRoster,
    InvalidKey,
    RosterMismatch,
    NonfiniteResidual,
}

pub(crate) fn valid_key(key: RowKey<'_>) -> bool {
    [key.group, key.row].iter().all(|s| {
        s.len() <= MAX_IDENTIFIER_BYTES && !s.trim().is_empty() && !s.chars().any(char::is_control)
    })
}

/// Compute the 1-indexed rank ceil((m+1) numerator/denominator), where
/// m is the number of calibration groups. Coverage is an exact rational in
/// (0, 1). Even when the rank cannot yield a finite cutoff, validate all rows.
/// Roster identity is order-independent and is matched before computing scores.
pub fn calibrate_group_max(
    roster: &[RowKey<'_>],
    rows: &[CalibrationRow<'_>],
    numerator: u64,
    denominator: u64,
) -> Result<GroupMaxCutoff, GroupMaxError> {
    if roster.len() > MAX_CALIBRATION_ROWS || rows.len() > MAX_CALIBRATION_ROWS {
        return Err(GroupMaxError::Capacity);
    }
    if numerator == 0 || numerator >= denominator {
        return Err(GroupMaxError::Coverage);
    }
    if roster.is_empty() {
        return Err(GroupMaxError::EmptyRoster);
    }
    let same_length = roster.len() == rows.len();
    let first_difference = roster.iter().zip(rows).position(|(key, row)| *key != row.key);
    let aligned = same_length && first_difference.is_none();
    // Exactly aligned keys are already checked by the declared roster.
    // Otherwise keep the original precedence: malformed supplied keys
    // outrank roster mismatch, even when lengths differ.
    if !roster.iter().copied().all(valid_key)
        || (!aligned && !rows.iter().all(|r| valid_key(r.key)))
    {
        return Err(GroupMaxError::InvalidKey);
    }
    let order = match_roster(roster, rows, first_difference).ok_or(GroupMaxError::RosterMismatch)?;
    let (groups, rank, maxima) = match order {
        RosterMatch::InGroupOrder => {
            let maxima = ordered_group_maxima(rows)?;
            let groups = maxima.len();
            let rank = cutoff_rank(groups, numerator, denominator);
            if rank > groups {
                return Ok(GroupMaxCutoff::InsufficientFinite { groups, rank });
            }
            (groups, rank, maxima)
        }
        RosterMatch::Arbitrary => {
            let (groups, rank, maxima) = arbitrary_group_maxima(rows, numerator, denominator)?;
            if rank > groups {
                return Ok(GroupMaxCutoff::InsufficientFinite { groups, rank });
            }
            (groups, rank, maxima)
        }
    };
    Ok(GroupMaxCutoff::Finite {
        groups,
        rank,
        radius: select_radius(maxima, rank),
    })
}

fn arbitrary_group_maxima(
    rows: &[CalibrationRow<'_>],
    numerator: u64,
    denominator: u64,
) -> Result<(usize, usize, Vec<f64>), GroupMaxError> {
    // A tree is cheap for a handful of keys. Promote once at higher
    // cardinality, where repeated lexical comparisons dominate.
    const PROMOTE_AT: usize = 32;
    let mut small = BTreeMap::<_, f64>::new();
    let mut remaining = rows.iter();
    while let Some(row) = remaining.next() {
        let score = residual(row)?;
        let current = small.entry(row.key.group).or_insert(0.0);
        *current = current.max(score);
        if small.len() == PROMOTE_AT && remaining.len() >= PROMOTE_AT {
            // Finite residuals always have nonnegative bits (+0 included).
            // Hash iteration does not affect the selected order statistic.
            let mut large = HashMap::with_capacity(PROMOTE_AT * 2);
            large.extend(small);
            for rest in remaining {
                let score = residual(rest)?;
                let current = large.entry(rest.key.group).or_insert(0.0);
                *current = current.max(score);
            }
            let groups = large.len();
            let rank = cutoff_rank(groups, numerator, denominator);
            let maxima = if rank <= groups { large.into_values().collect() } else { Vec::new() };
            return Ok((groups, rank, maxima));
        }
    }
    let groups = small.len();
    let rank = cutoff_rank(groups, numerator, denominator);
    let maxima = if rank <= groups { small.into_values().collect() } else { Vec::new() };
    Ok((groups, rank, maxima))
}

fn select_radius(mut maxima: Vec<f64>, rank: usize) -> f64 {
    debug_assert!((1..=maxima.len()).contains(&rank));
    // Only the selected order statistic is exposed; the scores on either
    // side need not be sorted or returned.
    if rank == 1 {
        maxima.into_iter().min_by(f64::total_cmp).unwrap()
    } else if rank == maxima.len() {
        maxima.into_iter().max_by(f64::total_cmp).unwrap()
    } else {
        *maxima.select_nth_unstable_by(rank - 1, f64::total_cmp).1
    }
}

fn cutoff_rank(groups: usize, numerator: u64, denominator: u64) -> usize {
    // groups <= MAX_CALIBRATION_ROWS; even the largest u64 fraction fits u128.
    let n = (groups as u128 + 1) * u128::from(numerator);
    let d = u128::from(denominator);
    ((n + d - 1) / d) as usize
}

fn residual(row: &CalibrationRow<'_>) -> Result<f64, GroupMaxError> {
    if !row.observed.is_finite() || !row.prediction.is_finite() {
        return Err(GroupMaxError::NonfiniteResidual);
    }
    let score = (row.observed - row.prediction).abs();
    if !score.is_finite() {
        return Err(GroupMaxError::NonfiniteResidual);
    }
    Ok(score)
}

fn ordered_group_maxima(rows: &[CalibrationRow<'_>]) -> Result<Vec<f64>, GroupMaxError> {
    let mut maxima = Vec::new();
    let Some(first) = rows.first() else {
        return Ok(maxima);
    };
    let mut previous_group = first.key.group;
    let mut maximum = 0.0_f64;
    for row in rows {
        let score = residual(row)?;
        let group = row.key.group;
        if previous_group != group {
            maxima.push(maximum);
            previous_group = group;
            maximum = 0.0;
        }
        maximum = maximum.max(score);
    }
    maxima.push(maximum);
    Ok(maxima)
}

enum RosterMatch {
    InGroupOrder,
    Arbitrary,
}

fn match_roster(
    roster: &[RowKey<'_>],
    rows: &[CalibrationRow<'_>],
    first_difference: Option<usize>,
) -> Option<RosterMatch> {
    if roster.len() != rows.len() {
        return None;
    }
    // A strictly ordered roster proves uniqueness. If the supplied keys
    // follow it directly or in reverse, there is nothing to index.
    if first_difference.is_some_and(|i| i + 1 == roster.len()) {
        // With every other pair unchanged, one differing final key cannot
        // be a permutation. This also rejects a repeated last key cheaply.
        return None;
    }
    if first_difference.is_none()
        || roster
            .iter()
            .zip(rows.iter().rev())
            .all(|(key, row)| *key == row.key)
    {
        let ascending = roster.first() <= roster.last();
        if roster.windows(2).all(|pair| {
            if ascending {
                pair[0] < pair[1]
            } else {
                pair[0] > pair[1]
            }
        }) {
            // Whole keys are ordered by (group, row). Exact alignment or
            // reversal therefore keeps each group's rows contiguous.
            return Some(RosterMatch::InGroupOrder);
        }
    }

    // Consume each unique declared pair exactly once, irrespective of order.
    // A repeated or unexpected supplied key cannot remove an entry.
    let mut remaining = HashSet::with_capacity(roster.len());
    if roster.iter().any(|key| !remaining.insert((key.group, key.row)))
        || rows
            .iter()
            .any(|r| !remaining.remove(&(r.key.group, r.key.row)))
    {
        return None;
    }
    debug_assert!(remaining.is_empty());
    Some(RosterMatch::Arbitrary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key<'a>(group: &'a str, row: &'a str) -> RowKey<'a> {
        RowKey { group, row }
    }

    fn obs(key: RowKey<'_>, observed: f64) -> CalibrationRow<'_> {
        CalibrationRow {
            key,
            observed,
            prediction: 0.0,
        }
    }

    #[test]
    fn promoted_arbitrary_group_maxima_preserve_ranks_and_validate_the_tail() {
        let names: Vec<_> = (0..80).map(|i| format!("g{i:02}")).collect();
        let roster: Vec<_> = names.iter().flat_map(|name| [key(name, "a"), key(name, "b")]).collect();
        let sorted: Vec<_> = roster.iter().enumerate().map(|(i, &key)| {
            obs(key, if i % 7 == 0 { -0.0 } else { ((i >> 1) % 13) as f64 })
        }).collect();
        let mut shuffled = sorted.clone();
        shuffled.rotate_left(51);
        for numerator in 1..=80 {
            assert_eq!(
                calibrate_group_max(&roster, &shuffled, numerator, 81),
                calibrate_group_max(&roster, &sorted, numerator, 81)
            );
        }
        assert_eq!(
            calibrate_group_max(&roster, &shuffled, 99, 100),
            Ok(GroupMaxCutoff::InsufficientFinite { groups: 80, rank: 81 })
        );
        shuffled[159].observed = f64::NAN;
        assert_eq!(
            calibrate_group_max(&roster, &shuffled, 99, 100),
            Err(GroupMaxError::NonfiniteResidual)
        );
    }

    #[test]
    fn aligned_keys_are_still_validated_before_duplicate_and_numeric_checks() {
        let key = RowKey { group: "g", row: "\n" };
        let row = CalibrationRow { key, observed: f64::NAN, prediction: 0.0 };
        assert_eq!(calibrate_group_max(&[key, key], &[row, row], 1, 2), Err(GroupMaxError::InvalidKey));
        let valid = RowKey { group: "g", row: "r" };
        assert_eq!(calibrate_group_max(&[valid, valid], &[row], 1, 2), Err(GroupMaxError::InvalidKey));
    }

    #[test]
    fn selected_radii_match_sorted_binary64_scores_at_every_rank() {
        let values = [
            f64::MAX, 0.0, f64::from_bits(1), 1.0,
            f64::from_bits(1.0_f64.to_bits() + 1), 1.0, f64::MIN_POSITIVE,
        ];
        let mut ordered = values;
        ordered.sort_unstable_by(f64::total_cmp);
        for rank in 1..=values.len() {
            assert_eq!(select_radius(values.to_vec(), rank).to_bits(), ordered[rank - 1].to_bits());
        }
    }

    #[test]
    fn group_maxima_preserve_tied_ranks_and_opaque_pair_keys() {
        let a = key("a:b", "c");
        let b = key("a", "b:c");
        let c = key("a", "d");
        let d = key("z", "d");
        let rows = [obs(c, -7.0), obs(d, 9.0), obs(b, 1.0), obs(a, 7.0)];
        assert_eq!(
            calibrate_group_max(&[a, b, c, d], &rows, 1, 2),
            Ok(GroupMaxCutoff::Finite {
                groups: 3,
                rank: 2,
                radius: 7.0
            })
        );
        // Retaining both equal scores is essential; rank 2 would be 9 if deduplicated.
    }

    #[test]
    fn exact_ninety_five_percent_finite_boundary() {
        let labels: Vec<_> = (1..=19).map(|i| i.to_string()).collect();
        let keys: Vec<_> = labels.iter().map(|g| key(g, "one")).collect();
        let rows: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| obs(*k, i as f64))
            .collect();
        assert_eq!(
            calibrate_group_max(&keys[..18], &rows[..18], 95, 100),
            Ok(GroupMaxCutoff::InsufficientFinite {
                groups: 18,
                rank: 19
            })
        );
        assert_eq!(
            calibrate_group_max(&keys, &rows, 95, 100),
            Ok(GroupMaxCutoff::Finite {
                groups: 19,
                rank: 19,
                radius: 18.0
            })
        );
        assert_eq!(
            calibrate_group_max(&keys, &rows, u64::MAX - 1, u64::MAX),
            Ok(GroupMaxCutoff::InsufficientFinite {
                groups: 19,
                rank: 20
            })
        );
    }

    #[test]
    fn exhaustive_distinct_group_ranks_cover_nineteen_of_twenty() {
        let labels: Vec<_> = (0..20).map(|i| i.to_string()).collect();
        let mut covered = 0;
        for held in 0..20 {
            let keys: Vec<_> = labels
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != held)
                .flat_map(|(_, g)| [key(g, "small"), key(g, "maximum")])
                .collect();
            let rows: Vec<_> = keys
                .iter()
                .map(|k| {
                    obs(
                        *k,
                        if k.row == "maximum" {
                            k.group.parse::<f64>().unwrap() + 1.0
                        } else {
                            0.0
                        },
                    )
                })
                .collect();
            match calibrate_group_max(&keys, &rows, 95, 100).unwrap() {
                GroupMaxCutoff::Finite {
                    groups: 19,
                    rank: 19,
                    radius,
                } => {
                    covered += usize::from((held + 1) as f64 <= radius);
                }
                other => panic!("unexpected cutoff: {other:?}"),
            }
        }
        assert_eq!(covered, 19);
    }

    #[test]
    fn contiguous_maxima_agree_with_shuffled_groups_in_both_directions() {
        let roster = [
            key("a", "1"),
            key("a", "2"),
            key("b", "1"),
            key("b", "2"),
            key("c", "1"),
            key("d", "1"),
            key("d", "2"),
        ];
        let rows = [
            obs(roster[0], -0.0),
            obs(roster[1], 9.0),
            obs(roster[2], -8.0),
            obs(roster[3], 1.0),
            obs(roster[4], -0.0),
            obs(roster[5], -2.0),
            obs(roster[6], 3.0),
        ];
        let reversed = [rows[6], rows[5], rows[4], rows[3], rows[2], rows[1], rows[0]];
        let shuffled = [rows[3], rows[1], rows[0], rows[6], rows[4], rows[2], rows[5]];
        for (numerator, expected) in [0.0, 3.0, 8.0, 9.0].into_iter().enumerate() {
            for order in [rows, reversed, shuffled] {
                let result = calibrate_group_max(&roster, &order, (numerator + 1) as u64, 5);
                let Ok(GroupMaxCutoff::Finite { groups, rank, radius }) = result else {
                    panic!("expected a finite cutoff, got {result:?}");
                };
                assert_eq!(groups, 4);
                assert_eq!(rank, numerator + 1);
                assert_eq!(radius.to_bits(), f64::to_bits(expected));
            }
        }
        let mut invalid = rows;
        invalid[6].prediction = f64::NEG_INFINITY;
        assert_eq!(
            calibrate_group_max(&roster, &invalid, 99, 100),
            Err(GroupMaxError::NonfiniteResidual)
        );
    }

    #[test]
    fn missing_max_and_duplicate_same_length_are_rejected() {
        let a = key("a", "lo");
        let b = key("a", "hi");
        let c = key("b", "one");
        assert_eq!(
            calibrate_group_max(&[a, b, c], &[obs(a, 1.0), obs(c, 3.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
        );
        assert_eq!(
            calibrate_group_max(&[a, b, c], &[obs(a, 1.0), obs(a, 1.0), obs(c, 3.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
        );
        assert_eq!(
            calibrate_group_max(&[a, b, c], &[obs(a, 1.0), obs(b, 1.0), obs(b, 1.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
        );
        assert_eq!(
            calibrate_group_max(&[a, a, c], &[obs(a, 1.0), obs(b, 1.0), obs(c, 3.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
        );
    }

    #[test]
    fn aligned_or_reversed_rosters_still_require_unique_pairs() {
        let a = key("a", "only");
        let b = key("b", "only");
        let c = key("c", "only");
        let order = [a, b, c];
        let duplicate = [a, a, c];
        for roster in [order, duplicate] {
            let rows = roster.map(|key| obs(key, 1.0));
            let expected = if roster == order {
                Ok(GroupMaxCutoff::Finite {
                    groups: 3,
                    rank: 2,
                    radius: 1.0,
                })
            } else {
                Err(GroupMaxError::RosterMismatch)
            };
            assert_eq!(calibrate_group_max(&roster, &rows, 1, 2), expected);
            let reversed = [rows[2], rows[1], rows[0]];
            assert_eq!(calibrate_group_max(&roster, &reversed, 1, 2), expected);
        }
    }

    #[test]
    fn validates_all_keys_before_matching_and_roster_before_residuals() {
        let a = key("group", "first");
        let b = key("group", "second");
        let bad = key("group", "\n");
        assert_eq!(
            calibrate_group_max(&[a, a], &[obs(a, 0.0), obs(bad, f64::NAN)], 1, 2),
            Err(GroupMaxError::InvalidKey)
        );
        assert_eq!(
            calibrate_group_max(&[a, b], &[obs(a, f64::NAN), obs(a, 0.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
        );
        assert_eq!(
            calibrate_group_max(&[a, b], &[obs(b, 0.0), obs(a, f64::NAN)], 1, 2),
            Err(GroupMaxError::NonfiniteResidual)
        );
    }

    #[test]
    fn nonfinite_fails_even_if_cutoff_would_be_insufficient() {
        let a = key("a", "one");
        for row in [
            obs(a, f64::NAN),
            obs(a, f64::INFINITY),
            CalibrationRow {
                key: a,
                observed: f64::MAX,
                prediction: -f64::MAX,
            },
        ] {
            assert_eq!(
                calibrate_group_max(&[a], &[row], 19, 20),
                Err(GroupMaxError::NonfiniteResidual)
            );
        }
        assert_eq!(
            calibrate_group_max(&[a], &[obs(a, 0.0)], 0, 0),
            Err(GroupMaxError::Coverage)
        );
        assert_eq!(
            calibrate_group_max(&[], &[], 1, 2),
            Err(GroupMaxError::EmptyRoster)
        );
    }
}
