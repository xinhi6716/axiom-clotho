//! Fit-free split calibration of one maximum residual per declared group.
//!
//! Any group-level guarantee requires exchangeable calibration and new-group
//! scores under the same roster construction rule and a predictor fixed
//! independently of calibration. This function checks neither assumption.
//! It does not consume query labels, construct bounds, or read any files.

use std::collections::{BTreeMap, BTreeSet};

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

fn valid_key(key: RowKey<'_>) -> bool {
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
    if !roster
        .iter()
        .copied()
        .chain(rows.iter().map(|r| r.key))
        .all(valid_key)
    {
        return Err(GroupMaxError::InvalidKey);
    }
    let expected: BTreeSet<_> = roster.iter().copied().collect();
    let supplied: BTreeSet<_> = rows.iter().map(|r| r.key).collect();
    if expected.len() != roster.len() || supplied.len() != rows.len() || expected != supplied {
        return Err(GroupMaxError::RosterMismatch);
    }

    let mut maxima = BTreeMap::<_, f64>::new();
    for r in rows {
        if !r.observed.is_finite() || !r.prediction.is_finite() {
            return Err(GroupMaxError::NonfiniteResidual);
        }
        let score = (r.observed - r.prediction).abs();
        if !score.is_finite() {
            return Err(GroupMaxError::NonfiniteResidual);
        }
        let current = maxima.entry(r.key.group).or_insert(0.0);
        *current = current.max(score);
    }
    let groups = maxima.len();
    // groups <= MAX_CALIBRATION_ROWS; even the largest u64 fraction fits u128.
    let n = (groups as u128 + 1) * u128::from(numerator);
    let d = u128::from(denominator);
    let rank = ((n + d - 1) / d) as usize;
    if rank > groups {
        return Ok(GroupMaxCutoff::InsufficientFinite { groups, rank });
    }
    let mut ordered: Vec<_> = maxima.into_values().collect();
    ordered.sort_unstable_by(f64::total_cmp);
    Ok(GroupMaxCutoff::Finite {
        groups,
        rank,
        radius: ordered[rank - 1],
    })
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
            calibrate_group_max(&[a, a, c], &[obs(a, 1.0), obs(b, 1.0), obs(c, 3.0)], 1, 2),
            Err(GroupMaxError::RosterMismatch)
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
