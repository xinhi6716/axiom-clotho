//! Paired descriptive coverage transitions on one complete interval roster.
//! Not a statistical test, a calibration procedure, or a claim of independence.
use crate::group_interval_diagnostics::{
    IntervalDiagnosticError, IntervalObservation, IntervalRoster, ValidatedBounds,
};
use crate::group_max_calibration::RowKey;
use crate::group_max_intervals::GroupMaxIntervals;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverageTransitions {
    pub both: usize,
    pub left_only: usize,
    pub right_only: usize,
    pub neither: usize,
}

impl CoverageTransitions {
    fn add(&mut self, left: bool, right: bool) {
        match (left, right) {
            (true, true) => self.both += 1,
            (true, false) => self.left_only += 1,
            (false, true) => self.right_only += 1,
            (false, false) => self.neither += 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedIntervalCoverage {
    pub rows: CoverageTransitions,
    pub whole_groups: CoverageTransitions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalComparisonError {
    Left(IntervalDiagnosticError),
    Right(IntervalDiagnosticError),
    InsufficientFinite { left: bool, right: bool },
}

/// Evaluate both arms over exactly the same declared row roster and outcomes.
/// A method covers a whole group only when it covers every row in that group.
/// Neither arm is ever dropped or converted to zero coverage.
pub fn compare_interval_coverage(
    roster: &[RowKey<'_>],
    left: &GroupMaxIntervals<'_>,
    right: &GroupMaxIntervals<'_>,
    observations: &[IntervalObservation<'_>],
) -> Result<PairedIntervalCoverage, IntervalComparisonError> {
    // Both arms use the same observation validation. Preserve left-first
    // error precedence, including validating the right arm when the left
    // arm has an insufficient (but otherwise valid) cutoff.
    let matched = IntervalRoster::new(roster, observations).map_err(IntervalComparisonError::Left)?;
    let left = matched.bounds(left).map_err(IntervalComparisonError::Left)?;
    let right = matched.bounds(right).map_err(IntervalComparisonError::Right)?;
    let (left_bounds, right_bounds) = match (left, right) {
        (ValidatedBounds::Finite(left), ValidatedBounds::Finite(right)) => (left, right),
        (left, right) => {
            return Err(IntervalComparisonError::InsufficientFinite {
                left: matches!(left, ValidatedBounds::Insufficient { .. }),
                right: matches!(right, ValidatedBounds::Insufficient { .. }),
            });
        }
    };
    // Each map has already matched the exact same sorted composite keys.
    // Zip them in key order, so each group is one contiguous run.
    let mut rows = CoverageTransitions::default();
    let mut whole_groups = CoverageTransitions::default();
    let mut current = None;
    let (mut left_all, mut right_all) = (true, true);
    for (((key, &value), (left_key, &(left_lo, left_hi))), (right_key, &(right_lo, right_hi))) in
        matched.observed.iter().zip(left_bounds.iter()).zip(right_bounds.iter())
    {
        debug_assert_eq!(key, left_key);
        debug_assert_eq!(key, right_key);
        if current != Some(key.group) {
            if current.is_some() {
                whole_groups.add(left_all, right_all);
            }
            current = Some(key.group);
            (left_all, right_all) = (true, true);
        }
        let left = left_lo <= value && value <= left_hi;
        let right = right_lo <= value && value <= right_hi;
        rows.add(left, right);
        left_all &= left;
        right_all &= right;
    }
    if current.is_some() {
        whole_groups.add(left_all, right_all);
    }
    Ok(PairedIntervalCoverage { rows, whole_groups })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_max_intervals::QueryInterval;

    fn key(group: &'static str, row: &'static str) -> RowKey<'static> {
        RowKey { group, row }
    }

    fn arm(rows: &[(RowKey<'static>, bool)]) -> GroupMaxIntervals<'static> {
        GroupMaxIntervals::Finite {
            groups: 2, rank: 2, radius: 1.0,
            intervals: rows.iter().map(|&(key, hit)| QueryInterval {
                key, lower: if hit { 0.0 } else { 1.0 }, upper: 2.0,
            }).collect(),
        }
    }

    #[test]
    fn pairs_groups_instead_of_comparing_only_identical_marginals() {
        let a = key("a", "1");
        let b = key("a", "2");
        let c = key("b", "1");
        let d = key("b", "2");
        let roster = [a,b,c,d];
        let observed: Vec<_> = roster.iter().rev().map(|&key| IntervalObservation { key, value: 0.0 }).collect();
        let l = arm(&[(a,true),(b,true),(c,true),(d,false)]);
        let r = arm(&[(d,true),(c,true),(b,true),(a,false)]);
        let score = compare_interval_coverage(&roster, &l, &r, &observed).unwrap();
        assert_eq!(score.rows, CoverageTransitions { both:2, left_only:1, right_only:1, neither:0 });
        assert_eq!(score.whole_groups, CoverageTransitions { both:0, left_only:1, right_only:1, neither:0 });
        let same = compare_interval_coverage(&roster, &l, &l, &observed).unwrap();
        assert_eq!(same.whole_groups, CoverageTransitions { both:1, left_only:0, right_only:0, neither:1 });
    }

    #[test]
    fn all_transition_types_use_each_exact_group_after_independent_permutations() {
        let a = key("a", "b:c");
        let b = key("a:b", "c");
        let c = key("a:b", "d");
        let d = key("m", "one");
        let e = key("z", "one");
        let f = key("z", "two");
        let roster = [d, f, b, a, e, c];
        let observed = [e, c, d, a, b, f].map(|key| IntervalObservation { key, value: -0.0 });
        let left = arm(&[(f, true), (b, true), (a, true), (d, false), (c, true), (e, false)]);
        let right = arm(&[(c, true), (d, true), (e, true), (b, false), (f, false), (a, true)]);
        let score = compare_interval_coverage(&roster, &left, &right, &observed).unwrap();
        assert_eq!(
            score,
            PairedIntervalCoverage {
                rows: CoverageTransitions { both: 2, left_only: 2, right_only: 2, neither: 0 },
                whole_groups: CoverageTransitions { both: 1, left_only: 1, right_only: 1, neither: 1 },
            }
        );
        let GroupMaxIntervals::Finite { mut intervals, groups, rank, radius } = right else {
            unreachable!()
        };
        intervals[0].key = intervals[1].key;
        let duplicate = GroupMaxIntervals::Finite { intervals, groups, rank, radius };
        assert_eq!(
            compare_interval_coverage(&roster, &left, &duplicate, &observed),
            Err(IntervalComparisonError::Right(IntervalDiagnosticError::RosterMismatch)),
        );
    }

    #[test]
    fn fails_closed_on_equal_length_wrong_roster_and_insufficient_arm() {
        let a = key("a", "1");
        let b = key("b", "1");
        let observed = [IntervalObservation { key:a, value:0.0 }];
        let l = arm(&[(a,true)]);
        let r = arm(&[(b,true)]);
        assert_eq!(compare_interval_coverage(&[a], &l, &r, &observed),
            Err(IntervalComparisonError::Right(IntervalDiagnosticError::RosterMismatch)));
        let insufficient = GroupMaxIntervals::InsufficientFinite { groups:1,rank:2 };
        assert_eq!(compare_interval_coverage(&[a], &l, &insufficient, &observed),
            Err(IntervalComparisonError::InsufficientFinite { left:false,right:true }));
    }

    #[test]
    fn validates_both_arms_before_reporting_insufficient_cutoff() {
        let a = key("a", "1");
        let roster = [a];
        let mut right = arm(&[(a, true)]);
        let GroupMaxIntervals::Finite { intervals, .. } = &mut right else { unreachable!() };
        intervals[0].upper = f64::NAN;
        let insufficient = GroupMaxIntervals::InsufficientFinite { groups: 1, rank: 2 };
        let observations = [IntervalObservation { key: a, value: 0.0 }];
        assert_eq!(
            compare_interval_coverage(&roster, &insufficient, &right, &observations),
            Err(IntervalComparisonError::Right(IntervalDiagnosticError::InvalidBounds))
        );
        let invalid_observations = [IntervalObservation { key: a, value: f64::NAN }];
        assert_eq!(
            compare_interval_coverage(&roster, &insufficient, &right, &invalid_observations),
            Err(IntervalComparisonError::Left(IntervalDiagnosticError::NonfiniteObservation))
        );
        assert_eq!(
            compare_interval_coverage(&roster, &insufficient, &insufficient, &observations),
            Err(IntervalComparisonError::InsufficientFinite { left: true, right: true })
        );
    }

}
