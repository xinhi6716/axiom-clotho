//! Paired descriptive coverage transitions on one complete interval roster.
//! Not a statistical test, a calibration procedure, or a claim of independence.
use crate::group_interval_diagnostics::{
    evaluate_group_intervals, IntervalDiagnosticError, IntervalObservation,
};
use crate::group_max_calibration::RowKey;
use crate::group_max_intervals::GroupMaxIntervals;
use std::collections::BTreeMap;

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
    use GroupMaxIntervals::Finite;
    evaluate_group_intervals(roster, left, observations).map_err(IntervalComparisonError::Left)?;
    evaluate_group_intervals(roster, right, observations).map_err(IntervalComparisonError::Right)?;
    let (left, right) = match (left, right) {
        (Finite { intervals: l, .. }, Finite { intervals: r, .. }) => (l, r),
        (l, r) => {
            return Err(IntervalComparisonError::InsufficientFinite {
                left: !matches!(l, Finite { .. }),
                right: !matches!(r, Finite { .. }),
            });
        }
    };
    let observed: BTreeMap<_, _> = observations.iter().map(|r| (r.key, r.value)).collect();
    let left_bounds: BTreeMap<_, _> = left.iter().map(|r| (r.key, (r.lower, r.upper))).collect();
    let right_bounds: BTreeMap<_, _> = right.iter().map(|r| (r.key, (r.lower, r.upper))).collect();
    let mut group: BTreeMap<&str, (bool, bool)> = BTreeMap::new();
    let mut rows = CoverageTransitions::default();
    for &key in roster {
        let v = observed[&key];
        let (lo, hi) = left_bounds[&key];
        let l = lo <= v && v <= hi;
        let (lo, hi) = right_bounds[&key];
        let r = lo <= v && v <= hi;
        rows.add(l, r);
        let entry = group.entry(key.group).or_insert((true, true));
        entry.0 &= l;
        entry.1 &= r;
    }
    let mut whole_groups = CoverageTransitions::default();
    for (l, r) in group.into_values() {
        whole_groups.add(l, r);
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
}
