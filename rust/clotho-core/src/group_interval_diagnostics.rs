//! Descriptive scoring of frozen, group-keyed prediction intervals.
//! A group is covered only if every declared row is inside its interval.
//! This audit is not calibration, independence evidence, or a population guarantee.

use crate::group_max_calibration::{valid_key, RowKey, MAX_CALIBRATION_ROWS};
use crate::group_max_intervals::GroupMaxIntervals;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct IntervalObservation<'a> {
    pub key: RowKey<'a>,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalDiagnosticError {
    Capacity,
    EmptyRoster,
    InvalidKey,
    RosterMismatch,
    NonfiniteObservation,
    InvalidBounds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupIntervalDiagnostic {
    InsufficientFinite {
        calibration_groups: usize,
        rank: usize,
        rows: usize,
        groups: usize,
    },
    Scored {
        rows: usize,
        rows_covered: usize,
        groups: usize,
        groups_fully_covered: usize,
        below_rows: usize,
        above_rows: usize,
    },
}

/// Match predictions and observations independently to the exact declared roster.
/// Keys are literal pairs, row and group order are irrelevant, and endpoints
/// are inclusive. An insufficient cutoff reports the supplied roster denominators, not zero
/// coverage; no forecast roster is available for matching in that state.
pub fn evaluate_group_intervals(
    roster: &[RowKey<'_>],
    prediction: &GroupMaxIntervals<'_>,
    observations: &[IntervalObservation<'_>],
) -> Result<GroupIntervalDiagnostic, IntervalDiagnosticError> {
    use IntervalDiagnosticError as Error;

    if roster.len() > MAX_CALIBRATION_ROWS || observations.len() > MAX_CALIBRATION_ROWS {
        return Err(Error::Capacity);
    }
    if roster.is_empty() {
        return Err(Error::EmptyRoster);
    }
    if !roster.iter().copied().chain(observations.iter().map(|r| r.key)).all(valid_key) {
        return Err(Error::InvalidKey);
    }
    let expected: BTreeSet<_> = roster.iter().copied().collect();
    let observed: BTreeMap<_, _> = observations.iter().map(|r| (r.key, r.value)).collect();
    if expected.len() != roster.len() || observed.len() != observations.len()
        || !expected.iter().copied().eq(observed.keys().copied()) {
        return Err(Error::RosterMismatch);
    }
    if !observed.values().all(|x| x.is_finite()) {
        return Err(Error::NonfiniteObservation);
    }
    let groups: BTreeSet<_> = roster.iter().map(|row| row.group).collect();
    let count = groups.len();
    let intervals = match prediction {
        GroupMaxIntervals::InsufficientFinite { groups, rank } => {
            return Ok(GroupIntervalDiagnostic::InsufficientFinite {
                calibration_groups: *groups, rank: *rank, rows: roster.len(), groups: count,
            });
        }
        GroupMaxIntervals::Finite { intervals, .. } => intervals,
    };
    if intervals.len() > MAX_CALIBRATION_ROWS {
        return Err(Error::Capacity);
    }
    if !intervals.iter().all(|r| valid_key(r.key)) {
        return Err(Error::InvalidKey);
    }
    let predicted: BTreeMap<_, _> = intervals.iter().map(|r| (r.key, (r.lower, r.upper))).collect();
    if predicted.len() != intervals.len() || !expected.iter().copied().eq(predicted.keys().copied()) {
        return Err(Error::RosterMismatch);
    }
    if !predicted.values().all(|(l, u)| l.is_finite() && u.is_finite() && l <= u) {
        return Err(Error::InvalidBounds);
    }
    let mut missed_groups = BTreeSet::new();
    let (mut below, mut above) = (0, 0);
    for (key, (lower, upper)) in predicted {
        let value = observed[&key];
        if value < lower {
            below += 1;
            missed_groups.insert(key.group);
        } else if value > upper {
            above += 1;
            missed_groups.insert(key.group);
        }
    }
    Ok(GroupIntervalDiagnostic::Scored {
        rows: roster.len(), rows_covered: roster.len() - below - above,
        groups: count, groups_fully_covered: count - missed_groups.len(),
        below_rows: below, above_rows: above,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_max_intervals::QueryInterval;

    const A: RowKey<'static> = RowKey { group: "A", row: "one" };
    const B: RowKey<'static> = RowKey { group: "A", row: "two" };
    const C: RowKey<'static> = RowKey { group: "B", row: "one" };

    fn obs(key: RowKey<'static>, value: f64) -> IntervalObservation<'static> {
        IntervalObservation { key, value }
    }

    fn bounds(keys: &[RowKey<'static>]) -> GroupMaxIntervals<'static> {
        GroupMaxIntervals::Finite {
            groups: 2, rank: 2, radius: 1.0,
            intervals: keys.iter().copied().map(|key| QueryInterval { key, lower: -1., upper: 1. }).collect(),
        }
    }

    #[test]
    fn simultaneous_group_coverage_differs_from_row_counts_and_ties_include() {
        assert_eq!(evaluate_group_intervals(&[A,B,C], &bounds(&[B,C,A]), &[obs(C,1.),obs(A,-1.),obs(B,-2.)]),
            Ok(GroupIntervalDiagnostic::Scored { rows:3, rows_covered:2, groups:2, groups_fully_covered:1, below_rows:1, above_rows:0 }));
    }

    #[test]
    fn exact_pair_rosters_for_both_inputs() {
        assert_eq!(evaluate_group_intervals(&[A,B,C], &bounds(&[A,A,C]), &[obs(A,0.),obs(B,0.),obs(C,0.)]),
            Err(IntervalDiagnosticError::RosterMismatch));
        assert_eq!(evaluate_group_intervals(&[A,B,C], &bounds(&[A,B,C]), &[obs(A,0.),obs(A,0.),obs(C,0.)]),
            Err(IntervalDiagnosticError::RosterMismatch));
        assert_eq!(evaluate_group_intervals(&[A,B,C], &bounds(&[A,C]), &[obs(A,0.),obs(B,0.),obs(C,0.)]),
            Err(IntervalDiagnosticError::RosterMismatch));
    }

    #[test]
    fn insufficient_has_no_zero_coverage_and_nonfinite_is_rejected() {
        let pred = GroupMaxIntervals::InsufficientFinite { groups: 1, rank: 2 };
        assert_eq!(evaluate_group_intervals(&[A,B], &pred, &[obs(A,1.),obs(B,1.)]),
            Ok(GroupIntervalDiagnostic::InsufficientFinite { calibration_groups:1,rank:2,rows:2,groups:1 }));
        assert_eq!(evaluate_group_intervals(&[A], &pred, &[obs(A,f64::NAN)]),
            Err(IntervalDiagnosticError::NonfiniteObservation));
        let mut wrong = bounds(&[A]);
        if let GroupMaxIntervals::Finite { intervals, .. } = &mut wrong { intervals[0].upper = -2.; }
        assert_eq!(evaluate_group_intervals(&[A], &wrong, &[obs(A,0.)]),
            Err(IntervalDiagnosticError::InvalidBounds));
    }
}
