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

pub(crate) struct IntervalRoster<'a> {
    expected: BTreeSet<RowKey<'a>>,
    pub(crate) observed: BTreeMap<RowKey<'a>, f64>,
}

pub(crate) enum ValidatedBounds<'a> {
    Insufficient { groups: usize, rank: usize },
    Finite(BTreeMap<RowKey<'a>, (f64, f64)>),
}

impl<'a> IntervalRoster<'a> {
    pub(crate) fn new(
        roster: &[RowKey<'a>],
        observations: &[IntervalObservation<'a>],
    ) -> Result<Self, IntervalDiagnosticError> {
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
        Ok(Self { expected, observed })
    }

    // Carry the same bound checks into either single-arm scoring or paired
    // scoring without rebuilding shared roster/observation maps.
    pub(crate) fn bounds(
        &self,
        prediction: &GroupMaxIntervals<'a>,
    ) -> Result<ValidatedBounds<'a>, IntervalDiagnosticError> {
        use IntervalDiagnosticError as Error;
        let intervals = match prediction {
            GroupMaxIntervals::Finite { intervals, .. } => intervals,
            GroupMaxIntervals::InsufficientFinite { groups, rank } => {
                return Ok(ValidatedBounds::Insufficient { groups: *groups, rank: *rank });
            }
        };
        if intervals.len() > MAX_CALIBRATION_ROWS {
            return Err(Error::Capacity);
        }
        if !intervals.iter().all(|r| valid_key(r.key)) {
            return Err(Error::InvalidKey);
        }
        let predicted: BTreeMap<_, _> = intervals
            .iter()
            .map(|r| (r.key, (r.lower, r.upper)))
            .collect();
        if predicted.len() != intervals.len()
            || !self.expected.iter().copied().eq(predicted.keys().copied())
        {
            return Err(Error::RosterMismatch);
        }
        if !predicted.values().all(|(l, u)| l.is_finite() && u.is_finite() && l <= u) {
            return Err(Error::InvalidBounds);
        }
        Ok(ValidatedBounds::Finite(predicted))
    }
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
    evaluate_group_intervals_by_group(roster, prediction, observations, |_, _| {})
}

/// Visit each finite group's (declared rows, covered rows) once.
pub(crate) fn evaluate_group_intervals_by_group(
    roster: &[RowKey<'_>],
    prediction: &GroupMaxIntervals<'_>,
    observations: &[IntervalObservation<'_>],
    mut visit_group: impl FnMut(usize, usize),
) -> Result<GroupIntervalDiagnostic, IntervalDiagnosticError> {
    let matched = IntervalRoster::new(roster, observations)?;
    let predicted = match matched.bounds(prediction)? {
        ValidatedBounds::Finite(predicted) => predicted,
        ValidatedBounds::Insufficient { groups, rank } => {
            let mut count = 0;
            let mut previous_group = None;
            for key in matched.observed.keys() {
                if previous_group != Some(key.group) {
                    count += 1;
                    previous_group = Some(key.group);
                }
            }
            return Ok(GroupIntervalDiagnostic::InsufficientFinite {
                calibration_groups: groups, rank, rows: roster.len(), groups: count,
            });
        }
    };

    // The two maps were independently matched to the exact same sorted
    // roster. Therefore corresponding entries have identical row keys, and
    // rows of each group are contiguous without another lookup or index.
    let mut previous_group = None;
    let (mut group_rows, mut group_covered) = (0, 0);
    let (mut groups, mut groups_fully_covered) = (0, 0);
    let (mut below, mut above) = (0, 0);
    let mut finish_group = |rows, covered| {
        groups += 1;
        groups_fully_covered += usize::from(rows == covered);
        visit_group(rows, covered);
    };
    for ((key, &(lower, upper)), (observed_key, &value)) in
        predicted.iter().zip(matched.observed.iter())
    {
        debug_assert_eq!(key, observed_key);
        if previous_group != Some(key.group) {
            if group_rows > 0 {
                finish_group(group_rows, group_covered);
            }
            previous_group = Some(key.group);
            group_rows = 0;
            group_covered = 0;
        }
        group_rows += 1;
        if value < lower {
            below += 1;
        } else if value > upper {
            above += 1;
        } else {
            group_covered += 1;
        }
    }
    if group_rows > 0 {
        finish_group(group_rows, group_covered);
    }

    Ok(GroupIntervalDiagnostic::Scored {
        rows: roster.len(), rows_covered: roster.len() - below - above,
        groups, groups_fully_covered,
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
