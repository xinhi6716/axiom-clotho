//! Descriptive scoring of frozen, group-keyed prediction intervals.
//! A group is covered only if every declared row is inside its interval.
//! This audit is not calibration, independence evidence, or a population guarantee.

use crate::group_max_calibration::{valid_key, RowKey, MAX_CALIBRATION_ROWS};
use crate::group_max_intervals::{GroupMaxIntervals, QueryInterval};
use std::borrow::Cow;

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

pub(crate) struct IntervalRoster<'a, 's> {
    pub(crate) observed: Cow<'s, [IntervalObservation<'a>]>,
}

pub(crate) enum ValidatedBounds<'a, 's> {
    Insufficient { groups: usize, rank: usize },
    Finite(Cow<'s, [QueryInterval<'a>]>),
}

// Preserve the original rows when already ordered. Copy only when keys
// decrease; duplicate detection is a separate exact roster check.
fn ordered_rows<'a, 's, T: Clone>(
    rows: &'s [T],
    key: impl Fn(&T) -> RowKey<'a>,
) -> Cow<'s, [T]> {
    if rows.windows(2).all(|w| key(&w[0]) <= key(&w[1])) {
        Cow::Borrowed(rows)
    } else {
        let mut ordered = rows.to_vec();
        ordered.sort_unstable_by(|a, b| key(a).cmp(&key(b)));
        Cow::Owned(ordered)
    }
}

impl<'a, 's> IntervalRoster<'a, 's> {
    pub(crate) fn new(
        roster: &[RowKey<'a>],
        observations: &'s [IntervalObservation<'a>],
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
        if roster.len() != observations.len() {
            return Err(Error::RosterMismatch);
        }
        let expected = ordered_rows(roster, |key| *key);
        let observed = ordered_rows(observations, |r| r.key);
        if expected.windows(2).any(|w| w[0] == w[1])
            || !expected.iter().copied().eq(observed.iter().map(|r| r.key))
        {
            return Err(Error::RosterMismatch);
        }
        if !observed.iter().all(|r| r.value.is_finite()) {
            return Err(Error::NonfiniteObservation);
        }
        Ok(Self { observed })
    }

    // Carry the same bound checks into either single-arm scoring or paired
    // scoring. The observed keys have already matched the declared roster.
    pub(crate) fn bounds<'p>(
        &self,
        prediction: &'p GroupMaxIntervals<'a>,
    ) -> Result<ValidatedBounds<'a, 'p>, IntervalDiagnosticError> {
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
        if intervals.len() != self.observed.len() {
            return Err(Error::RosterMismatch);
        }
        let predicted = ordered_rows(intervals, |r| r.key);
        if !self.observed.iter().map(|r| r.key).eq(predicted.iter().map(|r| r.key)) {
            return Err(Error::RosterMismatch);
        }
        if !predicted.iter().all(|r| r.lower.is_finite() && r.upper.is_finite() && r.lower <= r.upper) {
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
            for key in matched.observed.iter().map(|r| r.key) {
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

    // The two inputs were independently matched to the exact same sorted
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
    for (interval, observed) in predicted.iter().zip(matched.observed.iter()) {
        let QueryInterval { key, lower, upper } = *interval;
        let value = observed.value;
        debug_assert_eq!(key, observed.key);
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
    #[test]
    fn sorted_rosters_borrow_rows_and_shuffled_inputs_use_the_same_order() {
        let observations = [obs(A, -1.0), obs(B, 1.0), obs(C, 2.0)];
        let prediction = bounds(&[A, B, C]);
        let ordered = IntervalRoster::new(&[A, B, C], &observations).unwrap();
        assert!(matches!(ordered.observed, Cow::Borrowed(_)));
        assert!(matches!(
            ordered.bounds(&prediction),
            Ok(ValidatedBounds::Finite(Cow::Borrowed(_)))
        ));
        let reversed = [observations[2], observations[1], observations[0]];
        let unordered = IntervalRoster::new(&[B, C, A], &reversed).unwrap();
        assert!(matches!(unordered.observed, Cow::Owned(_)));
        assert_eq!(
            unordered.observed.iter().map(|r| r.key).collect::<Vec<_>>(),
            vec![A, B, C]
        );
        assert_eq!(
            evaluate_group_intervals(&[B, C, A], &bounds(&[B, A, C]), &reversed),
            evaluate_group_intervals(&[A, B, C], &prediction, &observations)
        );

        // A bad value/bound remains subordinate to the exact-roster check.
        let duplicated = [observations[0], observations[0], obs(C, f64::NAN)];
        assert_eq!(
            evaluate_group_intervals(&[A, B, C], &prediction, &duplicated),
            Err(IntervalDiagnosticError::RosterMismatch)
        );
        let mut wrong = bounds(&[A, A, C]);
        if let GroupMaxIntervals::Finite { intervals, .. } = &mut wrong {
            intervals[2].lower = f64::INFINITY;
        }
        assert_eq!(
            evaluate_group_intervals(&[A, B, C], &wrong, &observations),
            Err(IntervalDiagnosticError::RosterMismatch)
        );
    }

}
