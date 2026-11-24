//! Observed interval coverage by exact declared group size.
//! No pooling across sizes or inferential claims.
use crate::group_interval_diagnostics::{
    evaluate_group_intervals, GroupIntervalDiagnostic, IntervalDiagnosticError, IntervalObservation,
};
use crate::group_max_calibration::RowKey;
use crate::group_max_intervals::GroupMaxIntervals;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSizeCoverage {
    pub rows_per_group: usize,
    pub groups: usize,
    pub groups_fully_covered: usize,
    pub rows_covered: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StratifiedIntervalCoverage {
    pub overall: GroupIntervalDiagnostic,
    /// Sorted by exact group size. Absent when no finite cutoff was returned.
    pub strata: Option<Vec<GroupSizeCoverage>>,
}

/// Describe coverage on the same complete roster as the main diagnostic.
/// Group size is the number of declared rows, not the count of covered rows.
pub fn stratify_group_interval_coverage(
    roster: &[RowKey<'_>],
    prediction: &GroupMaxIntervals<'_>,
    observations: &[IntervalObservation<'_>],
) -> Result<StratifiedIntervalCoverage, IntervalDiagnosticError> {
    let overall = evaluate_group_intervals(roster, prediction, observations)?;
    let GroupMaxIntervals::Finite { intervals, .. } = prediction else {
        return Ok(StratifiedIntervalCoverage { overall, strata: None });
    };
    let values: BTreeMap<_, _> = observations.iter().map(|r| (r.key, r.value)).collect();
    let mut groups: BTreeMap<_, (usize, usize)> = BTreeMap::new();
    for row in intervals {
        let value = values[&row.key];
        let (len, covered) = groups.entry(row.key.group).or_default();
        *len += 1;
        *covered += usize::from(row.lower <= value && value <= row.upper);
    }
    let mut strata = BTreeMap::new();
    for (_, (len, covered)) in groups {
        let stratum = strata.entry(len).or_insert(GroupSizeCoverage {
            rows_per_group: len, groups: 0, groups_fully_covered: 0, rows_covered: 0,
        });
        stratum.groups += 1;
        stratum.groups_fully_covered += usize::from(len == covered);
        stratum.rows_covered += covered;
    }
    Ok(StratifiedIntervalCoverage { overall, strata: Some(strata.into_values().collect()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_max_intervals::QueryInterval;

    #[test]
    fn one_miss_in_large_group_preserves_size_and_denominators() {
        let a = RowKey { group: "short", row: "1" };
        let b = RowKey { group: "long", row: "1" };
        let c = RowKey { group: "long", row: "2" };
        let rows = [a, b, c];
        let observations = [
            IntervalObservation { key: c, value: 2.0 },
            IntervalObservation { key: b, value: 1.0 },
            IntervalObservation { key: a, value: 0.0 },
        ];
        let prediction = GroupMaxIntervals::Finite {
            groups: 1, rank: 1, radius: 1.0,
            intervals: rows.into_iter().rev().map(|key| QueryInterval {
                key, lower: 0.0, upper: 1.0,
            }).collect(),
        };
        let result = stratify_group_interval_coverage(&rows, &prediction, &observations).unwrap();
        assert_eq!(result.strata, Some(vec![
            GroupSizeCoverage { rows_per_group: 1, groups: 1, groups_fully_covered: 1, rows_covered: 1 },
            GroupSizeCoverage { rows_per_group: 2, groups: 1, groups_fully_covered: 0, rows_covered: 1 },
        ]));
        assert_eq!(stratify_group_interval_coverage(&rows, &prediction, &observations[..2]),
            Err(IntervalDiagnosticError::RosterMismatch));
        let insufficient = GroupMaxIntervals::InsufficientFinite { groups: 1, rank: 2 };
        assert_eq!(stratify_group_interval_coverage(&rows, &insufficient, &observations).unwrap().strata, None);
    }
}
