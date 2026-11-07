//! Bounded, explicitly SI/binary64 FET branch features. Not legacy exact-decimal
//! compatibility, physical threshold identification, or automatic window search.
use std::collections::BTreeSet;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Acquisition {
    pub lot: String,
    pub wafer: String,
    pub device: String,
    pub condition: String,
    pub replicate: String,
    pub branch: String,
}
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub source_row: usize,
    pub temperature_k: f64,
    pub drain_voltage_v: f64,
    pub body_voltage_v: f64,
    pub gate_voltage_v: f64,
    pub drain_current_a: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub criterion_a: f64,
    pub positive_current: bool,
    pub max_bracket_span_v: f64,
    pub window_min_v: f64,
    pub window_max_v: f64,
    pub assumptions_accepted: bool,
}
#[derive(Clone, Debug)]
pub struct Features {
    pub acquisition: Acquisition,
    pub criterion_gate_voltage_v: f64,
    pub criterion_source_rows: Vec<usize>,
    pub bracket_span_v: f64,
    pub log_slope_decades_per_v: f64,
    pub log_intercept: f64,
    pub log_rmse_decades: f64,
    pub log_selected_source_rows: Vec<usize>,
    pub all_source_rows: Vec<usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetError {
    InvalidSettings,
    InvalidIdentity,
    InvalidSample,
    InvalidConditions,
    InvalidGateOrder,
    InvalidCurrent,
    NonUniqueCriterion,
    CriterionNotBracketed,
    NonMonotonicCurrent,
    BracketTooWide,
    InsufficientWindow,
    NumericalRange,
    DuplicateDevice,
    ResourceLimit,
}
impl fmt::Display for FetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FET extraction rejected: {:?}", self)
    }
}
impl std::error::Error for FetError {}
fn bounded(x: f64) -> bool {
    x.is_finite() && (x == 0.0 || (1e-150..=1e150).contains(&x.abs()))
}
fn id(x: &str) -> bool {
    !x.is_empty() && x.trim() == x && x.chars().count() <= 128 && !x.chars().any(|c| c.is_control())
}

/// Extract every branch, rejecting duplicate physical devices and the entire
/// cohort if any branch fails. Preserves input order; never selects a best branch.
pub fn extract_cohort(
    branches: &[(Acquisition, Vec<Sample>)],
    settings: Settings,
) -> Result<Vec<Features>, FetError> {
    if branches.is_empty()
        || branches.len() > 1000
        || branches.iter().map(|(_, s)| s.len()).sum::<usize>() > 10000
    {
        return Err(FetError::ResourceLimit);
    }
    let mut devices = BTreeSet::new();
    let mut source_rows = BTreeSet::new();
    for (a, s) in branches {
        if !devices.insert((&a.lot, &a.wafer, &a.device)) {
            return Err(FetError::DuplicateDevice);
        }
        if s.iter().any(|p| !source_rows.insert(p.source_row)) {
            return Err(FetError::InvalidSample);
        }
    }
    branches
        .iter()
        .map(|(a, s)| extract_branch(a, s, settings))
        .collect()
}
/// Values and parameters must already be SI binary64. Exact original decimals,
/// unit parsing, bundle provenance and train/endpoint joins are separate work.
pub fn extract_branch(a: &Acquisition, s: &[Sample], p: Settings) -> Result<Features, FetError> {
    let sign = if p.positive_current { 1.0 } else { -1.0 };
    if !p.assumptions_accepted
        || ![
            p.criterion_a,
            p.max_bracket_span_v,
            p.window_min_v,
            p.window_max_v,
        ]
        .iter()
        .all(|x| bounded(*x))
        || p.criterion_a * sign <= 0.0
        || p.max_bracket_span_v <= 0.0
        || p.window_min_v >= p.window_max_v
    {
        return Err(FetError::InvalidSettings);
    }
    if [
        &a.lot,
        &a.wafer,
        &a.device,
        &a.condition,
        &a.replicate,
        &a.branch,
    ]
    .iter()
    .any(|x| !id(x))
    {
        return Err(FetError::InvalidIdentity);
    }
    if s.len() < 3 || s.len() > 1000 {
        return Err(FetError::ResourceLimit);
    }
    let mut rows = BTreeSet::new();
    for q in s {
        if q.source_row < 2
            || !rows.insert(q.source_row)
            || ![
                q.temperature_k,
                q.drain_voltage_v,
                q.body_voltage_v,
                q.gate_voltage_v,
                q.drain_current_a,
            ]
            .iter()
            .all(|x| bounded(*x))
        {
            return Err(FetError::InvalidSample);
        }
        if q.temperature_k <= 0.0
            || q.drain_voltage_v == 0.0
            || q.temperature_k != s[0].temperature_k
            || q.drain_voltage_v != s[0].drain_voltage_v
            || q.body_voltage_v != s[0].body_voltage_v
        {
            return Err(FetError::InvalidConditions);
        }
        if q.drain_current_a * sign < 0.0 {
            return Err(FetError::InvalidCurrent);
        }
    }
    if !(s
        .windows(2)
        .all(|w| w[1].gate_voltage_v > w[0].gate_voltage_v)
        || s.windows(2)
            .all(|w| w[1].gate_voltage_v < w[0].gate_voltage_v))
    {
        return Err(FetError::InvalidGateOrder);
    }
    let exact: Vec<_> = s
        .iter()
        .filter(|q| q.drain_current_a == p.criterion_a)
        .collect();
    let crossing: Vec<_> = s
        .windows(2)
        .filter(|w| {
            p.criterion_a > w[0].drain_current_a.min(w[1].drain_current_a)
                && p.criterion_a < w[0].drain_current_a.max(w[1].drain_current_a)
        })
        .collect();
    match exact.len() + crossing.len() {
        0 => return Err(FetError::CriterionNotBracketed),
        1 => {}
        _ => return Err(FetError::NonUniqueCriterion),
    }
    if !(s
        .windows(2)
        .all(|w| w[1].drain_current_a >= w[0].drain_current_a)
        || s.windows(2)
            .all(|w| w[1].drain_current_a <= w[0].drain_current_a))
    {
        return Err(FetError::NonMonotonicCurrent);
    }
    let (criterion, span, criterion_rows) = if !exact.is_empty() {
        (exact[0].gate_voltage_v, 0.0, vec![exact[0].source_row])
    } else {
        let w = crossing[0];
        let span = (w[1].gate_voltage_v - w[0].gate_voltage_v).abs();
        if span > p.max_bracket_span_v {
            return Err(FetError::BracketTooWide);
        }
        let t =
            (p.criterion_a - w[0].drain_current_a) / (w[1].drain_current_a - w[0].drain_current_a);
        if !(0.0 < t && t < 1.0) {
            return Err(FetError::NumericalRange);
        }
        (
            w[0].gate_voltage_v * (1.0 - t) + w[1].gate_voltage_v * t,
            span,
            vec![w[0].source_row, w[1].source_row],
        )
    };
    let selected: Vec<_> = s
        .iter()
        .filter(|q| q.gate_voltage_v >= p.window_min_v && q.gate_voltage_v <= p.window_max_v)
        .collect();
    if selected.len() < 3 {
        return Err(FetError::InsufficientWindow);
    }
    if selected.iter().any(|q| q.drain_current_a * sign <= 0.0) {
        return Err(FetError::InvalidCurrent);
    }
    let logs: Vec<_> = selected
        .iter()
        .map(|q| (q.drain_current_a * sign).log10())
        .collect();
    // Reject loss of distinctions introduced by binary64 logarithms.
    for i in 0..selected.len() {
        for j in 0..i {
            if selected[i].drain_current_a != selected[j].drain_current_a && logs[i] == logs[j] {
                return Err(FetError::NumericalRange);
            }
        }
    }
    let n = selected.len() as f64;
    let x0 = selected[0].gate_voltage_v;
    let scale = selected
        .iter()
        .map(|q| (q.gate_voltage_v - x0).abs())
        .fold(0.0, f64::max);
    let xs: Vec<_> = selected
        .iter()
        .map(|q| (q.gate_voltage_v - x0) / scale)
        .collect();
    let xm = xs.iter().sum::<f64>() / n;
    let ym = logs.iter().sum::<f64>() / n;
    let xx = xs.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
    let b = xs
        .iter()
        .zip(&logs)
        .map(|(x, y)| (x - xm) * (y - ym))
        .sum::<f64>()
        / xx;
    let slope = b / scale;
    let intercept = ym - slope * x0 - b * xm;
    let rmse = (xs
        .iter()
        .zip(&logs)
        .map(|(x, y)| (y - (ym + b * (x - xm))).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    if ![criterion, span, slope, intercept, rmse]
        .iter()
        .all(|x| x.is_finite())
        || (b != 0.0 && slope == 0.0)
    {
        return Err(FetError::NumericalRange);
    }
    Ok(Features {
        acquisition: a.clone(),
        criterion_gate_voltage_v: criterion,
        criterion_source_rows: criterion_rows,
        bracket_span_v: span,
        log_slope_decades_per_v: slope,
        log_intercept: intercept,
        log_rmse_decades: rmse,
        log_selected_source_rows: selected.iter().map(|q| q.source_row).collect(),
        all_source_rows: s.iter().map(|q| q.source_row).collect(),
    })
}
