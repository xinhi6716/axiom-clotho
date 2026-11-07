//! Explicit SI/binary64 RRAM readpoints and paired descriptive features.
//! State is declared, never inferred; cycles are not independent devices.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    Hrs,
    Lrs,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Acquisition {
    pub batch: String,
    pub wafer: String,
    pub device: String,
    pub cycle: String,
    pub replicate: String,
    pub branch: String,
    pub state: State,
    pub condition: String,
    pub protocol: String,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditions {
    pub read_voltage_v: f64,
    pub temperature_k: f64,
    pub read_delay_s: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub source_row: usize,
    pub voltage_v: f64,
    pub current_a: f64,
}
#[derive(Clone, Debug)]
pub struct Branch {
    pub acquisition: Acquisition,
    pub conditions: Conditions,
    pub samples: Vec<Sample>,
}
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub max_bracket_span_v: f64,
    pub assumptions_accepted: bool,
    pub by_condition: bool,
}
#[derive(Clone, Debug)]
pub struct Readpoint {
    pub acquisition: Acquisition,
    pub conditions: Conditions,
    pub current_a: f64,
    pub resistance_ohm: f64,
    pub bracket_span_v: f64,
    pub selected_source_rows: Vec<usize>,
    pub all_source_rows: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct Pair {
    pub hrs: Readpoint,
    pub lrs: Readpoint,
    pub resistance_ratio: f64,
}
#[derive(Clone, Debug)]
pub struct Features {
    pub batch: String,
    pub wafer: String,
    pub device: String,
    pub condition: String,
    pub protocol: String,
    pub conditions: Conditions,
    pub pairs: Vec<Pair>,
    pub hrs_resistance_median_ohm: f64,
    pub lrs_resistance_median_ohm: f64,
    pub resistance_ratio_median: f64,
    pub ratio_below_one_cycle_count: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RramError {
    InvalidSettings,
    InvalidIdentity,
    InvalidConditions,
    InvalidSample,
    InvalidVoltageOrder,
    ReadVoltageNotBracketed,
    BracketTooWide,
    InvalidCurrent,
    NumericalRange,
    DuplicateAcquisition,
    DuplicateState,
    MissingState,
    IdentityMismatch,
    ReplicateMismatch,
    ResourceLimit,
}
impl fmt::Display for RramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RRAM extraction rejected: {:?}", self)
    }
}
impl std::error::Error for RramError {}
fn bounded(x: f64) -> bool {
    x.is_finite() && (x == 0.0 || (1e-150..=1e150).contains(&x.abs()))
}
fn id(x: &str) -> bool {
    !x.is_empty()
        && x.trim() == x
        && x.chars().count() <= 128
        && !x.chars().any(|c| c.is_control())
        && !x.starts_with(['=', '+', '-', '@'])
}
fn settings(p: Settings) -> Result<(), RramError> {
    if !p.assumptions_accepted || !bounded(p.max_bracket_span_v) || p.max_bracket_span_v <= 0.0 {
        return Err(RramError::InvalidSettings);
    }
    Ok(())
}
/// Extract measured current or adjacent linear interpolation in acquisition order.
/// No sorting to repair branches, extrapolation, abs(), or state inference.
pub fn extract_branch(b: &Branch, p: Settings) -> Result<Readpoint, RramError> {
    settings(p)?;
    let a = &b.acquisition;
    if [
        &a.batch,
        &a.wafer,
        &a.device,
        &a.cycle,
        &a.replicate,
        &a.branch,
        &a.condition,
        &a.protocol,
    ]
    .iter()
    .any(|s| !id(s))
    {
        return Err(RramError::InvalidIdentity);
    }
    let c = b.conditions;
    if ![c.read_voltage_v, c.temperature_k, c.read_delay_s]
        .iter()
        .all(|x| bounded(*x))
        || c.read_voltage_v == 0.0
        || c.temperature_k <= 0.0
        || c.read_delay_s < 0.0
    {
        return Err(RramError::InvalidConditions);
    }
    let s = &b.samples;
    if s.is_empty() || s.len() > 1000 {
        return Err(RramError::ResourceLimit);
    }
    let mut rows = BTreeSet::new();
    if s.iter().any(|q| {
        q.source_row < 2
            || !rows.insert(q.source_row)
            || !bounded(q.voltage_v)
            || !bounded(q.current_a)
    }) {
        return Err(RramError::InvalidSample);
    }
    if !(s.windows(2).all(|w| w[1].voltage_v > w[0].voltage_v)
        || s.windows(2).all(|w| w[1].voltage_v < w[0].voltage_v))
    {
        return Err(RramError::InvalidVoltageOrder);
    }
    let (current, span, selected) =
        if let Some(q) = s.iter().find(|q| q.voltage_v == c.read_voltage_v) {
            (q.current_a, 0.0, vec![q.source_row])
        } else {
            let w = s
                .windows(2)
                .find(|w| {
                    c.read_voltage_v > w[0].voltage_v.min(w[1].voltage_v)
                        && c.read_voltage_v < w[0].voltage_v.max(w[1].voltage_v)
                })
                .ok_or(RramError::ReadVoltageNotBracketed)?;
            let span = (w[1].voltage_v - w[0].voltage_v).abs();
            if span > p.max_bracket_span_v {
                return Err(RramError::BracketTooWide);
            }
            let t = (c.read_voltage_v - w[0].voltage_v) / (w[1].voltage_v - w[0].voltage_v);
            if !(0.0 < t && t < 1.0) {
                return Err(RramError::NumericalRange);
            }
            (
                w[0].current_a * (1.0 - t) + w[1].current_a * t,
                span,
                vec![w[0].source_row, w[1].source_row],
            )
        };
    if current == 0.0 || (current > 0.0) != (c.read_voltage_v > 0.0) {
        return Err(RramError::InvalidCurrent);
    }
    let resistance = c.read_voltage_v / current;
    if !current.is_finite() || !resistance.is_finite() || resistance <= 0.0 {
        return Err(RramError::NumericalRange);
    }
    Ok(Readpoint {
        acquisition: a.clone(),
        conditions: c,
        current_a: current,
        resistance_ohm: resistance,
        bracket_span_v: span,
        selected_source_rows: selected,
        all_source_rows: s.iter().map(|q| q.source_row).collect(),
    })
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        values[mid - 1] + (values[mid] - values[mid - 1]) / 2.0
    }
}
/// Reject the entire cohort on invalid or incomplete acquisitions. Device IDs are
/// globally unique, condition labels denote one protocol, and medians never pool
/// devices or conditions. Output is sorted by device and condition, then cycle.
pub fn extract_cohort(branches: &[Branch], p: Settings) -> Result<Vec<Features>, RramError> {
    settings(p)?;
    if branches.is_empty()
        || branches.len() > 1000
        || branches.iter().map(|b| b.samples.len()).sum::<usize>() > 10000
    {
        return Err(RramError::ResourceLimit);
    }
    let mut devices = BTreeMap::new();
    let mut declarations = BTreeMap::new();
    let mut repeats = BTreeMap::new();
    let mut acquisitions = BTreeSet::new();
    let mut rows = BTreeSet::new();
    let mut pairs: BTreeMap<(&str, &str), BTreeMap<State, Readpoint>> = BTreeMap::new();
    for b in branches {
        let point = extract_branch(b, p)?;
        let a = &b.acquisition;
        let identity = (&a.batch, &a.wafer, &a.condition);
        let prev = devices.entry(&a.device).or_insert(identity);
        if prev.0 != identity.0 || prev.1 != identity.1 || (!p.by_condition && prev.2 != identity.2)
        {
            return Err(RramError::IdentityMismatch);
        }
        let declaration = (b.conditions, &a.protocol);
        if *declarations.entry(&a.condition).or_insert(declaration) != declaration {
            return Err(RramError::InvalidConditions);
        }
        if *repeats.entry((&a.device, &a.replicate)).or_insert(&a.cycle) != &a.cycle {
            return Err(RramError::ReplicateMismatch);
        }
        if !acquisitions.insert((&a.device, &a.cycle, &a.branch)) {
            return Err(RramError::DuplicateAcquisition);
        }
        if b.samples.iter().any(|q| !rows.insert(q.source_row)) {
            return Err(RramError::InvalidSample);
        }
        if pairs
            .entry((&a.device, &a.cycle))
            .or_default()
            .insert(a.state, point)
            .is_some()
        {
            return Err(RramError::DuplicateState);
        }
    }
    let mut groups: BTreeMap<(String, String), Vec<Pair>> = BTreeMap::new();
    for (_, mut states) in pairs {
        let hrs = states.remove(&State::Hrs).ok_or(RramError::MissingState)?;
        let lrs = states.remove(&State::Lrs).ok_or(RramError::MissingState)?;
        if hrs.acquisition.replicate != lrs.acquisition.replicate {
            return Err(RramError::ReplicateMismatch);
        }
        if hrs.acquisition.condition != lrs.acquisition.condition
            || hrs.conditions != lrs.conditions
        {
            return Err(RramError::InvalidConditions);
        }
        let ratio = hrs.resistance_ohm / lrs.resistance_ohm;
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(RramError::NumericalRange);
        }
        groups
            .entry((
                hrs.acquisition.device.clone(),
                hrs.acquisition.condition.clone(),
            ))
            .or_default()
            .push(Pair {
                hrs,
                lrs,
                resistance_ratio: ratio,
            });
    }
    Ok(groups
        .into_iter()
        .map(|((device, condition), pairs)| {
            let first = &pairs[0].hrs;
            Features {
                batch: first.acquisition.batch.clone(),
                wafer: first.acquisition.wafer.clone(),
                device,
                condition,
                protocol: first.acquisition.protocol.clone(),
                conditions: first.conditions,
                hrs_resistance_median_ohm: median(
                    pairs.iter().map(|p| p.hrs.resistance_ohm).collect(),
                ),
                lrs_resistance_median_ohm: median(
                    pairs.iter().map(|p| p.lrs.resistance_ohm).collect(),
                ),
                resistance_ratio_median: median(pairs.iter().map(|p| p.resistance_ratio).collect()),
                ratio_below_one_cycle_count: pairs
                    .iter()
                    .filter(|p| p.resistance_ratio < 1.0)
                    .count(),
                pairs,
            }
        })
        .collect())
}
