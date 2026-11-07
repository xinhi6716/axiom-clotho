//! Explicit SI CSV adapter; deliberately distinct from legacy decimal bundles.
use super::{create_file, required, Result};
use clotho_core::fet::{extract_cohort, Acquisition, Sample, Settings};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
const COLUMNS: [&str; 11] = [
    "lot",
    "wafer",
    "device",
    "condition",
    "replicate",
    "branch",
    "temperature_K",
    "drain_source_voltage_V",
    "body_source_voltage_V",
    "gate_source_voltage_V",
    "drain_current_A",
];
fn parse_si(text: &str) -> Result<f64> {
    if text.len() > 64 {
        return Err("FET numeric text exceeds 64 bytes".into());
    }
    let value: f64 = text.parse().map_err(|_| "Invalid FET SI numeric value")?;
    let mantissa = text.split(['e', 'E']).next().unwrap_or("");
    if !value.is_finite() || (value == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)))
    {
        return Err("FET numeric value is outside the supported range".into());
    }
    Ok(value)
}
pub fn run(opts: &HashMap<String, String>) -> Result<()> {
    const ALLOWED: [&str; 8] = [
        "--csv",
        "--out",
        "--criterion-a",
        "--polarity",
        "--max-bracket-v",
        "--window-min-v",
        "--window-max-v",
        "--accept-assumptions",
    ];
    if opts.keys().any(|k| !ALLOWED.contains(&k.as_str())) {
        return Err("Unknown FET extraction option".into());
    }
    if required(opts, "--accept-assumptions")? != "true" {
        return Err("Explicit --accept-assumptions true is required".into());
    }
    let positive_current = match required(opts, "--polarity")? {
        "positive" => true,
        "negative" => false,
        _ => return Err("Polarity must be positive or negative".into()),
    };
    let settings = Settings {
        criterion_a: parse_si(required(opts, "--criterion-a")?)?,
        positive_current,
        max_bracket_span_v: parse_si(required(opts, "--max-bracket-v")?)?,
        window_min_v: parse_si(required(opts, "--window-min-v")?)?,
        window_max_v: parse_si(required(opts, "--window-max-v")?)?,
        assumptions_accepted: true,
    };
    let input = Path::new(required(opts, "--csv")?);
    let output = Path::new(required(opts, "--out")?);
    if output.symlink_metadata().is_ok() {
        return Err("Output already exists".into());
    }
    if !input
        .metadata()
        .map_err(|_| "Unable to inspect FET input")?
        .is_file()
    {
        return Err("FET input must be a regular file".into());
    }
    let file = std::fs::File::open(input).map_err(|_| "Unable to open FET input")?;
    if !file.metadata()?.is_file() {
        return Err("FET input must be a regular file".into());
    }
    let mut raw = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut raw)?;
    if raw.len() > 8 * 1024 * 1024 {
        return Err("FET input exceeds 8 MiB".into());
    }
    let mut reader = csv::Reader::from_reader(raw.as_slice());
    let headers = reader
        .headers()
        .map_err(|_| "Invalid FET CSV header")?
        .clone();
    if headers.len() != COLUMNS.len()
        || COLUMNS
            .iter()
            .any(|c| headers.iter().filter(|h| h == c).count() != 1)
    {
        return Err("FET CSV requires exact documented SI columns".into());
    }
    let indices: Vec<_> = COLUMNS
        .iter()
        .map(|c| headers.iter().position(|h| h == *c).unwrap())
        .collect();
    let mut branches: Vec<(Acquisition, Vec<Sample>)> = Vec::new();
    let mut lookup = HashMap::new();
    let mut count = 0;
    for record in reader.records() {
        let r = record.map_err(|_| "Invalid FET CSV record")?;
        count += 1;
        if count > 10000 {
            return Err("FET input exceeds 10000 records".into());
        }
        let cell = |i: usize| r.get(indices[i]).unwrap();
        // Spreadsheet-safe identifiers for the CSV output, no automatic escaping.
        if (0..6).any(|i| cell(i).starts_with(['=', '+', '-', '@'])) {
            return Err("FET identifiers are unsafe for CSV export".into());
        }
        let a = Acquisition {
            lot: cell(0).into(),
            wafer: cell(1).into(),
            device: cell(2).into(),
            condition: cell(3).into(),
            replicate: cell(4).into(),
            branch: cell(5).into(),
        };
        let parse = |i| parse_si(cell(i));
        // Logical data record index + header, not physical line number.
        let sample = Sample {
            source_row: count + 1,
            temperature_k: parse(6)?,
            drain_voltage_v: parse(7)?,
            body_voltage_v: parse(8)?,
            gate_voltage_v: parse(9)?,
            drain_current_a: parse(10)?,
        };
        let key = (
            a.lot.clone(),
            a.wafer.clone(),
            a.device.clone(),
            a.condition.clone(),
            a.replicate.clone(),
            a.branch.clone(),
        );
        let index = if let Some(&i) = lookup.get(&key) {
            i
        } else {
            let i = branches.len();
            lookup.insert(key, i);
            branches.push((a, Vec::new()));
            i
        };
        if branches.len() > 1000 || branches[index].1.len() >= 1000 {
            return Err("FET branch limit exceeded".into());
        }
        branches[index].1.push(sample);
    }
    let features = extract_cohort(&branches, settings)?;
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "lot",
        "wafer",
        "device",
        "condition",
        "replicate",
        "branch",
        "criterion_gate_source_voltage_V",
        "log_current_slope_decades_per_V",
    ])?;
    let mut lineage = Vec::new();
    for f in &features {
        let a = &f.acquisition;
        writer.write_record([
            a.lot.clone(),
            a.wafer.clone(),
            a.device.clone(),
            a.condition.clone(),
            a.replicate.clone(),
            a.branch.clone(),
            f.criterion_gate_voltage_v.to_string(),
            f.log_slope_decades_per_v.to_string(),
        ])?;
        lineage.push(serde_json::json!({"identity":[a.lot,a.wafer,a.device,a.condition,a.replicate,a.branch],"all_source_rows":f.all_source_rows,"criterion_source_rows":f.criterion_source_rows,"log_selected_source_rows":f.log_selected_source_rows,"criterion_bracket_span_V":f.bracket_span_v,"log_intercept":f.log_intercept,"log_rmse_decades":f.log_rmse_decades}));
    }
    let csv = writer.into_inner()?;
    let report = serde_json::to_vec_pretty(
        &serde_json::json!({"schema_version":"clotho.fet_si_binary64.v1","status":"complete","measurement_rows":count,"device_rows":features.len(),"ml_fits":0,"source_row_convention":"logical CSV data record index plus header; not physical lines","input_units":"explicit SI binary64; no conversion","settings":{"criterion_A":settings.criterion_a,"positive_current":positive_current,"max_bracket_span_V":settings.max_bracket_span_v,"window_min_V":settings.window_min_v,"window_max_V":settings.window_max_v},"limitations":["Not legacy exact-decimal numerical or bundle compatibility","Criterion voltage and descriptive log slope do not certify physical threshold or subthreshold regime","No target joins, model training, confidence intervals or physical independence validation","Input snapshot is retained without cryptographic provenance authentication"],"lineage":lineage}),
    )?;
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(output)?;
    create_file(&output.join("features.csv"), &csv)?;
    create_file(&output.join("transfer_input.csv"), &raw)?;
    create_file(&output.join("extraction.json"), &report)?;
    create_file(&output.join("COMPLETE"), b"clotho.fet_si_binary64.v1\n")?;
    println!("Extracted {} FET device rows; zero ML fits", features.len());
    Ok(())
}
