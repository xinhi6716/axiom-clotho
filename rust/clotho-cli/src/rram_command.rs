//! Flattened explicit-SI CSV adapter, not a legacy exact-decimal bundle reader.
use super::device_csv::has_complete_csv_quoting;
use super::{create_file, required, Result};
use clotho_core::rram::{
    extract_cohort, Acquisition, Branch, Conditions, Readpoint, Sample, Settings, State,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
const COLUMNS: [&str; 14] = [
    "batch",
    "wafer",
    "device",
    "cycle",
    "replicate",
    "branch",
    "state",
    "condition",
    "protocol",
    "read_voltage_V",
    "temperature_K",
    "read_delay_s",
    "voltage_V",
    "current_A",
];
fn parse_si(text: &str) -> Result<f64> {
    if text.len() > 64 {
        return Err("RRAM numeric text exceeds 64 bytes".into());
    }
    let value: f64 = text.parse().map_err(|_| "Invalid RRAM SI numeric value")?;
    let mantissa = text.split(['e', 'E']).next().unwrap_or("");
    if !value.is_finite() || (value == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)))
    {
        return Err("RRAM numeric value is outside the supported range".into());
    }
    Ok(value)
}
fn lineage(p: &Readpoint) -> serde_json::Value {
    let a = &p.acquisition;
    serde_json::json!({"branch":a.branch,"replicate":a.replicate,"all_source_rows":p.all_source_rows,"selected_source_rows":p.selected_source_rows,"current_A":p.current_a,"resistance_ohm":p.resistance_ohm,"bracket_span_V":p.bracket_span_v})
}
// Successful extraction admits no control characters in any field, so each
// logical record occupies one physical line. Keep offsets tied to the exact
// retained bytes, including UTF-8 BOM, blank lines and mixed CSV terminators.
fn source_record_locations(raw: &[u8], count: usize) -> Result<Vec<[usize; 2]>> {
    let bom = if raw.starts_with(b"\xef\xbb\xbf") { 3 } else { 0 };
    let mut locations = Vec::with_capacity(count + 1);
    let mut start = bom;
    let mut line = 1;
    while start < raw.len() {
        let end = raw[start..]
            .iter()
            .position(|&b| b == b'\r' || b == b'\n')
            .map_or(raw.len(), |n| start + n);
        if start < end {
            locations.push([line, start]);
        }
        if end == raw.len() {
            break;
        }
        start = end + 1;
        if raw[end] == b'\r' && raw.get(start) == Some(&b'\n') {
            start += 1;
        }
        line += 1;
    }
    if locations.len() != count + 1 {
        return Err("RRAM physical source locations do not match admitted CSV records".into());
    }
    // The first nonempty record is the header, not a measurement.
    locations.remove(0);
    Ok(locations)
}
pub fn run(opts: &HashMap<String, String>) -> Result<()> {
    const ALLOWED: [&str; 7] = [
        "--csv",
        "--out",
        "--max-bracket-v",
        "--accept-assumptions",
        "--feature-row-unit",
        "--require-sampled-readpoints",
        "--include-source-locations",
    ];
    if opts.keys().any(|k| !ALLOWED.contains(&k.as_str())) {
        return Err("Unknown RRAM extraction option".into());
    }
    if required(opts, "--accept-assumptions")? != "true" {
        return Err("Explicit --accept-assumptions true is required".into());
    }
    let require_sampled = match opts.get("--require-sampled-readpoints").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => return Err("--require-sampled-readpoints must be true or false".into()),
    };
    let include_locations = match opts.get("--include-source-locations").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(_) => return Err("--include-source-locations must be true or false".into()),
    };
    let by_condition = match opts
        .get("--feature-row-unit")
        .map(String::as_str)
        .unwrap_or("device")
    {
        "device" => false,
        "device-condition" => true,
        _ => return Err("Feature row unit must be device or device-condition".into()),
    };
    let settings = Settings {
        max_bracket_span_v: parse_si(required(opts, "--max-bracket-v")?)?,
        assumptions_accepted: true,
        by_condition,
    };
    let input = Path::new(required(opts, "--csv")?);
    let output = Path::new(required(opts, "--out")?);
    if output.symlink_metadata().is_ok() {
        return Err("Output already exists".into());
    }
    if !input
        .metadata()
        .map_err(|_| "Unable to inspect RRAM input")?
        .is_file()
    {
        return Err("RRAM input must be a regular file".into());
    }
    let file = std::fs::File::open(input).map_err(|_| "Unable to open RRAM input")?;
    if !file.metadata()?.is_file() {
        return Err("RRAM input must be a regular file".into());
    }
    let mut raw = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut raw)?;
    if raw.len() > 8 * 1024 * 1024 {
        return Err("RRAM input exceeds 8 MiB".into());
    }
    if !has_complete_csv_quoting(&raw) {
        return Err("Invalid RRAM CSV quoting".into());
    }
    let mut reader = csv::Reader::from_reader(raw.as_slice());
    let headers = reader
        .headers()
        .map_err(|_| "Invalid RRAM CSV header")?
        .clone();
    if headers.len() != COLUMNS.len()
        || COLUMNS
            .iter()
            .any(|c| headers.iter().filter(|h| h == c).count() != 1)
    {
        return Err("RRAM CSV requires exact documented SI columns".into());
    }
    let indices: Vec<_> = COLUMNS
        .iter()
        .map(|c| headers.iter().position(|h| h == *c).unwrap())
        .collect();
    let mut branches: Vec<Branch> = Vec::new();
    let mut lookup = HashMap::<Vec<u8>, usize>::new();
    let mut key = Vec::new();
    let mut r = csv::StringRecord::new();
    let mut count = 0;
    while reader
        .read_record(&mut r)
        .map_err(|_| "Invalid RRAM CSV record")?
    {
        count += 1;
        if count > 10000 {
            return Err("RRAM input exceeds 10000 records".into());
        }
        let cell = |i: usize| r.get(indices[i]).unwrap();
        let state = match cell(6) {
            "HRS" => State::Hrs,
            "LRS" => State::Lrs,
            _ => return Err("RRAM state must be HRS or LRS".into()),
        };
        let conditions = Conditions {
            read_voltage_v: parse_si(cell(9))?,
            temperature_k: parse_si(cell(10))?,
            read_delay_s: parse_si(cell(11))?,
        };
        let sample = Sample {
            source_row: count + 1,
            voltage_v: parse_si(cell(12))?,
            current_a: parse_si(cell(13))?,
        };
        // StringRecord has already validated UTF-8, so 0xff is an
        // unambiguous component separator, even before ID validation.
        key.clear();
        key.extend_from_slice(cell(2).as_bytes());
        key.push(0xff);
        key.extend_from_slice(cell(3).as_bytes());
        key.push(0xff);
        key.extend_from_slice(cell(5).as_bytes());
        if let Some(&index) = lookup.get(key.as_slice()) {
            let branch = &mut branches[index];
            if branch.samples.len() >= 1000 {
                return Err("RRAM branch limit exceeded".into());
            }
            let a = &branch.acquisition;
            if a.batch != cell(0)
                || a.wafer != cell(1)
                || a.replicate != cell(4)
                || a.state != state
                || a.condition != cell(7)
                || a.protocol != cell(8)
                || branch.conditions != conditions
            {
                return Err("RRAM branch declarations must remain fixed".into());
            }
            branch.samples.push(sample);
        } else {
            if branches.len() >= 1000 {
                return Err("RRAM branch limit exceeded".into());
            }
            lookup.insert(key.clone(), branches.len());
            branches.push(Branch {
                acquisition: Acquisition {
                    batch: cell(0).into(),
                    wafer: cell(1).into(),
                    device: cell(2).into(),
                    cycle: cell(3).into(),
                    replicate: cell(4).into(),
                    branch: cell(5).into(),
                    state,
                    condition: cell(7).into(),
                    protocol: cell(8).into(),
                },
                conditions,
                samples: vec![sample],
            });
        }
    }
    let features = extract_cohort(&branches, settings)?;
    if require_sampled
        && features.iter().flat_map(|f| &f.pairs).any(|pair| {
            pair.hrs.selected_source_rows.len() != 1 || pair.lrs.selected_source_rows.len() != 1
        })
    {
        return Err("Sampled-only policy requires sampled RRAM readpoints for both states".into());
    }
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "batch",
        "wafer",
        "device",
        "condition",
        "physical_device_group",
        "split_group",
        "protocol",
        "read_voltage_V",
        "temperature_K",
        "read_delay_s",
        "cycle_count",
        "ratio_below_one_cycle_count",
        "hrs_resistance_median_ohm",
        "lrs_resistance_median_ohm",
        "resistance_ratio_median",
    ])?;
    let mut records = Vec::new();
    for f in &features {
        writer.write_record([
            f.batch.clone(),
            f.wafer.clone(),
            f.device.clone(),
            f.condition.clone(),
            f.device.clone(),
            f.batch.clone(),
            f.protocol.clone(),
            f.conditions.read_voltage_v.to_string(),
            f.conditions.temperature_k.to_string(),
            f.conditions.read_delay_s.to_string(),
            f.pairs.len().to_string(),
            f.ratio_below_one_cycle_count.to_string(),
            f.hrs_resistance_median_ohm.to_string(),
            f.lrs_resistance_median_ohm.to_string(),
            f.resistance_ratio_median.to_string(),
        ])?;
        let mut readpoint_support = [0_usize; 4];
        for pair in &f.pairs {
            // extract_cohort reports one source row for a sampled read
            // and two adjacent rows for an interpolated read.
            let hrs = usize::from(pair.hrs.selected_source_rows.len() == 2);
            let lrs = usize::from(pair.lrs.selected_source_rows.len() == 2);
            readpoint_support[hrs + 2 * lrs] += 1;
        }
        let pairs: Vec<_> = f.pairs.iter().map(|p| serde_json::json!({"cycle":p.hrs.acquisition.cycle,"HRS":lineage(&p.hrs),"LRS":lineage(&p.lrs),"resistance_ratio":p.resistance_ratio,"ratio_below_one":p.resistance_ratio < 1.0})).collect();
        records.push(
            serde_json::json!({
                "identity": [f.batch, f.wafer, f.device, f.condition],
                "cycle_readpoint_support_counts": {
                    "both_sampled": readpoint_support[0],
                    "hrs_only_interpolated": readpoint_support[1],
                    "lrs_only_interpolated": readpoint_support[2],
                    "both_interpolated": readpoint_support[3],
                },
                "pairs": pairs
            }),
        );
    }
    let csv = writer.into_inner()?;
    let input_sha256 = format!("{:x}", Sha256::digest(&raw));
    let features_sha256 = format!("{:x}", Sha256::digest(&csv));
    let mut report = serde_json::json!({"schema_version":"clotho.rram_si_binary64.v1","status":"complete","input_sha256":input_sha256,"features_sha256":features_sha256,"measurement_rows":count,"feature_rows":features.len(),"ml_fits":0,"feature_row_unit":if by_condition {"device-condition"} else {"device"},"source_row_convention":"logical CSV data record index plus header; not physical lines","input_units":"explicit SI binary64; no conversion","settings":{"max_bracket_span_V":settings.max_bracket_span_v,"assumptions_accepted":true},"feature_allowlist":["hrs_resistance_median_ohm","lrs_resistance_median_ohm","resistance_ratio_median"],"limitations":["Not legacy exact-decimal numerical or bundle compatibility","Declared states and conditions are not independently verified","Linear interpolation assumes local linearity; static V/I is not differential resistance","Cycles and condition rows are repeated observations, not independent devices; group downstream by physical_device_group or broader split_group","Ratio is derived from HRS/LRS observations; do not predict it from those resistances","No target joins, model training, confidence intervals or physical independence validation","Input and feature hashes identify exact bytes, not authenticated provenance"],"lineage":records});
    if require_sampled {
        report["settings"]["require_sampled_readpoints"] = true.into();
    }
    if include_locations {
        report["source_locations"] = serde_json::json!({
            "format": "[one-based physical line, zero-based byte offset in measurement_input.csv]",
            "first_logical_source_row": 2,
            "locations": source_record_locations(&raw, count)?,
        });
    }
    let report = serde_json::to_vec_pretty(&report)?;
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(output)?;
    create_file(&output.join("features.csv"), &csv)?;
    create_file(&output.join("measurement_input.csv"), &raw)?;
    create_file(&output.join("extraction.json"), &report)?;
    create_file(&output.join("COMPLETE"), b"clotho.rram_si_binary64.v1\n")?;
    println!(
        "Extracted {} RRAM feature rows; zero ML fits",
        features.len()
    );
    Ok(())
}
