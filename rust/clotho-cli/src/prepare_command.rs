//! Exact, no-fit measurement-table joins; distinct from legacy prepared bundles.
use super::{create_file, number, required, table, Result, Table};
use clotho_core::measurement::{
    prepare_measurements, DeviceFeatures, DevicePartition, Partition, TrainingEndpoint,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
fn columns(text: &str) -> Result<Vec<String>> {
    let v: Vec<String> = text.split(',').map(str::to_owned).collect();
    let mut seen = HashSet::new();
    if v.is_empty()
        || v.iter().any(|x| {
            x.is_empty()
                || x.chars().count() > 128
                || x.trim() != x
                || x.starts_with(['=', '+', '-', '@'])
                || x.chars().any(char::is_control)
                || !seen.insert(x)
        })
    {
        return Err("Column names must be nonempty and unique".into());
    }
    Ok(v)
}
fn positions(t: &Table, names: &[String]) -> Result<Vec<usize>> {
    names
        .iter()
        .map(|n| {
            t.headers
                .iter()
                .position(|h| h == n)
                .ok_or_else(|| "Required join column is missing".into())
        })
        .collect()
}
fn identity(row: &[String], indices: &[usize]) -> Vec<String> {
    indices.iter().map(|&i| row[i].clone()).collect()
}
fn finite(s: &str) -> Result<f64> {
    let x: f64 = s.parse().map_err(|_| "Invalid measurement numeric value")?;
    if !x.is_finite()
        || (x == 0.0
            && s.split(['e', 'E'])
                .next()
                .unwrap_or("")
                .bytes()
                .any(|b| (b'1'..=b'9').contains(&b)))
    {
        return Err("Measurement numeric value out of range".into());
    }
    Ok(x)
}
pub fn run(opts: &HashMap<String, String>) -> Result<()> {
    let allowed = [
        "--csv",
        "--partitions",
        "--training-targets",
        "--identity",
        "--features",
        "--group-prefix",
        "--out",
    ];
    if opts.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err("Unknown measurement preparation option".into());
    }
    let ids = columns(required(opts, "--identity")?)?;
    let feats = columns(required(opts, "--features")?)?;
    if ids.len() > 6
        || feats.len() > 100
        || ids
            .iter()
            .chain(&feats)
            .any(|x| ["partition", "target", "row_id", "group"].contains(&x.as_str()))
        || ids.iter().any(|x| feats.contains(x))
    {
        return Err("Identity and features must be disjoint and exclude reserved columns".into());
    }
    let group_prefix = number(opts, "--group-prefix", "0")?;
    let out = Path::new(required(opts, "--out")?);
    if out.symlink_metadata().is_ok() {
        return Err("Output already exists".into());
    }
    for key in ["--csv", "--partitions", "--training-targets"] {
        if !std::fs::metadata(required(opts, key)?)
            .map_err(|_| "Unable to inspect measurement input")?
            .is_file()
        {
            return Err("Measurement inputs must be regular files".into());
        }
    }
    let data = table(required(opts, "--csv")?)?;
    let parts = table(required(opts, "--partitions")?)?;
    let targets = table(required(opts, "--training-targets")?)?;
    // Endpoint and partition schemas are deliberately exact: no future labels,
    // target aliases or hidden secondary selection fields are admitted.
    if parts.headers.len() != ids.len() + 1 || targets.headers.len() != ids.len() + 1 {
        return Err(
            "Partition/endpoint tables require exact identity plus partition/target schemas".into(),
        );
    }
    let di = positions(&data, &ids)?;
    let fi = positions(&data, &feats)?;
    let pi = positions(&parts, &ids)?;
    let ti = positions(&targets, &ids)?;
    let pc = positions(&parts, &["partition".into()])?[0];
    let tc = positions(&targets, &["target".into()])?[0];
    let mut f = Vec::new();
    for r in &data.rows {
        f.push(DeviceFeatures {
            identity: identity(r, &di),
            values: fi
                .iter()
                .map(|&i| finite(&r[i]))
                .collect::<Result<Vec<_>>>()?,
        });
    }
    let mut p = Vec::new();
    for r in &parts.rows {
        p.push(DevicePartition {
            identity: identity(r, &pi),
            partition: match r[pc].as_str() {
                "train" => Partition::Train,
                "predict" => Partition::Predict,
                _ => return Err("Partition must be train or predict".into()),
            },
        });
    }
    let mut t = Vec::new();
    for r in &targets.rows {
        t.push(TrainingEndpoint {
            identity: identity(r, &ti),
            target: finite(&r[tc])?,
        });
    }
    let prepared = prepare_measurements(&f, &p, &t, group_prefix)?;
    let mut payloads = Vec::new();
    let mut mapping = Vec::new();
    for (name, rows, training) in [
        ("training.csv", &prepared.training, true),
        ("future.csv", &prepared.prediction, false),
    ] {
        let mut w = csv::Writer::from_writer(Vec::new());
        let mut header = vec!["row_id".to_string(), "group".into()];
        if training {
            header.push("target".into());
        }
        header.extend(feats.clone());
        w.write_record(&header)?;
        for (i, r) in rows.iter().enumerate() {
            let row_id = format!("{}_{}", if training { "train" } else { "predict" }, i + 1);
            let group = serde_json::to_string(&r.group)?;
            let mut values = vec![row_id.clone(), group];
            if training {
                values.push(
                    r.target
                        .ok_or("Internal missing training endpoint")?
                        .to_string(),
                );
            }
            values.extend(r.values.iter().map(ToString::to_string));
            w.write_record(&values)?;
            mapping.push(serde_json::json!({"row_id":row_id,"identity":r.identity,"group_tuple":r.group,"partition":if training{"train"}else{"predict"}}));
        }
        payloads.push((name, w.into_inner()?));
    }
    let manifest = serde_json::to_vec_pretty(
        &serde_json::json!({"schema_version":"clotho.measurement_join.v1","identity_columns":ids,"feature_columns":feats,"group_prefix":group_prefix,"training_rows":prepared.training.len(),"prediction_rows":prepared.prediction.len(),"ml_fits":0,"mapping":mapping,"limitations":["Exact declared identities only; aliases and unrecorded dependence are not detected","No source authentication or historical feature-timing validation","Not legacy prepared-bundle compatibility","Only explicit finite numeric selected features; no derived or imputed endpoints"]}),
    )?;
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(out)?;
    for (name, bytes) in payloads {
        create_file(&out.join(name), &bytes)?;
    }
    create_file(&out.join("preparation.json"), &manifest)?;
    create_file(&out.join("COMPLETE"), b"clotho.measurement_join.v1\n")?;
    println!(
        "Prepared {} training and {} prediction rows; zero fits",
        prepared.training.len(),
        prepared.prediction.len()
    );
    Ok(())
}
