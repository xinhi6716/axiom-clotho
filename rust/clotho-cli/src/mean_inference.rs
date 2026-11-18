//! Explicit opt-in Mean inference state, never raw training rows or refitting.
use super::*;

const SCHEMA: &str = "clotho.mean_inference.v1";
const LIMIT: u64 = 64 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Summary {
    schema_version: String,
    model: String,
    features: Vec<String>,
    target: String,
    mean: f64,
}

impl Summary {
    fn validate(&self) -> Result<MeanRegressor> {
        if self.schema_version != SCHEMA || self.model != "mean" {
            return Err("Unsupported mean inference schema or model".into());
        }
        validate_features(&self.features, &self.target, None)?;
        if self.features.iter().chain(std::iter::once(&self.target)).any(|s| {
            s.len() > 256 || s.trim() != s || s.chars().any(char::is_control)
        }) {
            return Err("Mean inference names must be bounded nonblank literal names".into());
        }
        Ok(MeanRegressor::from_mean(self.features.len(), self.mean)?)
    }

    // Reuse existing feature alignment and saved-export contract validation only.
    // This metadata has no training records and is never passed to Artifact::fit.
    fn metadata(&self) -> Artifact {
        Artifact {
            schema_version: "clotho.numeric.v3".into(),
            model: Kind::Mean,
            features: self.features.clone(),
            target: self.target.clone(),
            k: 5,
            alpha: 1.0,
            training_x: Vec::new(),
            training_y: Vec::new(),
            training_rows: None,
            categorical_features: Vec::new(),
        }
    }
}

pub(super) fn export(artifact: &Artifact, output: &Path) -> Result<()> {
    let fitted = artifact.fit()?;
    let mean = match fitted.model {
        Model::Mean(model) if fitted.preprocessing.is_none() => model.mean(),
        _ => return Err("Mean inference requires strict numeric mean".into()),
    };
    let summary = Summary {
        schema_version: SCHEMA.into(),
        model: "mean".into(),
        features: artifact.features.clone(),
        target: artifact.target.clone(),
        mean,
    };
    summary.validate()?;
    let bytes = serde_json::to_vec_pretty(&summary)?;
    if bytes.len() as u64 > LIMIT {
        return Err("Mean inference artifact exceeds 64 KiB".into());
    }
    create_file(output, &bytes)?;
    println!("Wrote mean inference summary (no training rows; not a confidentiality guarantee)");
    Ok(())
}

pub(super) fn predict(
    opts: &HashMap<String, String>,
    output: &Path,
    data: &Table,
    query_bytes: Option<&[u8]>,
) -> Result<()> {
    if opts.contains_key("--threshold") {
        return Err("threshold applies only to binary models".into());
    }
    let mut bytes = Vec::new();
    File::open(required(opts, "--artifact")?)?.take(LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err("Mean inference artifact exceeds 64 KiB".into());
    }
    if bytes.iter().copied().find(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r')) != Some(b'{') {
        return Err("Mean inference artifact must be a JSON object".into());
    }
    reject_numeric_underflow(&bytes)?;
    let summary: Summary = serde_json::from_slice(&bytes)
        .map_err(|_| "Invalid strict mean inference JSON")?;
    let model = summary.validate()?;
    let metadata = summary.metadata();
    let rows = data.feature_rows(&metadata)?;
    let x = match rows {
        SourceRows::Numeric(x) => x,
        _ => return Err("Mean inference requires numeric query rows".into()),
    };
    let values = model.predict(&x)?;
    let output_bytes = if let Some(query) = query_bytes {
        predict_saved::Export::prepare(
            required(opts, "--saved-contract")?, required(opts, "--row-id")?, data, &metadata,
        )?.bytes(&values, query)?
    } else {
        predictions_bytes(&values)?
    };
    create_file(output, &output_bytes)?;
    println!("Wrote {} predictions without refitting", values.len());
    Ok(())
}

// Reject decimal nonzero-to-zero loss; serde still validates grammar/duplicates.
fn reject_numeric_underflow(bytes: &[u8]) -> Result<()> {
    let (mut i, mut in_string) = (0, false);
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' { in_string = false; }
        } else if bytes[i] == b'"' {
            in_string = true;
        } else if bytes[i].is_ascii_digit() || bytes[i] == b'-' {
            let start = i;
            while i < bytes.len() && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                i += 1;
            }
            let token = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            if let Ok(value) = token.parse::<f64>() {
                let mantissa = token.split(['e', 'E']).next().unwrap_or("");
                if value == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                    return Err("Nonzero mean inference number underflows binary64".into());
                }
            }
            continue;
        }
        i += 1;
    }
    Ok(())
}
