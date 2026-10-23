"""Fresh-only binary bundles, using the unchanged generic safe text writer."""
from __future__ import annotations
import html
import json
from pathlib import Path

KNOWN_OUTPUTS = ("report.html", "summary.json", "manifest.json", "predictions.csv", "checksums.json")
MAX_BINARY_PAYLOAD_BYTES = 64 * 1024 * 1024


def ensure_fresh_binary_output(output_dir):
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise ValueError("Binary output must be a fresh local directory.")
    if any((out / name).exists() or (out / name).is_symlink() for name in KNOWN_OUTPUTS):
        raise FileExistsError("Binary output already contains a known bundle name; choose a fresh directory. Binary overwrite is not supported.")


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


def render_binary_html(result):
    esc = lambda value: html.escape(str(value), quote=True)
    models, m = result.summary["models"], result.manifest
    selected = models["selected"]["overall"]
    rows = []
    for name, metrics in sorted(models.items(), key=lambda item: (item[1]["overall"]["group_brier"], item[0])):
        s = metrics["overall"]
        rows.append(f'<tr><th>{esc(name)}</th><td>{_number(s["group_brier"])}</td><td>{_number(s["brier"])}</td><td>{_number(s["group_log_loss_clipped"])}</td><td>{_number(s["accuracy"])}</td><td>{s["confusion"]["fn"]}</td><td>{s["confusion"]["fp"]}</td></tr>')
    worst = []
    for row in sorted(models["selected"]["per_group"], key=lambda r: -r["brier"])[:15]:
        worst.append(f'<tr><th>{esc(row["group"])}</th><td>{row["n_rows"]}</td><td>{row["n_true_0"]} / {row["n_true_1"]}</td><td>{_number(row["brier"])}</td><td>{_number(row["log_loss_clipped"])}</td><td>{row["confusion"]["fn"]} / {row["confusion"]["fp"]}</td><td>{_number(row["recall_1"])}</td><td>{_number(row["specificity_0"])}</td></tr>')
    adverse = result.summary["adverse_comparison_to_dummy"]
    if adverse["available"]:
        comparison = (f'Selected minus Dummy group Brier: <strong>{_number(adverse["group_brier_delta"])}</strong> '
                      f'(positive is worse). Groups: {adverse["groups_better"]} better, '
                      f'{adverse["groups_tied"]} tied, {adverse["groups_worse"]} worse. Descriptive, not significance.')
    else:
        comparison = esc(adverse["reason"])
    audits = "".join(f'<tr><th>{f["fold"]}</th><td>{f["train_group_count"]} / {f["test_group_count"]}</td><td>{esc(f["selected_candidate"])}</td><td>{f["overlap_count"]}</td></tr>' for f in m["outer_folds"])
    warnings = "".join(f"<li>{esc(item)}</li>" for item in m["limitations"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>ML Super Calculator | Binary {esc(m['config']['target'])}</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;background:#f3f7f8;color:#183240;margin:0}}header{{background:#123744;color:white;padding:36px max(24px,calc((100vw - 1100px)/2))}}main{{max-width:1100px;margin:auto;padding:24px}}h1{{margin:0;font-size:32px}}section{{background:white;padding:24px;border:1px solid #d9e4e7;border-radius:10px;margin-bottom:20px}}.cards{{display:flex;gap:20px;flex-wrap:wrap}}.cards div{{flex:1;min-width:180px}}strong.value{{display:block;font-size:32px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:right;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #d9e4e7}}th:first-child{{text-align:left}}.note{{border-left:4px solid #128775;padding-left:16px}}small{{color:#526b76}}code{{overflow-wrap:anywhere}}@media print{{section{{break-inside:avoid}}header{{background:white;color:#183240}}}}</style></head><body>
<header><small style="color:#bfeada">LOCAL · EXPLICIT BINARY 0/1 · NO HOSTED API</small><h1>ML Super Calculator</h1><p>Unseen-group evaluation of {esc(m['config']['target'])}, holding out {esc(m['config']['group'])} groups.</p></header>
<main><section class="cards"><div><small>Selected procedure · group Brier</small><strong class="value">{_number(selected['group_brier'])}</strong><small>Lower is better · binary [0,1] scale</small></div><div><small>Held-out rows / groups</small><strong class="value">{m['n_rows']} / {m['n_groups']}</strong><small>One held-out probability pair per entry per row</small></div><div><small>Observed label 0 / label 1</small><strong class="value">{selected['n_true_0']} / {selected['n_true_1']}</strong><small>Meaning of label 1 is user-defined</small></div></section>
<section><h2>Read this correctly</h2><p class="note">The primary entry is <strong>selected</strong>, chosen only inside training folds. The display ranking is not another tuning step. Probabilities are not established calibrated failure risks. A fixed p1 ≥ 0.5 predicts label 1, including ties; this is not an optimized process decision.</p><p>Group Brier averages row squared probability error within each group, then weights groups equally. Models train with equal row weights. Natural-log loss is secondary and clips a metric-only copy at float64 epsilon; raw probabilities and clipping counts remain in CSV/JSON.</p></section>
<section><h2>All baseline results</h2><div class="scroll"><table><thead><tr><th>Entry</th><th>Group Brier</th><th>Row Brier</th><th>Group clipped log loss</th><th>Accuracy</th><th>False negatives</th><th>False positives</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>{comparison}</p></section>
<section><h2>Where it struggles</h2><p>Up to 15 groups with largest selected Brier. Small-group extremes are unstable. Undefined metrics remain undefined; no group is removed. Full details are in summary.json.</p><div class="scroll"><table><thead><tr><th>Group</th><th>Rows</th><th>True 0 / 1</th><th>Brier</th><th>Clipped log loss</th><th>FN / FP</th><th>Recall 1</th><th>Specificity 0</th></tr></thead><tbody>{''.join(worst)}</tbody></table></div></section>
<section><h2>Split audit</h2><div class="scroll"><table><thead><tr><th>Fold</th><th>Train / test groups</th><th>Inner-selected candidate</th><th>Overlap</th></tr></thead><tbody>{audits}</tbody></table></div></section>
<section><h2>Limits and reproducibility</h2><ul>{warnings}</ul><p>Protocol: {esc(m['protocol'])}<br>Seed: {m['config']['random_state']} · {m['fit_budget']['total_fits']} pipeline fits · one thread<br>Run key: <code>{esc(m['run_key'])}</code><br>Generated locally: {esc(m['created_at_utc'])}</p><p>No model files, external assets, telemetry or automatic physical decisions. Checksums verify bytes only, not scientific correctness or authenticity.</p></section></main></body></html>'''


def _write(payloads, output_dir):
    ensure_fresh_binary_output(output_dir)
    for name, content in payloads.items():
        if len(content.encode("utf-8")) > MAX_BINARY_PAYLOAD_BYTES:
            raise ValueError(f"Binary payload {name} exceeds the {MAX_BINARY_PAYLOAD_BYTES}-byte verifier limit; nothing was written.")
    from ..report import write_payloads
    write_payloads(payloads, output_dir, overwrite=False)


def write_binary_evaluation(result, output_dir):
    ensure_fresh_binary_output(output_dir)
    payloads = {"summary.json": json.dumps(result.summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                "manifest.json": json.dumps(result.manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                "predictions.csv": result.predictions.to_csv(index=False, lineterminator="\n"),
                "report.html": render_binary_html(result)}
    _write(payloads, output_dir)
    return Path(output_dir) / "report.html"


def write_binary_prediction(result, output_dir):
    ensure_fresh_binary_output(output_dir)
    payloads = {"manifest.json": json.dumps(result.manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                "predictions.csv": result.predictions.to_csv(index=False, lineterminator="\n")}
    _write(payloads, output_dir)
    return Path(output_dir) / "predictions.csv"
