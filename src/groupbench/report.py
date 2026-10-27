"""Offline, self-contained reports. All input-derived HTML is escaped."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

from .package_source import package_source_display


def _number(value):
    return "undefined" if value is None else f"{value:,.4f}"


def render_html(result) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    m = result.manifest
    models = result.summary["models"]
    primary = models["selected"]["overall"]
    ranking = sorted(models.items(), key=lambda pair: (pair[1]["overall"]["group_mae"], pair[0]))
    max_mae = max(value["overall"]["group_mae"] for _, value in ranking) or 1
    labels = {
        name: esc(name) + (' <small class="entry-label">Selection procedure</small>' if name == "selected" else "")
        for name, _ in ranking
    }
    bars = "".join(
        f'<div class="bar-row"><span>{labels[name]}</span><div class="track"><div class="bar {"primary" if name == "selected" else ""}" style="width:{max(1, 100 * value["overall"]["group_mae"] / max_mae):.2f}%"></div></div><b>{_number(value["overall"]["group_mae"])}</b></div>'
        for name, value in ranking
    )
    table = "".join(
        f'<tr><th scope="row">{esc(name)}</th><td>{_number(value["overall"]["group_mae"])}</td><td>{_number(value["overall"]["mae"])}</td><td>{_number(value["overall"]["rmse"])}</td><td>{_number(value["overall"]["r2"])}</td><td>{_number(value["overall"]["worst_group_mae"])}</td></tr>'
        for name, value in ranking
    )
    worst = sorted(models["selected"]["per_group"], key=lambda row: -row["mae"])[:15]
    group_rows = "".join(f'<tr><th scope="row">{esc(row["group"])}</th><td>{row["n_rows"]}</td><td>{_number(row["mae"])}</td><td>{_number(row["rmse"])}</td><td>{_number(row["bias"])}</td></tr>' for row in worst)
    fold_rows = "".join(
        f'<tr><th scope="row">{fold["fold"]}</th><td>{fold["train_group_count"]} / {fold["test_group_count"]}</td><td>{fold["train_rows"]} / {fold["test_rows"]}</td><td>{esc(fold["selected_candidate"])}</td><td>{fold["overlap_count"]}</td></tr>' for fold in m["outer_folds"]
    )
    warnings = "".join(f"<li>{esc(item)}</li>" for item in m["limitations"])
    singleton_note = (
        '<p>Recorded rows = groups: every supplied group has one row. There are no repeated-observation groups for this split to protect; check that related rows have not been assigned different IDs.</p>'
        if type(m['n_rows']) is int and type(m['n_groups']) is int
        and m['n_rows'] > 0 and m['n_rows'] == m['n_groups'] else ''
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>ML Super Calculator | {esc(m['config']['target'])}</title>
<style>
:root{{--ink:#172b3a;--muted:#526778;--paper:#f4f7f8;--line:#dce5e9;--accent:#087f75;--mint:#bcf2dc}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.6 system-ui,-apple-system,sans-serif}}header{{background:#102e3b;color:white;padding:44px max(24px,calc((100vw - 1080px)/2));}}header p{{max-width:720px;color:#c3d7e0;margin:10px 0}}h1{{font-size:40px;letter-spacing:-1.5px;margin:8px 0}}h2{{font-size:23px;margin:0 0 14px}}h3{{font-size:16px;margin:0 0 8px}}.eyebrow{{color:var(--mint);text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:750}}main{{max-width:1128px;margin:auto;padding:28px 24px 48px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.card,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:20px}}.card{{margin:0}}.value{{display:block;font-size:34px;font-weight:750;letter-spacing:-1px}}.muted,small{{color:var(--muted)}}.note{{border-left:4px solid var(--accent);background:#edf9f4;padding:14px 18px;margin:20px 0}}section p{{max-width:920px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;text-align:right;white-space:nowrap}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line)}}thead th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}}th:first-child{{text-align:left}}tbody th{{font-weight:600}}.bar-row{{display:grid;grid-template-columns:110px 1fr 90px;align-items:center;gap:14px;margin:12px 0;font-size:13px}}.track{{background:#eef3f4;height:20px;border-radius:4px;overflow:hidden}}.bar{{height:100%;background:#9eb3be;border-radius:4px}}.bar.primary{{background:var(--accent)}}.bar-row b{{text-align:right}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{word-break:break-all;font-size:12px}}ul{{padding-left:20px}}footer{{color:var(--muted);font-size:12px}}@media(max-width:700px){{.grid,.split{{grid-template-columns:1fr}}h1{{font-size:32px}}header{{padding:30px 24px}}.bar-row{{grid-template-columns:90px 1fr 70px;gap:8px}}section,.card{{padding:18px}}}}@media print{{header{{color:#172b3a;background:white}}header p,.eyebrow{{color:#526778}}section,.card{{break-inside:avoid}}body{{background:white}}}}

/* Offline reading path: no scripts, remote assets, or color-only state. */
a{{color:#075e57;text-underline-offset:3px}}a:focus-visible,[tabindex]:focus-visible{{outline:3px solid #925400;outline-offset:4px}}.skip-link{{position:absolute;left:16px;top:-100px;background:white;padding:12px 18px;z-index:2}}.skip-link:focus{{top:12px}}.report-nav{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}}.report-nav a{{display:flex;gap:10px;align-items:center;padding:10px 14px;border:1px solid #bcced3;border-radius:8px;background:white;text-decoration:none;font-size:12px;font-variant-numeric:tabular-nums}}.report-nav span{{color:var(--ink);font-size:14px;font-weight:650}}.report-nav a:hover,.evidence-links a:hover{{background:#edf9f4}}section{{scroll-margin-top:20px}}caption{{text-align:left;color:var(--ink);font-size:13px;font-weight:650;padding:10px 12px;background:#f4f7f8;border-bottom:1px solid var(--line)}}.table-wrap{{border:1px solid var(--line);border-radius:8px}}.entry-label{{display:block;font-size:10px;line-height:1.3;color:#075e57;font-weight:650}}.bar.primary{{border:2px solid #064a44}}.evidence{{padding:24px;background:#eaf1f3;border:1px solid #cbdadf;border-radius:12px;margin-bottom:24px}}.evidence p{{max-width:850px}}.evidence-links{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.evidence-links a{{display:block;background:white;border:1px solid #bcced3;border-radius:8px;padding:14px 16px;text-decoration:none}}.evidence-links strong{{text-decoration:underline}}.evidence-links span{{display:block;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}@media(max-width:700px){{.evidence-links{{grid-template-columns:1fr}}.report-nav a{{flex:1 1 140px}}.evidence{{padding:18px}}}}@media print{{.skip-link,.report-nav{{display:none}}.evidence{{background:white;break-inside:avoid}}}}
</style></head><body>
<a class="skip-link" href="#overview">Skip to report</a>
<header><div class="eyebrow">Local-first ML · group-aware evaluation</div><h1>ML Super Calculator</h1><p>How does the selection procedure perform on held-out groups?</p><p>Nested, group-disjoint evaluation of <strong>{esc(m['config']['target'])}</strong>, holding out complete <strong>{esc(m['config']['group'])}</strong> groups.</p></header>
<main id="overview" tabindex="-1"><nav class="report-nav" aria-label="Report sections"><a href="#overview">01 <span>Overview</span></a><a href="#comparison">02 <span>Comparison</span></a><a href="#difficult-groups">03 <span>Difficult groups</span></a><a href="#split-audit">04 <span>Split audit</span></a><a href="#run-details">05 <span>Run details</span></a></nav><div class="grid"><div class="card"><small>Selected pipeline · group-balanced MAE</small><span class="value">{_number(primary['group_mae'])}</span><small>Lower is better · target units</small></div><div class="card"><small>Held-out observations</small><span class="value">{m['n_rows']:,}</span><small>{m['n_groups']} distinct groups · one prediction per row</small></div><div class="card"><small>Outer / inner folds</small><span class="value">{m['config']['outer_splits']} / {m['config']['inner_splits']}</span><small>Single-thread fits · seed {m['config']['random_state']}</small></div></div>
<div class="note"><strong>Read this correctly:</strong> “selected” is the full selection procedure, chosen only on inner folds. The outer leaderboard is for evaluation, not another tuning set. This report does not establish performance on real materials when run on the synthetic fixture.</div>
<section id="comparison" aria-labelledby="comparison-heading"><h2 id="comparison-heading">Baseline comparison</h2>
<div class="note" aria-labelledby="split-scope-heading"><h3 id="split-scope-heading">What this split can answer</h3><p>Whole-group holdout on: <code>{esc(m['config']['group'])}</code>. Zero group-ID overlap checks only the supplied IDs, not their physical independence.</p><ul><li><strong>Domain-heldout?</strong> Separation is only along the supplied group column; other domain/site axes are not separately audited. If that column defines the intended domain boundary, its whole values are held out, but the scientific meaning of the IDs is not verified.</li><li><strong>Random-row comparison?</strong> Not run. The seed assigns whole groups to folds; it does not split rows within a group.</li></ul><p>Different group IDs may still share a site, time period or another dependence axis. Review the grouping boundary before interpreting scores.</p>{singleton_note}</div>
<p class="muted">Group-balanced MAE gives each group equal weight, regardless of its number of rows. Row MAE and RMSE pool all held-out observations. No confidence intervals are implied.</p>{bars}<div class="table-wrap" role="region" aria-labelledby="comparison-table" tabindex="0"><table><caption id="comparison-table">Held-out metrics for each entry</caption><thead><tr><th>Entry</th><th>Group MAE</th><th>Row MAE</th><th>Row RMSE</th><th>Pooled R²</th><th>Worst-group MAE</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section id="difficult-groups" aria-labelledby="groups-heading"><h2 id="groups-heading">Where it struggles</h2><p class="muted">Up to 15 groups with the largest selected-pipeline MAE. Bias is prediction minus observation; all groups are available in summary.json.</p><div class="table-wrap" role="region" aria-labelledby="groups-table" tabindex="0"><table><caption id="groups-table">Selected procedure: highest-error groups</caption><thead><tr><th>Held-out group</th><th>Rows</th><th>MAE</th><th>RMSE</th><th>Bias</th></tr></thead><tbody>{group_rows}</tbody></table></div></section>
<section id="split-audit" aria-labelledby="audit-heading"><h2 id="audit-heading">Split audit</h2><p class="muted">No group may occur in both training and validation within a fold. Inner selections and exact group assignments are recorded in manifest.json.</p><div class="table-wrap" role="region" aria-labelledby="audit-table" tabindex="0"><table><caption id="audit-table">Training and test split counts</caption><thead><tr><th>Outer fold</th><th>Train / test groups</th><th>Train / test rows</th><th>Inner-selected candidate</th><th>Group overlap</th></tr></thead><tbody>{fold_rows}</tbody></table></div></section>
<section id="run-details" aria-labelledby="run-heading"><div class="split"><div><h2 id="run-heading">Reproduce this run</h2><p><strong>Protocol:</strong> {esc(m['protocol'])}<br><strong>Version:</strong> GroupBench {esc(m['groupbench_version'])}<br><strong>Selection metric:</strong> {esc(m['config']['selection_objective'])}<br><strong>Elapsed:</strong> {m['elapsed_seconds']:.2f} seconds</p><p><strong>Observed package-source digest</strong><br><code>{esc(package_source_display(m))}</code><br><small>Self-reported source bytes; not proof of loaded code, build, hardware, reproducibility, authenticity or safety. Full record in manifest.json; checksums do not authenticate this metadata.</small></p><p><strong>Data fingerprint</strong><br><code>{esc(m['data_sha256'])}</code></p><p><strong>Run key</strong><br><code>{esc(m['run_key'])}</code></p></div><div><h2>Scope &amp; limitations</h2><ul>{warnings}</ul></div></div></section>
<aside class="evidence" aria-labelledby="evidence-heading"><h2 id="evidence-heading">Keep the evidence close</h2><p>Start with the selected procedure, inspect difficult groups, then check the split audit before interpreting the comparison.</p><div class="evidence-links"><a href="summary.json"><strong>All metrics</strong><span>summary.json · includes every group</span></a><a href="manifest.json"><strong>Run record</strong><span>manifest.json · settings and splits</span></a><a href="predictions.csv"><strong>Row predictions</strong><span>predictions.csv · observations and predictions</span></a><a href="checksums.json"><strong>File checksums</strong><span>checksums.json · byte-integrity record</span></a></div><p class="muted">These files sit beside this report. Keep the bundle together; review group IDs, targets and predictions before sharing. Checksums record file bytes, not scientific validity.</p></aside>
<footer>Powered by the GroupBench engine. Generated locally at {esc(m['created_at_utc'])}. No hosted inference, remote assets, telemetry, or pretrained weights. Original MIT-licensed toolkit; dependencies retain their own licenses.</footer></main></body></html>'''


def write_report(result, output_dir, *, overwrite: bool = False) -> Path:
    """Write strict JSON, CSV and standalone HTML; refuse existing files by default."""
    payloads = {
        "summary.json": json.dumps(result.summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "manifest.json": json.dumps(result.manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "predictions.csv": result.predictions.to_csv(index=False, lineterminator="\n"),
        "report.html": render_html(result),
    }
    write_payloads(payloads, output_dir, overwrite=overwrite)
    return Path(output_dir) / "report.html"


def write_payloads(payloads: dict[str, str], output_dir, *, overwrite: bool = False):
    """Write a same-directory staged bundle with atomic per-file no-clobber."""
    if not payloads:
        raise ValueError("At least one output payload is required.")
    for name, content in payloads.items():
        if not isinstance(name, str) or name in ("", ".", "..", "checksums.json") or "/" in name or "\\" in name or Path(name).is_absolute():
            raise ValueError("Payload names must be plain basenames and cannot be checksums.json.")
        if not isinstance(content, str):
            raise TypeError("Output payloads must be UTF-8 text strings.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = tuple(payloads) + ("checksums.json",)
    if not overwrite and any((out / name).exists() or (out / name).is_symlink() for name in names):
        raise FileExistsError(f"Output files already exist in {out}; choose a new directory or pass overwrite=True.")
    payloads = dict(payloads)
    hashes = {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in payloads.items()}
    payloads["checksums.json"] = json.dumps({"algorithm": "sha256", "files": hashes}, indent=2) + "\n"
    with tempfile.TemporaryDirectory(prefix=".groupbench-", dir=out) as staging:
        for name, content in payloads.items():
            # Write the exact UTF-8 bytes used by the checksum calculation.
            # Text-mode default newlines may translate LF to CRLF on Windows.
            Path(staging, name).write_bytes(content.encode("utf-8"))
        for name in names:
            if overwrite:
                os.replace(Path(staging, name), out / name)
            else:
                # Atomic no-clobber creation: a racing writer or symlink must
                # never be replaced after the initial convenience check.
                os.link(Path(staging, name), out / name)
