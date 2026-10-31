"""Offline, self-contained reports. All input-derived HTML is escaped."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

from .package_source import package_source_display


def _text(english, traditional_chinese):
    """Pair trusted UI copy; supplied names and diagnostics stay verbatim."""
    return (f'<span class="copy-en" lang="en">{html.escape(english, quote=True)}</span>'
            f'<span class="copy-zh" lang="zh-Hant">{html.escape(traditional_chinese, quote=True)}</span>')


def _number(value):
    if value is None:
        return _text("undefined", "未定義")
    if 0 < abs(value) < 0.0001:
        return f"{value:.4e}"
    return f"{value:,.4f}"


def render_html(result) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    t = _text
    m = result.manifest
    models = result.summary["models"]
    primary = models["selected"]["overall"]
    ranking = sorted(models.items(), key=lambda pair: (pair[1]["overall"]["group_mae"], pair[0]))
    max_mae = max(value["overall"]["group_mae"] for _, value in ranking) or 1
    labels = {
        name: esc(name) + (' <small class="entry-label">' + t('Selection procedure', '模型選擇程序') + '</small>' if name == "selected" else "")
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
    groups = models["selected"]["per_group"]

    def group_row(row):
        return f'<tr><th scope="row">{esc(row["group"])}</th><td>{row["n_rows"]}</td><td>{_number(row["mae"])}</td><td>{_number(row["rmse"])}</td><td>{_number(row["bias"])}</td></tr>'

    worst = sorted(groups, key=lambda row: -row["mae"])[:15]
    group_rows = "".join(group_row(row) for row in worst)
    all_groups = ''
    if len(groups) > 15:
        all_rows = ''.join(group_row(row) for row in sorted(groups, key=lambda row: str(row['group'])))
        view_all = t(f'View all {len(groups)} groups', f'查看全部 {len(groups)} 個群組')
        all_groups = f'''<details class="all-groups"><summary>{view_all}</summary><div class="table-wrap" role="region" aria-labelledby="all-groups-table" tabindex="0"><table><caption id="all-groups-table">{t('Selected procedure: all groups (ID order)', '模型選擇程序：全部群組（識別碼順序）')}</caption><thead><tr><th scope="col">{t('Held-out group', '保留群組')}</th><th scope="col">{t('Rows', '列數')}</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">{t('Bias', '偏差')}</th></tr></thead><tbody>{all_rows}</tbody></table></div></details>'''
    fold_rows = "".join(
        f'<tr><th scope="row">{fold["fold"]}</th><td>{fold["train_group_count"]} / {fold["test_group_count"]}</td><td>{fold["train_rows"]} / {fold["test_rows"]}</td><td>{esc(fold["selected_candidate"])}</td><td>{fold["overlap_count"]}</td></tr>' for fold in m["outer_folds"]
    )
    warnings = "".join(f"<li>{esc(item)}</li>" for item in m["limitations"])
    source_display = package_source_display(m)
    source_text = (t('not recorded', '未記錄') if source_display == 'not recorded' else
                   t('unavailable', '無法取得') if source_display == 'unavailable' else esc(source_display))
    singleton_note = (
        '<p>' + t('Recorded rows = groups: every supplied group has one row. There are no repeated-observation groups for this split to protect; check that related rows have not been assigned different IDs.',
                  '記錄列數等於群組數：每個提供的群組都只有一列。此切分沒有可保護的重複觀測群；請檢查相關列是否被指定為不同識別碼。') + '</p>'
        if type(m['n_rows']) is int and type(m['n_groups']) is int
        and m['n_rows'] > 0 and m['n_rows'] == m['n_groups'] else ''
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Axiom Clotho | {esc(m['config']['target'])}</title>
<style>
:root{{--ink:#172b3a;--muted:#526778;--paper:#f4f7f8;--line:#dce5e9;--accent:#087f75;--mint:#bcf2dc}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.6 system-ui,-apple-system,sans-serif}}header{{background:#102e3b;color:white;padding:44px max(24px,calc((100vw - 1080px)/2));}}header p{{max-width:720px;color:#c3d7e0;margin:10px 0}}h1{{font-size:40px;letter-spacing:-1.5px;margin:8px 0}}h2{{font-size:23px;margin:0 0 14px}}h3{{font-size:16px;margin:0 0 8px}}.eyebrow{{color:var(--mint);text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:750}}main{{max-width:1128px;margin:auto;padding:28px 24px 48px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.card,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:20px}}.card{{margin:0}}.value{{display:block;font-size:34px;font-weight:750;letter-spacing:-1px}}.muted,small{{color:var(--muted)}}.note{{border-left:4px solid var(--accent);background:#edf9f4;padding:14px 18px;margin:20px 0}}section p{{max-width:920px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;text-align:right;white-space:nowrap}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line)}}thead th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}}th:first-child{{text-align:left}}tbody th{{font-weight:600}}.bar-row{{display:grid;grid-template-columns:110px 1fr 90px;align-items:center;gap:14px;margin:12px 0;font-size:13px}}.track{{background:#eef3f4;height:20px;border-radius:4px;overflow:hidden}}.bar{{height:100%;background:#9eb3be;border-radius:4px}}.bar.primary{{background:var(--accent)}}.bar-row b{{text-align:right}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{word-break:break-all;font-size:12px}}ul{{padding-left:20px}}footer{{color:var(--muted);font-size:12px}}@media(max-width:700px){{.grid,.split{{grid-template-columns:1fr}}h1{{font-size:32px}}header{{padding:30px 24px}}.bar-row{{grid-template-columns:90px 1fr 70px;gap:8px}}section,.card{{padding:18px}}}}@media print{{header{{color:#172b3a;background:white}}header p,.eyebrow{{color:#526778}}section,.card{{break-inside:avoid}}body{{background:white}}}}

/* Offline reading path: no scripts, remote assets, or color-only state. */
a{{color:#075e57;text-underline-offset:3px}}a:focus-visible,summary:focus-visible,[tabindex]:focus-visible{{outline:3px solid #925400;outline-offset:4px}}.skip-link{{position:absolute;left:16px;top:-100px;background:white;padding:12px 18px;z-index:2}}.skip-link:focus{{top:12px}}.report-nav{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}}.report-nav a{{display:flex;gap:10px;align-items:center;padding:10px 14px;border:1px solid #bcced3;border-radius:8px;background:white;text-decoration:none;font-size:12px;font-variant-numeric:tabular-nums}}.report-nav span{{color:var(--ink);font-size:14px;font-weight:650}}.report-nav a:hover,.evidence-links a:hover{{background:#edf9f4}}section{{scroll-margin-top:20px}}caption{{text-align:left;color:var(--ink);font-size:13px;font-weight:650;padding:10px 12px;background:#f4f7f8;border-bottom:1px solid var(--line)}}.table-wrap{{border:1px solid var(--line);border-radius:8px}}.entry-label{{display:block;font-size:10px;line-height:1.3;color:#075e57;font-weight:650}}.bar.primary{{border:2px solid #064a44}}.evidence{{padding:24px;background:#eaf1f3;border:1px solid #cbdadf;border-radius:12px;margin-bottom:24px}}.evidence p{{max-width:850px}}.evidence-links{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.evidence-links a{{display:block;background:white;border:1px solid #bcced3;border-radius:8px;padding:14px 16px;text-decoration:none}}.evidence-links strong{{text-decoration:underline}}.evidence-links span{{display:block;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}@media(max-width:700px){{.evidence-links{{grid-template-columns:1fr}}.report-nav a{{flex:1 1 140px}}.evidence{{padding:18px}}}}@media print{{.skip-link,.report-nav{{display:none}}.evidence{{background:white;break-inside:avoid}}}}
.all-groups{{margin-top:18px}}.all-groups summary{{cursor:pointer;font-weight:650;padding:10px 0}}
.language-title{{margin:16px max(24px,calc((100vw - 1080px)/2)) 4px;font-size:12px;color:var(--muted)}}.language-choice{{accent-color:var(--accent);width:18px;height:18px;vertical-align:middle}}#report-en{{margin-left:max(24px,calc((100vw - 1080px)/2))}}.language-label{{display:inline-block;padding:8px 16px 16px 4px;cursor:pointer}}.report-content .copy-zh{{display:none}}#report-zh:checked ~ .report-content .copy-en{{display:none}}#report-zh:checked ~ .report-content .copy-zh{{display:inline}}input:focus-visible{{outline:3px solid #925400;outline-offset:4px}}@media print{{.language-title,.language-choice,.language-label{{display:none}}}}

/* Axiom Clotho / Moirai: offline editorial identity; data and controls unchanged. */
:root{{--ink:#101a22;--muted:#6d675a;--paper:#f4efe3;--panel:#fbf8f0;--line:#d4c7ac;--accent:#8d6b2d;--accent-text:#7e5c22;--mint:#8d6b2d;--wash:#ebe2cf;--header:#f4efe3}}
body{{background:var(--paper);color:var(--ink);font-size:16px;line-height:1.65}}
header{{position:relative;background:var(--header);color:var(--ink);border-top:1px solid var(--line);border-bottom:3px double var(--accent);padding:38px max(24px,calc((100vw - 1080px)/2)) 34px}}
header p{{color:var(--muted);max-width:780px;overflow-wrap:anywhere}}
h1,h2,h3{{font-family:Georgia,'DejaVu Serif',serif;font-weight:400;line-height:1.18;overflow-wrap:anywhere}}
h1{{font-size:clamp(34px,5vw,52px);letter-spacing:-.035em;margin:12px 0 16px}}
h2{{font-size:27px;margin-bottom:18px}}h3{{font-size:19px}}
.eyebrow{{color:var(--accent-text);font-size:11px;letter-spacing:.16em;line-height:1.6}}
.clotho-mark{{width:38px;height:38px;vertical-align:middle;margin:0 14px 3px 0;overflow:visible}}
.clotho-mark .spindle{{fill:var(--ink)}}.clotho-mark .thread{{stroke:var(--accent)}}
main{{max-width:1128px;padding-top:32px}}.grid,.cards{{gap:18px;grid-template-columns:repeat(3,minmax(0,1fr))}}.split{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid>*,.cards>*,.split>*{{min-width:0;overflow-wrap:anywhere}}
.card,section,.evidence{{background:var(--panel);border:1px solid var(--line);border-radius:0;padding:26px;margin-bottom:24px;min-width:0}}
.card{{margin-bottom:0;border-top:2px solid var(--accent)}}
.cards{{border-top:2px solid var(--accent)}}
.value,strong.value{{font-family:Georgia,'DejaVu Serif',serif;font-weight:400;font-size:36px;line-height:1.25;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}}
.muted,small,.language-title,footer{{color:var(--muted)}}
a,.entry-label{{color:var(--accent-text)}}a{{text-underline-offset:4px}}
a:focus-visible,input:focus-visible,summary:focus-visible,[tabindex]:focus-visible{{outline:3px solid var(--accent);outline-offset:4px}}
.note{{background:var(--wash);border-left:3px solid var(--accent);padding:16px 20px}}
.report-nav{{gap:8px;margin-bottom:28px}}.report-nav a,.evidence-links a{{background:var(--panel);color:var(--accent-text);border:1px solid var(--line);border-radius:0}}
.report-nav a{{padding:11px 14px}}.report-nav span{{color:var(--ink)}}
.report-nav a:hover,.evidence-links a:hover{{background:var(--wash)}}
.evidence{{background:var(--wash);border-top:2px solid var(--accent)}}
.evidence-links span{{color:var(--muted)}}
.table-wrap,.scroll{{border:1px solid var(--line);border-radius:0;max-width:100%}}
caption{{background:var(--wash);color:var(--ink)}}th,td{{border-bottom-color:var(--line)}}thead th{{color:var(--muted)}}
.track{{background:#eef3f4;border-radius:4px}}.bar{{background:#9eb3be;border-radius:4px}}.bar.primary{{background:#087f75;border:2px solid #064a44}}
.bar-row{{grid-template-columns:minmax(90px,140px) minmax(30px,1fr) minmax(70px,100px)}}.bar-row>span{{overflow-wrap:anywhere}}
code,.diagnostics,header p{{overflow-wrap:anywhere}}.language-choice{{accent-color:var(--accent)}}.skip-link{{background:var(--panel)}}
footer{{padding-top:10px;border-top:1px solid var(--line);line-height:1.7}}
@media(prefers-color-scheme:dark){{:root{{--ink:#f4efe3;--muted:#b8ad97;--paper:#101a22;--panel:#15232e;--line:#43525c;--accent:#d7bd82;--accent-text:#d7bd82;--mint:#d7bd82;--wash:#1b2b37;--header:#101a22}}}}
@media(max-width:700px){{header{{padding:28px 20px}}main{{padding:24px 16px 36px}}.grid,.cards,.split,.evidence-links{{grid-template-columns:minmax(0,1fr)}}section,.card,.evidence{{padding:20px 16px}}.report-nav a{{flex:1 1 130px}}h2{{font-size:24px}}.bar-row{{grid-template-columns:80px minmax(24px,1fr) 76px;gap:8px}}.clotho-mark{{width:30px;height:30px;margin-right:9px}}}}
@media print{{:root{{--ink:#101a22;--muted:#6d675a;--paper:#FFFFFF;--panel:#FFFFFF;--line:#d4c7ac;--accent:#8d6b2d;--accent-text:#7e5c22;--wash:#FFFFFF;--header:#FFFFFF}}header,body,section,.card,.evidence{{background:#FFFFFF;color:#101a22}}header p,.eyebrow{{color:#6d675a}}.skip-link,.report-nav,.language-title,.language-choice,.language-label{{display:none}}section,.card,.evidence{{break-inside:avoid}}a{{color:#101a22}}}}
</style></head><body>
<p class="language-title" id="language-help">Report language / 報告語言</p>
<input class="language-choice" type="radio" name="report-language" id="report-en" value="en" aria-describedby="language-help" checked><label class="language-label" for="report-en" lang="en">English</label>
<input class="language-choice" type="radio" name="report-language" id="report-zh" value="zh-Hant" aria-describedby="language-help"><label class="language-label" for="report-zh" lang="zh-Hant">繁體中文</label>
<div class="report-content">
<a class="skip-link" href="#overview">{t('Skip to report', '跳至報告')}</a>
<header><div class="eyebrow">{t('Local-first ML · group-aware evaluation', '本機優先 ML · 群組感知評估')}</div><h1><svg class="clotho-mark" viewBox="0 0 32 32" aria-hidden="true" focusable="false"><g transform="scale(.0625)"><g id="report-clotho-clotho-fate-instrument"><path d="M256 44 Q259 60 275 63 Q259 66 256 82 Q253 66 237 63 Q253 60 256 44Z" fill="var(--accent)" /><path d="M256 88V111" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" /><path d="M256 116V424" fill="none" stroke="var(--accent)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" /><path id="report-clotho-spindle-body" d="M256 126 C248 188 231 220 231 260 C231 300 248 360 256 405 C264 360 281 300 281 260 C281 220 264 188 256 126Z" fill="var(--ink)" /><path d="M256 188V361" fill="none" stroke="var(--accent)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" /><path id="report-clotho-unspun-thread" d="M186 131 C119 151 118 207 168 235 C212 259 294 249 331 282 C363 312 350 358 307 379 C353 372 388 328 365 288 C338 243 247 241 193 216 C148 194 148 163 186 131Z" fill="var(--accent)" /><path id="report-clotho-woven-return" d="M326 132 C371 149 384 179 365 205 C339 241 278 260 224 285 C178 307 155 341 168 378 C122 334 154 284 208 258 C264 232 322 219 342 193 C359 171 345 149 326 132Z" fill="var(--ink)" /><path id="report-clotho-thread-release" d="M168 378 C191 408 229 420 256 421 C303 424 343 402 363 371 C356 403 321 438 270 441 C226 444 189 422 168 378Z" fill="var(--accent)" /><path d="M158 180C155 201 184 218 216 226" fill="none" stroke="var(--header)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" /><path d="M343 304C354 329 336 350 319 360" fill="none" stroke="var(--header)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" /><path d="M355 170C357 188 335 207 314 217" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" /><path d="M256 441L264.4 455 256 469 247.6 455Z" fill="var(--ink)" /><path d="M121 252 Q124 259 131 262 Q124 265 121 272 Q118 265 111 262 Q118 259 121 252Z" fill="var(--accent)" /><path d="M392 243 Q395 248 400 251 Q395 254 392 259 Q389 254 384 251 Q389 248 392 243Z" fill="var(--accent)" /></g></g></svg>Axiom Clotho</h1><p>{t('How does the selection procedure perform on held-out groups?', '模型選擇程序在保留群組上的表現如何？')}</p><p>{t('Nested, group-disjoint evaluation of ', '以巢狀、群組互斥方式評估 ')}<strong>{esc(m['config']['target'])}</strong>{t(', holding out complete ', '，完整保留 ')}<strong>{esc(m['config']['group'])}</strong>{t(' groups.', ' 群組。')}</p></header>
<main id="overview" tabindex="-1"><nav class="report-nav" aria-label="Report sections / 報告章節"><a href="#overview">01 <span>{t('Overview', '總覽')}</span></a><a href="#comparison">02 <span>{t('Comparison', '比較')}</span></a><a href="#difficult-groups">03 <span>{t('Difficult groups', '困難群組')}</span></a><a href="#split-audit">04 <span>{t('Split audit', '切分稽核')}</span></a><a href="#run-details">05 <span>{t('Run details', '執行詳情')}</span></a></nav><div class="grid"><div class="card"><small>{t('Selected pipeline · group-balanced MAE', '模型選擇程序 · 群組等權 MAE')}</small><span class="value">{_number(primary['group_mae'])}</span><small>{t('Lower is better · input target scale', '越低越好 · 輸入目標尺度')}</small></div><div class="card"><small>{t('Held-out observations', '保留觀測列數')}</small><span class="value">{m['n_rows']:,}</span><small>{m['n_groups']}{t(' distinct groups · one prediction per row', ' 個不同群組 · 每列一個預測')}</small></div><div class="card"><small>{t('Outer / inner folds', '外層 / 內層折數')}</small><span class="value">{m['config']['outer_splits']} / {m['config']['inner_splits']}</span><small>{t('Single-thread fits · seed ', '單執行緒擬合 · 隨機種子：')}{m['config']['random_state']}</small></div></div>
<div class="note"><strong>{t('Read this correctly:', '如何正確解讀：')}</strong> {t('“selected” is the full selection procedure, chosen only on inner folds. The outer leaderboard is for evaluation, not another tuning set. This report does not establish performance on real materials when run on the synthetic fixture.', 'selected 是完整的模型選擇程序，僅在內層折中選擇。外層排行榜用於評估，不是另一個調參資料集。使用合成範例產生的報告不能證明真實材料上的表現。')}</div>
<section id="comparison" aria-labelledby="comparison-heading"><h2 id="comparison-heading">{t('Baseline comparison', '基準模型比較')}</h2>
<div class="note" aria-labelledby="split-scope-heading"><h3 id="split-scope-heading">{t('What this split can answer', '此切分可回答的問題')}</h3><p>{t('Whole-group holdout on: ', '完整群組保留欄位：')}<code>{esc(m['config']['group'])}</code>. {t('Zero group-ID overlap checks only the supplied IDs, not their physical independence.', '群組識別碼零重疊僅檢查提供的識別碼，不驗證其實際獨立性。')}</p><ul><li><strong>{t('Domain-heldout?', '是否為跨領域保留？')}</strong> {t('Separation is only along the supplied group column; other domain/site axes are not separately audited. If that column defines the intended domain boundary, its whole values are held out, but the scientific meaning of the IDs is not verified.', '僅沿指定的群組欄位隔離，未另稽核其他領域／場域軸。若該欄位定義預定的領域邊界，其完整值會被保留，但識別碼的科學意義並未經驗證。')}</li><li><strong>{t('Random-row comparison?', '是否比較隨機逐列切分？')}</strong> {t('Not run. The seed assigns whole groups to folds; it does not split rows within a group.', '未執行。種子將完整群組分配至各折，不會拆開同一群組內的列。')}</li></ul><p>{t('Different group IDs may still share a site, time period or another dependence axis. Review the grouping boundary before interpreting scores.', '不同群組識別碼仍可能共享場域、時段或其他相依軸。解讀分數前，請先確認分組邊界。')}</p>{singleton_note}</div>
<div class="note" aria-labelledby="target-scale-heading"><h3 id="target-scale-heading">{t('Target scale is not verified', '目標尺度未經核實')}</h3><p>{t('Errors are computed from the supplied target numbers; this report does not establish their physical units or whether they are reported, normalized or transformed values. No automatic target-unit conversion or inverse transformation is performed.', '誤差依提供的目標數值計算；本報告不確認其物理單位，或它們是原報告值、正規化值還是轉換後的值。不會自動換算目標單位或反轉換。')}</p><p>{t('Before pooling rows or comparing errors across runs, confirm the same physical endpoint and a common scale (for example, ratio versus percent). Readable data, group separation and consistent saved metrics do not by themselves establish physical comparability or model readiness.', '合併資料列或跨執行比較誤差前，請確認物理終點相同且尺度一致（例如比值與百分比）。資料可讀、群組分離及保存指標一致，本身不代表物理上可比較或模型已適用。')}</p></div>
<p class="muted">{t('Group-balanced MAE gives each group equal weight, regardless of its number of rows. Row MAE and RMSE pool all held-out observations. No confidence intervals are implied. Nonzero values below 0.0001 in magnitude use scientific notation.', '群組等權 MAE 給各群組相同權重，不受組內列數影響。逐列 MAE 與 RMSE 合併所有保留觀測計算。這不表示提供信賴區間。絕對值小於 0.0001 的非零數值以科學記號顯示。')}</p>{bars}<div class="table-wrap" role="region" aria-labelledby="comparison-table" tabindex="0"><table><caption id="comparison-table">{t('Held-out metrics for each entry', '各項目的保留集指標')}</caption><thead><tr><th>{t('Entry', '項目')}</th><th>{t('Group MAE', '群組 MAE')}</th><th>{t('Row MAE', '逐列 MAE')}</th><th>{t('Row RMSE', '逐列 RMSE')}</th><th>{t('Pooled R²', '合併 R²')}</th><th>{t('Worst-group MAE', '最差群組 MAE')}</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section id="difficult-groups" aria-labelledby="groups-heading"><h2 id="groups-heading">{t('Where it struggles', '表現較差的群組')}</h2><p class="muted">{t('Up to 15 groups with the largest selected-pipeline MAE. Bias is prediction minus observation; all groups are available in summary.json.', '顯示模型選擇程序 MAE 最高的至多 15 組。偏差為預測值減去觀測值；所有群組均保留於 summary.json。')}</p><div class="table-wrap" role="region" aria-labelledby="groups-table" tabindex="0"><table><caption id="groups-table">{t('Selected procedure: highest-error groups', '模型選擇程序：誤差最高的群組')}</caption><thead><tr><th>{t('Held-out group', '保留群組')}</th><th>{t('Rows', '列數')}</th><th>MAE</th><th>RMSE</th><th>{t('Bias', '偏差')}</th></tr></thead><tbody>{group_rows}</tbody></table></div>{all_groups}</section>
<section id="split-audit" aria-labelledby="audit-heading"><h2 id="audit-heading">{t('Split audit', '切分稽核')}</h2><p class="muted">{t('No group may occur in both training and validation within a fold. Inner selections and exact group assignments are recorded in manifest.json.', '同一折的訓練集與驗證集不得包含相同群組。內層選擇與完整群組分配記錄於 manifest.json。')}</p><div class="table-wrap" role="region" aria-labelledby="audit-table" tabindex="0"><table><caption id="audit-table">{t('Training and test split counts', '訓練與測試切分數量')}</caption><thead><tr><th>{t('Outer fold', '外層折次')}</th><th>{t('Train / test groups', '訓練 / 測試群組')}</th><th>{t('Train / test rows', '訓練 / 測試列數')}</th><th>{t('Inner-selected candidate', '內層選定候選模型')}</th><th>{t('Group overlap', '重疊群組')}</th></tr></thead><tbody>{fold_rows}</tbody></table></div></section>
<section id="run-details" aria-labelledby="run-heading"><div class="split"><div><h2 id="run-heading">{t('Reproduce this run', '重做此執行')}</h2><p><strong>{t('Protocol:', '流程：')}</strong> {esc(m['protocol'])}<br><strong>{t('Version:', '版本：')}</strong> GroupBench {esc(m['groupbench_version'])}<br><strong>{t('Selection metric:', '選擇指標：')}</strong> {esc(m['config']['selection_objective'])}<br><strong>{t('Elapsed:', '耗時：')}</strong> {m['elapsed_seconds']:.2f} {t('seconds', '秒')}</p><p><strong>{t('Observed package-source digest', '觀察到的套件原始碼摘要')}</strong><br><code>{source_text}</code><br><small>{t('Self-reported source bytes; not proof of loaded code, build, hardware, reproducibility, authenticity or safety. Full record in manifest.json; checksums do not authenticate this metadata.', '僅為自行記錄的原始碼位元組；不證明載入的程式碼、建置、硬體、可重現性、真實性或安全性。完整記錄見 manifest.json；檢查碼不認證此中繼資料。')}</small></p><p><strong>{t('Data fingerprint', '資料指紋')}</strong><br><code>{esc(m['data_sha256'])}</code></p><p><strong>{t('Run key', '執行識別碼')}</strong><br><code>{esc(m['run_key'])}</code></p></div><div><h2>{t('Scope & limitations', '範圍與限制')}</h2><p class="muted">{t('Names and recorded diagnostics are kept as supplied, not translated.', '名稱與記錄診斷保留原文，不進行翻譯。')}</p><ul class="diagnostics">{warnings}</ul></div></div></section>
<aside class="evidence" aria-labelledby="evidence-heading"><h2 id="evidence-heading">{t('Keep the evidence close', '保留完整證據')}</h2><p>{t('Start with the selected procedure, inspect difficult groups, then check the split audit before interpreting the comparison.', '先查看模型選擇程序與表現較差的群組，再檢查切分記錄，之後才解讀模型比較。')}</p><div class="evidence-links"><a href="summary.json"><strong>{t('All metrics', '所有指標')}</strong><span>{t('summary.json · includes every group', 'summary.json · 包含每個群組')}</span></a><a href="manifest.json"><strong>{t('Run record', '執行記錄')}</strong><span>{t('manifest.json · settings and splits', 'manifest.json · 設定與切分')}</span></a><a href="predictions.csv"><strong>{t('Row predictions', '逐列預測')}</strong><span>{t('predictions.csv · observations and predictions', 'predictions.csv · 觀測與預測')}</span></a><a href="checksums.json"><strong>{t('File checksums', '檔案檢查碼')}</strong><span>{t('checksums.json · byte-integrity record', 'checksums.json · 位元組完整性記錄')}</span></a></div><p class="muted">{t('These files sit beside this report. Keep the bundle together; review group IDs, targets and predictions before sharing. Checksums record file bytes, not scientific validity.', '這些檔案位於報告旁。請保留完整檔案組；分享前先檢查群組識別碼、目標與預測。檢查碼記錄檔案位元組，不代表科學有效性。')}</p></aside>
<footer>{t('Powered by the GroupBench engine. Generated locally at ', '由 GroupBench 引擎產生。本機產生時間：')}{esc(m['created_at_utc'])}{t('. No hosted inference, remote assets, telemetry, or pretrained weights. Original MIT-licensed toolkit; dependencies retain their own licenses.', '。無託管推論、遠端資源、遙測或預訓練權重。原創工具採 MIT 授權；依賴保留各自授權。')} {t('Language controls change report copy only, not saved data or the CLI language.', '語言控制僅改變報告文字，不改變已儲存資料或命令列語言。')}</footer></main></div></body></html>'''


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
