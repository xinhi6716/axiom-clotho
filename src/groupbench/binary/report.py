"""Fresh-only binary bundles, using the unchanged generic safe text writer."""
from __future__ import annotations
import html
import json
from pathlib import Path

from ..package_source import package_source_display

KNOWN_OUTPUTS = ("report.html", "summary.json", "manifest.json", "predictions.csv", "checksums.json")
MAX_BINARY_PAYLOAD_BYTES = 64 * 1024 * 1024


def ensure_fresh_binary_output(output_dir):
    out = Path(output_dir)
    # Reject known impossible destinations before reading data or fitting.
    # This is not a reservation: write-time checks still handle races and permissions.
    for component in (out, *out.parents):
        if (component.exists() or component.is_symlink()) and not component.is_dir():
            raise ValueError("Binary output must be a local directory with directory parents, "
                             "not a file or broken link; choose a fresh destination.")
    if any((out / name).exists() or (out / name).is_symlink() for name in KNOWN_OUTPUTS):
        raise FileExistsError("Binary output already contains a known bundle name; choose a fresh directory. Binary overwrite is not supported.")


def _text(english, traditional_chinese):
    """Pair trusted UI copy; caller-provided names/diagnostics stay verbatim."""
    return (f'<span class="copy-en" lang="en">{html.escape(english, quote=True)}</span>'
            f'<span class="copy-zh" lang="zh-Hant">{html.escape(traditional_chinese, quote=True)}</span>')


def _number(value):
    return _text("undefined", "未定義") if value is None else f"{value:.6g}"


def _rate(value, label):
    if value is not None:
        return _number(value)
    return _text(f"undefined: no label {label}", f"未定義：沒有標籤 {label}")


def render_binary_html(result):
    esc = lambda value: html.escape(str(value), quote=True)
    t = _text
    models, m = result.summary["models"], result.manifest
    selected = models["selected"]["overall"]
    rows = []
    for name, metrics in sorted(models.items(), key=lambda item: (item[1]["overall"]["group_brier"], item[0])):
        s = metrics["overall"]
        entry = esc(name)
        if name == "selected":
            entry += '<small class="entry-label">' + t("Selection procedure", "模型選擇程序") + '</small>'
        rows.append(f'<tr><th scope="row">{entry}</th><td>{_number(s["group_brier"])}</td><td>{_number(s["brier"])}</td><td>{_number(s["group_log_loss_clipped"])}</td><td>{_number(s["accuracy"])}</td><td>{s["confusion"]["fn"]}</td><td>{s["confusion"]["fp"]}</td></tr>')
    worst = []
    for row in sorted(models["selected"]["per_group"], key=lambda r: -r["brier"])[:15]:
        worst.append(f'<tr><th scope="row">{esc(row["group"])}</th><td>{row["n_rows"]}</td><td>{row["n_true_0"]} / {row["n_true_1"]}</td><td>{_number(row["brier"])}</td><td>{_number(row["log_loss_clipped"])}</td><td>{row["confusion"]["fn"]} / {row["confusion"]["fp"]}</td><td>{_rate(row["recall_1"], 1)}</td><td>{_rate(row["specificity_0"], 0)}</td></tr>')
    adverse = result.summary["adverse_comparison_to_dummy"]
    if adverse["available"]:
        comparison = (t("Selected minus Dummy group Brier: ", "Selected 減去 Dummy 的群組 Brier：")
                      + f'<strong>{_number(adverse["group_brier_delta"])}</strong> '
                      + t("(positive is worse). Groups: ", "（正值表示較差）。群組數：")
                      + f'{adverse["groups_better"]} ' + t("better, ", "較好、")
                      + f'{adverse["groups_tied"]} ' + t("tied, ", "相同、")
                      + f'{adverse["groups_worse"]} ' + t("worse. Descriptive, not significance.", "較差。僅為描述性比較，不代表統計顯著。"))
    else:
        comparison = t("Comparison unavailable; recorded reason: ", "無法比較；記錄原因：") + esc(adverse["reason"])
    audits = "".join(f'<tr><th scope="row">{f["fold"]}</th><td>{f["train_group_count"]} / {f["test_group_count"]}</td><td>{esc(f["selected_candidate"])}</td><td>{f["overlap_count"]}</td></tr>' for f in m["outer_folds"])
    warnings = "".join(f"<li>{esc(item)}</li>" for item in m["limitations"])
    source_display = package_source_display(m)
    source_text = (t("not recorded", "未記錄") if source_display == "not recorded" else
                   t("unavailable", "無法取得") if source_display == "unavailable" else esc(source_display))
    singleton_note = (
        '<p>' + t('Recorded rows = groups: every supplied group has one row. There are no repeated-observation groups for this split to protect; check that related rows have not been assigned different IDs.',
                  '記錄列數等於群組數：每個提供的群組都只有一列。此切分沒有可保護的重複觀測群；請檢查相關列是否被指定為不同識別碼。') + '</p>'
        if type(m['n_rows']) is int and type(m['n_groups']) is int
        and m['n_rows'] > 0 and m['n_rows'] == m['n_groups'] else ''
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>ML Super Calculator | Binary {esc(m['config']['target'])}</title>
<style>
*{{box-sizing:border-box}}body{{font:15px/1.6 system-ui,sans-serif;background:#f3f7f8;color:#183240;margin:0}}header{{background:#123744;color:white;padding:36px max(24px,calc((100vw - 1052px)/2))}}main{{max-width:1100px;margin:auto;padding:24px}}h1{{margin:8px 0;font-size:36px;letter-spacing:-1px}}h2{{margin:0 0 12px;font-size:23px}}p{{max-width:920px}}section,.evidence{{background:white;padding:24px;border:1px solid #d9e4e7;border-radius:12px;margin-bottom:20px;scroll-margin-top:20px;min-width:0}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}}.cards div{{min-width:0}}strong.value{{display:block;font-size:32px}}.scroll{{overflow:auto;border:1px solid #d9e4e7;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:right;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #d9e4e7}}th:first-child{{text-align:left}}caption{{padding:10px 12px;text-align:left;background:#f3f7f8;font-weight:650}}thead th{{color:#526b76}}.note{{border-left:4px solid #087f75;padding:14px 18px;background:#edf9f4}}small,.muted{{color:#526b76}}code{{overflow-wrap:anywhere}}header p,th:first-child,.diagnostics{{overflow-wrap:anywhere}}.entry-label{{display:block;color:#075e57;font-size:11px}}.eyebrow{{color:#bfeada;text-transform:uppercase;font-size:12px;letter-spacing:1px}}a{{color:#075e57;text-underline-offset:3px}}a:focus-visible,input:focus-visible,[tabindex]:focus-visible{{outline:3px solid #925400;outline-offset:4px}}.skip-link{{position:absolute;top:-100px;left:16px;background:white;padding:12px;z-index:2}}.skip-link:focus{{top:12px}}.report-nav{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}}.report-nav a{{border:1px solid #bcced3;background:white;border-radius:8px;padding:10px 14px;text-decoration:none;font-weight:650}}.report-nav a:hover,.evidence-links a:hover{{background:#edf9f4}}.evidence{{background:#eaf1f3}}.evidence-links{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.evidence-links a{{display:block;background:white;padding:14px;border:1px solid #bcced3;border-radius:8px;text-decoration:none;overflow-wrap:anywhere}}.evidence-links strong{{text-decoration:underline}}.evidence-links small{{display:block}}.language-title{{margin:16px max(24px,calc((100vw - 1052px)/2)) 4px;font-size:12px;color:#526b76}}.language-choice{{accent-color:#087f75;width:18px;height:18px;vertical-align:middle}}#report-en{{margin-left:max(24px,calc((100vw - 1052px)/2))}}.language-label{{display:inline-block;padding:8px 16px 16px 4px;cursor:pointer}}.copy-zh{{display:none}}#report-zh:checked ~ .report-content .copy-en{{display:none}}#report-zh:checked ~ .report-content .copy-zh{{display:inline}}footer{{font-size:12px;color:#526b76}}@media(max-width:700px){{.cards,.evidence-links{{grid-template-columns:1fr}}h1{{font-size:30px}}section,.evidence{{padding:18px}}.report-nav a{{flex:1 1 130px}}}}@media print{{.skip-link,.report-nav,.language-title,.language-choice,.language-label{{display:none}}section,.evidence{{break-inside:avoid}}header{{background:white;color:#183240}}.eyebrow{{color:#526b76}}body{{background:white}}}}
</style></head><body>
<p class="language-title" id="language-help">Report language / 報告語言</p>
<input class="language-choice" type="radio" name="report-language" id="report-en" value="en" aria-describedby="language-help" checked><label class="language-label" for="report-en" lang="en">English</label>
<input class="language-choice" type="radio" name="report-language" id="report-zh" value="zh-Hant" aria-describedby="language-help"><label class="language-label" for="report-zh" lang="zh-Hant">繁體中文</label>
<div class="report-content"><a class="skip-link" href="#overview">{t('Skip to report', '跳至報告')}</a>
<header><div class="eyebrow">{t('Local · explicit binary 0/1 · no hosted API', '本機執行 · 明確二元 0/1 分類 · 無託管 API')}</div><h1>ML Super Calculator</h1><p>{t('Held-out evaluation · not probability calibration', '保留群組評估 · 並非機率校準')}</p><p>{t('Target: ', '目標：')}<strong>{esc(m['config']['target'])}</strong> · {t('Whole groups held out: ', '完整保留群組：')}<strong>{esc(m['config']['group'])}</strong></p></header>
<main id="overview" tabindex="-1"><nav class="report-nav" aria-label="Report sections / 報告章節"><a href="#overview">01 {t('Overview', '總覽')}</a><a href="#comparison">02 {t('Comparison', '比較')}</a><a href="#difficult-groups">03 {t('Difficult groups', '困難群組')}</a><a href="#split-audit">04 {t('Split audit', '切分稽核')}</a><a href="#run-details">05 {t('Run details', '執行詳情')}</a></nav>
<section class="cards" aria-label="Evaluation overview / 評估總覽"><div><small>{t('Selected procedure · group Brier', '模型選擇程序 · 群組 Brier')}</small><strong class="value">{_number(selected['group_brier'])}</strong><small>{t('Lower is better · binary [0,1] scale', '越低越好 · 二元 [0,1] 尺度')}</small></div><div><small>{t('Held-out rows / groups', '保留觀測列數 / 群組數')}</small><strong class="value">{m['n_rows']} / {m['n_groups']}</strong><small>{t('One held-out probability pair per entry per row', '每個項目對每列產生一組保留集機率')}</small></div><div><small>{t('Observed label 0 / label 1', '實際標籤 0 / 標籤 1')}</small><strong class="value">{selected['n_true_0']} / {selected['n_true_1']}</strong><small>{t('Meaning of label 1 is user-defined', '標籤 1 的意義由使用者定義')}</small></div></section>
<section aria-labelledby="reading-heading"><h2 id="reading-heading">{t('Read this correctly', '如何正確解讀')}</h2><p class="note">{t('The primary entry is selected, chosen only inside training folds. The display ranking is not another tuning step. Probabilities are not established calibrated failure risks. A fixed p1 ≥ 0.5 predicts label 1, including ties; this is not an optimized process decision.', '主要項目為 selected，僅在訓練折內選擇。顯示排名不是另一輪調參。這些機率尚未證實為已校準的失敗風險。固定以 p1 ≥ 0.5 預測標籤 1，包含等於 0.5 的情況；這不是最佳化的製程決策。')}</p><p>{t('Group Brier averages row squared probability error within each group, then weights groups equally. Models train with equal row weights. Natural-log loss is secondary and clips a metric-only copy at float64 epsilon; raw probabilities and clipping counts remain in CSV/JSON.', '群組 Brier 先平均每組內各列的機率平方誤差，再給各群組相同權重。模型訓練時各列權重相同。自然對數損失為次要指標，僅將計算指標的副本截限於 float64 epsilon；原始機率與截限次數保留於 CSV/JSON。')}</p></section>
<section id="comparison" aria-labelledby="comparison-heading"><h2 id="comparison-heading">{t('All baseline results', '所有基準結果')}</h2>
<div class="note" aria-labelledby="split-scope-heading"><h3 id="split-scope-heading">{t('What this split can answer', '此切分可回答的問題')}</h3><p>{t('Whole-group holdout on: ', '完整群組保留欄位：')}<code>{esc(m['config']['group'])}</code>. {t('Zero group-ID overlap checks only the supplied IDs, not their physical independence.', '群組識別碼零重疊僅檢查提供的識別碼，不驗證其實際獨立性。')}</p><ul><li><strong>{t('Domain-heldout?', '是否為跨領域保留？')}</strong> {t('Separation is only along the supplied group column; other domain/site axes are not separately audited. If that column defines the intended domain boundary, its whole values are held out, but the scientific meaning of the IDs is not verified.', '僅沿指定的群組欄位隔離，未另稽核其他領域／場域軸。若該欄位定義預定的領域邊界，其完整值會被保留，但識別碼的科學意義並未經驗證。')}</li><li><strong>{t('Random-row comparison?', '是否比較隨機逐列切分？')}</strong> {t('Not run. The seed assigns whole groups to folds; it does not split rows within a group.', '未執行。種子將完整群組分配至各折，不會拆開同一群組內的列。')}</li></ul><p>{t('Different group IDs may still share a site, time period or another dependence axis. Review the grouping boundary before interpreting scores.', '不同群組識別碼仍可能共享場域、時段或其他相依軸。解讀分數前，請先確認分組邊界。')}</p>{singleton_note}</div>
<div class="scroll" role="region" aria-labelledby="comparison-table" tabindex="0"><table><caption id="comparison-table">{t('Held-out metrics for each entry', '各項目的保留集指標')}</caption><thead><tr><th scope="col">{t('Entry', '項目')}</th><th scope="col">{t('Group Brier', '群組 Brier')}</th><th scope="col">{t('Row Brier', '逐列 Brier')}</th><th scope="col">{t('Group clipped log loss', '群組截限對數損失')}</th><th scope="col">{t('Accuracy', '準確率')}</th><th scope="col">{t('False negatives', '偽陰性')}</th><th scope="col">{t('False positives', '偽陽性')}</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>{comparison}</p></section>
<section id="difficult-groups" aria-labelledby="groups-heading"><h2 id="groups-heading">{t('Where it struggles', '表現較差的群組')}</h2><p>{t('Up to 15 groups with largest selected Brier. Small-group extremes are unstable. Recall 1 is undefined when a group has no observed label 1; specificity 0 is undefined with no observed label 0. This is not zero performance or a missing prediction. No group is removed; all groups remain in summary.json.', '顯示 selected Brier 最高的至多 15 組。小群組的極端值不穩定。若群組沒有實際標籤 1，召回率 1 未定義；沒有實際標籤 0 時，特異度 0 未定義。這不代表表現為零或缺少預測。沒有群組被刪除；全部群組保留於 summary.json。')}</p><div class="scroll" role="region" aria-labelledby="groups-table" tabindex="0"><table><caption id="groups-table">{t('Selected procedure: highest-error groups', '模型選擇程序：誤差最高的群組')}</caption><thead><tr><th scope="col">{t('Group', '群組')}</th><th scope="col">{t('Rows', '列數')}</th><th scope="col">{t('True 0 / 1', '實際 0 / 1')}</th><th scope="col">Brier</th><th scope="col">{t('Clipped log loss', '截限對數損失')}</th><th scope="col">FN / FP</th><th scope="col">{t('Recall 1', '召回率 1')}</th><th scope="col">{t('Specificity 0', '特異度 0')}</th></tr></thead><tbody>{''.join(worst)}</tbody></table></div><p class="muted">{t('FN / FP = false negatives / false positives. On small screens, scroll tables sideways; keyboard users can focus each table and use the arrow keys.', 'FN / FP = 偽陰性 / 偽陽性。小螢幕可左右捲動表格；鍵盤使用者可將焦點移至表格並使用方向鍵。')}</p></section>
<section id="split-audit" aria-labelledby="audit-heading"><h2 id="audit-heading">{t('Split audit', '切分稽核')}</h2><p>{t('No group may occur in both training and test within a fold. Candidate selection uses inner training folds only; exact group assignments are in manifest.json.', '同一折的訓練集與測試集不得包含相同群組。候選模型僅於內層訓練折中選擇；完整群組分配記錄於 manifest.json。')}</p><div class="scroll" role="region" aria-labelledby="audit-table" tabindex="0"><table><caption id="audit-table">{t('Training and test split counts', '訓練與測試切分數量')}</caption><thead><tr><th scope="col">{t('Fold', '折次')}</th><th scope="col">{t('Train / test groups', '訓練 / 測試群組')}</th><th scope="col">{t('Inner-selected candidate', '內層選定候選模型')}</th><th scope="col">{t('Overlap', '重疊群組')}</th></tr></thead><tbody>{audits}</tbody></table></div></section>
<section id="run-details" aria-labelledby="run-heading"><h2 id="run-heading">{t('Limits and run record', '限制與執行記錄')}</h2><p class="muted">{t('Names and recorded diagnostics below are kept as supplied, not translated.', '名稱與下列記錄診斷保留原文，不進行翻譯。')}</p><ul class="diagnostics">{warnings}</ul><p>{t('Package version: ', '套件版本：')}{esc(m.get('groupbench_version', 'not recorded'))}<br>{t('Observed package-source digest: ', '觀察到的套件原始碼摘要：')}<code>{source_text}</code></p><p>{t('Self-reported source bytes; not proof of loaded code, build, hardware, reproducibility, authenticity or safety. Full record in manifest.json; checksums do not authenticate this metadata.', '僅為自行記錄的原始碼位元組；不證明載入的程式碼、建置、硬體、可重現性、真實性或安全性。完整記錄見 manifest.json；檢查碼不認證此中繼資料。')}</p><p>{t('Protocol: ', '流程：')}{esc(m['protocol'])}<br>{t('Seed: ', '隨機種子：')}{m['config']['random_state']} · {m['fit_budget']['total_fits']} {t('pipeline fits · one thread', '次管線擬合 · 單執行緒')}<br>{t('Run key: ', '執行識別碼：')}<code>{esc(m['run_key'])}</code><br>{t('Generated locally: ', '本機產生時間：')}{esc(m['created_at_utc'])}</p><p>{t('Synthetic examples show software behavior, not real-world accuracy. No model files, external assets, telemetry or automatic physical decisions. Checksums verify bytes only, not scientific correctness or authenticity.', '合成範例僅展示軟體行為，不代表真實世界準確度。不含模型檔、外部資源、遙測或自動實體決策。檢查碼僅驗證位元組，不驗證科學正確性或真實性。')}</p></section>
<aside class="evidence" aria-labelledby="evidence-heading"><h2 id="evidence-heading">{t('Keep the evidence close', '保留完整證據')}</h2><div class="evidence-links"><a href="summary.json"><strong>{t('All metrics', '所有指標')}</strong><small>summary.json</small></a><a href="manifest.json"><strong>{t('Run record', '執行記錄')}</strong><small>manifest.json</small></a><a href="predictions.csv"><strong>{t('Row probabilities and labels', '逐列機率與標籤')}</strong><small>predictions.csv</small></a><a href="checksums.json"><strong>{t('File checksums', '檔案檢查碼')}</strong><small>checksums.json</small></a></div><p>{t('These files sit beside this report. Keep the bundle together; review group IDs, observed labels and probabilities before sharing.', '這些檔案位於報告旁。請保留完整檔案組；分享前先檢查群組識別碼、實際標籤與機率。')}</p></aside>
<footer>{t('Powered by the GroupBench engine. This offline language switch uses only native controls and CSS; it does not change scores, saved data or the CLI language.', '由 GroupBench 引擎產生。離線語言切換僅使用原生控制元件與 CSS，不改變分數、已儲存資料或命令列語言。')}</footer></main></div></body></html>'''


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
