<p align="right"><a href="README.md">English</a> · <strong>繁體中文</strong></p>

<h1 id="axiom-clotho" align="center">Axiom Clotho</h1>

<picture>
  <source media="(prefers-color-scheme: dark) and (max-width: 600px)" srcset="docs/assets/clotho-ritual-hero-dark-mobile.svg">
  <source media="(max-width: 600px)" srcset="docs/assets/clotho-ritual-hero-light-mobile.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/clotho-ritual-hero-dark.svg">
  <img src="docs/assets/clotho-ritual-hero-light.svg" width="100%" alt="Clotho 的紡錘在星空框景下收攏一條連續的線。">
</picture>

<p align="center"><strong>元件量測、群組模型、本機報告。</strong></p>

Clotho 將支援的 RRAM、FET 量測表整理成元件特徵，依你宣告的群組切分比較模型，再預測新元件的結果。也可直接使用整理好的表格。

**[Beta 0.3.0b7](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b7)** · [開始使用](#begin) · [量測流程](#thread) · [自己的表格](#tabular-start) · [報告](#evidence) · [使用限制](#limits)

### 選擇你的起點

| 手上有什麼 | 從這裡開始 |
| --- | --- |
| 還沒有資料 | [小型合成範例](#try-the-demo) |
| RRAM 或 FET 量測 | [量測工作流程](#thread) |
| 特徵資料表 | [分組建模](#tabular-start) |
| 已保存的預測 | [報告與評估](#evidence) |

<a id="begin"></a>
<a id="get-the-reviewed-beta-core"></a>
<a id="先選擇要使用的版本"></a>

## 開始使用

需要 Python 3.10 以上版本，建議使用新的環境。以下固定版本的原始碼包含範例檔案：

```bash
git clone https://github.com/xinhi6716/axiom-clotho.git clotho-source
cd clotho-source
git checkout --detach v0.3.0b7
python -m venv .venv
# POSIX；Windows 改用 .venv\Scripts\activate。
. .venv/bin/activate
python -m pip install -e .
clotho --version
```

安裝時可能下載建置與執行相依套件。Clotho 的運算在本機進行，不需要帳號、託管 ML API 或模型權重下載。發行套件名稱仍為 **`ml-super-calculator`**，並保留 `mlcalc`、`groupbench` 與既有 Python 匯入名稱。[安裝與相容性說明](docs/CLOTHO_BETA_MIGRATION.md)。

<a id="try-the-demo"></a>

### 產生第一份報告

```bash
clotho tabular demo --out runs/demo
```

開啟 **`runs/demo/report.html`**。指令會使用一份小型虛構表格擬合模型。每次範例請使用新的輸出路徑。舊有的 `clotho demo` 是另一個固定半導體演練；這裡請使用 `clotho tabular demo`。

<details>
<summary>改用發行版 wheel 安裝</summary>

從 [b7 發行頁](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b7)下載 wheel 與 `SHA256SUMS.txt`，先核對校驗碼。若環境已備妥相容的相依套件：

```bash
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b7-py3-none-any.whl
clotho --version
```

需要 `numpy>=1.24,<3`、`pandas>=2,<3`、`scikit-learn>=1.4,<2`、`threadpoolctl>=3,<4`。這是支援版本範圍，不代表每種組合都已測試；上面的指令不會下載缺少的套件。範例 CSV 請從同版原始碼封存檔解壓取得。目前不宣稱已發布至 PyPI。[完整安裝指南](docs/RELEASE_NOTES_0.3.0b7.md#installation-and-reproducible-example)。

</details>

<a id="thread"></a>
<a id="what-this-beta-completes"></a>
<a id="four-stage-quickstart"></a>

## 從量測到預測

**Prepare → Plan → Run → Score。** 訓練端點與日後用來評估預測的觀測值分開提供。

<picture>
  <source media="(prefers-color-scheme: dark) and (max-width: 600px)" srcset="docs/assets/clotho-ritual-process-dark-mobile.svg">
  <source media="(max-width: 600px)" srcset="docs/assets/clotho-ritual-process-light-mobile.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/clotho-ritual-process-dark.svg">
  <img src="docs/assets/clotho-ritual-process-light.svg" width="100%" alt="整理元件特徵與訓練端點、檢查計畫、擬合並預測，最後以日後觀測值計分，不重新擬合。">
</picture>

| 你的輸入 | 支援路線 |
| --- | --- |
| RRAM I–V 量測 | 在固定宣告條件下整理電阻特徵，每個元件一列。[RRAM 範例](examples/measurement-regression/README.md) |
| 標準格式的 FET 轉移曲線 | 每個 `(lot, wafer, device)` 一次宣告量測，擷取準則 VGS 與指定視窗的對數斜率。[FET 範例](examples/fet-device-regression/README.md) |
| 自己的量測檔案 | `clotho init` 建立空白 CSV 表頭、未完成的設定檔與檢查清單。[輸入設定指南](docs/WORKFLOW_INIT.md) |

```bash
clotho init --workflow rram --out my-measurements
# FET 輸入請改用 --workflow fet。
```

`init` 不會產生假量測、猜測單位或擬合模型。請先填妥檔案與必要宣告，再執行 `prepare`。

<details>
<summary>執行完整 RRAM 範例</summary>

從固定版本的 checkout 或同版原始碼封存檔執行，並使用新的輸出目錄：

```bash
clotho prepare \
  --input-dir examples/measurement-regression/fixtures/raw \
  --partitions examples/measurement-regression/fixtures/partitions.csv \
  --training-targets examples/measurement-regression/fixtures/training_targets.csv \
  --config examples/measurement-regression/fixtures/config.json \
  --output-dir clotho-prepared

clotho plan --prepared clotho-prepared
clotho run --prepared clotho-prepared --output-dir clotho-run

# 只有計分步驟接收預測群組日後的觀測值。
clotho score --run-dir clotho-run \
  --outcomes examples/measurement-regression/fixtures/later_outcomes.csv \
  --output-dir clotho-evaluation
```

虛構範例共有 **18 個元件：12 個訓練、6 個預測**，宣告的批次互不重疊。`plan` 顯示 **40 次預計擬合**，本身不訓練；`run` 擬合模型並保存預測；`score` 評估全部六個日後觀測值，**不增加擬合**。這些數量用來檢查流程，不代表真實元件準確度。[輸入、輸出與預期結果](examples/measurement-regression/README.md)。

</details>

<a id="tabular-start"></a>

## 使用自己的表格

指定數值目標、批次或晶圓等群組欄位，以及在觀測結果之前就能取得的特徵清單。支援數值、類別特徵與特徵缺值。

```bash
mlcalc plan run measurements.csv \
  --target mobility --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family

mlcalc run measurements.csv \
  --target mobility --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/mobility
```

計畫會檢查輸入，列出切分、候選設定與擬合次數，不訓練模型。迴歸比較 **Dummy 平均值、Ridge 與 Extra Trees**；前處理與選模都在訓練分割內進行。

請使用 `--features` 或 `--features-file` 排除識別碼、備註及結果發生後才取得的量測。若未指定清單，除了目標與群組，其他欄位都會成為特徵。CSV 請使用 UTF-8、逗號分隔與小數點；小數逗號不會自動轉換。

要預測新資料，第二份 CSV 必須恰好包含選定的特徵，不含目標與群組欄位。預測流程會用已標記訓練資料選模並重新擬合，不產生新的測試集分數。[預測指南](docs/USAGE_REFERENCE.md#predict-new-rows)。

0/1 結果請明確選用 `mlcalc binary`；預設指令仍做迴歸。二元路線比較 Dummy、Logistic 與 Extra Trees，決策門檻固定為 0.5。[二元流程](docs/BINARY_PROTOCOL.md) · [全部範例](examples/README.md) · [Python API](docs/USAGE_REFERENCE.md#python-api)。

<a id="evidence"></a>

## 閱讀結果

| 開啟檔案 | 查看內容 |
| --- | --- |
| `report.html` | 模型比較、各群組誤差與保存結果的連結。 |
| 量測 run：`prediction_rows.csv` | 尚未提供日後觀測值的元件預測。 |
| 量測 score：`report.html` | 將保存預測與日後觀測值分開評估。 |
| 表格 run：`summary.json`、`predictions.csv` | 指標、觀測值、預測值與各折分配。 |

表格報告提供 **English / 繁體中文** 切換；此 beta 的量測摘要頁仍為英文。語言切換不改動保存數值。分享報告或提交執行目錄前，請先檢查識別碼、目標與預測是否含私人資料。

manifest、校驗碼與驗證方式見[報告指南](docs/USAGE_REFERENCE.md#outputs)。`mlcalc verify` 不擬合模型，只檢查保存檔案的位元組；它需要 POSIX 的 no-follow 檔案存取。校驗通過不代表科學有效性或來源真實性。

<a id="limits"></a>
<a id="horizon"></a>

## 使用限制

- **群組由你宣告。** 隔開批次標籤，無法找出未記錄的相依性，也不證明能推廣至新領域。
- **量測支援有明確範圍。** RRAM、FET 路線各有指定格式與特徵，並非通用儀器匯入器。量測流程目前只支援迴歸。
- **預測需要獨立評估。** 不提供因果識別、經校準的不確定性、物理單位驗證或外推保證。虛構範例是教學資料。
- **輸入限制不等於運算配額。** 檔案、列數與特徵上限會拒絕不支援的輸入，但不是硬性記憶體或時間限制。[輸入與資源限制](docs/USAGE_REFERENCE.md#limits-to-understand-before-trusting-a-result)。

b7 更新 Clotho 圖像、報告呈現與文件；保留 b6 拒絕已擬合迴歸 ExtraTrees 中非有限數值的檢查。既有流程、模型、切分與指標維持不變。[版本說明](docs/RELEASE_NOTES_0.3.0b7.md)。更多領域模組仍屬[規劃工作](ROADMAP.md)。

## 下一步

[範例](examples/README.md) · [量測輸入](docs/CLOTHO_CSV_BETA.md) · [評估協定](docs/PROTOCOL.md) · [完整參考](docs/USAGE_REFERENCE.md) · [參與開發](CONTRIBUTING.md) · [安全說明](SECURITY.md)

上述技術指南目前以英文為主。英文首頁也保留了舊版段落連結。

<details>
<summary>舊版與回退</summary>

請依同版指南，使用獨立環境回退。歷史 tag 與下載檔保持不變：[b6](docs/RELEASE_NOTES_0.3.0b6.md)、[b5](docs/RELEASE_NOTES_0.3.0b5.md)、[b4](docs/RELEASE_NOTES_0.3.0b4.md)、[b3](docs/RELEASE_NOTES_0.3.0b3.md)、[b2](docs/RELEASE_NOTES_0.3.0b2.md)、[b1](docs/RELEASE_NOTES_0.3.0b1.md)、[a4](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.2.0a4)。新安裝請使用 b7；b1 有已知的封包清單路徑問題。

</details>

原創程式與合成資料產生器採 [MIT](LICENSE) 授權；相依套件保留各自授權。[參考來源](docs/REFERENCES.md) · [美術與字體聲明](docs/assets/CLOTHO_EDITORIAL_NOTICES.md)。

**Moirai** 系列 · [Axiom](https://github.com/xinhi6716) · [Chronos](https://github.com/xinhi6716/axiom-chronos) · [Atropos](https://github.com/xinhi6716/axiom-atropos)

