<p align="right"><a href="README.md">English</a> · <strong>繁體中文</strong></p>

# ML Super Calculator

**保留整批資料做評估，再在本機預測新資料。**

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/readme-workflow.zh-TW.mobile.svg">
  <img src="docs/assets/readme-workflow.zh-TW.svg" alt="GroupBench 工作流程：準備資料表、以保留群組比較模型、檢查報告">
</picture>

材料、半導體與製程的小型資料集，往往不是彼此獨立的資料列。計算器的第一個引擎 GroupBench 會在比較回歸基線時，將你指定的同批資料保留在同一側。

訓練、報告和預測都留在你的電腦。不需帳號，不提供託管運算 API，也不下載模型權重或傳送遙測資料。

> **已發布預覽版：0.2.0a3。** 從 [a3 發布頁](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a3)安裝已審核的輸入／輸出防護修正，安裝前核對同版 SHA-256 紀錄。[變更與安裝細節（英文）](docs/RELEASE_NOTES_0.2.0a3.md)。先前的 a2 檔案保留不變，可供回退。

[下載預發布版](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a3) · [選擇版本](#先選擇要使用的版本) · [試跑範例](#試跑範例) · [使用自己的資料](#使用自己的資料) · [閱讀結果](#閱讀結果) · [範例](#範例) · [完整技術參考（英文）](docs/USAGE_REFERENCE.md)

> **下方示範的下載：** v0.2.0a3，含須明確選用的二元分類流程。這是以 sklearn 為基礎、原創的模型選擇與評估工具，不是預訓練基礎模型、完整 AutoML 系統或 TabPFN 的同等替代品。合成範例展示用法，不代表真實資料的準確度。

## 試跑範例

### 先選擇要使用的版本

- **已發布的 a3 下載：** `v0.2.0a3` 對應 [4bbfe99 原始碼](https://github.com/xinhi6716/ml-super-calculator/tree/4bbfe99ed55d2f9e6cb931d6f568af4fa4a887d3)。wheel 已包含迴歸與二元流程的輸出路徑檢查、已驗證 CSV 欄名保留，以及 Python API 特徵清單防護；模型家族與選模規則不變。
- **先前的 a2 下載：** [v0.2.0a2](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a2) 仍固定於 [20d17c1 快照](https://github.com/xinhi6716/ml-super-calculator/tree/20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a)，不包含上述四項修正。中間的 [73dc3f8 原始碼修正](https://github.com/xinhi6716/ml-super-calculator/tree/73dc3f81885a1f8d9acca14198bce7ebe6aad755) 保留為歷史紀錄；現在不必為了這些修正改用原始碼安裝。

回退 a2 時請使用另一個環境。從原始碼安裝時，除了套件版本，也記下 Git revision。下方固定 checkout 對應 a3 發布原始碼；之後 `main` 更新不會改動已發布檔案。

**安裝已發布的 wheel：** 下載 [0.2.0a3 wheel](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a3/ml_super_calculator-0.2.0a3-py3-none-any.whl)，並與 [SHA256SUMS.txt](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a3/SHA256SUMS.txt) 核對。在[全新的 Python 虛擬環境](docs/RELEASE_NOTES_0.2.0a3.md#install-a-verified-a3-wheel)中，切換至下載目錄後執行：

```bash
python -m pip install ./ml_super_calculator-0.2.0a3-py3-none-any.whl
mlcalc demo --out runs/demo
```

安裝時可能從設定的套件索引下載相依套件；wheel 並未包含這些相依套件。模型運算仍在本機執行。這是 GitHub Release 下載，不是 PyPI 或 GitHub Packages 發布。下方使用範例檔案的指令，請搭配 [原始碼封存檔](https://github.com/xinhi6716/ml-super-calculator/releases/download/v0.2.0a3/ml_super_calculator-0.2.0a3.tar.gz) 或 checkout。

**或從 checkout 安裝：**

若要取得新的固定版本 checkout，請選擇尚不存在的目的目錄；下列指令不會更新既有 checkout。需要 Git 與 Python。請使用全新的 Python 虛擬環境；[既有的 POSIX 設定範例](docs/RELEASE_NOTES_0.2.0a3.md#install-a-verified-a3-wheel)示範如何建立及啟用環境，再接著執行下方的 checkout 安裝指令。

```bash
git clone https://github.com/xinhi6716/ml-super-calculator.git mlcalc-source
cd mlcalc-source
git checkout --detach 4bbfe99ed55d2f9e6cb931d6f568af4fa4a887d3
git rev-parse HEAD
python -m pip install -e .
mlcalc demo --out runs/demo
```

開啟 `runs/demo/report.html`。報告是獨立的本機檔案，不含遠端資源或腳本。範例包含 **24 批、288 筆完全合成的觀測值**。

宣告支援 Python 3.10+。目前開發環境已驗證 Python 3.12.14、NumPy 2.3.5、pandas 2.2.3 與 scikit-learn 1.8.0；尚未完成跨版本 CI 測試矩陣。`groupbench` 是 `mlcalc` 的別名。

<details>
<summary>已備妥相依套件？不安裝、不連網也能試跑。</summary>

```bash
PYTHONPATH=src python -m groupbench demo --out runs/demo
```

</details>

## 使用自己的資料

先準備包含三類欄位的 CSV：

| 欄位 | 應放什麼 |
| --- | --- |
| 目標 | 有限的數值結果，例如實測材料性質。 |
| 群組 | 不能跨越評估邊界的批次、試片、晶圓或來源實驗。 |
| 特徵 | 在量測結果**之前**就能取得的數值或類別輸入。允許特徵缺值。 |

```bash
mlcalc run measurements.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/mobility
```

目標與群組欄位會自動排除於預測特徵之外。請用 `--features` 或 `--features-file` 也排除識別碼、結果發生後的量測與備註。若未指定允許清單，**其他所有欄位都會成為特徵**；計算器無法判斷哪些輸入在科學上合理。

想先用現成輸入？可試 [合成訓練 CSV](examples/synthetic_materials.csv) 與 [新製程設定](examples/new_process_settings.csv)。這些數值都是虛構的。

### 先查看計畫，再開始訓練

```bash
mlcalc plan run examples/synthetic_materials.csv --target response --group batch
mlcalc plan predict examples/synthetic_materials.csv examples/new_process_settings.csv --target response --group batch
```

計畫會驗證輸入，將分割大小、候選設定與預計擬合次數，以 JSON 印至標準輸出；不擬合模型、不寫入結果。若每次擬合都成功，預設評估需 **105 次擬合**，預測需 **21 次擬合**。這不是執行時間估算，也不是硬性的運算配額。[計畫細節（英文）](docs/USAGE_REFERENCE.md#preview-the-work-before-fitting)

### 評估後，再預測新資料

第二份 CSV 必須**恰好包含選定的特徵欄位**，不含目標與群組欄位：

```bash
mlcalc predict measurements.csv new_process_settings.csv \
  --target mobility \
  --group wafer_batch \
  --features temperature_c,pressure_mpa,duration_min,material_family \
  --out runs/new-predictions
```

預測會以全部已標記資料進行群組內層選模，並在本機重新擬合。輸出使用目標原單位，不是新的測試集表現估計。不載入或儲存可執行模型，CLI 每次呼叫都會重新擬合。簡單的輸入支持範圍標記能提示未見過的值，但**不是不確定性區間，也不是分布外檢定**。[預測與輸入支持範圍細節（英文）](docs/USAGE_REFERENCE.md#predict-new-rows)

## 閱讀結果

| 檔案 | 用途 |
| --- | --- |
| `report.html` | 比較模型，檢查誤差最大的群組。 |
| `summary.json` | 查看整體、各群組與各折的指標。 |
| `predictions.csv` | 逐列檢查真值、預測與所屬折。 |
| `manifest.json` | 核對分割、候選選擇、版本、種子、雜湊與診斷。 |
| `checksums.json` | 核對其他四份評估檔案的位元組。 |

**僅目前原始碼提供：** 結果 manifest 與評估報告另記錄
[觀察到的套件原始碼摘要](docs/PACKAGE_SOURCE_IDENTITY.md) 或無法取得狀態。
這是自行記錄的原始碼位元組中繼資料，不含於封存 a3 下載；
不認證載入的程式碼、建置或可重現性。

目前原始碼的報告也說明[所選切分可回答的問題（英文）](docs/PROTOCOL.md#read-the-split-scope)：保留群組不自動等於保留領域，也沒有另外執行隨機逐列比較。當記錄中的每個群組都只有一列時，會顯示提醒。此報告說明不含於封存 a3，不改變分數或切分規則。

群組超過 15 個？目前原始碼的報告可在最高誤差表下選擇 **View all N groups**（二元繁中介面為「查看全部 N 個群組」），依識別碼順序查看各組已存指標。原本的前 15 組摘要仍保留，不重新計算分數。封存 a3 報告維持不變。

完整表格不會為縮檔而截掉群組：二元輸出會拒絕超過 64 MiB 的單一檔案，`verify` 也有相同的單檔上限。

回歸流程使用相同分割與前處理，比較 **Dummy 平均值、Ridge 與 Extra Trees**。超參數只在各外折的訓練部分內選擇。`selected` 評估的是整個選模程序，不是看完外層測試分數後才挑勝者。不同模型家族**沒有取得相同運算預算**。

Group MAE 給每個群組相同權重；row MAE 與 RMSE 合併所有觀測列計算。目標為常數或只有一列時，R² 為 `null`。不要反覆依外層排行榜調參，再把同一批分數當成無偏證據。[完整評估協定（英文）](docs/PROTOCOL.md)

```bash
mlcalc verify runs/demo
```

`verify` 不擬合或載入模型，只比對保存檔案與 checksum 的位元組。它需要 POSIX 的 no-follow 檔案存取能力，不支援的平台會被拒絕。通過不代表來源真實或科學正確。另一個 `validate-prediction-bundle` 指令可檢查受支援的預測 manifest／CSV 關係。[驗證限制（英文）](docs/USAGE_REFERENCE.md#verify-an-existing-bundle-offline) · [預測封包規格（英文）](docs/PREDICTION_BUNDLE_SCHEMA.md)

**分享前先檢查：** 報告含群組識別碼、目標值、預測值與特徵名稱。不要把私人資料與執行目錄提交至公開倉庫。CSV 未針對試算表公式做清理；使用試算表軟體開啟或上傳前，請先確認內容。

## 範例

以下詳細範例文件目前以英文撰寫。

| 範例 | 展示內容 |
| --- | --- |
| [Python API](examples/python_api.py) | `benchmark`、`fit_predict` 與本機輸出。另有類 sklearn 估計器，見 [API 參考](docs/USAGE_REFERENCE.md#python-api)。 |
| [跨環境保留評估](examples/environment-heldout-comparison/README.md) | 五條固定流程、兩組預先指定特徵，以及訓練標籤打亂診斷。這是預測壓力測試，不是因果推論。 |
| [半導體串接範例](examples/semiconductor-heldout-walkthrough/README.md) | 虛構 I–V 量測經可信任的本機 Toolbox checkout 轉成元件特徵，再做批次保留、只用訓練資料擬合、不讀取保留元件目標值的預測，並另行計分。 |
| [因果設計演練](examples/causal-design-rehearsal/README.md) | 用虛構方程與成對介入真值演練調整變項的選擇。不是經校準的元件模型或真實資料因果估計器。 |
| [已保存的預算比較](docs/SAVED_BUDGET_COMPARISON.md) | 兩方法 × 兩預算的描述性比較，保留不利案例。不重新訓練，也不驗證來源。 |
| [已保存的區間評估](docs/SAVED_INTERVAL_EVALUATION.md) | 從已保存的上下界計算覆蓋率、寬度與區間分數。不產生或校準區間。 |

### 二元通過／失敗：須明確選用

```bash
mlcalc binary plan run examples/synthetic_binary_process.csv --target fail --group lot
mlcalc binary run examples/synthetic_binary_process.csv --target fail --group lot --out runs/binary
mlcalc binary predict examples/synthetic_binary_process.csv examples/new_binary_process.csv --target fail --group lot --out runs/binary-new
```

這條預發布流程使用數值 0/1 標籤，以群組平衡 Brier 分數，在先驗 Dummy、正則化 Logistic 與有界 Extra Trees 中選擇。每個訓練分割都須包含兩種標籤；只有單一類別的保留群組仍會保留。門檻固定為 p1 ≥ 0.5，未針對製造決策最佳化。二元流程的輸出目錄必須是新的。預設回歸指令即使收到 0/1 目標也仍做回歸。[二元流程協定與限制（英文）](docs/BINARY_PROTOCOL.md)

**目前原始碼的二元報告介面（不含於封存 a3 下載）：** 支援 English / 繁體中文切換、章節導覽與可用鍵盤橫向捲動的表格。切換不會改變分數或已儲存資料。[閱讀指南與版本範圍（英文）](docs/BINARY_PROTOCOL.md#read-the-offline-evaluation-report)

## 使用界線

- **謹慎選擇群組。** 批次標籤互斥，不代表未記錄的相依性已隔離；`batch × specimen` 組合標籤也不會自動隔離兩個交叉因素。目前沒有多軸或時間保留評估。
- **小群組可能改變排名。** 各折差異不是信賴區間；單一種子不是穩健性研究。用於重要決策前，應另行獨立驗證。
- **輸入留在本機，並控制大小。** 上限為 20,000 列、100 個特徵；CSV 必須是本機一般 UTF-8 檔案，每份最多 64 MiB。缺少目標、格式不完整的列、NUL 字元或無效分割都會失敗，不會默默修補。這些防護不是硬性的記憶體或時間限制。
- **注意類別缺值標記。** 相容性預設 `legacy-v1` 可能把字串 `__GROUPBENCH_MISSING__` 與缺值合併。選用 `distinct-v2` 可使用不碰撞的內部鍵。[編碼規格（英文）](docs/CATEGORICAL_ENCODING.md)
- **把預測與科學主張分開。** 不提供經校準的不確定性、因果推論、物理單位驗證、外推保證、基礎模型訓練或正式環境監控。合成範例不是經驗證的材料模擬器。

[完整技術參考（英文）](docs/USAGE_REFERENCE.md) 保留輸入、schema、算術、資源、輸出與相容性的詳細規格。文件範例不構成真實元件的準確度證據；有效的特徵 schema 也不代表沒有資料洩漏。

## 開發與貢獻

```bash
python -m unittest discover -s tests -v
python examples/python_api.py
```

測試使用標準函式庫 `unittest`。若尚未安裝 checkout，請在兩個指令前加上 `PYTHONPATH=src`。另請閱讀 [貢獻指南](CONTRIBUTING.md)、[路線圖](ROADMAP.md) 與 [安全說明](SECURITY.md)，目前皆為英文。

TabPFN 提供的是易用 `fit`／`predict` 介面與套件設計的參考，不是模型實作來源。本倉庫不包含 TabPFN 程式、權重、下載流程或 API 用戶端，也不宣稱具同等能力。[參考與授權說明（英文）](docs/REFERENCES.md)

原創專案程式與合成資料產生器採 [MIT](LICENSE) 授權；相依套件保留各自授權。路線圖描述預定工作，不代表功能已經交付。

<details>
<summary>在找舊版 README 的某個段落？</summary>

詳細段落已移至 [完整技術參考（英文）](docs/USAGE_REFERENCE.md)。英文首頁保留舊段落錨點，方便沿用原本的連結。

</details>
