# 選用分組模型 → 新資料預測 → 後續結果評分

[範例總覽](../README.md) · [English](README.md)

這是**全虛構資料的軟體操作示範**，不是科學效能基準。18 筆訓練資料分成
6 個虛構群組；3 筆新資料在評分時分成 2 組。`x`、`response` 使用任意虛構
單位，沒有真實物理量的意義。`kind` 是虛構類別；訓練資料的空值與新資料的
`new` 類別用來走過缺值／未見類別處理流程。「後續結果」其實預先編造，
不是真正後來量測、也不是盲測。

## 環境與範圍

請使用包含這些範例的對應原始碼版本，以及已具備 Clotho 相依套件的 Python
環境。選用模型與 `score-predictions` 是**目前原始碼的功能**，不要假設舊版
已安裝套件也提供。在專案根目錄先查看 `clotho tabular predict --help` 與
`clotho tabular score-predictions --help`。直接執行原始碼時，指令前加
`PYTHONPATH=src`，並將 `clotho tabular` 換成 `python -m groupbench`。
不需要額外模型套件。

以下指令使用 POSIX shell，從專案根目錄執行。請確保兩個輸出目錄
`runs/optional-later-evaluation`、`runs/optional-later-prediction` 是新路徑，
不要加 `--overwrite`。訓練步驟會產生檔案；規劃及評分只輸出 JSON。
不要把 shell 輸出重新導向任何輸入檔，因為 shell 可能在程式啟動前就覆寫它。

## 1. 預覽分組驗證預算，不訓練

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
clotho tabular plan run examples/optional-model-delayed-scoring/training.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --outer-splits 3
```

規劃會列出 30 次管線擬合，但此步驟實際擬合 0 次。兩個模型都明確選用，
不改變預設模型。`study` 用於分組切分，不是特徵。`group_median` 給訓練群組
相同權重，不會針對每個新群組建立查表預測。
詳見[選用模型說明](../../docs/OPTIONAL_REGRESSION_FAMILIES.md)。

## 2. 以留出訓練群組評估，執行 30 次微型管線擬合

```sh
clotho tabular run examples/optional-model-delayed-scoring/training.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --outer-splits 3 --out runs/optional-later-evaluation
```

開啟 `runs/optional-later-evaluation/report.html`。虛構分數只驗證軟體流程，
不能證明哪個模型在真實研究較好，也不應用後續結果檔反覆調參。

## 3. 預測三筆只有特徵的新資料，執行 10 次微型管線擬合

```sh
clotho tabular plan predict examples/optional-model-delayed-scoring/training.csv \
  examples/optional-model-delayed-scoring/new_features.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2
clotho tabular predict examples/optional-model-delayed-scoring/training.csv \
  examples/optional-model-delayed-scoring/new_features.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --out runs/optional-later-prediction
clotho tabular validate-prediction-bundle runs/optional-later-prediction
```

`predict` 只使用訓練資料選模並重新擬合，不會沿用步驟 2 的模型。
預測規劃列出 10 次擬合，規劃本身不擬合。`new_features.csv` 沒有標籤或群組。
保存的列位置從 0 起算：0 對應 x 1.5、1 對應 x 9.5、2 對應 x 20。
請保留這個對照；列位置不是實體元件識別碼。驗證預測產物只檢查結構與完整性，
不代表量測為真或資料獨立。

## 4. 對完整的後續結果評分，完全不重新訓練

```sh
clotho tabular score-predictions --run-dir runs/optional-later-prediction \
  --outcomes examples/optional-model-delayed-scoring/later_outcomes.csv
```

標頭必須是 `row_position,group,observed`。範例刻意使用 **2、0、1** 的順序，
按明確列位置對應，不依 CSV 列順序配對。每個預測位置必須恰好出現一次。
`queryA` 有 1 筆、`queryB` 有 2 筆；這是使用者提供的彙總群組，
不是程式推斷的訓練群組，也不是已驗證的實體身分。

成功時輸出 JSON：`fits_performed: 0`、3 筆評分、2 組、逐列與群組等權摘要，
以及確切輸入雜湊。單筆群組的 R² 是 null；逐列平均與群組等權平均可能不同。
評分讀取已保存的預測，不載入模型、不重讀訓練特徵、不重新擬合或校準，
不悄悄刪除資料，也不更動預測產物。缺漏／重複位置、無效數值或不完整資料群
會讓整次評分被拒絕。結構有效的預測產物仍可能不符合較嚴格的數值轉換限制；
詳見[完整評分契約](../../docs/SAVED_PREDICTION_SCORING.md)。

這個編造的資料群不是獨立研究證據。真實研究應在看結果前固定特徵順序、
列身分對照、群組定義與結果量測規範。雜湊證明讀入的位元組，不保證資料來源真實。
評分 JSON 包含使用者提供的群組、目標與預測值，分享前請檢查。
本範例不隨附執行產生的模型結果檔。
