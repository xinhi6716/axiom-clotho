# Axiom Clotho 0.3.0b3 — FET device regression

Clotho b3 packages the FET measurement adapter alongside the existing RRAM and
grouped tabular workflows. The distribution remains `ml-super-calculator`; all
four console entrypoints and existing imports retain their names.

## What changes from b2

- Canonical FET transfer CSV → signed-current criterion VGS and descriptive
  windowed log-current slope → exact external endpoint joins → the same grouped
  ML core. The two measurement algorithms are pinned MIT components, not rewrites.
- One declared acquisition per composite physical device `(lot, wafer, device)`.
  Additional branches, conditions or recorded repeats reject rather than being
  averaged, selected by success, or treated as independent devices.
- Explicit `measurement-regression-v2` and versioned bundles, including a
  reversible identity map. Training/prediction grouping uses the corresponding
  physical-key prefixes; later outcomes enter only separate saved scoring.
- An original invented FET example with 105 observations, 12 training devices
  across four lots, and three prediction devices from a separate lot.

The accepted FET integration was already on `main` before b3. This prerelease
packages it under its own version: no new estimator, feature-extraction algorithm,
scientific dataset or dependency is added by the release step. RRAM v1 and its
saved bundles remain supported. Frozen b2, b1, a4 and earlier tags/assets are not
rewritten; b2 does not contain the FET adapter.

## Installation and reproducible example

Download the wheel, matching source archive and `SHA256SUMS.txt` from the
[b3 release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b3).
Compare the full downloaded-file digests with that release's checksum file. In a
separate environment with compatible dependencies already present:

```bash
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b3-py3-none-any.whl
clotho --version
mlcalc --version
groupbench --version
mlcalc-workflow --version
clotho components
```

All four version commands report `0.3.0b3`. This is a GitHub prerelease, not a PyPI
publication. `--no-deps` does not supply missing dependencies. Requirements remain
Python >=3.10, NumPy >=1.24,<3, pandas >=2,<3, scikit-learn >=1.4,<2 and
threadpoolctl >=3,<4; the complete version matrix is not certified.

The **wheel contains runtime adapters, pinned extractors and licenses**. The
**matching source archive contains the genuine invented CSV fixtures and their
generator**, tests and guides. Fixtures are not downloaded by the CLI. From the
extracted source archive or fixed b3 checkout, choose fresh output directories:

```bash
clotho prepare \
  --input-dir examples/fet-device-regression/fixtures/raw \
  --partitions examples/fet-device-regression/fixtures/partitions.csv \
  --training-targets examples/fet-device-regression/fixtures/training_targets.csv \
  --config examples/fet-device-regression/fixtures/config.json \
  --output-dir fet-prepared
clotho plan --prepared fet-prepared
clotho run --prepared fet-prepared --output-dir fet-run
clotho score --run-dir fet-run \
  --outcomes examples/fet-device-regression/fixtures/heldout_outcomes.csv \
  --output-dir fet-score
```

Expected FET software counts: **12 training + 3 prediction devices, 30 planned ML
pipeline fits, zero ML fits at saved scoring**. Preparation includes descriptive
log-slope least squares, but no ML model fit. These invented values do not establish
useful device prediction, a physical threshold, subthreshold swing or a causal
mechanism. Read the [FET contract](FET_DEVICE_REGRESSION.md) before adapting it.

For the unchanged resistance-feature route, follow the
[RRAM four-stage example](../examples/measurement-regression/README.md): 12 training
and six prediction devices, 40 planned fits, then separate saved scoring.

## Verification and limits

The accepted FET source integration passed **579 source tests with zero skips**
on Linux/Python 3.12, plus a separate **20-method independent adversarial review**.
The 27 focused FET tests are included in the 579, not an additional total. The
independent review covered composite identity, leakage boundaries, semantic
mapping checks and RRAM v1 compatibility; its engine serialization stubs performed
zero actual ML fits. These are inherited integration results, not a claim that
changing a version string repeats an experiment.

The release receipt records the exact b3 source pin, final test runs, deterministic
build checks, wheel RECORD/license inventory, isolated installed-wheel FET and
RRAM smoke results, and artifact SHA-256 values. Local artifacts and installation
checks do not establish a full remote-binary download/readback. No alternative
URL or browser route is used to bypass a blocked binary download.

**Native Windows/macOS, hosted CI and a minimum-dependency matrix remain pending.**
Windows receipt paths have only been simulated in tests. The wheel's
`py3-none-any` tag is not native-platform certification. No new hosted workflow,
PyPI publication, dependency download, account permission, or credential is part
of this release preparation. Scientific validity and real-device generalization
remain outside the synthetic software evidence.

Real bundles retain original transfer bytes, physical IDs, endpoints and
predictions. Keep them private unless sharing is separately authorized. There is
no telemetry, automatic upload, private-toolbox fetch or hosted ML service.

## 繁體中文：安裝與版本範圍

b3 將既有 FET v2 轉接器正式納入自己的發行套件；同時保留 RRAM v1、表格計算器
及原本的 CLI／匯入名稱。b2、b1、a4 的 tag 與資產不變，b2 本身不含 FET。

請下載 b3 的 wheel、相符原始碼封存檔及 `SHA256SUMS.txt`，核對完整校驗碼後，
於已備妥相依套件的獨立環境執行上方安裝指令。wheel 提供執行程式、固定來源的
萃取器與授權；原始碼封存檔提供虛構 CSV 範例及產生器。CLI 不會另行下載資料。
四個版本指令均應顯示 `0.3.0b3`；本次不是 PyPI 發布。

FET 範例每個 `(lot, wafer, device)` 僅接受一段完整宣告量測，以帶符號電流準則
對應的 VGS 與指定視窗的描述性對數斜率接上原有群組 ML 核心。12 個裝置用於訓練，
另 3 個用於預測；計畫擬合 30 次，獨立評分不重新擬合 ML。這不代表物理閾值、
次臨界擺幅、因果機制或真實裝置準確度已獲驗證。

驗證目前限 Linux、本機建置／安裝與虛構資料；原生 Windows／macOS、託管 CI、
最低相依版本組合及完整遠端二進位檔回讀尚未完成。正式發布收據另列精確來源、
測試範圍與資產校驗碼。含真實量測、身分與目標的封包仍應保留私有。
