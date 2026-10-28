# Axiom Clotho 0.3.0b4 — Blank-input setup

Clotho b4 includes blank-input setup alongside RRAM v1, FET v2
and grouped tabular workflows. The distribution remains `ml-super-calculator`;
`clotho`, `mlcalc-workflow`, `mlcalc` and `groupbench` remain available.

## What changes from b3

`clotho init --workflow rram|fet --out ...` creates header-only CSVs, an
intentionally unfinished configuration and a self-contained English/Traditional
Chinese checklist. It creates **no measurement rows, endpoints, predictions or
scientific results**. Units and scientific choices are not guessed; declarations
start false. Init needs only the Python standard library. An untouched scaffold
fails prepare before a prepared output is created.

Init first appeared in source commit
[`972e4dd`](https://github.com/xinhi6716/axiom-clotho/commit/972e4dddbec1ea266a4c2b8f6ad1bf20cca88b18).
This release makes it available in the b4 package. No new estimator, extraction
algorithm, schema, dependency, scientific dataset or model weight is introduced. The existing synthetic
fixtures and four-stage workflows remain unchanged. Frozen **b3 has FET but no
init**; b2 has neither FET nor init. All old tags and assets remain unchanged.

## Installation and reproducible example

Download the b4 wheel, matching source archive and `SHA256SUMS.txt` from the
[b4 release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b4).
Compare complete local-file SHA-256 values with that release's checksum file.
Use a separate environment with compatible dependencies already available:

```sh
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b4-py3-none-any.whl
clotho --version
mlcalc-workflow --version
mlcalc --version
groupbench --version
clotho components
clotho init --workflow rram --out my-measurements
# OR, using another fresh directory:
clotho init --workflow fet --out my-transfer-data
```

All four version commands report `0.3.0b4`. The offline install command does not
fetch missing dependencies. Requirements stay Python >=3.10, NumPy >=1.24,<3,
pandas >=2,<3, scikit-learn >=1.4,<2 and threadpoolctl >=3,<4. Declared bounds are
not a tested version matrix. This is a GitHub prerelease route, not a claim of
PyPI or GitHub Packages publication. Do not co-install an overlapping distribution.

Open the generated README before editing inputs. Choose a fresh output directory
with an existing parent, supply your own measurements and justified configuration,
and follow prepare → plan → run → score. Later outcomes belong only in scoring.
[The setup guide](WORKFLOW_INIT.md) details exact IDs, units and editable choices.
The status `READY_TO_EDIT_NOT_VALIDATED` means only that blank setup succeeded;
it is not an analysis-completion or scientific-validation receipt.

For complete invented examples, extract the matching b4 source archive and use
[RRAM](../examples/measurement-regression/README.md) or
[FET](../examples/fet-device-regression/README.md). The wheel includes runtime
code, pinned components and their licenses; the source archive also contains the
synthetic fixtures, generators, tests and documentation. The CLI downloads none.
RRAM uses seed 7, 12 training / 6 prediction devices and 40 planned pipeline fits;
FET uses seed 1729, 12 / 3 devices and 30 fits. Both use dummy/ridge; planning and
saved scoring perform zero ML fits. These are software examples, not settings
recommended for a real scientific study.

## Verification and limits

Runtime checks used Linux/Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0 and threadpoolctl 3.6.0:

- The author-run full suite passed **595 tests with zero skips**.
- A separate wheel installation passed all four CLI version checks and the
  **16 init tests**, including blank inputs, invalid configuration, existing
  paths, interruption and observed directory/status replacement.
- Installed synthetic workflows made **40 RRAM / 30 FET pipeline fits**.
  Init, prepare, plan and saved scoring made no ML fits; later outcomes were
  supplied only after prediction.
- Two clean local builds produced identical wheels and source archives after
  source-archive metadata normalization. Archive contents, wheel RECORD hashes
  and license notices were checked.

The earlier independent init review covered **11 adversarial test groups** and
**16 focused init tests** against a separately built wheel. It did not rerun the
full suite. These checks overlap; they are not additional tests to add to 595.
All workflow inputs were invented examples, so these results establish software
behavior rather than real-device prediction quality.

From the matching source checkout, the setup and documentation checks can be run
without model fitting or dependency downloads:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_workflow_init.py -v
python tools/check_doc_versions.py
```

Native Windows/macOS, Python 3.10, minimum-dependency combinations and hosted CI
remain unverified. A `py3-none-any` wheel does not certify native-platform behavior.
Verification covers local build and installation; downloaded release binaries
have not been independently byte-verified.

## Filesystem safety, licensing and privacy

Use **trusted, idle local directories**. Init rejects pre-existing outputs and
symlink components at preflight. Exclusive writes preserve competing files;
observed root/status replacement, changed payloads or unexpected inventory fail.
The status is updated only through its original still-open handle; no status file
is unlinked or replaced. Failed creation can leave partial output. Retry a fresh
name; do not edit a status to imply success. A failed command remains a failure
even if a status file appears ready.

This is **not a filesystem transaction or comprehensive TOCTOU defense**. Hostile
concurrent ancestor replacement can still cause new files outside the intended
path before detection.

Original code and invented fixtures remain MIT-licensed; bundled code, artwork
and typography retain their existing notices. Release files contain code,
documentation and invented examples, not user measurements or model weights.
Review filled inputs and reports before sharing: they can contain physical IDs,
measurements, targets and predictions. There is no telemetry, automatic upload
or hosted ML service.
Passing software tests does not establish useful real-device prediction, physical
validity, causal identification, independence or calibrated uncertainty.

## 繁體中文：版本與使用範圍

b4 提供 `clotho init` 空白輸入設定，同時保留 RRAM、FET 及表格流程。
凍結的 b3 有 FET、沒有 init；b2 兩者皆無。舊 tag 與資產均保持不變。
安裝前請核對發布頁提供的來源及完整資產校驗碼。

安裝相符的 b4 wheel 後，四個版本指令應顯示 `0.3.0b4`。init 只建立空白標頭、
未完成設定與雙語操作指南，不產生虛構量測、不猜單位、不代替你接受科學聲明。
先讀產生的 README，填入自己的量測與獨立目標，再 prepare、plan、run、score；
後續結果僅供 score。未填寫範本會在寫入 prepared 目錄前被拒絕。

驗證限 Linux／Python 3.12.14 與上述相依套件：作者完整測試通過 595 項、沒有略過；
安裝後的虛構 RRAM／FET 流程分別擬合 40／30 次，評分不重新擬合。先前獨立的
11 組與 16 項測試不等於獨立完整測試，也不與 595 項相加。原生 Windows／macOS、
Python 3.10、最低相依版本矩陣及託管 CI 尚未驗證；下載的發布二進位檔尚未完成
獨立逐位元組核驗。
只在可信且沒有其他寫入者的本機目錄使用。這不是檔案系統交易或完整 TOCTOU
防護：惡意並行替換上層目錄仍可能在偵測前造成外部新檔建立。真實輸入與輸出
保持私有，分享前請先檢查內容；虛構案例與軟體測試不代表科學驗證。
